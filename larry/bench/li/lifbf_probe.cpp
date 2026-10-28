// lifbf_probe.cpp - larry, 2026-10-28: the fused prompt indexer's two modes on one card (patch-lifbf16.py): mode 1 = the
// whole key table converted to fp16 a call + the fp16 kernel, mode 2 = the bf16 kernel on the keys as stored (only the
// queries converted).  512 queries (a span) of FP4-rounded values like the engine's (e2m1 x a power of two a 32-block),
// every key visible; the scores of both compared bit for bit, then the median time of each at 4K..512K keys.
// lif_kernel.inc = the kernel as lightning-indexer.cpp has it (LIF_HSUM .. lif_run_t).  Build: icpx -fsycl -O2 (AOT for
// bmg-g31) lifbf_probe.cpp.  Card: ONEAPI_DEVICE_SELECTOR.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace li_mx = sycl::ext::oneapi::experimental::matrix;
using li_bf16 = sycl::ext::oneapi::bfloat16;
using lif_C = li_mx::joint_matrix<sycl::sub_group, float, li_mx::use::accumulator, 8, 16>;
using queue_ptr = sycl::queue *;

#include "lif_kernel.inc"

static float fp4(std::mt19937_64 & rng) {   // an e2m1 value
    static const float v[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const float x = v[rng() % 8];
    return (rng() & 1) ? -x : x;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::mt19937_64 rng(77);
    const int64_t B = 512;
    std::vector<float> hq((size_t) B * 4096), hw((size_t) B * 32), hnv((size_t) B);
    for (int64_t i = 0; i < B * 128; ++i) {   // a 32-block shares its power of two
        const float s = std::ldexp(1.0f, (int) (rng() % 7) - 3);
        for (int j = 0; j < 32; ++j) hq[(size_t) (i * 32 + j)] = fp4(rng) * s;
    }
    for (auto & x : hw) x = 0.01f * (float) (rng() % 100);
    const int64_t maxk = 524288;
    std::vector<float> hk((size_t) maxk * 128);
    for (int64_t i = 0; i < maxk * 4; ++i) {
        const float s = std::ldexp(1.0f, (int) (rng() % 7) - 3);
        for (int j = 0; j < 32; ++j) hk[(size_t) (i * 32 + j)] = fp4(rng) * s;
    }
    float * dq = sycl::malloc_device<float>(hq.size(), q);
    float * dw = sycl::malloc_device<float>(hw.size(), q);
    float * dnv = sycl::malloc_device<float>(hnv.size(), q);
    li_bf16 * kb = sycl::malloc_device<li_bf16>((size_t) maxk * 128, q);
    sycl::half * kh = sycl::malloc_device<sycl::half>((size_t) maxk * 128, q);
    sycl::half * qh = sycl::malloc_device<sycl::half>((size_t) B * 4096, q);
    li_bf16 * qb = sycl::malloc_device<li_bf16>((size_t) B * 4096, q);
    float * o1 = sycl::malloc_device<float>((size_t) B * maxk, q);
    float * o2 = sycl::malloc_device<float>((size_t) B * maxk, q);
    q.memcpy(dq, hq.data(), hq.size() * 4);
    q.memcpy(dw, hw.data(), hw.size() * 4);
    {
        std::vector<li_bf16> t(hk.size());
        for (size_t i = 0; i < hk.size(); ++i) t[i] = (li_bf16) hk[i];
        q.memcpy(kb, t.data(), t.size() * 2).wait();
    }
    int fails = 0;
    for (int64_t NKV : { (int64_t) 4096, (int64_t) 8192, (int64_t) 16384, (int64_t) 32768, (int64_t) 65536, (int64_t) 131072,
                         (int64_t) 262144, (int64_t) 524288 }) {
        for (auto & x : hnv) x = (float) NKV;
        q.memcpy(dnv, hnv.data(), hnv.size() * 4).wait();
        const int qw = NKV > 131072 ? 16 : 8;
        const int64_t KR = NKV <= 16384 ? 128 : NKV <= 65536 ? 256 : 512;
        auto mode1 = [&]() {   // as li_fused mode 1: q and the whole key table to fp16, then the fp16 kernel
            q.parallel_for(sycl::range<1>((size_t) B * 4096), [=](sycl::id<1> i) { qh[i[0]] = (sycl::half) dq[i[0]]; });
            q.parallel_for(sycl::range<1>((size_t) NKV * 128), [=](sycl::id<1> i) { kh[i[0]] = (sycl::half) (float) kb[i[0]]; });
            lif_run_t<sycl::half>(qh, kh, 128, NKV, dw, 32, nullptr, 0, dnv, o1, NKV, B, qw, KR, &q);
        };
        auto mode2 = [&]() {   // as mode 2: q to bf16, the keys as stored
            q.parallel_for(sycl::range<1>((size_t) B * 4096), [=](sycl::id<1> i) { qb[i[0]] = (li_bf16) dq[i[0]]; });
            lif_run_t<li_bf16>(qb, kb, 128, NKV, dw, 32, nullptr, 0, dnv, o2, NKV, B, qw, KR, &q);
        };
        mode1(); mode2(); q.wait();
        std::vector<float> a((size_t) B * NKV), b((size_t) B * NKV);
        q.memcpy(a.data(), o1, a.size() * 4);
        q.memcpy(b.data(), o2, b.size() * 4).wait();
        size_t differ = 0;
        for (size_t i = 0; i < a.size(); ++i) differ += std::memcmp(&a[i], &b[i], 4) != 0;
        fails += differ != 0;
        auto tm = [&](auto f) {
            std::vector<double> t;
            for (int r = 0; r < 7; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                f();
                q.wait();
                t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(t.begin(), t.end());
            return t[3];
        };
        const double t1 = tm(mode1), t2 = tm(mode2);
        std::printf("%7lld keys: mode 1 (fp16 copies) %8.3f ms, mode 2 (bf16 in place) %8.3f ms (%.2fx); %zu of %zu scores differ\n",
                    (long long) NKV, t1, t2, t1 / t2, differ, a.size());
    }
    std::printf("%s\n", fails ? "SCORES DIFFER" : "scores the same bits");
    return fails ? 2 : 0;
}
