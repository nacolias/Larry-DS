// test_mqa_pf.cpp - larry: the prompt attention kernels in isolation - dsv4pf::launch (fattn-dsv4-pf.hpp, one fused
// kernel) against fattn-dsv4.cpp's oneMKL path (Q -> f16, gemm_batch S, softmax, gemm_batch O: copied here as it is)
// and a naive reference with the same roundings (Q scaled to f16, S f32, P f16, O f32 sums).  DeepSeek-V4.1's prompt
// shapes: 64 MQA heads x 512, a 512-query span, M keys a query (window 128 + top-k compressed), sinks.
// Reports max |diff| / max |O| of both paths against the reference, and GPU times (event profiling, median of 20).
//   build: see build.sh (AOT bmg-g31, oneMKL SYCL BLAS)
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "fattn-dsv4-pf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static double ev_us(const sycl::event & e) {
    return (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
            e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-3;
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// fattn-dsv4.cpp's path (Q any strides -> here contiguous [b][h][d]); returns the four events' summed time
static double mkl_path(sycl::queue & q, const float * Q, const sycl::half * K, const sycl::half * mask, const float * sinks,
                       float * O, int n, int NH, int D, int M, float scale, sycl::half * qh, float * s, sycl::half * p) {
    namespace mkl = oneapi::mkl;
    const size_t total = (size_t) n * NH * D;
    sycl::event e1 = q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) { qh[id[0]] = (sycl::half) (Q[id[0]] * scale); });
    sycl::event e2 = mkl::blas::row_major::gemm_batch(q, mkl::transpose::nontrans, mkl::transpose::trans, NH, M, D, 1.0f, qh, D,
                                                      (int64_t) NH * D, K, D, (int64_t) M * D, 0.0f, s, M, (int64_t) NH * M, n);
    const int WGS = 256;
    sycl::event e3 = q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n * NH * WGS), sycl::range<1>(WGS)), [=](sycl::nd_item<1> it) {
        const int64_t row = (int64_t) it.get_group(0);
        const int64_t b = row / NH, h = row % NH;
        const int lid = (int) it.get_local_id(0);
        const float * sr = s + row * M;
        const sycl::half * mr = mask + b * M;
        float mx = sinks ? sinks[h] : -1e30f;
        for (int64_t m = lid; m < M; m += WGS) mx = sycl::fmax(mx, sr[m] + (float) mr[m]);
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float sum = 0.0f;
        for (int64_t m = lid; m < M; m += WGS) sum += sycl::exp(sr[m] + (float) mr[m] - mx);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        if (sinks) sum += sycl::exp(sinks[h] - mx);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        sycl::half * pr = p + row * M;
        for (int64_t m = lid; m < M; m += WGS) pr[m] = (sycl::half) (sycl::exp(sr[m] + (float) mr[m] - mx) * inv);
    });
    sycl::event e4 = mkl::blas::row_major::gemm_batch(q, mkl::transpose::nontrans, mkl::transpose::nontrans, NH, D, M, 1.0f, p, M,
                                                      (int64_t) NH * M, K, D, (int64_t) M * D, 0.0f, O, D, (int64_t) NH * D, n);
    e4.wait();
    return ev_us(e1) + ev_us(e2) + ev_us(e3) + ev_us(e4);
}

// the naive reference: one work-item a (query, head) row of S / O
static void ref_path(sycl::queue & q, const float * Q, const sycl::half * K, const sycl::half * mask, const float * sinks,
                     float * O, int n, int NH, int D, int M, float scale, float * s) {
    q.parallel_for(sycl::range<2>((size_t) n * NH, (size_t) M), [=](sycl::id<2> id) {
        const size_t row = id[0], j = id[1], b = row / NH;
        const float * qr = Q + row * D;
        const sycl::half * kr = K + (b * M + j) * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc += (float) (sycl::half) (qr[d] * scale) * (float) kr[d];
        s[row * M + j] = acc + (float) mask[b * M + j];
    }).wait();
    q.parallel_for(sycl::range<2>((size_t) n * NH, (size_t) D), [=](sycl::id<2> id) {
        const size_t row = id[0], d = id[1], b = row / NH, h = row % NH;
        const float * sr = s + row * M;
        float mx = sinks ? sinks[h] : -1e30f;
        for (int j = 0; j < M; ++j) mx = sycl::fmax(mx, sycl::fmax(sr[j], -1e30f));
        float sum = sinks ? sycl::exp(sinks[h] - mx) : 0.0f;
        for (int j = 0; j < M; ++j) sum += sr[j] > -1e29f ? sycl::exp(sr[j] - mx) : 0.0f;
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        float acc = 0.0f;
        for (int j = 0; j < M; ++j) {
            const float pj = sr[j] > -1e29f ? sycl::exp(sr[j] - mx) * inv : 0.0f;
            acc += (float) (sycl::half) pj * (float) K[(b * M + j) * D + d];
        }
        O[row * D + d] = acc;
    }).wait();
}

int main(int argc, char ** argv) {
    sycl::queue q { sycl::gpu_selector_v, sycl::property_list { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() } };
    std::printf("%s, local mem %zu KB\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                (size_t) q.get_device().get_info<sycl::info::device::local_mem_size>() / 1024);
    const int NH = 64, D = 512, n = argc > 1 ? std::atoi(argv[1]) : 512;
    const float scale = 1.0f / std::sqrt((float) D);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    for (int M : { 128, 384, 640, 600 }) {
        std::vector<float> hq((size_t) n * NH * D), hs(NH);
        std::vector<sycl::half> hk((size_t) n * M * D), hm((size_t) n * M);
        for (auto & x : hq) x = 2.0f * nd(rng);   // scores of a few units (the model's are similar)
        for (auto & x : hk) x = (sycl::half) nd(rng);
        for (auto & x : hs) x = 2.0f * nd(rng);
        for (int b = 0; b < n; ++b)
            for (int j = 0; j < M; ++j) {   // ~10 % masked; query 0 fully masked (all -inf: only the sink)
                const bool off = b == 0 || ud(rng) < 0.1f || (j < 128 && j > 64 + b);   // window rows beyond the query
                hm[(size_t) b * M + j] = off ? (sycl::half) -INFINITY : (sycl::half) 0.0f;
            }
        float * Q = sycl::malloc_device<float>(hq.size(), q);
        sycl::half * K = sycl::malloc_device<sycl::half>(hk.size(), q);
        sycl::half * Mk = sycl::malloc_device<sycl::half>(hm.size(), q);
        float * S = sycl::malloc_device<float>(hs.size(), q);
        float * O1 = sycl::malloc_device<float>((size_t) n * NH * D, q);
        float * O2 = sycl::malloc_device<float>((size_t) n * NH * D, q);
        float * O3 = sycl::malloc_device<float>((size_t) n * NH * D, q);
        sycl::half * qh = sycl::malloc_device<sycl::half>((size_t) n * NH * D, q);
        float * s = sycl::malloc_device<float>((size_t) n * NH * M, q);
        sycl::half * p = sycl::malloc_device<sycl::half>((size_t) n * NH * M, q);
        q.memcpy(Q, hq.data(), hq.size() * 4);
        q.memcpy(K, hk.data(), hk.size() * 2);
        q.memcpy(Mk, hm.data(), hm.size() * 2);
        q.memcpy(S, hs.data(), hs.size() * 4).wait();
        ref_path(q, Q, K, Mk, S, O3, n, NH, D, M, scale, s);
        std::vector<double> tm, tf;
        for (int r = 0; r < 21; ++r) tm.push_back(mkl_path(q, Q, K, Mk, S, O1, n, NH, D, M, scale, qh, s, p));
        dsv4pf::Args a;
        a.q = Q; a.q_h = D; a.q_b = (size_t) NH * D;
        a.k = K; a.k_b = (size_t) M * D;
        a.mask = Mk; a.m_b = (size_t) M;
        a.sinks = S; a.o = O2; a.M = M; a.n = n; a.scale = scale;
        for (int r = 0; r < 21; ++r) { sycl::event e = dsv4pf::launch(q, a); e.wait(); tf.push_back(ev_us(e)); }
        std::vector<float> o1((size_t) n * NH * D), o2(o1.size()), o3(o1.size());
        q.memcpy(o1.data(), O1, o1.size() * 4);
        q.memcpy(o2.data(), O2, o2.size() * 4);
        q.memcpy(o3.data(), O3, o3.size() * 4).wait();
        double mx = 0, d1 = 0, d2 = 0, d12 = 0;
        size_t bad = 0;
        for (size_t i = 0; i < o3.size(); ++i) {
            mx = std::max(mx, (double) std::fabs(o3[i]));
            d1 = std::max(d1, (double) std::fabs(o1[i] - o3[i]));
            d2 = std::max(d2, (double) std::fabs(o2[i] - o3[i]));
            d12 = std::max(d12, (double) std::fabs(o1[i] - o2[i]));
            if (!std::isfinite(o2[i])) ++bad;
        }
        double o0 = 0;   // query 0 is fully masked: only the sink -> 0
        for (size_t i = 0; i < (size_t) NH * D; ++i) o0 = std::max(o0, (double) std::fabs(o2[i]));
        std::printf("n %d M %d: max|O| %.3f  oneMKL path vs ref %.2e, fused vs ref %.2e, fused vs oneMKL %.2e (relative %.2e); "
                    "non-finite %zu; masked query max %.1e | oneMKL path %.0f us, fused %.0f us (%.2fx)\n",
                    n, M, mx, d1, d2, d12, d12 / mx, bad, o0, med(tm), med(tf), med(tm) / med(tf));
        for (void * x : { (void *) Q, (void *) K, (void *) Mk, (void *) S, (void *) O1, (void *) O2, (void *) O3, (void *) qh, (void *) s, (void *) p })
            sycl::free(x, q);
    }
    return 0;
}
