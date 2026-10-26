// xmx_var_bench.cpp - larry: where the XMX gate/up kernel's time goes (3 synthetic Q2_0 experts, E 5120, F 2304,
// 3 tokens): VAR 0 full, 1 B constant (weights still loaded, no decode / fragment writes), 2 decode + B constant (no
// fragment writes), 3 decode only (no fragment writes, no DPAS)
#include "xe_xmx_var.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
using namespace strata::v41::xe;
int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    const int E = 5120, F = 2304, NE = 24, nj = 3;
    const size_t NB = (size_t) F * (E / 64), mbytes = NB * 18;
    std::vector<uint8_t> h(2 * mbytes);
    std::mt19937 rng(1);
    for (auto& b : h) b = (uint8_t) rng();
    std::vector<uint8_t*> blobs;
    for (int e = 0; e < NE; ++e) { uint8_t* d = sycl::malloc_device<uint8_t>(h.size(), q); q.memcpy(d, h.data(), h.size()).wait(); blobs.push_back(d); }
    sycl::half* xt = sycl::malloc_device<sycl::half>((size_t) 8 * E, q);
    q.memset(xt, 0, (size_t) 8 * E * 2).wait();
    sycl::half* ht = sycl::malloc_device<sycl::half>((size_t) 8 * F * 8, q);
    std::vector<XeJob> jl(NE);
    for (int e = 0; e < NE; ++e) {
        jl[e].blob = blobs[e]; jl[e].up_off = (uint32_t) mbytes; jl[e].gu_type = kQ20; jl[e].d_type = kQ20; jl[e].nt = 3;
        for (int t = 0; t < 3; ++t) { jl[e].tok[t] = t; jl[e].w[t] = 1.f; }
    }
    XeJob* dj = sycl::malloc_device<XeJob>(NE, q);
    q.memcpy(dj, jl.data(), NE * sizeof(XeJob)).wait();
    const double bytes = 2.0 * mbytes * nj;
    auto bench = [&](auto fn, const char* nm) {
        for (int i = 0; i < NE / nj; ++i) fn(dj + i * nj);
        q.wait();
        const int it = 200;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) fn(dj + (i % (NE / nj)) * nj);
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        std::printf("%-40s %6.1f us = %4.0f GB/s\n", nm, us, bytes / us / 1e3);
    };
    bench([&](XeJob* j) { xmxvar::gate_up_x<10, 256, 0>(q, j, nj, xt, ht, E, F, 10.f); }, "VAR 0 full");
    bench([&](XeJob* j) { xmxvar::gate_up_x<10, 256, 1>(q, j, nj, xt, ht, E, F, 10.f); }, "VAR 1 B constant (loads only)");
    bench([&](XeJob* j) { xmxvar::gate_up_x<10, 256, 2>(q, j, nj, xt, ht, E, F, 10.f); }, "VAR 2 decode, B constant");
    bench([&](XeJob* j) { xmxvar::gate_up_x<10, 256, 3>(q, j, nj, xt, ht, E, F, 10.f); }, "VAR 3 decode only, no DPAS");
    bench([&](XeJob* j) { xmxvar::gate_up_x<10, 256, 4>(q, j, nj, xt, ht, E, F, 10.f); }, "VAR 4 B via local memory (col-major)");
    bench([&](XeJob* j) { xmxvar::gate_up_x<10, 256, 5>(q, j, nj, xt, ht, E, F, 10.f); }, "VAR 5 = 4 without the * 1");
    return 0;
}
