// fa_dec_bench.cpp - fattn-dsv4-dec.hpp (DeepSeek-V4.1 decode attention) against a CPU reference with the same f16
// keys, and its speed at the decode / verify shapes.
#include "fattn-dsv4-dec.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const int D = 512, NH = 64;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int n : {1, 5}) {
        for (int M : {128 + 160, 640}) {
            std::vector<float> Q((size_t) n * NH * D), K((size_t) n * M * D), mask((size_t) n * M), sinks(NH);
            for (auto & v : Q) v = nd(rng);
            for (auto & v : K) v = nd(rng);
            for (size_t i = 0; i < mask.size(); ++i) mask[i] = (i % 7 == 3) ? -INFINITY : 0.0f;
            for (auto & v : sinks) v = nd(rng);
            const float scale = 1.0f / std::sqrt((float) D);
            // reference: K rounded to f16 like the kernel
            std::vector<double> ref((size_t) n * NH * D, 0.0);
            for (int b = 0; b < n; ++b)
                for (int h = 0; h < NH; ++h) {
                    std::vector<double> s(M);
                    double mx = sinks[h];
                    for (int j = 0; j < M; ++j) {
                        double a = 0;
                        for (int d = 0; d < D; ++d)
                            a += (double) (Q[((size_t) b * NH + h) * D + d] * scale) * (double) (float) (sycl::half) K[((size_t) b * M + j) * D + d];
                        s[j] = std::isinf(mask[(size_t) b * M + j]) ? -1e300 : a;
                        mx = std::max(mx, s[j]);
                    }
                    double L = std::exp(sinks[h] - mx);
                    for (int j = 0; j < M; ++j) { s[j] = s[j] < -1e299 ? 0.0 : std::exp(s[j] - mx); L += s[j]; }
                    for (int j = 0; j < M; ++j)
                        for (int d = 0; d < D; ++d)
                            ref[((size_t) b * NH + h) * D + d] += s[j] / L * (double) (float) (sycl::half) K[((size_t) b * M + j) * D + d];
                }
            float * dq = sycl::malloc_device<float>(Q.size(), q);
            float * dk = sycl::malloc_device<float>(K.size(), q);
            float * dm = sycl::malloc_device<float>(mask.size(), q);
            float * ds = sycl::malloc_device<float>(NH, q);
            float * dout = sycl::malloc_device<float>((size_t) n * NH * D, q);
            float * part = sycl::malloc_device<float>(dsv4dec::scratch_floats(NH, M, n, 16), q);
            q.memcpy(dq, Q.data(), Q.size() * 4); q.memcpy(dk, K.data(), K.size() * 4);
            q.memcpy(dm, mask.data(), mask.size() * 4); q.memcpy(ds, sinks.data(), NH * 4).wait();
            dsv4dec::Args a;
            a.q = dq; a.q_h = D; a.q_b = (size_t) NH * D;
            a.k = dk; a.k_f16 = false; a.k_b = (size_t) M * D;
            a.mask = dm; a.m_f16 = false; a.m_b = M;
            a.sinks = ds; a.o = dout; a.D = D; a.NH = NH; a.M = M; a.n = n; a.scale = scale; a.part = part;
            auto run_cfg = [&](const char * name, auto fn) {
                fn(nullptr).wait();
                std::vector<float> got((size_t) n * NH * D);
                q.memcpy(got.data(), dout, got.size() * 4).wait();
                double num = 0, den = 0;
                for (size_t i = 0; i < got.size(); ++i) { num += (got[i] - ref[i]) * (got[i] - ref[i]); den += ref[i] * ref[i]; }
                for (int i = 0; i < 5; ++i) fn(nullptr);
                q.wait();
                const int it = 300;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) fn(nullptr);
                q.wait();
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
                sycl::event e1;
                sycl::event e2 = fn(&e1);
                e2.wait();
                auto ns = [](const sycl::event & e) {
                    return (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000.0;
                };
                const double rel = std::sqrt(num / den);
                std::printf("n %d M %4d %-16s rel %.2e %s, %6.1f us (partial %.1f, combine %.1f)\n", n, M, name, rel,
                            rel < 5e-3 ? "ok" : "FAIL", us, ns(e1), ns(e2));
            };
            run_cfg("KC32 HG16 half", [&](sycl::event * e) { return dsv4dec::launch_t<32, 16, sycl::half>(q, a, e); });
            run_cfg("KC32 HG16 float", [&](sycl::event * e) { return dsv4dec::launch_t<32, 16, float>(q, a, e); });
            run_cfg("KC16 HG16 float", [&](sycl::event * e) { return dsv4dec::launch_t<16, 16, float>(q, a, e); });
            run_cfg("KC16 HG8 float", [&](sycl::event * e) { return dsv4dec::launch_t<16, 8, float>(q, a, e); });
            run_cfg("KC16 HG16 half", [&](sycl::event * e) { return dsv4dec::launch_t<16, 16, sycl::half>(q, a, e); });
            run_cfg("KC32 HG32 half", [&](sycl::event * e) { return dsv4dec::launch_t<32, 32, sycl::half>(q, a, e); });
            {
                float * px = sycl::malloc_device<float>(dsv4dec::xmx_scratch_floats(NH, M, n), q);
                dsv4dec::Args ax = a;
                ax.part = px;
                run_cfg("XMX", [&](sycl::event * e) { return dsv4dec::launch_xmx(q, ax, e); });
                sycl::free(px, q);
            }
            for (void * p : {(void *) dq, (void *) dk, (void *) dm, (void *) ds, (void *) dout, (void *) part}) sycl::free(p, q);
        }
    }
    return 0;
}
