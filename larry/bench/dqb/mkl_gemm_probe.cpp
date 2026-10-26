// mkl_gemm_probe.cpp - larry: does oneMKL's USM gemm block the host, and how fast is it, for the shared-expert
// shapes of a prompt call (STRATA_SHEXP_TIER: m tokens x [2 nf | H] outputs)?  Per variant: the host time of the gemm
// call while the queue is still busy with a ~20 ms kernel queued ahead (a call that waits for the queue shows ~20 ms),
// and the GPU time of 20 back-to-back calls.  Variants: half x half -> float (what patch-shexptier.py uses) and
// half x half -> half.
//   icpx -fsycl -O2 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" mkl_gemm_probe.cpp \
//        -DMKL_ILP64 -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -ltbb -o mkl_gemm_probe
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./mkl_gemm_probe
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace mkl = oneapi::mkl;
using sycl::half;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int64_t H = 5120, nf = 512;
    struct Shape { int64_t m, n, k; const char* what; };
    const Shape shapes[] = {{1024, 2 * nf, H, "gate|up 1024 x 1024 x 5120"}, {1024, H, nf, "down 1024 x 5120 x 512"},
                            {4096, 2 * nf, H, "gate|up 4096 x 1024 x 5120"}, {4096, H, nf, "down 4096 x 5120 x 512"}};
    const size_t maxa = 4096 * H, maxb = (size_t) H * 2 * nf, maxc = 4096 * H;
    half* A = sycl::malloc_device<half>(maxa, q);
    half* B = sycl::malloc_device<half>(maxb, q);
    float* Cf = sycl::malloc_device<float>(maxc, q);
    half* Ch = sycl::malloc_device<half>(maxc, q);
    float* busy = sycl::malloc_device<float>(1 << 20, q);
    q.fill(A, half(0.01f), maxa).wait();
    q.fill(B, half(0.02f), maxb).wait();
    // a kernel of ~20 ms: every work-item spins on a dependent FMA chain
    auto spin = [&](int iters) {
        return q.parallel_for(sycl::range<1>(1 << 20), [=](sycl::id<1> i) {
            float x = (float) i[0] * 1e-7f;
            for (int t = 0; t < iters; ++t) x = x * 0.999999f + 1e-7f;
            busy[i[0]] = x;
        });
    };
    spin(1000).wait();
    double t0 = now_ms();
    spin(200000).wait();
    const double spin_ms = now_ms() - t0;
    const int iters = (int) (200000 * 20.0 / std::max(spin_ms, 0.1));
    t0 = now_ms();
    spin(iters).wait();
    std::printf("spin kernel: %.1f ms\n", now_ms() - t0);
    for (const Shape& s : shapes) {
        for (int v = 0; v < 2; ++v) {
            auto call = [&]() {
                if (v == 0)
                    mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans, s.m, s.n, s.k, 1.0f, A, s.k,
                                               B, s.k, 0.0f, Cf, s.n);
                else
                    mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans, s.m, s.n, s.k, half(1.0f), A,
                                               s.k, B, s.k, half(0.0f), Ch, s.n);
            };
            call();   // the first call (kernel selection / JIT)
            q.wait();
            // host time of a call behind a busy queue
            spin(iters);
            const double th0 = now_ms();
            call();
            const double th1 = now_ms();
            q.wait();
            const double tq = now_ms();
            // GPU time of 20 calls
            const double tg0 = now_ms();
            for (int i = 0; i < 20; ++i) call();
            q.wait();
            const double tg = (now_ms() - tg0) / 20;
            const double flop = 2.0 * s.m * s.n * s.k;
            std::printf("%-30s %-18s host call %7.3f ms (behind the busy queue; it ends %5.1f ms later)  gpu %6.3f ms = %6.1f TFLOPS\n",
                        s.what, v == 0 ? "half,half->float" : "half,half->half", th1 - th0, tq - th1, tg, flop / tg / 1e9);
        }
    }
    sycl::free(A, q); sycl::free(B, q); sycl::free(Cf, q); sycl::free(Ch, q); sycl::free(busy, q);
    return 0;
}
