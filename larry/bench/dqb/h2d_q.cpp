// h2d_q.cpp - larry: H2D copies (pinned source) on an in-order queue of one card - sizes 64 KB .. 4 MB, alone and
// between small kernels (the decode sequence's shape): device time per copy and per-copy wall
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
int main() {
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const size_t MB = 1 << 20;
    uint8_t* h = sycl::malloc_host<uint8_t>(16 * MB, q);
    uint8_t* d = sycl::malloc_device<uint8_t>(16 * MB, q);
    float* k = sycl::malloc_device<float>(1024, q);
    for (size_t i = 0; i < 16 * MB; ++i) h[i] = (uint8_t) i;
    q.memcpy(d, h, 16 * MB).wait();
    for (size_t sz : {64 * 1024ul, 128 * 1024ul, 256 * 1024ul, MB, 4 * MB}) {
        for (int mix = 0; mix < 2; ++mix) {
            double dev = 0, best = 1e30;
            const int n = 50;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) {
                if (mix) q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> j) { k[j] += 1.0f; });
                sycl::event e = q.memcpy(d + (i % 4) * 4 * MB, h + (i % 4) * 4 * MB, sz);
                if (mix) q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> j) { k[j] *= 0.5f; });
                e.wait();
                const double us = (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e3;
                dev += us; best = std::min(best, us);
            }
            q.wait();
            const double wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
            std::printf("%7zu KB %s: device %.1f us mean (best %.1f) = %.2f GB/s; wall per iteration %.1f us\n", sz / 1024,
                        mix ? "between kernels" : "alone          ", dev / n, best, sz / (dev / n) / 1e3, wall);
        }
    }
    return 0;
}
