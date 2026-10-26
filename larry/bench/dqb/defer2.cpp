// defer2.cpp - larry: the "kernel after a host copy starts late" effect - H2D / D2H (pinned and pageable host), gap
// before the next kernel 0 / 20 / 100 / 500 us: submit -> start of that kernel
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
static void spin_us(double us) { auto t0 = std::chrono::steady_clock::now(); while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < us) {} }
int main() {
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const size_t B = 4096;
    uint8_t* h = sycl::malloc_host<uint8_t>(B, q);
    std::vector<uint8_t> pg(B);
    uint8_t* d = sycl::malloc_device<uint8_t>(B, q);
    float* k = sycl::malloc_device<float>(1024, q);
    q.memset(d, 0, B).wait();
    const char* nm[4] = {"H2D pinned  ", "D2H pinned  ", "H2D pageable", "D2H pageable"};
    for (int mode = 0; mode < 4; ++mode) {
        for (double gap : {0.0, 20.0, 100.0, 500.0}) {
            double s2s = 0;
            const int n = 30;
            for (int i = 0; i < n; ++i) {
                q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> j) { k[j] += 1.0f; });
                if (mode == 0) q.memcpy(d, h, B);
                else if (mode == 1) q.memcpy(h, d, B);
                else if (mode == 2) q.memcpy(d, pg.data(), B);
                else q.memcpy(pg.data(), d, B);
                spin_us(gap);
                sycl::event e2 = q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> j) { k[j] *= 0.5f; });
                e2.wait();
                s2s += (e2.get_profiling_info<sycl::info::event_profiling::command_start>() - e2.get_profiling_info<sycl::info::event_profiling::command_submit>()) / 1e3;
            }
            std::printf("%s 4 KB, kernel %5.0f us later: submit -> start %.1f us\n", nm[mode], gap, s2s / n);
        }
    }
    return 0;
}
