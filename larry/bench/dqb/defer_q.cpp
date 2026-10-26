// defer_q.cpp - larry: does a copy on an in-order queue start when submitted, or only once the next command comes?
// copy (H2D pinned / D2D) or a small kernel, then the host idles 500 us, then a kernel: submit -> start of each.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <thread>
static void spin_us(double us) { auto t0 = std::chrono::steady_clock::now(); while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < us) {} }
int main() {
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const size_t B = 128 * 1024;
    uint8_t* h = sycl::malloc_host<uint8_t>(B, q);
    uint8_t* d = sycl::malloc_device<uint8_t>(B, q);
    uint8_t* d2 = sycl::malloc_device<uint8_t>(B, q);
    float* k = sycl::malloc_device<float>(1024, q);
    q.memset(d, 0, B).wait();
    const char* nm[5] = {"H2D 128 KB", "D2D 128 KB", "kernel    ", "H2D + tiny kernel right after", "copy KERNEL reading host 128 KB"};
    for (int mode = 0; mode < 5; ++mode) {
        double s2s = 0, s2s2 = 0;
        const int n = 20;
        for (int i = 0; i < n; ++i) {
            sycl::event e1;
            if (mode == 0 || mode == 3) e1 = q.memcpy(d, h, B);
            else if (mode == 1) e1 = q.memcpy(d2, d, B);
            else if (mode == 2) e1 = q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> j) { k[j] += 1.0f; });
            else {
                const uint32_t* hs = (const uint32_t*) h;
                uint32_t* dd = (uint32_t*) d;
                e1 = q.parallel_for(sycl::range<1>(B / 16), [=](sycl::id<1> j) {
                    const sycl::vec<uint32_t, 4> v = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(hs + 4 * j[0]);
                    *reinterpret_cast<sycl::vec<uint32_t, 4>*>(dd + 4 * j[0]) = v;
                });
            }
            if (mode == 3) q.single_task([=] { k[0] += 1.0f; });
            spin_us(500);
            sycl::event e2 = q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> j) { k[j] *= 0.5f; });
            e2.wait();
            auto P = [](sycl::event& e, auto info) { return (double) e.get_profiling_info<decltype(info)::value>(); };
            const double sub1 = e1.get_profiling_info<sycl::info::event_profiling::command_submit>() / 1e3;
            const double st1 = e1.get_profiling_info<sycl::info::event_profiling::command_start>() / 1e3;
            const double sub2 = e2.get_profiling_info<sycl::info::event_profiling::command_submit>() / 1e3;
            const double st2 = e2.get_profiling_info<sycl::info::event_profiling::command_start>() / 1e3;
            (void) P;
            s2s += st1 - sub1; s2s2 += st2 - sub2;
        }
        std::printf("%s: submit -> start %.1f us (the kernel 500 us later: %.1f us)\n", nm[mode], s2s / n, s2s2 / n);
    }
    return 0;
}
