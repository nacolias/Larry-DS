// wake_probe.cpp - does an idle B70 start work late?  After an idle gap of G us, submit a one-lane kernel that sets a
// pinned host flag; time submit -> flag seen on the host (busy-polled).  Per gap: mean / median / max over 200.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    volatile int* f = (volatile int*) sycl::malloc_host<int>(1, q);
    for (int w = 0; w < 50; ++w) { *f = 0; q.single_task([=] { *f = 1; }); while (!*f) {} }
    for (int gap : {0, 50, 100, 200, 400, 700, 1000, 3000, 10000}) {
        std::vector<double> v;
        for (int i = 0; i < 200; ++i) {
            const double t0 = now_us();
            while (now_us() - t0 < gap) {}
            *f = 0;
            const double ts = now_us();
            q.single_task([=] { *f = 1; });
            while (!*f) {}
            v.push_back(now_us() - ts);
        }
        std::sort(v.begin(), v.end());
        double m = 0; for (double x : v) m += x;
        std::printf("idle %5d us: submit -> flag mean %6.1f us, median %6.1f, p90 %6.1f, max %6.1f\n", gap, m / v.size(), v[v.size() / 2], v[v.size() * 9 / 10], v.back());
    }
    return 0;
}
