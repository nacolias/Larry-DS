// copy_probe.cpp - memset / H2D / D2H on the device (the copy engines), timed
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    const size_t n = 64 << 20;
    char* d = sycl::malloc_device<char>(n, q);
    std::vector<char> h(n, 1);
    auto t0 = std::chrono::steady_clock::now();
    q.memset(d, 0, n).wait();
    q.memcpy(d, h.data(), n).wait();
    q.memcpy(h.data(), d, n).wait();
    std::printf("memset + H2D + D2H of 64 MiB: %.1f ms, byte %d\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), (int) h[n / 2]);
    sycl::free(d, q);
}
