// clock_probe.cpp - sycl_ext_oneapi_clock on the B70: is the device clock there, its tick rate, a stamp kernel's cost
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <chrono>
#include <cstdio>
#include <thread>
namespace sx = sycl::ext::oneapi::experimental;
int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("aspect ext_oneapi_clock_sub_group: %d\n", (int) q.get_device().has(sycl::aspect::ext_oneapi_clock_device));
    uint64_t* b = sycl::malloc_device<uint64_t>(4096, q);
    uint64_t h[4096];
    auto stamp = [&](int i) { q.single_task([=] { b[i] = sx::clock<sx::clock_scope::sub_group>(); }); };
    for (int w = 0; w < 3; ++w) { stamp(0); q.wait(); }
    auto t0 = std::chrono::steady_clock::now();
    stamp(0); q.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stamp(1); q.wait();
    auto t1 = std::chrono::steady_clock::now();
    q.memcpy(h, b, 16).wait();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::printf("ticks %llu over %.0f us -> %.3f ticks/us\n", (unsigned long long) (h[1] - h[0]), us, (h[1] - h[0]) / us);
    const int N = 2000;
    auto s0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) stamp(i);
    auto s1 = std::chrono::steady_clock::now();
    q.wait();
    q.memcpy(h, b, N * 8).wait();
    std::printf("stamp: host submit %.2f us each; device spacing %.2f ticks each\n",
                std::chrono::duration<double, std::micro>(s1 - s0).count() / N, (double) (h[N - 1] - h[0]) / (N - 1));
    return 0;
}
