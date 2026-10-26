// alloc_probe.cpp - larry: sycl::malloc_device of 1 GiB slabs on card 0 in a context of all four cards (the tier's):
// time per allocation and the host's "hidden" memory (MemTotal - MemAvailable - AnonPages) after allocating, touching
// (memset) and freeing them
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
static long hidden_mib() {
    std::ifstream f("/proc/meminfo");
    std::string k; long v, t = 0, a = 0, p = 0; std::string u;
    while (f >> k >> v >> u) { if (k == "MemTotal:") t = v; if (k == "MemAvailable:") a = v; if (k == "AnonPages:") p = v; }
    return (t - a - p) / 1024;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 8;
    const int single = argc > 2 ? std::atoi(argv[2]) : 0;   // 1: a context of card 0 only
    std::vector<sycl::device> devs;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) devs.push_back(d);
    sycl::context ctx = single == 1 ? sycl::context(devs[0]) : single == 2 ? devs[0].get_platform().khr_get_default_context() : sycl::context(devs);
    sycl::queue q(ctx, devs[0], sycl::property::queue::in_order());
    std::vector<sycl::queue> peers;
    if (!single) for (size_t i = 0; i < devs.size(); ++i) peers.emplace_back(ctx, devs[i], sycl::property::queue::in_order());
    std::vector<float*> pk;
    for (auto& pq : peers) pk.push_back(sycl::malloc_device<float>(1024, pq));
    auto peer_work = [&]() {   // a kernel on every card (the engine's cards are all busy)
        for (size_t i = 0; i < peers.size(); ++i) { float* k = pk[i]; peers[i].parallel_for(sycl::range<1>(1024), [=](sycl::id<1> j) { k[j] += 1.0f; }); }
        for (auto& pq : peers) pq.wait();
    };
    peer_work();
    std::printf("%zu devices in the context; hidden %ld MiB at start\n", single ? (size_t) 1 : devs.size(), hidden_mib());
    std::vector<void*> p;
    const double t0 = now_ms();
    for (int i = 0; i < n; ++i) p.push_back(sycl::malloc_device(1ull << 30, q));
    const double t1 = now_ms();
    std::printf("%d x 1 GiB malloc_device: %.0f ms (%.0f ms each); hidden %ld MiB\n", n, t1 - t0, (t1 - t0) / n, hidden_mib());
    peer_work();
    std::printf("after a kernel on every card: hidden %ld MiB\n", hidden_mib());
    for (void* x : p) q.memset(x, 1, 1ull << 30);
    q.wait();
    std::printf("after memset: hidden %ld MiB (%.0f ms)\n", hidden_mib(), now_ms() - t1);
    const double t2 = now_ms();
    for (void* x : p) sycl::free(x, ctx);
    std::printf("after free: hidden %ld MiB (%.0f ms)\n", hidden_mib(), now_ms() - t2);
    return 0;
}
