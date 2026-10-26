// h2d_probe.cpp - pinned host -> device copy bandwidth: one card at a time, then cards 1..3 together (one submitting
// thread, 10 MiB copies like the tier's prefetch), and pageable -> pinned memcpy rate on 1 / 4 threads
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<sycl::device> devs;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) devs.push_back(d);
    std::printf("%zu level-zero GPUs\n", devs.size());
    if (devs.size() < 4) return 1;
    sycl::context ctx = devs[0].get_platform().ext_oneapi_get_default_context();
    const size_t SZ = (size_t) 1 << 30, CH = (size_t) 10 << 20;
    std::vector<sycl::queue> qs;
    for (auto& d : devs) qs.emplace_back(ctx, d, sycl::property::queue::in_order());
    std::vector<uint8_t*> h, dv;
    for (size_t i = 0; i < devs.size(); ++i) {
        h.push_back(sycl::malloc_host<uint8_t>(SZ, qs[i]));
        std::memset(h.back(), (int) i + 1, SZ);
        dv.push_back(sycl::malloc_device<uint8_t>(SZ, qs[i]));
        std::printf("card %zu: host %p dev %p\n", i, (void*) h.back(), (void*) dv.back());
        if (!h.back() || !dv.back()) return 2;
    }
    auto run = [&](std::vector<int> cards) {
        for (int c : cards) qs[c].memcpy(dv[c], h[c], CH).wait();   // warm
        const double t0 = now();
        for (size_t off = 0; off < SZ; off += CH)
            for (int c : cards) qs[c].memcpy(dv[c] + off, h[c] + off, CH);
        for (int c : cards) qs[c].wait();
        const double t = now() - t0;
        std::printf("cards");
        for (int c : cards) std::printf(" %d", c);
        std::printf(": %.2f GB in %.3f s = %.1f GB/s\n", SZ * cards.size() / 1e9, t, SZ * cards.size() / 1e9 / t);
    };
    for (int c = 0; c < (int) devs.size(); ++c) run({c});
    run({1, 2, 3});
    run({0, 1, 2, 3});
    // pageable -> pinned memcpy
    const size_t PS = (size_t) 4 << 30;
    uint8_t* pg = (uint8_t*) aligned_alloc(2 << 20, PS);
    std::memset(pg, 7, PS);
    uint8_t* pin = sycl::malloc_host<uint8_t>(PS, ctx);
    for (int nt : {1, 2, 4, 6}) {
        const double t0 = now();
        std::vector<std::thread> th;
        for (int k = 0; k < nt; ++k) th.emplace_back([&, k] {
            for (size_t off = (size_t) k * CH; off < PS; off += (size_t) nt * CH) std::memcpy(pin + off, pg + off, CH);
        });
        for (auto& t : th) t.join();
        const double t = now() - t0;
        std::printf("memcpy pageable->pinned, %d threads: %.1f GB/s\n", nt, PS / 1e9 / t);
    }
    return 0;
}
