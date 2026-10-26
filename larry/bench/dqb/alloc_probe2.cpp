// alloc_probe2.cpp - larry: the dense card's 1 GiB slab allocations under the engine's memory state, built up step by
// step (pinned host, pageable host, the expert cards' VRAM): time per slab and hidden memory
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
static long hidden_mib() {
    std::ifstream f("/proc/meminfo");
    std::string k, u; long v, t = 0, a = 0, p = 0;
    while (f >> k >> v >> u) { if (k == "MemTotal:") t = v; if (k == "MemAvailable:") a = v; if (k == "AnonPages:") p = v; }
    return (t - a - p) / 1024;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
    const int pinned_gib = argc > 1 ? std::atoi(argv[1]) : 19, page_gib = argc > 2 ? std::atoi(argv[2]) : 50;
    const int exp_gib = argc > 3 ? std::atoi(argv[3]) : 28, c0_gib = argc > 4 ? std::atoi(argv[4]) : 7;
    std::vector<sycl::device> devs;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) devs.push_back(d);
    sycl::context ctx = devs[0].get_platform().khr_get_default_context();
    std::vector<sycl::queue> qs;
    for (auto& d : devs) qs.emplace_back(ctx, d, sycl::property::queue::in_order());
    auto slabs = [&](const char* tag) {
        std::vector<void*> p;
        const double t0 = now_ms();
        const long h0 = hidden_mib();
        for (int i = 0; i < 4; ++i) p.push_back(sycl::malloc_device(1ull << 30, qs[0]));
        const double t1 = now_ms();
        std::printf("%-40s 4 x 1 GiB on card 0: %.0f ms each, hidden %+ld MiB\n", tag, (t1 - t0) / 4, hidden_mib() - h0);
        for (void* x : p) sycl::free(x, ctx);
    };
    slabs("clean:");
    std::vector<void*> pin;
    for (int i = 0; i < pinned_gib; ++i) pin.push_back(sycl::malloc_host(1ull << 30, ctx));
    slabs("+ pinned host:");
    std::vector<char*> pg;
    for (int i = 0; i < page_gib; ++i) { char* x = (char*) std::malloc(1ull << 30); std::memset(x, 1, 1ull << 30); pg.push_back(x); }
    slabs("+ pageable host (touched):");
    std::vector<std::pair<int, void*>> dv;
    for (int c = 1; c < (int) devs.size(); ++c) for (int i = 0; i < exp_gib; ++i) dv.push_back({c, sycl::malloc_device(1ull << 30, qs[(size_t) c])});
    slabs("+ expert cards' VRAM:");
    for (int i = 0; i < c0_gib; ++i) dv.push_back({0, sycl::malloc_device(1ull << 30, qs[0])});
    slabs("+ card 0's VRAM:");
    for (auto& x : dv) sycl::free(x.second, ctx);
    for (void* x : pin) sycl::free(x, ctx);
    for (char* x : pg) std::free(x);
    return 0;
}
