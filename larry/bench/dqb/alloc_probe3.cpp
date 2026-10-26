// alloc_probe2.cpp - larry: the dense card's 1 GiB slab allocations under the engine's memory state, built up step by
// step (pinned host, pageable host, the expert cards' VRAM): time per slab and hidden memory
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
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
    const int native = argc > 5 ? std::atoi(argv[5]) : 0;   // 1: slabs natively; 2: the expert cards' VRAM natively too
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    auto zalloc = [&](int dev, size_t bytes) -> void* {   // zeMemAllocDevice: outside UR / UMF (no peer residency)
        ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devs[(size_t) dev]);
        ze_device_mem_alloc_desc_t d = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
        void* p = nullptr;
        if (zeMemAllocDevice(zc, &d, bytes, 4096, zd, &p) != ZE_RESULT_SUCCESS) return nullptr;
        return p;
    };
    auto zfree = [&](void* p) { zeMemFree(zc, p); };
    auto slabs = [&](const char* tag) {
        std::vector<void*> p;
        const double t0 = now_ms();
        const long h0 = hidden_mib();
        for (int i = 0; i < 4; ++i) p.push_back(native ? zalloc(0, 1ull << 30) : sycl::malloc_device(1ull << 30, qs[0]));
        const double t1 = now_ms();
        // used by a kernel on card 0 (residency where it runs)
        for (void* x : p) { float* f = (float*) x; qs[0].parallel_for(sycl::range<1>(1 << 20), [=](sycl::id<1> j) { f[j] = 1.0f; }); }
        qs[0].wait();
        std::printf("%-40s 4 x 1 GiB on card 0%s: %.0f ms each, hidden %+ld MiB (after a kernel wrote them)\n", tag, native ? " (native)" : "",
                    (t1 - t0) / 4, hidden_mib() - h0);
        for (void* x : p) { if (native) zfree(x); else sycl::free(x, ctx); }
    };
    slabs("clean:");
    std::vector<void*> pin;
    for (int i = 0; i < pinned_gib; ++i) pin.push_back(sycl::malloc_host(1ull << 30, ctx));
    slabs("+ pinned host:");
    std::vector<char*> pg;
    for (int i = 0; i < page_gib; ++i) { char* x = (char*) std::malloc(1ull << 30); std::memset(x, 1, 1ull << 30); pg.push_back(x); }
    slabs("+ pageable host (touched):");
    std::vector<std::pair<int, void*>> dv;
    const long hx = hidden_mib();
    for (int c = 1; c < (int) devs.size(); ++c) for (int i = 0; i < exp_gib; ++i) dv.push_back({c, native == 2 ? zalloc(c, 1ull << 30) : sycl::malloc_device(1ull << 30, qs[(size_t) c])});
    std::printf("the expert cards' %d GiB each%s: hidden %+ld MiB\n", exp_gib, native == 2 ? " (native)" : "", hidden_mib() - hx);
    slabs("+ expert cards' VRAM:");
    for (int i = 0; i < c0_gib; ++i) dv.push_back({0, sycl::malloc_device(1ull << 30, qs[0])});
    slabs("+ card 0's VRAM:");
    for (auto& x : dv) { if (native == 2) zfree(x.second); else sycl::free(x.second, ctx); }
    for (void* x : pin) sycl::free(x, ctx);
    for (char* x : pg) std::free(x);
    return 0;
}
