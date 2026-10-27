// vmm_span_probe3.cpp - larry, 2026-10-27: which card-0 VMM mappings card 1 reads right (vmm_span_probe2: past the first
// 2 MiB of a 130 MiB + 894 MiB reservation, card 1's copies returned other data, even after re-making it resident; the
// first probe's 64 MiB allocations read exact).  Each case a fresh reservation of one physical allocation of SIZE on card 0,
// resident on every other card, filled with a pattern by card 0 (FILL: a host-to-device copy on card 0's queue, or a
// kernel), then read back by card 0 itself (D2H) and by card 1 (a peer copy into its memory, then D2H); per view: exact,
// or the bad bytes and the first bad offset.  Baseline: a plain sycl::malloc_device buffer on card 0 read the same ways.
// Build: icpx -fsycl -O2 vmm_span_probe3.cpp -lze_loader.  Cards to itself (an abw.sh "@" step).
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main() {
    std::vector<sycl::device> gpus;
    sycl::platform plat;
    for (const auto & p : sycl::platform::get_platforms())
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            auto d = p.get_devices(sycl::info::device_type::gpu);
            if (!d.empty()) { gpus = d; plat = p; break; }
        }
    if (gpus.size() < 2) { std::printf("need 2 gpus\n"); return 1; }
    sycl::context ctx = plat.khr_get_default_context();
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order()), q1(ctx, gpus[1], sycl::property::queue::in_order());
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_device_handle_t z0 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[0]);
    std::vector<ze_device_handle_t> peers;
    for (size_t i = 1; i < gpus.size(); ++i) peers.push_back(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[i]));
    size_t pg = 0;
    zeVirtualMemQueryPageSize(zc, z0, (size_t) 1 << 30, &pg);
    std::printf("VMM page size for 1 GiB: %zu KiB\n", pg >> 10);
    const size_t MiB = (size_t) 1 << 20, MAX = 1024 * MiB;
    uint8_t * d1 = sycl::malloc_device<uint8_t>(MAX, q1);
    std::vector<uint8_t> host(MAX), back(MAX);
    auto pat = [](size_t i, uint8_t salt) { return (uint8_t) (((i * 2654435761ull) >> 13) ^ (i >> 21) ^ salt); };
    int fails = 0;
    auto verify = [&](const char * what, size_t n, uint8_t salt) {
        size_t bad = 0, first = (size_t) -1;
        for (size_t i = 0; i < n; ++i)
            if (back[i] != pat(i, salt)) { ++bad; if (first == (size_t) -1) first = i; }
        fails += bad != 0;
        if (bad) std::printf("    %-34s %zu bad bytes of %zu, first at %.2f MiB\n", what, bad, n, first / (double) MiB);
        else std::printf("    %-34s exact\n", what);
        std::fflush(stdout);
    };
    auto run = [&](const char * label, uint8_t * p, size_t n, bool kernel, uint8_t salt) {
        std::printf("  %s, %zu MiB, filled by %s:\n", label, n / MiB, kernel ? "a card-0 kernel" : "a host-to-device copy");
        if (kernel) {
            q0.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                const size_t i = id[0];
                p[i] = (uint8_t) (((i * 2654435761ull) >> 13) ^ (i >> 21) ^ salt);
            }).wait();
        } else {
            for (size_t i = 0; i < n; ++i) host[i] = pat(i, salt);
            q0.memcpy(p, host.data(), n).wait();
        }
        q0.memcpy(back.data(), p, n).wait();
        verify("card 0's own read", n, salt);
        std::memset(back.data(), 0, n);
        q1.memcpy(d1, p, n).wait();
        q1.memcpy(back.data(), d1, n).wait();
        verify("card 1's peer copy", n, salt);
    };
    const size_t sizes[] = { 64 * MiB, 130 * MiB, 512 * MiB, 1024 * MiB };
    uint8_t salt = 1;
    for (size_t sz : sizes)
        for (int kernel = 0; kernel < 2; ++kernel) {
            void * vp = nullptr;
            if (zeVirtualMemReserve(zc, nullptr, 2 * sz, &vp) != ZE_RESULT_SUCCESS || !vp) { std::printf("reserve failed\n"); return 1; }
            ze_physical_mem_desc_t d = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, sz };
            ze_physical_mem_handle_t h = nullptr;
            if (zePhysicalMemCreate(zc, z0, &d, &h) != ZE_RESULT_SUCCESS || !h) { std::printf("physical create failed\n"); return 1; }
            if (zeVirtualMemMap(zc, vp, sz, h, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE) != ZE_RESULT_SUCCESS) { std::printf("map failed\n"); return 1; }
            for (ze_device_handle_t pd : peers)
                if (zeContextMakeMemoryResident(zc, pd, vp, sz) != ZE_RESULT_SUCCESS) { std::printf("resident failed\n"); return 1; }
            run("VMM", (uint8_t *) vp, sz, kernel != 0, salt++);
            zeVirtualMemUnmap(zc, vp, sz);
            zePhysicalMemDestroy(zc, h);
            zeVirtualMemFree(zc, vp, 2 * sz);
        }
    {
        uint8_t * p = sycl::malloc_device<uint8_t>(MAX, q0);
        run("malloc_device", p, MAX, false, salt++);
        run("malloc_device", p, MAX, true, salt++);
        sycl::free(p, q0);
    }
    std::printf("%s\n", fails ? "FAILURES" : "all exact");
    sycl::free(d1, q1);
    return fails ? 2 : 0;
}
