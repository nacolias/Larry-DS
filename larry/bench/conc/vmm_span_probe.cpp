// vmm_span_probe.cpp - larry, 2026-10-27: does ONE peer copy that spans two physical allocations of a card-0 Level Zero
// virtual memory reservation work?  Behind the question: STRATA_DENSE_LAZY_KV + patch-kvbucket.py faulted card 3's copy
// engine (a read, DEVICE_LOST) in the TP core's first replica growth after a 131K prompt - that copy, rows [8192, 262144)
// of card 0's compressed KV, started in the pages mapped at load and ended in the pages mapped just before the prompt
// (two zePhysicalMemCreate allocations).  vmm_peer_read only ever copied inside one allocation (all exact).
//   F  card 0 maps [0, 64 MiB) as allocation 1, then [64, 128 MiB) as allocation 2; card 1 copies [32, 96 MiB) as two
//      copies split at 64 MiB (each inside one allocation)                                      - expected exact
//   G  allocation 3 = [128, 192 MiB) mapped now, zeContextMakeMemoryResident on card 1 for it; card 1 copies
//      [96, 160 MiB) in ONE copy (spans allocations 2 and 3)                                    - the residency question
//   E  allocation 4 = [192, 256 MiB) mapped now, nothing more; card 1 copies [160, 224 MiB) in ONE copy - may FAULT
//      (run last; a fault means a GT reset of card 1 afterwards: logs/gtreset.sh)
// Each copy read back through card 1's queue and compared.  Build: icpx -fsycl -O2 vmm_span_probe.cpp -lze_loader
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::string only = argc > 1 ? argv[1] : "FGE";
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
    ze_device_handle_t z1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[1]);
    size_t pg = 0;
    zeVirtualMemQueryPageSize(zc, z0, (size_t) 1 << 30, &pg);
    pg = std::max<size_t>(pg, (size_t) 2 << 20);
    const size_t part = (size_t) 64 << 20, cap = (size_t) 512 << 20;
    void * vp = nullptr;
    if (zeVirtualMemReserve(zc, nullptr, cap, &vp) != ZE_RESULT_SUCCESS || !vp) { std::printf("reserve failed\n"); return 1; }
    uint8_t * v0 = (uint8_t *) vp;
    std::vector<ze_physical_mem_handle_t> phys;
    std::vector<std::pair<void *, size_t>> maps;
    auto map = [&](uint8_t * p, size_t bytes) -> bool {
        ze_physical_mem_desc_t d = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, bytes };
        ze_physical_mem_handle_t h = nullptr;
        if (zePhysicalMemCreate(zc, z0, &d, &h) != ZE_RESULT_SUCCESS || !h) return false;
        if (zeVirtualMemMap(zc, p, bytes, h, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE) != ZE_RESULT_SUCCESS) return false;
        phys.push_back(h);
        maps.emplace_back(p, bytes);
        return true;
    };
    ze_device_mem_alloc_desc_t dd = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0 };
    uint8_t * d1 = nullptr;
    if (zeMemAllocDevice(zc, &dd, part, 4096, z1, (void **) &d1) != ZE_RESULT_SUCCESS || !d1) { std::printf("card 1 alloc failed\n"); return 1; }
    zeContextMakeMemoryResident(zc, z1, d1, part);
    std::vector<uint8_t> host(part), back(part);
    auto fill = [&](uint8_t * dev, uint8_t tag, size_t n) {
        for (size_t i = 0; i < n; ++i) host[i] = (uint8_t) (tag ^ (i * 131) ^ (i >> 13));
        q0.memcpy(dev, host.data(), n).wait();
    };
    int fails = 0;
    auto check = [&](const char * what, uint8_t tag, size_t n) {
        q1.memcpy(back.data(), d1, n).wait();
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += back[i] != (uint8_t) (tag ^ (i * 131) ^ (i >> 13));
        fails += bad != 0;
        std::printf("%s: %zu bad bytes of %zu\n", what, bad, n);
        std::fflush(stdout);
    };
    if (!map(v0, part) || !map(v0 + part, part)) { std::printf("map 1/2 failed\n"); return 1; }
    if (only.find('F') != std::string::npos) {
        fill(v0 + part / 2, 0x61, part);
        q1.memcpy(d1, v0 + part / 2, part / 2).wait();
        q1.memcpy(d1 + part / 2, v0 + part, part / 2).wait();
        check("F [32, 96 MiB) as two copies split at the allocation boundary", 0x61, part);
    }
    if (only.find('G') != std::string::npos) {
        if (!map(v0 + 2 * part, part)) { std::printf("map 3 failed\n"); return 1; }
        const ze_result_t r = zeContextMakeMemoryResident(zc, z1, v0 + 2 * part, part);
        std::printf("G zeContextMakeMemoryResident(card 1, allocation 3): %d\n", (int) r);
        fill(v0 + part + part / 2, 0x71, part);
        q1.memcpy(d1, v0 + part + part / 2, part).wait();
        check("G [96, 160 MiB) in one copy across allocations 2 | 3 (3 made resident on card 1)", 0x71, part);
    }
    if (only.find('E') != std::string::npos) {
        if (!map(v0 + 3 * part, part)) { std::printf("map 4 failed\n"); return 1; }
        fill(v0 + 2 * part + part / 2, 0x81, part);
        q1.memcpy(d1, v0 + 2 * part + part / 2, part).wait();
        check("E [160, 224 MiB) in one copy across allocations 3 | 4 (4 only mapped)", 0x81, part);
    }
    std::printf("%s\n", fails ? "FAILURES" : "all exact");
    q0.wait(); q1.wait();
    for (auto & m : maps) zeVirtualMemUnmap(zc, m.first, m.second);
    for (auto h : phys) zePhysicalMemDestroy(zc, h);
    zeVirtualMemFree(zc, v0, cap);
    zeMemFree(zc, d1);
    return fails ? 2 : 0;
}
