// vmm_p2p.cpp - larry, engine1: can the copy engines move data between card 0's Level Zero virtual memory (pages the
// engine maps and unmaps: the dense weights it would page in for prompts) and another card's zeMemAllocDevice memory
// (the TP core's weight shards)?  Copy-engine copies only (queue memcpy / memcpy2d), no kernel touches peer memory.
//   A  card 1's queue: card 1 (zeMemAllocDevice) -> card 0 (VMM)          (a peer write into virtual memory)
//   B  card 0's queue: card 1 (zeMemAllocDevice) -> card 0 (VMM)          (a peer read of zeMemAllocDevice memory)
//   C  card 1's queue: card 1 -> card 0 (VMM), 2-D (rows of 64 B at a 256 B pitch: a K-split shard into its columns)
//   D  card 0's queue: card 0 (SYCL USM staging) -> card 0 (VMM), 2-D (the local scatter)
// Each case: a pattern copied, read back through card 0's queue into host memory and compared; MB/s printed.  Every
// wait bounded by the queue (no spins).  Build: icpx -fsycl -O2 vmm_p2p.cpp -lze_loader
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

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
    ze_device_handle_t z1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[1]);
    const size_t bytes = (size_t) 64 << 20;
    // card 0: virtual memory, all pages mapped
    size_t pg = 0;
    if (zeVirtualMemQueryPageSize(zc, z0, bytes, &pg) != ZE_RESULT_SUCCESS) { std::printf("page size query failed\n"); return 1; }
    void * va = nullptr;
    if (zeVirtualMemReserve(zc, nullptr, bytes, &va) != ZE_RESULT_SUCCESS || !va) { std::printf("reserve failed\n"); return 1; }
    ze_physical_mem_desc_t pd = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, bytes };
    ze_physical_mem_handle_t ph = nullptr;
    if (zePhysicalMemCreate(zc, z0, &pd, &ph) != ZE_RESULT_SUCCESS) { std::printf("physical create failed\n"); return 1; }
    if (zeVirtualMemMap(zc, va, bytes, ph, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE) != ZE_RESULT_SUCCESS) { std::printf("map failed\n"); return 1; }
    std::printf("card 0 VMM: %zu MB at %p (page %zu KB)\n", bytes >> 20, va, pg >> 10);
    // card 1: zeMemAllocDevice (as the TP core's arena), made resident on card 1 only
    ze_device_mem_alloc_desc_t dd = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0 };
    void * m1 = nullptr;
    if (zeMemAllocDevice(zc, &dd, bytes, 4096, z1, &m1) != ZE_RESULT_SUCCESS || !m1) { std::printf("card 1 alloc failed\n"); return 1; }
    zeContextMakeMemoryResident(zc, z1, m1, bytes);
    // card 0: SYCL USM staging
    uint8_t * st0 = sycl::malloc_device<uint8_t>(bytes, q0);
    std::vector<uint8_t> pat(bytes), back(bytes);
    for (size_t i = 0; i < bytes; ++i) pat[i] = (uint8_t) (i * 131 + (i >> 12));
    q1.memcpy(m1, pat.data(), bytes).wait();
    q0.memcpy(st0, pat.data(), bytes).wait();
    auto verify = [&](const char * name, double us, size_t moved, const std::vector<std::pair<size_t, size_t>> * runs) {
        std::memset(back.data(), 0, bytes);
        q0.memcpy(back.data(), va, bytes).wait();
        size_t bad = 0;
        if (!runs) { for (size_t i = 0; i < bytes; ++i) bad += back[i] != pat[i]; }
        else for (const auto & r : *runs) for (size_t i = r.first; i < r.first + r.second; ++i) bad += back[i] != pat[i];
        std::printf("%s: %s, %.0f MB/s\n", name, bad ? ("MISMATCH (" + std::to_string(bad) + " bytes)").c_str() : "ok", moved / us);
        std::fflush(stdout);
    };
    auto clear = [&]() { q0.memset(va, 0, bytes).wait(); };
    for (int rep = 0; rep < 2; ++rep) {
        clear();
        double t = now_us();
        try { q1.memcpy(va, m1, bytes).wait(); verify("A card 1 queue: card 1 -> card 0 VMM", now_us() - t, bytes, nullptr); }
        catch (const std::exception & e) { std::printf("A: %s\n", e.what()); }
        clear();
        t = now_us();
        try { q0.memcpy(va, m1, bytes).wait(); verify("B card 0 queue: card 1 -> card 0 VMM", now_us() - t, bytes, nullptr); }
        catch (const std::exception & e) { std::printf("B: %s\n", e.what()); }
        // 2-D: rows of 64 bytes at a 256-byte pitch on the destination, packed on the source (a K-split shard's planes)
        const size_t w = 64, dp = 256, h = bytes / dp;
        std::vector<std::pair<size_t, size_t>> runs;
        for (size_t r = 0; r < h; r += 997) runs.emplace_back(r * dp, w);   // a sample of the rows
        std::vector<uint8_t> packed(w * h);
        for (size_t r = 0; r < h; ++r) std::memcpy(packed.data() + r * w, pat.data() + r * dp, w);
        q1.memcpy(m1, packed.data(), w * h).wait();
        q0.memcpy(st0, packed.data(), w * h).wait();
        clear();
        t = now_us();
        try { q1.ext_oneapi_memcpy2d(va, dp, m1, w, w, h).wait(); verify("C card 1 queue 2-D: card 1 -> card 0 VMM", now_us() - t, w * h, &runs); }
        catch (const std::exception & e) { std::printf("C: %s\n", e.what()); }
        clear();
        t = now_us();
        try { q0.ext_oneapi_memcpy2d(va, dp, st0, w, w, h).wait(); verify("D card 0 queue 2-D: card 0 USM -> card 0 VMM", now_us() - t, w * h, &runs); }
        catch (const std::exception & e) { std::printf("D: %s\n", e.what()); }
        q1.memcpy(m1, pat.data(), bytes).wait();   // the full pattern again for the next round
        q0.memcpy(st0, pat.data(), bytes).wait();
    }
    zeVirtualMemUnmap(zc, va, bytes);
    zePhysicalMemDestroy(zc, ph);
    zeVirtualMemFree(zc, va, bytes);
    zeMemFree(zc, m1);
    sycl::free(st0, q0);
    std::printf("done\n");
    return 0;
}
