// vmm_span_probe2.cpp - larry, 2026-10-27: the lazy card-0 KV map's wrong replica rows (patch-tpsynclog.py: right after
// sync_state, the replicas' rows that live in card 0's LATER physical allocation differ from card 0's, when the peer copy
// started in the first one) at the engine's sizes and in its order.  Card 0 reserves 2 GiB and maps A = [0, 130 MiB)
// (the KV_INIT pages: 65 x 2 MiB), resident on every other card, zeroed; card 1 reads A (its first peer access to the
// reservation); then B = [130, 1024 MiB), resident on the other cards, zeroed; card 0 writes a pattern over A + B (a
// kernel, as the engine's KV writes); then card 1 copies, each read back through card 1's queue and compared:
//   1  [128, 144 MiB) in one copy (A | B)
//   2  [256, 272 MiB) (inside B)
//   3  [0, 1024 MiB) in one copy (A | B, the engine's size)
//   4  [0, 130) + [130, 1024 MiB) as two copies (split at the boundary)
//   5  [128, 144 MiB) again after re-making the WHOLE mapped range resident on card 1 (one call for A + B)
// Build: icpx -fsycl -O2 vmm_span_probe2.cpp -lze_loader.  Cards to itself (an abw.sh "@" step); a fault: GT reset.
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
    const size_t MiB = (size_t) 1 << 20, A = 130 * MiB, TOT = 1024 * MiB, cap = 2048 * MiB;
    void * vp = nullptr;
    if (zeVirtualMemReserve(zc, nullptr, cap, &vp) != ZE_RESULT_SUCCESS || !vp) { std::printf("reserve failed\n"); return 1; }
    uint8_t * v0 = (uint8_t *) vp;
    std::vector<ze_physical_mem_handle_t> phys;
    std::vector<std::pair<void *, size_t>> maps;
    auto map = [&](uint8_t * p, size_t bytes) -> bool {   // as Ds41KvMap::Impl::map
        ze_physical_mem_desc_t d = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, bytes };
        ze_physical_mem_handle_t h = nullptr;
        if (zePhysicalMemCreate(zc, z0, &d, &h) != ZE_RESULT_SUCCESS || !h) return false;
        if (zeVirtualMemMap(zc, p, bytes, h, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE) != ZE_RESULT_SUCCESS) return false;
        phys.push_back(h);
        maps.emplace_back(p, bytes);
        for (ze_device_handle_t pd : peers)
            if (zeContextMakeMemoryResident(zc, pd, p, bytes) != ZE_RESULT_SUCCESS) { std::printf("make resident failed\n"); return false; }
        q0.memset(p, 0, bytes).wait();
        return true;
    };
    uint8_t * d1 = sycl::malloc_device<uint8_t>(TOT, q1);
    if (!d1) { std::printf("card 1 buffer failed\n"); return 1; }
    std::vector<uint8_t> back(TOT);
    auto pat = [](size_t i) { return (uint8_t) ((i * 2654435761u) >> 13 ^ (i >> 21) ^ 0x5a); };
    int fails = 0;
    auto check = [&](const char * what, size_t off, size_t n) {
        q1.memcpy(back.data(), d1, n).wait();
        size_t bad = 0, first = (size_t) -1, zero = 0;
        for (size_t i = 0; i < n; ++i)
            if (back[i] != pat(off + i)) { ++bad; if (first == (size_t) -1) first = i; zero += back[i] == 0; }
        fails += bad != 0;
        if (bad) std::printf("%s: %zu bad bytes of %zu (first at +%.2f MiB = %.2f MiB into the reservation; %zu of them zero)\n", what, bad, n,
                             first / (double) MiB, (off + first) / (double) MiB, zero);
        else std::printf("%s: exact (%zu bytes)\n", what, n);
        std::fflush(stdout);
    };
    if (!map(v0, A)) { std::printf("map A failed\n"); return 1; }
    q1.memcpy(d1, v0, A).wait();   // card 1's first peer access to the reservation
    std::printf("card 1 read A (its first peer access)\n");
    if (!map(v0 + A, TOT - A)) { std::printf("map B failed\n"); return 1; }
    q0.parallel_for(sycl::range<1>(TOT), [=](sycl::id<1> i) { v0[i] = (uint8_t) ((i * 2654435761u) >> 13 ^ (i >> 21) ^ 0x5a); }).wait();
    q1.memcpy(d1, v0 + 128 * MiB, 16 * MiB).wait();
    check("1 [128, 144 MiB) one copy across A | B", 128 * MiB, 16 * MiB);
    q1.memcpy(d1, v0 + 256 * MiB, 16 * MiB).wait();
    check("2 [256, 272 MiB) inside B", 256 * MiB, 16 * MiB);
    q1.memcpy(d1, v0, TOT).wait();
    check("3 [0, 1024 MiB) one copy across A | B", 0, TOT);
    q1.memcpy(d1, v0, A).wait();
    q1.memcpy(d1 + A, v0 + A, TOT - A).wait();
    check("4 [0, 130) + [130, 1024 MiB) split at the boundary", 0, TOT);
    {
        const ze_result_t r = zeContextMakeMemoryResident(zc, peers[0], v0, TOT);
        std::printf("5 zeContextMakeMemoryResident(card 1, the whole mapped range): %d\n", (int) r);
        q1.memcpy(d1, v0 + 128 * MiB, 16 * MiB).wait();
        check("5 [128, 144 MiB) one copy across A | B after it", 128 * MiB, 16 * MiB);
    }
    std::printf("%s\n", fails ? "FAILURES" : "all exact");
    q0.wait(); q1.wait();
    for (auto & m : maps) zeVirtualMemUnmap(zc, m.first, m.second);
    for (auto h : phys) zePhysicalMemDestroy(zc, h);
    zeVirtualMemFree(zc, v0, cap);
    sycl::free(d1, q1);
    return fails ? 2 : 0;
}
