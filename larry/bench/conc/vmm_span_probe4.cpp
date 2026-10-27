// vmm_span_probe4.cpp - larry, 2026-10-27: how card 0's VMM reservation must be grown for the other cards to keep reading
// it right (vmm_span_probe2: once a second allocation was mapped AND made resident on every peer after a peer had read the
// first, the peers' reads past the first 2 MiB were other data; vmm_span_probe3: a reservation of one allocation reads
// exact either way; vmm_span_probe: allocations never made resident explicitly read exact, and a later one made resident
// on card 1 alone too).  The engine's order, per strategy, in a fresh reservation each: map A = 130 MiB, zero it, card 1
// and card 2 read A; map B = 894 MiB after it, zero it; card 0 writes a pattern over A + B (a kernel); card 1 and card 2
// read A + B as copies split at the A | B boundary (and, where it cannot fault, as one copy across it):
//   S1  every mapping made resident on every other card at once (Ds41KvMap today, patch-kvresident.py)
//   S2  no explicit residency: the copies split at the boundary (each inside one allocation)
//   S3  a mapping made resident only on the cards that read it, just before their copy (card 1, then card 2)
//   S4  S1, then the reservation's mapped range evicted from every peer and made resident again (one call each)
// Build: icpx -fsycl -O2 vmm_span_probe4.cpp -lze_loader.  Cards to itself (an abw.sh "@" step).
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
    if (gpus.size() < 3) { std::printf("need 3 gpus\n"); return 1; }
    sycl::context ctx = plat.khr_get_default_context();
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order());
    sycl::queue qr[2] = { sycl::queue(ctx, gpus[1], sycl::property::queue::in_order()), sycl::queue(ctx, gpus[2], sycl::property::queue::in_order()) };
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_device_handle_t z0 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[0]);
    std::vector<ze_device_handle_t> peers;
    for (size_t i = 1; i < gpus.size(); ++i) peers.push_back(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[i]));
    const size_t MiB = (size_t) 1 << 20, A = 130 * MiB, TOT = 1024 * MiB, cap = 2048 * MiB;
    uint8_t * dr[2] = { sycl::malloc_device<uint8_t>(TOT, qr[0]), sycl::malloc_device<uint8_t>(TOT, qr[1]) };
    std::vector<uint8_t> back(TOT);
    auto pat = [](size_t i, uint8_t salt) { return (uint8_t) (((i * 2654435761ull) >> 13) ^ (i >> 21) ^ salt); };
    int fails = 0;
    auto verify = [&](const char * what, size_t n, uint8_t salt) {
        size_t bad = 0, first = (size_t) -1;
        for (size_t i = 0; i < n; ++i)
            if (back[i] != pat(i, salt)) { ++bad; if (first == (size_t) -1) first = i; }
        fails += bad != 0;
        if (bad) std::printf("    %-44s %zu bad bytes of %zu, first at %.2f MiB\n", what, bad, n, first / (double) MiB);
        else std::printf("    %-44s exact\n", what);
        std::fflush(stdout);
    };
    uint8_t salt = 7;
    for (int s = 1; s <= 4; ++s) {
        std::printf("  S%d:\n", s);
        void * vp = nullptr;
        if (zeVirtualMemReserve(zc, nullptr, cap, &vp) != ZE_RESULT_SUCCESS || !vp) { std::printf("reserve failed\n"); return 1; }
        uint8_t * v = (uint8_t *) vp;
        std::vector<ze_physical_mem_handle_t> phys;
        auto map = [&](uint8_t * p, size_t bytes) -> bool {
            ze_physical_mem_desc_t d = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, bytes };
            ze_physical_mem_handle_t h = nullptr;
            if (zePhysicalMemCreate(zc, z0, &d, &h) != ZE_RESULT_SUCCESS || !h) return false;
            if (zeVirtualMemMap(zc, p, bytes, h, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE) != ZE_RESULT_SUCCESS) return false;
            phys.push_back(h);
            if (s == 1 || s == 4)
                for (ze_device_handle_t pd : peers)
                    if (zeContextMakeMemoryResident(zc, pd, p, bytes) != ZE_RESULT_SUCCESS) { std::printf("resident failed\n"); return false; }
            q0.memset(p, 0, bytes).wait();
            return true;
        };
        auto reader_prep = [&](int k, uint8_t * p, size_t bytes) {   // S3: resident on the reading card just before its copy
            if (s == 3) zeContextMakeMemoryResident(zc, peers[(size_t) k], p, bytes);
        };
        if (!map(v, A)) { std::printf("map A failed\n"); return 1; }
        for (int k = 0; k < 2; ++k) { reader_prep(k, v, A); qr[k].memcpy(dr[k], v, A).wait(); }   // the readers' first access
        if (!map(v + A, TOT - A)) { std::printf("map B failed\n"); return 1; }
        if (s == 4)
            for (ze_device_handle_t pd : peers) {
                const ze_result_t e = zeContextEvictMemory(zc, pd, v, TOT);
                const ze_result_t r = zeContextMakeMemoryResident(zc, pd, v, TOT);
                if (e != ZE_RESULT_SUCCESS || r != ZE_RESULT_SUCCESS) std::printf("    S4 evict %d, resident %d\n", (int) e, (int) r);
            }
        q0.parallel_for(sycl::range<1>(TOT), [=](sycl::id<1> id) {
            const size_t i = id[0];
            v[i] = (uint8_t) (((i * 2654435761ull) >> 13) ^ (i >> 21) ^ salt);
        }).wait();
        for (int k = 0; k < 2; ++k) {
            reader_prep(k, v + A, TOT - A);
            qr[k].memcpy(dr[k], v, A).wait();
            qr[k].memcpy(dr[k] + A, v + A, TOT - A).wait();
            qr[k].memcpy(back.data(), dr[k], TOT).wait();
            verify(k == 0 ? "card 1, split at the boundary" : "card 2, split at the boundary", TOT, salt);
            if (s != 2) {   // (S2 has nothing resident explicitly: one copy across may fault)
                qr[k].memcpy(dr[k], v, TOT).wait();
                qr[k].memcpy(back.data(), dr[k], TOT).wait();
                verify(k == 0 ? "card 1, one copy across" : "card 2, one copy across", TOT, salt);
            }
        }
        zeVirtualMemUnmap(zc, v, A);
        zeVirtualMemUnmap(zc, v + A, TOT - A);
        for (auto h : phys) zePhysicalMemDestroy(zc, h);
        zeVirtualMemFree(zc, vp, cap);
        ++salt;
    }
    std::printf("%s\n", fails ? "FAILURES" : "all exact");
    sycl::free(dr[0], qr[0]);
    sycl::free(dr[1], qr[1]);
    return fails ? 2 : 0;
}
