// vmm_remap.cpp - larry, engine1: why the pager's fills corrupted card 0 when the OTHER cards wrote them (service,
// 2026-10-23: long prompts garbled with the copies on cards 1-3's queues, fine with them on card 0's).  Two suspects:
//   A  a barrier on card 0's queue does not wait for an event of card 1's queue (cross-device dependency): card 1
//      copies 128 MB into card 0, card 0's queue then reads it behind ext_oneapi_submit_barrier({that event})
//   B  card 1's view of card 0's virtual memory goes stale after a remap: VA X mapped to pages P1, card 1 writes into
//      X; X unmapped and mapped to fresh pages P2, card 1 writes again - do the bytes land in P2 (seen through X), or in
//      P1 (still mapped at a second address Y, kept to look)?
//   C  as B with card 0's queue doing the copies (peer reads; the fix)
// Copy-engine copies only, every wait bounded by its queue.  Build: icpx -fsycl -O2 vmm_remap.cpp -lze_loader
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include <cstdio>
#include <cstring>
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
    ze_device_handle_t z1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[1]);
    const size_t big = (size_t) 128 << 20, small = (size_t) 64 << 20;
    // card 1's source: zeMemAllocDevice (as the TP core's shards), resident on card 1
    ze_device_mem_alloc_desc_t dd = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0 };
    uint8_t * s1 = nullptr;
    if (zeMemAllocDevice(zc, &dd, big, 4096, z1, (void **) &s1) != ZE_RESULT_SUCCESS || !s1) { std::printf("card 1 alloc failed\n"); return 1; }
    zeContextMakeMemoryResident(zc, z1, s1, big);
    uint8_t * d0 = sycl::malloc_device<uint8_t>(big, q0);   // card 0: a plain destination / a scratch copy
    uint8_t * c0 = sycl::malloc_device<uint8_t>(big, q0);
    std::vector<uint8_t> host(big), back(big);
    auto fill = [&](uint8_t * dev, sycl::queue & q, uint8_t tag, size_t n) {
        for (size_t i = 0; i < n; ++i) host[i] = (uint8_t) (tag ^ (i * 131) ^ (i >> 13));
        q.memcpy(dev, host.data(), n).wait();
    };
    auto bad_of = [&](const uint8_t * h, uint8_t tag, size_t n) {
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += h[i] != (uint8_t) (tag ^ (i * 131) ^ (i >> 13));
        return bad;
    };

    // ---- A: card 1's copy -> card 0, card 0's barrier on its event, then card 0 copies the destination away at once
    for (int rep = 0; rep < 6; ++rep) {
        const uint8_t tag = (uint8_t) (0x10 + rep);
        fill(s1, q1, tag, big);
        q0.memset(d0, 0, big).wait();
        sycl::event e = q1.memcpy(d0, s1, big);
        q0.ext_oneapi_submit_barrier({ e });
        q0.memcpy(c0, d0, big);   // behind the barrier: must see the whole copy
        q0.wait();
        q0.memcpy(back.data(), c0, big).wait();
        std::printf("A rep %d: card 0's read behind the barrier: %zu bad bytes of %zu\n", rep, bad_of(back.data(), tag, big), big);
    }

    // ---- B / C: virtual memory on card 0 remapped between two writes
    size_t pg = 0;
    zeVirtualMemQueryPageSize(zc, z0, small, &pg);
    auto run_remap = [&](const char * name, sycl::queue & qc) {
        void * X = nullptr, * Y = nullptr;
        if (zeVirtualMemReserve(zc, nullptr, small, &X) != ZE_RESULT_SUCCESS || zeVirtualMemReserve(zc, nullptr, small, &Y) != ZE_RESULT_SUCCESS) {
            std::printf("%s: reserve failed\n", name); return;
        }
        ze_physical_mem_desc_t pd = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, small };
        ze_physical_mem_handle_t P1 = nullptr, P2 = nullptr;
        zePhysicalMemCreate(zc, z0, &pd, &P1);
        zePhysicalMemCreate(zc, z0, &pd, &P2);
        for (int rep = 0; rep < 3; ++rep) {
            const uint8_t ta = (uint8_t) (0x40 + rep), tb = (uint8_t) (0x80 + rep);
            // X -> P1, the first write (tag a), done
            zeVirtualMemMap(zc, X, small, P1, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
            fill(s1, q1, ta, small);
            qc.memcpy(X, s1, small).wait();
            // X -> P2 (P1 still mapped, at Y), the second write (tag b) into X
            zeVirtualMemUnmap(zc, X, small);
            zeVirtualMemMap(zc, X, small, P2, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
            zeVirtualMemMap(zc, Y, small, P1, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
            q0.memset(X, 0, small).wait();
            fill(s1, q1, tb, small);
            qc.memcpy(X, s1, small).wait();
            std::vector<uint8_t> hx(small), hy(small);
            q0.memcpy(hx.data(), X, small).wait();
            q0.memcpy(hy.data(), Y, small).wait();
            std::printf("%s rep %d: X (P2 now) holds the 2nd write: %zu bad bytes; P1 (old pages, at Y) still the 1st write: %zu bad; "
                        "P1 overwritten by the 2nd: %s\n", name, rep, bad_of(hx.data(), tb, small), bad_of(hy.data(), ta, small),
                        bad_of(hy.data(), tb, small) == 0 ? "YES" : "no");
            std::fflush(stdout);
            zeVirtualMemUnmap(zc, X, small);
            zeVirtualMemUnmap(zc, Y, small);
        }
        zePhysicalMemDestroy(zc, P1);
        zePhysicalMemDestroy(zc, P2);
        zeVirtualMemFree(zc, X, small);
        zeVirtualMemFree(zc, Y, small);
    };
    run_remap("B card 1's queue (peer write)", q1);
    run_remap("C card 0's queue (peer read)", q0);
    zeMemFree(zc, s1);
    sycl::free(d0, q0);
    sycl::free(c0, q0);
    std::printf("done\n");
    return 0;
}
