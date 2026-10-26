// vmm_peer_read.cpp - larry: can another card's copy engine READ card 0's Level Zero virtual memory, including pages
// mapped after the first peer reads?  The question behind card 0's KV state mapped as the context grows: the TP core's
// replicas copy their new rows from card 0 on THEIR queues (card k reads card 0).  vmm_p2p covered peer writes into
// card 0's VMM and card 0 reading its peers; vmm_remap showed a peer's view going stale after a REMAP (grow-only here).
//   A  card 0 VMM [0, 64 MiB) filled by card 0 -> card 1's queue copies it into card 1 memory (zeMemAllocDevice)
//   B  card 0 maps [64, 128 MiB) NOW (after A's peer reads), fills it -> card 1 copies it
//   C  the reservation's last page (mapped at the start, like the KV's spare row): card 0 fills, card 1 copies
//   D  card 0 VMM -> card 1 VMM (a grow-only reservation on card 1, as the replicas): card 1's queue, then card 2's
// Each: read back through the destination card's queue, compared.  Copy-engine copies only; every wait bounded by its
// queue.  Build: icpx -fsycl -O2 vmm_peer_read.cpp -lze_loader
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
    if (gpus.size() < 3) { std::printf("need 3 gpus\n"); return 1; }
    sycl::context ctx = plat.khr_get_default_context();
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order()), q1(ctx, gpus[1], sycl::property::queue::in_order()),
        q2(ctx, gpus[2], sycl::property::queue::in_order());
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_device_handle_t z0 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[0]);
    ze_device_handle_t z1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[1]);
    size_t pg = 0;
    zeVirtualMemQueryPageSize(zc, z0, (size_t) 1 << 30, &pg);
    pg = std::max<size_t>(pg, (size_t) 2 << 20);
    const size_t part = (size_t) 64 << 20, cap = (size_t) 256 << 20;
    auto reserve = [&](size_t bytes) -> uint8_t * {
        void * p = nullptr;
        return zeVirtualMemReserve(zc, nullptr, bytes, &p) == ZE_RESULT_SUCCESS ? (uint8_t *) p : nullptr;
    };
    std::vector<ze_physical_mem_handle_t> phys;
    auto map = [&](ze_device_handle_t zd, uint8_t * p, size_t bytes) -> bool {
        ze_physical_mem_desc_t d = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, bytes };
        ze_physical_mem_handle_t h = nullptr;
        if (zePhysicalMemCreate(zc, zd, &d, &h) != ZE_RESULT_SUCCESS || !h) return false;
        if (zeVirtualMemMap(zc, p, bytes, h, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE) != ZE_RESULT_SUCCESS) return false;
        phys.push_back(h);
        return true;
    };
    uint8_t * v0 = reserve(cap);
    if (!v0 || !map(z0, v0 + cap - pg, pg) || !map(z0, v0, part)) { std::printf("card 0 VMM failed\n"); return 1; }
    ze_device_mem_alloc_desc_t dd = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0 };
    uint8_t * d1 = nullptr;
    if (zeMemAllocDevice(zc, &dd, 2 * part, 4096, z1, (void **) &d1) != ZE_RESULT_SUCCESS || !d1) { std::printf("card 1 alloc failed\n"); return 1; }
    zeContextMakeMemoryResident(zc, z1, d1, 2 * part);
    std::vector<uint8_t> host(part), back(part);
    auto fill = [&](uint8_t * dev, uint8_t tag, size_t n) {
        for (size_t i = 0; i < n; ++i) host[i] = (uint8_t) (tag ^ (i * 131) ^ (i >> 13));
        q0.memcpy(dev, host.data(), n).wait();
    };
    auto bad_of = [&](uint8_t tag, size_t n) {
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += back[i] != (uint8_t) (tag ^ (i * 131) ^ (i >> 13));
        return bad;
    };
    int fails = 0;
    auto check = [&](const char * what, sycl::queue & q, uint8_t * dst, uint8_t tag, size_t n) {
        q.memcpy(back.data(), dst, n).wait();
        const size_t b = bad_of(tag, n);
        fails += b != 0;
        std::printf("%s: %zu bad bytes of %zu\n", what, b, n);
    };
    // A
    for (int rep = 0; rep < 3; ++rep) {
        const uint8_t tag = (uint8_t) (0x21 + rep);
        fill(v0, tag, part);
        q1.memcpy(d1, v0, part).wait();
        check("A card 1 reads card 0's VMM [0, 64 MiB)", q1, d1, tag, part);
    }
    // B: grow, then peer-read the new pages
    if (!map(z0, v0 + part, part)) { std::printf("card 0 grow failed\n"); return 1; }
    for (int rep = 0; rep < 3; ++rep) {
        const uint8_t tag = (uint8_t) (0x31 + rep);
        fill(v0 + part, tag, part);
        q1.memcpy(d1 + part, v0 + part, part).wait();
        check("B card 1 reads card 0's pages mapped after A [64, 128 MiB)", q1, d1 + part, tag, part);
    }
    // C: the tail page
    {
        const uint8_t tag = 0x41;
        fill(v0 + cap - pg, tag, pg);
        q1.memcpy(d1, v0 + cap - pg, pg).wait();
        check("C card 1 reads card 0's last page (mapped at the start)", q1, d1, tag, pg);
    }
    // D: card 0 VMM -> a grow-only VMM reservation on card 1 (card 1's queue), then on card 2 (card 2's queue)
    ze_device_handle_t z2 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[2]);
    uint8_t * v1 = reserve(cap);
    uint8_t * v2 = reserve(cap);
    if (!v1 || !v2 || !map(z1, v1, part) || !map(z2, v2, part)) { std::printf("card 1/2 VMM failed\n"); return 1; }
    for (int rep = 0; rep < 2; ++rep) {
        const uint8_t tag = (uint8_t) (0x51 + rep);
        fill(v0 + part, tag, part);
        q1.memcpy(v1, v0 + part, part).wait();
        check("D card 1: card 0 VMM -> card 1 VMM", q1, v1, tag, part);
        q2.memcpy(v2, v0 + part, part).wait();
        check("D card 2: card 0 VMM -> card 2 VMM", q2, v2, tag, part);
    }
    std::printf("%s\n", fails ? "FAILURES" : "all exact");
    q0.wait(); q1.wait(); q2.wait();
    zeVirtualMemUnmap(zc, v0, part); zeVirtualMemUnmap(zc, v0 + part, part); zeVirtualMemUnmap(zc, v0 + cap - pg, pg);
    zeVirtualMemUnmap(zc, v1, part); zeVirtualMemUnmap(zc, v2, part);
    for (auto h : phys) zePhysicalMemDestroy(zc, h);
    zeVirtualMemFree(zc, v0, cap); zeVirtualMemFree(zc, v1, cap); zeVirtualMemFree(zc, v2, cap);
    zeMemFree(zc, d1);
    return fails ? 2 : 0;
}
