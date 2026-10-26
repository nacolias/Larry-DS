#!/usr/bin/env python3
# patch-barstage.py [TIER_CPP] - larry, STRATA_XE_BAR_STAGE=1 (default 0 until measured): the prompt path's pageable
# arena experts written by the CPU straight into the expert cards' VRAM through their BARs (larry: 32 GiB BARs, all
# of VRAM CPU-visible) instead of a pinned hot ring + a DMA a chunk.  The cards' flex regions (the prefetch slots;
# decode's promotion slots) and stream buffers become buffer objects made through the xe uAPI (VRAM only,
# NEEDS_VISIBLE_VRAM, WC), mmap'd, exported and imported into Level Zero as device allocations of their card.
# bench/dqb/bar_probe (abwT79): the export takes no host RAM, the card reads exactly what the CPU wrote, one thread
# writes 21 GB/s (the CCD's write link; plain or streaming stores) and with pinned DMAs beside it the two share ~27
# GB/s (the x16 uplink) - while the hot rings' stagers wait half their time for ring DMAs queued behind the pinned
# chunks in the card's one copy engine (prompt staging 14.5-16.5 GB/s).  Pinned experts keep their DMAs.  The slot's
# barrier on the copy queue and the calls' dependencies stay as they were: every CPU write is out (sfence + a read
# through the BAR) before its stager marks the expert done.  STRATA_XE_HOTRING_CHECK=1 compares every staged expert.
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_BAR_STAGE" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""#include <pthread.h>
#include <sys/mman.h>
""", """#include <pthread.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <dirent.h>
#include <fcntl.h>
#include <drm/drm.h>
#include <drm/xe_drm.h>
""")
rep("""inline bool local_alloc_on() {""", """// larry, STRATA_XE_BAR_STAGE (patch-barstage.py): card memory the CPU writes through the card's BAR - a buffer object
// in the card's VRAM made through the xe uAPI (VRAM only, NEEDS_VISIBLE_VRAM, WC), mmap'd, and imported into Level Zero
// as a device allocation of its card (made resident there).  g_bar: by device address; g_bar_dev: per PCI address the
// card's render node and its full-BAR VRAM region (-1: not possible there).
struct BarBuf { int fd; uint32_t handle; uint8_t* cpu; size_t bytes; };
std::mutex g_bar_mu;
std::map<uintptr_t, BarBuf> g_bar;
std::map<std::string, std::pair<int, int>> g_bar_dev;
std::atomic<int64_t> g_bar_put{0};   ///< bytes written through BARs (the prefetch summary)
inline bool bar_stage_on() { static const bool on = env_i("STRATA_XE_BAR_STAGE", 0) != 0; return on; }
/// a BAR-mapped buffer of `bytes` on q's card; nullptr: not possible (the caller allocates as before)
inline uint8_t* bar_alloc(sycl::queue& q, size_t bytes) {
    std::string pci;
    try { pci = q.get_device().get_info<sycl::ext::intel::info::device::pci_address>(); } catch (...) { return nullptr; }
    std::lock_guard<std::mutex> lk(g_bar_mu);
    auto it = g_bar_dev.find(pci);
    if (it == g_bar_dev.end()) {
        int fd = -1, inst = -1;
        const std::string dir = "/sys/bus/pci/devices/" + pci + "/drm";
        if (DIR* d = opendir(dir.c_str())) {
            while (dirent* de = readdir(d))
                if (fd < 0 && !std::strncmp(de->d_name, "renderD", 7))
                    fd = open(("/dev/dri/" + std::string(de->d_name)).c_str(), O_RDWR | O_CLOEXEC);
            closedir(d);
        }
        drm_xe_device_query dq = {};
        dq.query = DRM_XE_DEVICE_QUERY_MEM_REGIONS;
        if (fd >= 0 && ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &dq) == 0 && dq.size) {
            std::vector<uint8_t> qb(dq.size);
            dq.data = (uintptr_t) qb.data();
            if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &dq) == 0) {
                const auto* mr = (const drm_xe_query_mem_regions*) qb.data();
                for (uint32_t i = 0; i < mr->num_mem_regions && inst < 0; ++i) {
                    const drm_xe_mem_region& r = mr->mem_regions[i];
                    if (r.mem_class == DRM_XE_MEM_REGION_CLASS_VRAM && r.cpu_visible_size >= r.total_size) inst = r.instance;
                }
            }
        }
        it = g_bar_dev.emplace(pci, std::make_pair(fd, inst)).first;
        std::fprintf(stderr, "xe tier: BAR staging on %s: %s\\n", pci.c_str(),
                     fd >= 0 && inst >= 0 ? "VRAM fully CPU-visible" : "not possible (card memory as before)");
    }
    const int fd = it->second.first, inst = it->second.second;
    if (fd < 0 || inst < 0) return nullptr;
    const size_t sz = (bytes + 65535) & ~(size_t) 65535;
    drm_xe_gem_create gc = {};
    gc.size = sz;
    gc.placement = 1u << inst;
    gc.flags = DRM_XE_GEM_CREATE_FLAG_NEEDS_VISIBLE_VRAM;
    gc.cpu_caching = DRM_XE_GEM_CPU_CACHING_WC;
    if (ioctl(fd, DRM_IOCTL_XE_GEM_CREATE, &gc)) return nullptr;
    auto close_bo = [&] { drm_gem_close cl = {}; cl.handle = gc.handle; ioctl(fd, DRM_IOCTL_GEM_CLOSE, &cl); };
    drm_xe_gem_mmap_offset mo = {};
    mo.handle = gc.handle;
    void* cpu = MAP_FAILED;
    if (ioctl(fd, DRM_IOCTL_XE_GEM_MMAP_OFFSET, &mo) == 0) cpu = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mo.offset);
    if (cpu == MAP_FAILED) { close_bo(); return nullptr; }
    drm_prime_handle ph = {};
    ph.handle = gc.handle;
    ph.flags = DRM_CLOEXEC | DRM_RDWR;
    if (ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &ph)) { munmap(cpu, sz); close_bo(); return nullptr; }
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ze_external_memory_import_fd_t imp = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD, nullptr, ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, ph.fd};
    ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &imp, 0, 0};
    void* p = nullptr;
    const bool ok = zeMemAllocDevice(zc, &dd, sz, 4096, zd, &p) == ZE_RESULT_SUCCESS && p &&
                    zeContextMakeMemoryResident(zc, zd, p, sz) == ZE_RESULT_SUCCESS;
    close(ph.fd);
    if (!ok) {
        if (p) zeMemFree(zc, p);
        munmap(cpu, sz);
        close_bo();
        return nullptr;
    }
    // the CPU mapping's pages now (TTM maps a 64 KiB run a fault): not inside the first prompt's staging
    for (size_t o = 0; o < sz; o += 65536) (void) *(volatile const uint8_t*) ((const uint8_t*) cpu + o);
    g_bar[(uintptr_t) p] = BarBuf{fd, gc.handle, (uint8_t*) cpu, sz};
    return (uint8_t*) p;
}
/// device address `p` inside a BAR buffer -> the CPU's address of it (nullptr: not in one)
inline uint8_t* bar_cpu(const void* p) {
    std::lock_guard<std::mutex> lk(g_bar_mu);
    if (g_bar.empty()) return nullptr;
    auto it = g_bar.upper_bound((uintptr_t) p);
    if (it == g_bar.begin()) return nullptr;
    --it;
    const uintptr_t off = (uintptr_t) p - it->first;
    return off < it->second.bytes ? it->second.cpu + off : nullptr;
}
inline bool bar_free(sycl::queue& q, void* p) {
    BarBuf b;
    {
        std::lock_guard<std::mutex> lk(g_bar_mu);
        auto it = g_bar.find((uintptr_t) p);
        if (it == g_bar.end()) return false;
        b = it->second;
        g_bar.erase(it);
    }
    zeMemFree(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context()), p);
    munmap(b.cpu, b.bytes);
    drm_gem_close cl = {};
    cl.handle = b.handle;
    ioctl(b.fd, DRM_IOCTL_GEM_CLOSE, &cl);
    return true;
}
/// `n` bytes into a BAR mapping (streaming stores), then out to the card: a read through the BAR completes only
/// after the posted writes before it
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
__attribute__((target("avx2")))
#endif
inline void bar_put(uint8_t* dst, const uint8_t* src, size_t n) {
    if (!n) return;
    size_t i = 0;
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
    if (((uintptr_t) dst & 31) == 0)
        for (; i + 128 <= n; i += 128) {
            const __m256i a = _mm256_loadu_si256((const __m256i*) (src + i));
            const __m256i b = _mm256_loadu_si256((const __m256i*) (src + i + 32));
            const __m256i c = _mm256_loadu_si256((const __m256i*) (src + i + 64));
            const __m256i d = _mm256_loadu_si256((const __m256i*) (src + i + 96));
            _mm256_stream_si256((__m256i*) (dst + i), a);
            _mm256_stream_si256((__m256i*) (dst + i + 32), b);
            _mm256_stream_si256((__m256i*) (dst + i + 64), c);
            _mm256_stream_si256((__m256i*) (dst + i + 96), d);
        }
    if (i < n) std::memcpy(dst + i, src + i, n - i);
    _mm_sfence();
#else
    std::memcpy(dst, src, n);
#endif
    (void) *(volatile const uint8_t*) (dst + n - 1);
    g_bar_put += (int64_t) n;
}
inline bool local_alloc_on() {""")
rep("""inline void local_free(sycl::queue& q, void* p) {
    if (!p) return;
""", """inline void local_free(sycl::queue& q, void* p) {
    if (!p) return;
    if (bar_free(q, p)) return;   // larry, patch-barstage
""")
rep("""    addr_log("tier local alloc", q, p, bytes);
    return (uint8_t*) p;
}
""", """    addr_log("tier local alloc", q, p, bytes);
    return (uint8_t*) p;
}
/// larry, patch-barstage: a card's flex region / stream buffer - BAR-mapped with STRATA_XE_BAR_STAGE=1 when possible
inline uint8_t* card_alloc(sycl::queue& q, size_t bytes) {
    if (bar_stage_on())
        if (uint8_t* p = bar_alloc(q, bytes)) { addr_log("tier BAR alloc", q, p, bytes); return p; }
    return local_alloc(q, bytes);
}
""")
# the flex regions and the stream buffer
rep("""            c.flex = local_alloc(c.q, (size_t) want);
            c.flex_cap = c.flex ? (size_t) want : 0;
""", """            c.flex = card_alloc(c.q, (size_t) want);   // larry, patch-barstage
            c.flex_cap = c.flex ? (size_t) want : 0;
""")
rep("""    c.flex = left >= 8 * (int64_t) im.blob_max ? local_alloc(c.q, (size_t) left) : nullptr;
""", """    c.flex = left >= 8 * (int64_t) im.blob_max ? card_alloc(c.q, (size_t) left) : nullptr;   // larry, patch-barstage
""")
rep("""            c.flex = local_alloc(c.q, 2 * cap);
            c.flex_cap = c.flex ? 2 * cap : 0;
""", """            c.flex = card_alloc(c.q, 2 * cap);   // larry, patch-barstage
            c.flex_cap = c.flex ? 2 * cap : 0;
""")
rep("""    c.stream = local_alloc(c.q, want);
    c.stream_bytes = c.stream ? want : 0;
""", """    c.stream = card_alloc(c.q, want);   // larry, patch-barstage
    c.stream_bytes = c.stream ? want : 0;
""")
# the prefetch stagers: pageable experts through the slot's BAR mapping
rep("""        if (hot) {
            XeCard& cd = *im.cards[(size_t) ps.card[(size_t) e]];
            hot_put(cd.pq, cd.pbuf[s] + ps.off[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e),
                    ps.soff[(size_t) e] == SIZE_MAX, im.pf_rings[(size_t) idx], hev, hnext);
        } else {""", """        uint8_t* bc = nullptr;   // larry, STRATA_XE_BAR_STAGE: a pageable expert straight into the slot through the BAR
        if (hot && ps.soff[(size_t) e] != SIZE_MAX) bc = bar_cpu(im.cards[(size_t) ps.card[(size_t) e]]->pbuf[s] + ps.off[(size_t) e]);
        if (bc) {
            bar_put(bc, im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));
        } else if (hot) {
            XeCard& cd = *im.cards[(size_t) ps.card[(size_t) e]];
            hot_put(cd.pq, cd.pbuf[s] + ps.off[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e),
                    ps.soff[(size_t) e] == SIZE_MAX, im.pf_rings[(size_t) idx], hev, hnext);
        } else {""")
# the overlap helper's round-0 streams and the later rounds' streams
rep("""                    if (!im.apin[key] && ring_ovl) {   // larry, patch-ovlring
                        oev = hot_put(c.oq, c.stream + soff[(size_t) e], src, bytes, false, c.hring, c.hring_ev, c.hring_next);""",
    """                    if (!im.apin[key]) {   // larry, STRATA_XE_BAR_STAGE: through the stream buffer's BAR mapping
                        if (uint8_t* bc = bar_cpu(c.stream + soff[(size_t) e])) {
                            bar_put(bc, src, bytes);
                            ++pcie;
                            ++g_bp.streamed;
                            continue;
                        }
                    }
                    if (!im.apin[key] && ring_ovl) {   // larry, patch-ovlring
                        oev = hot_put(c.oq, c.stream + soff[(size_t) e], src, bytes, false, c.hring, c.hring_ev, c.hring_next);""")
rep("""                    if (pinned) {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.stream + soff[(size_t) e], hb, bytes);
                    } else if (hot_on()""", """                    uint8_t* bc = pinned ? nullptr : bar_cpu(c.stream + soff[(size_t) e]);   // larry, STRATA_XE_BAR_STAGE
                    if (pinned) {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.stream + soff[(size_t) e], hb, bytes);
                    } else if (bc) {
                        bar_put(bc, hb, bytes);
                        if (hot_check_on())
                            hot_check(im, c, c.q, c.stream + soff[(size_t) e], hb, bytes, l, e, "stream (BAR)");
                    } else if (hot_on()""")
rep("""    if (xe_prof())
        std::fprintf(stderr, "xe prefetch: %lld experts (%.2f GB) planned, %lld used\\n", (long long) im.pf_planned,
                     im.pf_bytes / 1e9, (long long) im.pf_used);
""", """    if (xe_prof())
        std::fprintf(stderr, "xe prefetch: %lld experts (%.2f GB) planned, %lld used; %.2f GB through BARs so far\\n",
                     (long long) im.pf_planned, im.pf_bytes / 1e9, (long long) im.pf_used, g_bar_put.load() / 1e9);
""")
open(P, "w").write(s)
print("patched", P)
