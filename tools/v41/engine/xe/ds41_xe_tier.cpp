// tools/v41/engine/xe/ds41_xe_tier.cpp - see ds41_xe_tier.hpp.
#include "ds41_xe_tier.hpp"
#include "xe_kernels.hpp"
#include "xe_cpus.hpp"
#include "../../../../third_party/llama.cpp/ggml/src/ggml-sycl/fastmv.hpp"   // the head split's mat-vec

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <oneapi/mkl/blas.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <condition_variable>
#include <deque>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <thread>

#include <pthread.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <dirent.h>
#include <fcntl.h>
#include <drm/drm.h>
#include <drm/xe_drm.h>
#include <cerrno>
#include <sched.h>
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
#include <immintrin.h>
#endif
#include <dlfcn.h>

namespace strata::v41 {

namespace cpu = strata::kernels::cpu;
using xe::XeJob;
using xe::kMaxJobTok;

namespace {

// larry, STRATA_XE_FN16=2 (patch-fn16k.py): every expert card can read the dense card's memory (peer access on)
bool g_fn16_peer = true;

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double env_d(const char* k, double def) {
    const char* v = std::getenv(k);
    return v && *v ? std::atof(v) : def;
}
int env_i(const char* k, int def) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoi(v) : def;
}

int64_t mem_available_bytes() {
    std::ifstream f("/proc/meminfo");
    std::string k;
    int64_t v = 0;
    std::string unit;
    while (f >> k >> v >> unit)
        if (k == "MemAvailable:") return v * 1024;
    return -1;
}

constexpr double kGiB = 1073741824.0;
constexpr size_t kSlab = (size_t) 1 << 30;       ///< device expert storage is bump-allocated in 1 GiB slabs

/// larry, STRATA_XE_LOCAL_ALLOC (default 1): device memory that no other card touches (expert slabs, flex regions) is
/// allocated by Level Zero directly.  SYCL's allocations go through UR / UMF, which make each one resident on every P2P
/// peer in the context, and on larry (card pairs behind ACS redirect) that costs ~1 GiB of host RAM per GiB once the
/// peers hold memory: the expert cards' 3 x 28 GiB of slabs +13.8 GiB of driver-held RAM, the dense card's 1 GiB seed
/// slabs 60-600 ms and +1 GiB each (bench/dqb/alloc_probe3.cpp; natively: +28 MiB, 0 ms).
/// larry, STRATA_ADDR_LOG=1: a line per device allocation / free (diagnostic)
inline bool addr_log_on() { static const bool on = std::getenv("STRATA_ADDR_LOG") != nullptr; return on; }
inline void addr_log(const char* what, sycl::queue& q, const void* p, size_t bytes) {
    if (!addr_log_on()) return;
    std::string pci = "?";
    try { pci = q.get_device().get_info<sycl::ext::intel::info::device::pci_address>(); } catch (...) {}
    std::fprintf(stderr, "addr: %s %s %p +0x%zx\n", what, pci.c_str(), p, bytes);
}
// larry, STRATA_XE_BAR_STAGE (patch-barstage.py): card memory the CPU writes through the card's BAR - a buffer object
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
        std::fprintf(stderr, "xe tier: BAR staging on %s: %s\n", pci.c_str(),
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
    g_bar[(uintptr_t) p] = BarBuf{fd, gc.handle, (uint8_t*) cpu, sz};
    return (uint8_t*) p;   // (its CPU pages fault in at first touch: bar_touch, patch-barstage2)
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
/// larry, patch-barstage2: fault a BAR buffer's CPU pages in (TTM maps a 64 KiB run a fault); ms taken
inline double bar_touch(const void* p, size_t bytes) {
    uint8_t* c = bar_cpu(p);
    if (!c) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t o = 0; o < bytes; o += 65536) (void) *(volatile const uint8_t*) (c + o);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
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
inline bool local_alloc_on() { static const bool on = std::getenv("STRATA_XE_LOCAL_ALLOC") == nullptr || std::atoi(std::getenv("STRATA_XE_LOCAL_ALLOC")) != 0; return on; }
inline uint8_t* local_alloc(sycl::queue& q, size_t bytes) {
    if (!local_alloc_on()) return sycl::malloc_device<uint8_t>(bytes, q);
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ze_device_mem_alloc_desc_t d = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    void* p = nullptr;
    if (zeMemAllocDevice(zc, &d, bytes, 4096, zd, &p) != ZE_RESULT_SUCCESS || !p) return nullptr;
    // resident on this card now: such memory is otherwise committed at first touch (the free-VRAM query does not see
    // it, and a card out of VRAM then fails the touching command - no spill into host RAM as UR / UMF memory does)
    if (zeContextMakeMemoryResident(zc, zd, p, bytes) != ZE_RESULT_SUCCESS) { zeMemFree(zc, p); return nullptr; }
    addr_log("tier local alloc", q, p, bytes);
    return (uint8_t*) p;
}
/// larry, patch-barstage: a card's flex region / stream buffer - BAR-mapped with STRATA_XE_BAR_STAGE=1 when possible
inline uint8_t* card_alloc(sycl::queue& q, size_t bytes) {
    if (bar_stage_on())
        if (uint8_t* p = bar_alloc(q, bytes)) { addr_log("tier BAR alloc", q, p, bytes); return p; }
    return local_alloc(q, bytes);
}
inline void local_free(sycl::queue& q, void* p) {
    if (!p) return;
    if (bar_free(q, p)) return;   // larry, patch-barstage
    addr_log("tier local free", q, p, 0);
    if (!local_alloc_on()) { sycl::free(p, q); return; }
    zeMemFree(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context()), p);
}
constexpr int kMaxDecodeJobs = 64;               ///< a verify pass of 8 tokens routes at most 48 distinct experts
constexpr size_t kStageAlign = 256;
constexpr int kPcBounce = 3;                     ///< STRATA_XE_PROMOTE: pinned bounces (promotions in flight)

/// larry, STRATA_XE_GROUTE: where a (layer, expert) is, for the dense card's route kernel (a pinned table kept with
/// card_of / dptr: rt_put), and what the kernel leaves the host (pinned): its card entries' count and the CPU pool's
/// entries (Ent's fields)
struct RtEnt {
    const uint8_t* blob;                ///< the device address on `card`
    uint32_t up_off, down_off;
    uint16_t gu_type, d_type, gu_row, d_row;   ///< (larry, engine1: CPU-pool experts too - the TP core's balance sizes them)
    int32_t card;                       ///< -1: on no card
    int32_t alt;                        ///< larry, engine1: a second card holding a copy (STRATA_XE_REPLICA_GIB), or -1
    const uint8_t* alt_blob;            ///< the copy on `alt`
};
struct RtCpu { int32_t e, nt, tok[kMaxJobTok], k[kMaxJobTok]; float w[kMaxJobTok]; };
struct RtHdr { int32_t ncard, ncpu, pad[2]; RtCpu cpu[kMaxDecodeJobs]; };

}  // namespace

/// One card.
struct XeCard {
    int idx = 0;
    int gpu = -1;                   ///< its Level Zero GPU index (STRATA_XE_DEVICES' numbering)
    sycl::device dev;
    sycl::queue q;
    std::string name;
    bool dense = false;          ///< holds the dense half (elastic expert set: released by release_cache)
    std::vector<uint8_t*> slabs; ///< expert storage
    size_t slab_used = 0;        ///< bytes used in the last slab
    int64_t budget = 0, used = 0, n_experts = 0;
    // decode / verify scratch
    uint8_t* stage_dev = nullptr;   ///< [jobs | X]
    uint8_t* stage_host = nullptr;
    size_t stage_bytes = 0;
    float* H = nullptr;             ///< [kMaxDecodeJobs][8][n_ff]
    float* OUT = nullptr;           ///< [2][8][n_embd]: the kernels' sum, then (dense card) the host's
    float* part_host = nullptr;     ///< [8][n_embd]
    float* dslots = nullptr;        ///< dense card only: [cards + 1][8][n_embd], every card's decode part (+ the CPU's)
    bool busy = false;              ///< this call enqueued work on it
    sycl::event stage_ev;           ///< the last copy out of stage_host (async decode: wait before refilling it)
    // STRATA_XE_ARM: the call queued ahead of its routing (Ds41XeTier::arm)
    int* arm_flag = nullptr;        ///< device (this card): releases, written by the dense card's release queue
    int* arm_src = nullptr;         ///< pinned: a ring of 64 source slots (one cache line each) for those copies
    int arm_src_i = 0;
    int* arm_seen = nullptr;        ///< device: releases consumed by the spin kernel
    int arm_rel = 0;                ///< host copy of *arm_flag
    bool armed = false;             ///< an armed sequence waits for its release
    bool arm_routed = false;        ///< STRATA_XE_GROUTE: it pulls its jobs from the dense card's route list (released there)
    /// STRATA_XE_ARM_GRAPH: a routed armed sequence per shape (jobs, tokens, kernel width), recorded once and replayed
    std::map<uint32_t, std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>>> agraphs;
    std::map<uint32_t, const float*> agraph_x;   ///< per recording: the X its quant kernel reads (baked in)
    int arm_ng = 0;                 ///< its job count (null-padded)
    sycl::event arm_ev;             ///< its last command (the part's copy to the dense card)
    XeJob arm_null{};               ///< a null job (nt 0) on a blob this card holds
    bool arm_null_ok = false;
    // larry, STRATA_XE_PROMOTE: this card's share of the decode-time cache of promoted experts (Ds41XeImpl::pc_*)
    uint8_t* pc_mem = nullptr;      ///< pc_n slots of Ds41XeImpl::pc_slot bytes (planar experts)
    int pc_n = 0;
    std::vector<int64_t> pc_key;    ///< slot -> expert key; -1 free, -2 being filled
    std::vector<uint64_t> pc_tick;  ///< slot -> the decode call that last used it
    int pc_busy = 0;                ///< promotions of this card in flight (at most the tier's pc_par)
    uint8_t* pc_tmp = nullptr;      ///< larry, engine1 (pc_devrepack): one slot's bytes the card repacks a promoted blob into
    /// decode / verify: the card's command sequence of a call (stage copy, quantise, gate/up, down, copy to the dense
    /// card's slot) recorded once per shape (jobs, tokens, kernel variant) and replayed - one submission, not five
    std::map<uint32_t, std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>>> dgraphs;
    std::map<uint32_t, const float*> dgraph_x;   ///< per recording: the direct X it reads (null: the staged one)
    // prompt-chunk scratch (grown on demand)
    float* cX = nullptr; size_t cX_cap = 0;          ///< floats
    sycl::half* cXh = nullptr; size_t cXh_cap = 0;   ///< larry, STRATA_XE_FN16: X as it arrives (halves; card-local)
    float* cOUT = nullptr; size_t cOUT_cap = 0;      ///< floats
    // larry, STRATA_SHEXP_TIER (patch-shexptier.py): this card's shared-expert slices of layer sh_layer in fp16 (per
    // slice gate | up rows, then down) and a call's scratch for them (X rows, gate | up, SwiGLU), the row indices 0..
    sycl::half* sh_w = nullptr; size_t sh_w_cap = 0; int64_t sh_layer = -1;
    uint8_t* sh_buf = nullptr; sycl::half* sh_x = nullptr; float* sh_g = nullptr; sycl::half* sh_h = nullptr;
    size_t sh_rows = 0, sh_nf = 0;
    int32_t* sh_iota = nullptr; size_t sh_iota_n = 0;
    float* cOUT_host = nullptr; size_t cOUT_host_cap = 0;
    int32_t* cS = nullptr; float* cW = nullptr; size_t cS_cap = 0;   ///< entries
    XeJob* cjobs = nullptr; float* cH = nullptr; float* cY = nullptr; int cjobs_cap = 0;
    XeJob* cjobs_host = nullptr;    ///< pinned: cjobs' host copy (no shared USM: its page-fault handler deadlocked)
    uint8_t* dq_host = nullptr; size_t dq_host_cap = 0;   ///< pinned: xe_dqgemm's job / tile tables of a call
    uint8_t* nw_host = nullptr; size_t nw_cap = 0;   ///< larry, STRATA_XE_DENSE_LITE: pinned per-call staging (no host waits)
    uint8_t* stream = nullptr; size_t stream_bytes = 0;   ///< streamed (arena / file) experts of one chunk round
    /// the experts the last prompt call left in `stream` (expert, offset), all of layer st_layer, ending at st_end
    int64_t st_layer = -1; size_t st_end = 0;
    std::vector<std::pair<int32_t, int64_t>> st_have;
    uint8_t* gscratch = nullptr; size_t gscratch_cap = 0;  ///< the GEMM path's operands (bytes)
    float* cX2 = nullptr; size_t cX2_cap = 0;              ///< the dense card's copy of the other cards' sum
    sycl::half* cX2h = nullptr; size_t cX2h_cap = 0;       ///< STRATA_XE_COMBINE=2: the parts as fp16 (slot = card)
    uint8_t* bounce = nullptr; size_t bounce_cap = 0;      ///< pinned staging for streamed experts in pageable memory
    uint8_t* hring = nullptr;                              ///< larry, STRATA_XE_HOTRING: the card thread's pinned ring
    std::vector<sycl::event> hring_ev;                     ///< its chunks' DMAs (a slot is reused once its copy is done)
    int hring_next = 0;
    uint8_t* hchk = nullptr;                               ///< STRATA_XE_HOTRING_CHECK: [blob_max] pinned readback
    // prompt prefetch (Ds41XeTier::prefetch_begin): a copy queue and two layer buffers (ping-pong by layer parity)
    sycl::queue pq;
    sycl::queue oq;                 ///< prompt calls: the overflow experts' DMAs (STRATA_XE_OVERLAP)
    uint8_t* pbuf[2] = {nullptr, nullptr};
    size_t pbuf_cap = 0;
    /// larry: device memory kept for the card's lifetime (allocated at load): a prompt's prefetch buffers (pbuf), the
    /// decode promotion cache's slots (pc_mem) - allocating either on the way costs 100s of ms
    uint8_t* flex = nullptr;
    size_t flex_cap = 0;
    sycl::event pev[2];             ///< the last prefetch copy into pbuf[s]
    // the planar int8 path (xe_planar.cpp): every expert a card holds is stored repacked
    uint8_t* rstage = nullptr;      ///< [blob_max]: a file-layout blob on its way to its planar slot
    std::mutex rmu;                 ///< makes (copy into rstage, repack out of it) one unit for concurrent uploaders
    xe::XeQ8 xq, hq;                ///< decode / verify: X [8][n_embd], H [kMaxDecodeJobs][8][n_ff]
    sycl::half* xt = nullptr;       ///< STRATA_XE_XMX: X as [n_embd][8] fp16, H as [kMaxDecodeJobs][n_ff][8] fp16
    sycl::half* ht = nullptr;
    xe::XeQ8 cxq; size_t cxq_rows = 0;   ///< prompt chunks: X [n][n_embd]
    xe::XeQ8 chq; int chq_jobs = 0;      ///< prompt chunks: H [jobs][8][n_ff]

    uint8_t* alloc_expert(size_t bytes) {
        if (slabs.empty() || slab_used + bytes > kSlab) {
            uint8_t* p = local_alloc(q, kSlab);
            if (!p) return nullptr;
            slabs.push_back(p);
            slab_used = 0;
        }
        uint8_t* p = slabs.back() + slab_used;
        slab_used += (bytes + 255) & ~(size_t) 255;
        used += (int64_t) bytes;
        ++n_experts;
        return p;
    }
    void free_experts() {
        q.wait();
        for (uint8_t* p : slabs) local_free(q, p);
        slabs.clear();
        slab_used = 0;
        used = 0;
        n_experts = 0;
    }
    int64_t free_vram() const {
        if (dev.has(sycl::aspect::ext_intel_free_memory)) return (int64_t) dev.get_info<sycl::ext::intel::info::device::free_memory>();
        return -1;
    }
};

struct Ds41XeImpl {
    // STRATA_GPU_TL: the decode timeline - stamps (a one-lane kernel setting a pinned flag) whose arrival a monitor
    // thread times on the host clock: rings[0] = the dense queue, rings[1 + c] = card c's queue
    struct TlRing {
        volatile uint32_t* flag = nullptr;   // host USM [kTlCap]
        std::vector<int> id;
        std::vector<double> t;               // host us when the monitor saw the flag
        std::atomic<uint32_t> issued{0}, seen{0};
        uint32_t done = 0;                   // the flush's position
    };
    static constexpr uint32_t kTlCap = 1u << 16;
    std::vector<std::unique_ptr<TlRing>> tl_rings;
    std::thread tl_mon;
    std::atomic<bool> tl_stop{false};
    double tl_sum[17][17] = {};   // dense: [from id][to id] us
    long tl_n[17][17] = {};
    std::vector<double> tl_cbusy, tl_clag, tl_csum_lag;   // per card: start->end, dense 1->start, end->dense 3
    std::vector<double> tl_chunk, tl_chunk_max;          // per card: promotion chunks' device time (stamps 12 -> 13)
    std::vector<double> tl_op_busy, tl_op_lag;           // per card: oproj part 20 -> 21, dense 5 -> 21
    std::vector<long> tl_op_n;
    std::vector<long> tl_chunk_n, tl_chunk_late;         // ... how many, how many ended after the next readback
    std::vector<long> tl_cn;
    long tl_tokens = 0;
    std::vector<double> tl_copy_lag, tl_copy_sum; std::vector<long> tl_copy_n;   // per card: end -> copy done -> sum
    std::vector<std::pair<double, double>> tl_hmark;   // decode calls: host us at the tier's entry, at the sum's submit
    double tl_h_entry = 0, tl_h_sum = 0; long tl_h_n = 0;
    Ds41MoeGeom g;
    Ds41MoeConfig cfg;
    Ds41BlobSource* blobs = nullptr;
    std::vector<cpu::NativeFmt> fl;    ///< per class
    std::vector<int64_t> bl;           ///< per class: blob bytes
    int64_t blob_max = 0;
    float L = std::numeric_limits<float>::infinity();
    sycl::context ctx;
    std::vector<std::unique_ptr<XeCard>> cards;
    int dense = -1;                    ///< index of the dense card in `cards`
    // residency: per (layer, expert)
    std::vector<int8_t> card_of;       ///< -1 = not on a card
    std::vector<const uint8_t*> dptr;  ///< device address on its card
    std::vector<uint8_t*> aptr;        ///< host address in the arena, null = not in the arena (file tier)
    std::vector<uint8_t*> achunks;     ///< the arena's allocations (SYCL host USM, pinned; or pageable)
    bool arena_usm = false;
    int64_t arena_pinned = 0;
    int64_t arena_bytes = 0, arena_n = 0, file_n = 0;
    bool placed = false;
    bool cache_enabled = true;
    double slot_gib = 0.0;
    std::vector<std::pair<int32_t, int32_t>> dense_set;   ///< experts currently on the dense card (seeded in ranked order)
    // larry, engine1 (STRATA_XE_REPLICA_GIB): copies of the expert cards' hottest experts on the dense card; the TP core
    // hands a pass's job of one to whichever holder finishes sooner (route_jobs).  rep_set in upload order, their planar
    // bytes pinned once (rep_host + rep_off), the dense card's copies by key (rep_ptr) in slabs of their own
    std::vector<std::pair<int32_t, int32_t>> rep_set;
    std::vector<size_t> rep_off;
    uint8_t* rep_host = nullptr;
    std::vector<uint8_t*> rep_ptr;
    std::vector<uint8_t*> rep_slabs;
    size_t rep_slab_used = 0;
    int64_t rep_n = 0, rep_bytes = 0;
    std::vector<std::pair<int32_t, int32_t>> seed_ranked; ///< the ranked list of the last seed (regrow_cache continues it)
    int64_t resident_n = 0;
    // CPU pool for the arena experts of a decode / verify call
    std::unique_ptr<cpu::ExpertPool> pool;
    // the LM head split over the cards (mv_hook): per weight, the expert cards' row slices and their scratch
    struct HeadSplit {
        bool ok = false;
        int64_t r[9] = {};              ///< rows [r[i], r[i+1]) on part i (part 0: the dense card)
        std::vector<XeCard*> c;         ///< part i >= 1
        std::vector<uint8_t*> w;
        std::vector<char> own;          ///< w[i] is the split's own copy (else the TP core's shard)
        std::vector<int8_t*> xq;
        std::vector<float*> xd, y;
        // larry, patch-headstage.py: card-0 staging, the only card-0 memory the other cards' copies touch (imported once)
        uint8_t* sx = nullptr;          ///< ncols * K int8 then ncols * K / 32 scales (as the cards' xq: xd follows it)
        float* sy = nullptr;            ///< part i >= 1: its rows x ncols at (r[i] - r[1]) * 8
        sycl::queue* sq = nullptr;      ///< the dense queue sx / sy were made for (another queue: the direct copies)
    };
    std::map<const void*, HeadSplit> heads;
    /// larry, engine1: rows the TP core holds already as planar slices, by weight: (GPU, r0, r1, address)
    struct GivenSlice { int gpu; int64_t r0, r1; const void* p; };
    std::map<const void*, std::vector<GivenSlice>> head_given;
    // larry, patch-regrowasync: a queued regrow of the dense card's cache - its experts, published by regrow_poll
    struct RgEnt { size_t key; uint8_t* p; std::pair<int32_t, int32_t> le; };
    std::vector<RgEnt> rg_list;
    std::vector<uint8_t> rg_mark;   ///< by key: in rg_list (resident() counts them for the route bias)
    sycl::event rg_ev;              ///< the dense card's queue behind the last copy / repack of rg_list
    bool rg_pending = false;
    double rg_t0 = 0;
    // larry, STRATA_SHEXP_TIER: per layer the shared expert's TP shards on the expert cards (card index, gate | up
    // 2 nf x H, down H x nf - planar Q6_K, nf); copies made here (rank 0's) in sh_own
    struct ShSlice { int card; const uint8_t* gu; const uint8_t* dn; int64_t nf; };
    std::vector<std::vector<ShSlice>> sh_slices;
    std::vector<std::pair<XeCard*, uint8_t*>> sh_own;
    bool sh_next = false, sh_call = false;
    // STRATA_TP_OPROJ: per layer (keyed by wo_a), the attention output's parts - part 0 the dense card, part i >= 1 an
    // expert card: groups [g0[i], g0[i + 1]) of wo_a (its slices as they are) and wo_b's matching columns (gathered)
    struct OSplit {
        bool ok = false;
        int np = 0;
        int64_t g0[9] = {};
        std::vector<XeCard*> c;           ///< part i >= 1: c[i - 1]
        std::vector<uint8_t*> wa, wb;     ///< part i's weights (part 0's wa: wo_a itself)
        std::vector<int8_t*> xq, aq;      ///< part i >= 1: its groups' int8 o, its int8 wo_a output (8 columns)
        std::vector<float*> xd, oa, ad, y;
        /// STRATA_TP_GRAPH (default 1): part i's commands for a width, recorded once (g[i][ncols])
        std::vector<std::array<std::shared_ptr<sycl::ext::oneapi::experimental::command_graph<
            sycl::ext::oneapi::experimental::graph_state::executable>>, 9>> g;
    };
    std::map<const void*, OSplit> osplits;
    // the dense card's side, shared by the layers (in-order on its queue): o quantised (all groups), its own part's
    // wo_a output and its quantisation, the expert cards' partial sums (8 columns each)
    int8_t* os_xq = nullptr; int8_t* os_aq = nullptr;
    float* os_xd = nullptr; float* os_oa = nullptr; float* os_ad = nullptr; float* os_slots = nullptr;
    int64_t os_calls = 0, os_declined = 0;
    // STRATA_XE_CPU_ASYNC (default 1): a decode / verify call's CPU-pool experts run on a helper thread (it drives
    // the pool) while the main thread dispatches the cards - they used to start only after every card's submission
    struct CpuTask { int64_t l; const void* ents; const float* x; float* hs; size_t n_out; };   ///< ents: std::vector<const Ent*>
    bool cpu_async = true;
    std::thread cpu_thr;
    std::mutex cpu_mu;
    std::condition_variable cpu_cv;
    std::atomic<int> cpu_state{0};     ///< 0 idle, 1 posted, 2 done, 3 stop
    CpuTask cpu_task{};
    std::vector<cpu::ExpertJobMulti> cjobs;
    std::vector<cpu::ActQ> actq;           ///< per token
    std::vector<uint8_t> nact;             ///< [token][format slot][kNativeActBytes]
    std::vector<int> nact_fmt;             ///< format slots: distinct gu_act values
    std::vector<float> cpart;              ///< [job][8][n_embd] CPU expert outputs
    // STRATA_XE_CPU_LIGHT (prompt calls, run_big): the light arena experts' CPU-pool buffers
    std::vector<cpu::ActQ> lt_actq;        ///< per token of the call
    std::vector<uint8_t> lt_nact;          ///< [token][format slot][kNativeActBytes]
    std::vector<uint8_t> lt_have;          ///< [token][1 + format slot]: quantised already this call
    std::vector<float> lt_cpart;           ///< [job][8][n_embd] their outputs
    std::vector<cpu::ExpertJobMulti> lt_jobs;
    float* lt_hs = nullptr;                ///< pinned: their weighted sum [n][n_embd]
    size_t lt_hs_cap = 0;
    float* lt_dev = nullptr;               ///< the dense card: lt_hs copied there for the combine
    size_t lt_dev_cap = 0;
    float* lt_x = nullptr;                 ///< pinned: a pipelined call's X read back off the dense card
    size_t lt_x_cap = 0;
    // larry, STRATA_XE_LT_COMPACT (patch-ltcompact.py): the call's light rows only
    std::vector<int32_t> lt_slot;          ///< [token]: its compact row, -1
    int32_t* lt_rows = nullptr;            ///< pinned: compact row -> token (read by card 0's gather / scatter kernels)
    size_t lt_rows_cap = 0;
    float* lt_xg = nullptr;                ///< the dense card: the gathered X rows (read back into lt_x)
    size_t lt_xg_cap = 0;
    int lt_nrows = -1;                     ///< this call's compact rows (-1: lt_hs is full width)
    int64_t lt_calls = 0, lt_experts = 0, lt_streamed = 0;
    double lt_bytes = 0, lt_q_ms = 0, lt_run_ms = 0, lt_sum_ms = 0, lt_wait_ms = 0;
    float* hsum = nullptr;                 ///< [8][n_embd] host-side sum (pinned: its copy to the dense card is async)
    float* hslots = nullptr;               ///< [cards][8][n_embd] pinned: the expert cards' decode parts (STRATA_XE_HSLOTS)
    bool host_slots = false;
    std::vector<uint8_t> ftmp;             ///< file-tier reads (decode)
    // larry, STRATA_XE_PROMOTE (default 1): a decode / verify expert the CPU pool ran (it is on no card) is copied into
    // a slot cache on an expert card and runs there from then on: a helper thread repacks its arena copy planar into a
    // pinned bounce (the card's own repack of a Q3_K expert took 739 us and held up its next call) -> the card's queue
    // copies it into the slot, STRATA_XE_PROMOTE_CHUNK_MB (1) a call, after that call's experts (the card idles there;
    // a second queue of a spinning card does not run, and the dense card's copy engine writing 11 MB into a card took
    // ~1 s and held up the dense card's own copies) -> resident.  LRU over the slots; dropped when a prompt's prefetch
    // begins (its buffers want the VRAM).  Every submission is the main thread's (submitting from several threads
    // faulted a copy engine once).
    struct PcJob {
        int64_t key = -1;
        int card = -1, slot = -1, bounce = -1;
        int stage = 0;                 ///< 0 bounce being filled (helper), 1 copying (chunks), 3 last chunk queued
        const uint8_t* src = nullptr;
        size_t sent = 0;               ///< bytes of the copy queued so far
        std::atomic<int> filled{0};
        sycl::event ev;
        double t_queued = 0;           ///< larry: when it went to the helpers (ms)
    };
    int pc_mode = -1;                  ///< -1 not read yet, 0 off, 1 on
    bool pc_ready = false;             ///< the slots exist (allocated a few calls into a decode)
    int64_t pc_calls = 0;              ///< decode calls since the last drop
    bool pc_tried = false;             ///< pc_setup ran since the last drop
    size_t pc_slot = 0;                ///< bytes a slot
    std::vector<int32_t> pc_where;     ///< key -> card << 16 | slot (published); -1; -2 in flight
    std::vector<uint8_t> pc_uses;      ///< key -> its CPU-pool uses (saturating; STRATA_XE_PROMOTE_MIN)
    uint8_t* pc_bounce = nullptr;      ///< pinned: pc_nb x pc_slot
    int pc_nb = kPcBounce;             ///< bounces (larry, engine1: more with STRATA_TP_PROMOTE_PAR)
    int pc_par = 1;                    ///< promotions in flight a card (1; engine1's between-pass driving: more)
    std::vector<sycl::queue*> pc_q;    ///< larry, engine1: the queue a card's promotion copies go to (null: the card's own)
    std::map<int, size_t> tp_defer;    ///< larry, engine1: by GPU, the TP core's VRAM not mapped yet (the flex holds it)
    /// larry, engine1, STRATA_XE_TP_EVICT: per card its static experts in upload order (slabs fill in that order) and the
    /// index from which they may go back to the arena (whole slabs; each has an arena copy)
    std::vector<std::vector<std::pair<int32_t, int32_t>>> card_set;
    std::vector<size_t> evict_from;
    int64_t evicted_n = 0;
    std::vector<uint8_t*> evict_bufs;  ///< the evicted experts' host copies (read from the model file at eviction)
    double dense_bonus_gib = 0;        ///< larry, engine1: card 0 VRAM the pager gave back (on top of STRATA_XE_DENSE_GIB)
    std::map<int, std::pair<uint8_t*, size_t>> dbg;   ///< debug captures (pinned), by slot
    /// larry, engine1 (STRATA_TP, STRATA_TP_PC_DEVREPACK=1; default 0 since 2026-10-24): a promotion copies the expert's
    /// FILE layout into its slot (pinned arena blobs straight from the arena, pageable ones through a bounce by a plain
    /// memcpy) and the card repacks it there (slot -> pc_tmp -> slot).  The host's old repack (10-20 ms an expert) took
    /// the cores the ggml draft's submission needed; the streaming 16-block one costs what the plain copy did, and the
    /// card's repack kernels ran beside the TP core's draft on every card (its first cross-card sum up to 5.5 ms late)
    bool pc_devrepack = false;
    std::vector<int> pc_bfree;
    std::list<PcJob> pc_jobs;          ///< in flight (stable addresses: the helper holds pointers)
    std::thread pc_thr;
    std::vector<std::thread> pc_thr_more;   ///< larry, engine1: more repack helpers (STRATA_TP_PC_THREADS)
    std::mutex pc_mu;
    std::condition_variable pc_cv;
    std::deque<PcJob*> pc_fill;        ///< bounces for the helper to fill (under pc_mu)
    bool pc_stop = false;
    uint64_t pc_clock = 0;
    int pc_rr = 0;
    int64_t pc_done = 0, pc_evicted = 0, pc_hit = 0, pc_busy_skip = 0, pc_bad = 0;
    double pc_push_us = 0, pc_poll_us = 0;
    int64_t pc_chunks = 0;
    std::atomic<int64_t> pc_fill_us{0}, pc_fill_n{0}, pc_wait_us{0};   ///< the helpers' repack time, count, time a job sat queued
    std::atomic<int> pool_busy{0};     ///< larry, engine1: a TP pass's CPU pool is computing (the fills step aside: STRATA_TP_PC_PACE)
    Ds41MoeStats st;
    std::mutex mu;
    bool planar = true;                ///< STRATA_XE_PLANAR (default 1): experts repacked on the cards, int8 kernels
    bool tier_graphs = true;           ///< STRATA_XE_GRAPH (default 1): decode / verify card work as replayed SYCL graphs
    bool tier_graphs_multi = false;    ///< STRATA_XE_GRAPH_MULTI (default 0): verify calls too (jobs padded to a bucket)
    bool arm_on = false, arm_multi = false;   ///< STRATA_XE_ARM / STRATA_XE_ARM_MULTI
    int64_t armed_l = -1;              ///< the layer / width the armed sequences are for
    int armed_nt = 0;
    const float* arm_x = nullptr;      ///< the X they read (the dense half's pinned readback mirror)
    int* arm_err = nullptr;            ///< pinned: a spin kernel gave up waiting (its card ran null jobs)
    std::unique_ptr<sycl::queue> relq; ///< on the dense card: the releases (copies into the armed cards' flags)
    int64_t arm_used = 0, arm_disarmed = 0;
    // larry, STRATA_XE_GROUTE (default 0): the dense card's route kernel dispatches decode / verify calls (route)
    bool groute_on = false;
    bool groute_want = false;          ///< groute_next(): the coming attn_router's tier call is one route() may take
    RtEnt* rt = nullptr;               ///< pinned [layers * experts]
    volatile uint64_t rt_version = 0;  ///< larry, engine1: bumped by every rt_put (the TP core refreshes its copies)
    std::vector<uint32_t> rt_dirty;    ///< larry, engine1: the keys rt_put changed since tp_take_dirty (bounded; overflow: all)
    bool rt_dirty_all = true;
    std::vector<std::pair<int64_t, size_t>> tp_want;   ///< larry, engine1: the last pass's CPU-pool experts (promotion candidates)
    // larry, engine1, STRATA_XE_CPU_LOG=<file> (diagnostics): a line per TP pass (T), per CPU-pool expert (E: layer,
    // expert, its tokens, class, arena pinned / pageable / file, earlier CPU uses, promotion state, bytes), per layer
    // with any (L: experts, bytes, wall us, the pool's phase ms) and per promotion event (Q queued, P published, V
    // evicted, S skipped); STRATA_XE_CPU_TWICE=1 also runs each layer's CPU experts again (weights hot in the caches)
    std::FILE* cpu_log = nullptr;
    int64_t tp_pass = 0;
    // larry, engine1, STRATA_XE_USAGE_OUT=<file> (diagnostics): per routed expert the TP passes that ran it (a job) and its
    // picks (tokens), written at close with its card when decode began (-1: the CPU pool) and its bytes - the input of
    // placement studies (bench/engine1/place.py)
    std::vector<uint32_t> use_jobs, use_picks;
    std::vector<int8_t> use_card;
    XeJob* rt_jobs = nullptr;          ///< dense card: [cards][kMaxDecodeJobs] - the routed call's jobs, per card
    XeJob* rt_null = nullptr;          ///< dense card: [cards] - each card's null job (the lists' padding)
    RtHdr* rt_hdr = nullptr;           ///< pinned: the route kernel's counts and CPU entries
    int64_t routed_l = -1;             ///< the routed call: layer, width, job bucket, kernel width, the cards it fed
    int routed_nt = 0, routed_ng = 0, routed_mv = 0;
    uint32_t routed_mask = 0;
    const float* rt_x = nullptr;       ///< its X on the dense card (the router's fn)
    bool rt_dense_armed = false;       ///< the dense card's own share is queued (arm: behind the shared expert)
    sycl::event rt_dense_ev;
    int64_t groute_used = 0, groute_miss = 0, groute_bad = 0;
    bool x_direct = true;              ///< STRATA_XE_XDIRECT (default 1): the cards read a pinned X straight from the host
    sycl::queue* dense_q = nullptr;    ///< set_dense_queue: decode sums go there, no host wait
    sycl::event last_sum;              ///< the previous decode call's sum on dense_q (its slots / stages are reused)
    sycl::event hsum_ev;               ///< the copy of hsum (the CPU part) into the dense card's slot
    // prompt prefetch (Ds41XeTier::prefetch_begin)
    std::vector<uint8_t> apin;         ///< per (layer, expert): its arena copy is pinned (DMA-able as it is)
    struct PfSlot {
        int64_t layer = -1;            ///< the layer it holds or is filling (-1: none)
        bool enqueued = false;         ///< its copies are on the cards' copy queues
        std::vector<int8_t> card;      ///< [n_experts]: the card the expert goes to, -1 = not prefetched
        std::vector<size_t> off;       ///< [n_experts]: offset in that card's pbuf
        std::vector<size_t> soff;      ///< [n_experts]: offset in stage, SIZE_MAX = straight from the pinned arena
        uint8_t* stage = nullptr;      ///< pinned host staging for the layer's pageable experts
        int staged = 0;                ///< 0 nothing to stage / done, 1 requested (stager threads), under pf_mu
        int64_t next = 0;              ///< the next expert a stager takes (under pf_mu)
        int active = 0;                ///< stagers copying for it (under pf_mu)
        int64_t n = 0, bytes = 0;      ///< experts / bytes planned
        double t_enq = 0;              ///< when its copies went on the queues (STRATA_XE_PROF)
        bool timed = false;
        double t_plan = 0, t_staged = 0;   ///< larry, STRATA_XE_PF_TRACE: planned / last copy submitted (now_ms)
        bool rec = false;              ///< larry, STRATA_XE_PF_NOBLOCK: its last stager recorded the cards' barriers
        bool traced = false;
    };
    PfSlot pf[2];
    sycl::half* fn16 = nullptr; size_t fn16_cap = 0;   ///< larry, STRATA_XE_FN16: the call's X as fp16 on the dense card
    size_t pf_stage_cap = 0;           ///< bytes per slot's stage
    bool pf_on = false;
    int64_t pf_last = -1;
    int64_t pf_used = 0, pf_planned = 0, pf_bytes = 0;   ///< over a batch (the prof line)
    std::vector<std::thread> pf_thr;
    std::vector<uint8_t*> pf_rings;    ///< larry, STRATA_XE_HOTRING: a pinned ring per stager thread
    std::atomic<int64_t> hc_n{0}, hc_bad{0};   ///< STRATA_XE_HOTRING_CHECK: experts compared / differing
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    bool pf_stop = false;
    // pipelined prompt batch: the tier call in flight (run_chunk_dev_async)
    std::thread run_thr;
    bool run_ok = true;

    int64_t key(int64_t l, int64_t e) const { return l * g.n_experts + e; }
    int cls(int64_t l, int64_t e) const { return g.cls(l, e); }
    size_t bbytes(int64_t l, int64_t e) const { return (size_t) bl[(size_t) cls(l, e)]; }
};

namespace {

/// The card's routed-expert job for (layer, expert) with up to 8 tokens.
XeJob make_job(const Ds41XeImpl& im, int64_t l, int64_t e, const uint8_t* dptr) {
    const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
    XeJob j;
    j.blob = dptr;
    j.up_off = (uint32_t) f.up_off;
    j.down_off = (uint32_t) f.down_off;
    j.gu_type = (uint16_t) f.gu_type;
    j.d_type = (uint16_t) f.d_type;
    j.gu_row = (uint16_t) f.gu_row;
    j.d_row = (uint16_t) f.d_row;
    j.nt = 0;
    return j;
}

/// STRATA_XE_GROUTE: the route table's entry of `key` from card_of / dptr - after every change of them.  The host never
/// changes them while a route kernel is queued (it waits for each call's readbacks, which are behind the kernel).
void rt_put(Ds41XeImpl& im, size_t key) {
    if (!im.rt) return;
    im.rt_version = im.rt_version + 1;
    if (!im.rt_dirty_all) {
        if (im.rt_dirty.size() < 4096) im.rt_dirty.push_back((uint32_t) key);
        else im.rt_dirty_all = true;
    }
    RtEnt& r = im.rt[key];
    const int64_t l = (int64_t) key / im.g.n_experts, e = (int64_t) key % im.g.n_experts;
    if (im.g.routed(l)) {   // the shapes of every routed expert (larry, engine1: the CPU pool's too - the TP core's balance)
        const XeJob j = make_job(im, l, e, nullptr);
        r.up_off = j.up_off;
        r.down_off = j.down_off;
        r.gu_type = j.gu_type;
        r.d_type = j.d_type;
        r.gu_row = j.gu_row;
        r.d_row = j.d_row;
    }
    r.alt = -1;
    r.alt_blob = nullptr;
    const int c = im.card_of[key];
    if (c < 0 || !im.dptr[key]) { r.card = -1; r.blob = nullptr; return; }
    r.blob = im.dptr[key];
    r.card = c;
    if (key < im.rep_ptr.size() && im.rep_ptr[key]) {   // larry, engine1: its copy on the dense card
        r.alt = im.dense;
        r.alt_blob = im.rep_ptr[key];
    }
}

/// Copies the file-layout blob of (l, e) at `src` (host USM, or device-readable) into its planar slot `dst` on card
/// `c`: through c.rstage, repacked there (in queue order: the next upload's copy waits for this repack).  Returns the
/// copy's event (`src` is free once it completes).
sycl::event upload_planar(const Ds41XeImpl& im, XeCard& c, int64_t l, int64_t e, const uint8_t* src, uint8_t* dst) {
    const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
    if (!im.planar) return c.q.memcpy(dst, src, f.bytes);
    std::lock_guard<std::mutex> lk(c.rmu);
    sycl::event ev = c.q.memcpy(c.rstage, src, f.bytes);
    xe::xe_repack_blob(c.q, c.rstage, dst, f.gu_type, f.d_type, f.up_off, f.down_off, (int) im.g.n_embd, (int) im.g.n_ff);
    return ev;
}

bool alloc_q8(xe::XeQ8& a, size_t rows, size_t n, sycl::queue& q) {
    if (a.q) { sycl::free(a.q, q); sycl::free(a.d, q); sycl::free(a.s, q); }
    a.q = sycl::malloc_device<int8_t>(rows * n, q);
    a.d = sycl::malloc_device<float>(rows * n / 32, q);
    a.s = sycl::malloc_device<int32_t>(rows * n / 16, q);
    return a.q && a.d && a.s;
}

/// Where the host copy of (l, e) is: the arena, or (file tier) a read into `tmp`.
const uint8_t* host_blob(Ds41XeImpl& im, int64_t l, int64_t e, std::vector<uint8_t>& tmp) {
    uint8_t* a = im.aptr[(size_t) im.key(l, e)];
    if (a) return a;
    if (tmp.size() < im.bbytes(l, e)) tmp.resize(im.bbytes(l, e));
    im.blobs->read(l, e, tmp.data());
    ++im.st.file_tier;
    return tmp.data();
}

/// Distinct experts of a call with their tokens (<= 8 per entry; an expert with more tokens takes several entries).
struct Ent {
    int32_t e = -1;
    int nt = 0;
    int32_t tok[kMaxJobTok];
    int32_t k[kMaxJobTok];   ///< the routing slot (t * top_k + k) each token used
    float w[kMaxJobTok];
};

/// Groups (token, slot) pairs by expert, routing order kept within an expert.
void group(const int32_t* ids, const float* w, int n, int K, int64_t n_experts, std::vector<Ent>& out,
           std::vector<int32_t>& first) {
    out.clear();
    first.assign((size_t) n_experts, -1);   // the expert's LAST entry (the one still filling)
    for (int t = 0; t < n; ++t)
        for (int k = 0; k < K; ++k) {
            const int32_t e = ids[(size_t) t * K + k];
            if (e < 0 || e >= n_experts) continue;
            int32_t& fi = first[(size_t) e];
            if (fi < 0 || out[(size_t) fi].nt == kMaxJobTok) {
                fi = (int32_t) out.size();
                Ent en;
                en.e = e;
                out.push_back(en);
            }
            Ent& en = out[(size_t) fi];
            en.tok[en.nt] = t;
            en.k[en.nt] = t * K + k;
            en.w[en.nt] = w[(size_t) t * K + k];
            ++en.nt;
        }
}

}  // namespace

Ds41XeTier::Ds41XeTier() : im_(new Ds41XeImpl) {}
Ds41XeTier::~Ds41XeTier() { close(); }

bool Ds41XeTier::init(const Ds41MoeGeom& geom, Ds41BlobSource* blobs, const Ds41MoeConfig& cfg, const void* dense_dev,
                      std::string& err) {
    close();
    Ds41XeImpl& im = *im_;
    im.g = geom;
    im.cfg = cfg;
    im.blobs = blobs;
    im.L = geom.swiglu_limit;
    im.planar = env_i("STRATA_XE_PLANAR", 1) != 0;
    im.tier_graphs = env_i("STRATA_XE_GRAPH", 1) != 0;
    im.x_direct = env_i("STRATA_XE_XDIRECT", 1) != 0;
    im.cpu_async = env_i("STRATA_XE_CPU_ASYNC", 1) != 0;
    im.tier_graphs_multi = env_i("STRATA_XE_GRAPH_MULTI", 0) != 0;   // measured: no gain (the cards' dispatch is not submit-bound)
    const int64_t NCL = geom.n_classes();
    im.fl.assign((size_t) NCL, cpu::NativeFmt());
    im.bl.assign((size_t) NCL, 0);
    for (int64_t c = 0; c < NCL; ++c) {
        if (!geom.per_expert() && !geom.routed(c)) continue;
        if (!ds41_moe_class_layout(geom, c, im.fl[(size_t) c], err)) return false;
        const cpu::NativeFmt& f = im.fl[(size_t) c];
        if (!xe::xe_type_ok(f.gu_type) || !xe::xe_type_ok(f.d_type)) {
            err = "xe tier: class " + std::to_string(c) + " has gate/up type " + std::to_string(f.gu_type) +
                  " / down type " + std::to_string(f.d_type) + " (the Xe kernels run Q3_K, Q2_0 and PTQ1_0)";
            return false;
        }
        if (f.gu_row > 65535 || f.d_row > 65535) { err = "xe tier: row too wide"; return false; }
        im.bl[(size_t) c] = (int64_t) f.bytes;
        im.blob_max = std::max(im.blob_max, im.bl[(size_t) c]);
    }
    if (geom.n_embd % 256 || geom.n_ff % 256) { err = "xe tier: n_embd / n_ff must be multiples of 256"; return false; }
    if (!blobs->open(err)) return false;

    // ---- the cards: the dense half's platform, its GPUs (STRATA_XE_DEVICES picks / orders them)
    sycl::platform plat;
    bool have_plat = false;
    for (const sycl::platform& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        if (p.get_devices(sycl::info::device_type::gpu).empty()) continue;
        if (dense_dev) {
            const sycl::context c = p.khr_get_default_context();
            if (sycl::get_pointer_type(dense_dev, c) == sycl::usm::alloc::unknown) continue;
        }
        plat = p;
        have_plat = true;
        break;
    }
    if (!have_plat) { err = "xe tier: no Level Zero GPU platform (that owns the dense half's buffers)"; return false; }
    im.ctx = plat.khr_get_default_context();
    std::vector<sycl::device> devs = plat.get_devices(sycl::info::device_type::gpu);
    std::vector<int> pick;
    if (const char* s = std::getenv("STRATA_XE_DEVICES")) {
        for (const char* p = s; *p;) {
            pick.push_back(std::atoi(p));
            while (*p && *p != ',') ++p;
            if (*p == ',') ++p;
        }
    } else {
        for (int i = 0; i < (int) devs.size(); ++i) pick.push_back(i);
    }
    int dense_card = -1;
    sycl::device ddev;
    if (dense_dev) ddev = sycl::get_pointer_device(dense_dev, im.ctx);
    for (int i : pick) {
        if (i < 0 || i >= (int) devs.size()) { err = "xe tier: STRATA_XE_DEVICES names device " + std::to_string(i); return false; }
        auto c = std::make_unique<XeCard>();
        c->idx = (int) im.cards.size();
        c->gpu = i;
        c->dev = devs[(size_t) i];
        // larry, STRATA_XE_QPRIO (patch-qprio.py): the expert cards' prefetch queue low (1), their call queue high (2)
        static const int qprio = env_i("STRATA_XE_QPRIO", 0);
        const bool expert_card = !(dense_dev && devs[(size_t) i] == ddev);
        namespace qp = sycl::ext::oneapi::property::queue;
        if (qprio >= 2 && expert_card)
            c->q = sycl::queue(im.ctx, c->dev, sycl::property_list{sycl::property::queue::in_order(), qp::priority_high()});
        else
            c->q = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        if (qprio >= 1 && expert_card)
            c->pq = sycl::queue(im.ctx, c->dev, sycl::property_list{sycl::property::queue::in_order(), qp::priority_low()});
        else
            c->pq = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        c->oq = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        c->name = c->dev.get_info<sycl::info::device::name>();
        if (dense_dev && c->dev == ddev) {
            c->dense = true;
            dense_card = c->idx;
        }
        im.cards.push_back(std::move(c));
    }
    if (dense_dev && dense_card < 0) { err = "xe tier: the dense half's card is not among STRATA_XE_DEVICES"; return false; }
    im.dense = dense_card;
    if (env_i("STRATA_XE_FN16", 0) >= 2 && im.dense >= 0) {   // larry, patch-fn16k.py: the peer reads of FN16's X
        const sycl::device d0 = im.cards[(size_t) im.dense]->dev;
        for (auto& cp : im.cards) {
            if (cp->dense) continue;
            if (!cp->dev.ext_oneapi_can_access_peer(d0)) { g_fn16_peer = false; continue; }
            try { cp->dev.ext_oneapi_enable_peer_access(d0); } catch (const std::exception&) {}   // on already: fine
        }
        std::fprintf(stderr, "xe tier: STRATA_XE_FN16=2: the expert cards read X by peer loads%s\n",
                     g_fn16_peer ? "" : " - NO PEER ACCESS: the copy engine reads it");
    }

    // ---- scratch per card
    const int64_t H = geom.n_embd, F = geom.n_ff;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.stage_bytes = kMaxDecodeJobs * sizeof(XeJob) + kStageAlign + (size_t) kMaxJobTok * H * 4;
        c.stage_dev = sycl::malloc_device<uint8_t>(c.stage_bytes, c.q);
        c.stage_host = sycl::malloc_host<uint8_t>(c.stage_bytes, c.q);
        c.H = sycl::malloc_device<float>((size_t) kMaxDecodeJobs * kMaxJobTok * F, c.q);
        c.OUT = sycl::malloc_device<float>((size_t) 2 * kMaxJobTok * H, c.q);
        c.part_host = sycl::malloc_host<float>((size_t) kMaxJobTok * H, c.q);
        if (!c.stage_dev || !c.stage_host || !c.H || !c.OUT || !c.part_host) { err = "xe tier: scratch allocation failed"; return false; }
        c.rstage = sycl::malloc_device<uint8_t>((size_t) im.blob_max, c.q);
        if (!c.rstage || !alloc_q8(c.xq, kMaxJobTok, (size_t) H, c.q) || !alloc_q8(c.hq, (size_t) kMaxDecodeJobs * kMaxJobTok, (size_t) F, c.q)) {
            err = "xe tier: int8 scratch allocation failed";
            return false;
        }
        c.xt = sycl::malloc_device<sycl::half>((size_t) 8 * H, c.q);
        c.ht = sycl::malloc_device<sycl::half>((size_t) kMaxDecodeJobs * F * 8, c.q);
        if (!c.xt || !c.ht) { err = "xe tier: fp16 scratch allocation failed"; return false; }
        if (c.dense) {
            c.dslots = sycl::malloc_device<float>((im.cards.size() + 1) * kMaxJobTok * H, c.q);
            if (!c.dslots) { err = "xe tier: slot allocation failed"; return false; }
        } else {
            c.arm_flag = sycl::malloc_device<int>(1, c.q);
            c.arm_seen = sycl::malloc_device<int>(1, c.q);
            c.arm_src = sycl::malloc_host<int>(64 * 16, c.q);
            if (!c.arm_flag || !c.arm_seen || !c.arm_src) { err = "xe tier: arm counters allocation failed"; return false; }
            c.q.memset(c.arm_flag, 0, sizeof(int)).wait();
            c.q.memset(c.arm_seen, 0, sizeof(int)).wait();
            std::memset(c.arm_src, 0, 64 * 16 * sizeof(int));
        }
    }
    const size_t NK = (size_t) geom.n_layers * geom.n_experts;
    im.card_of.assign(NK, -1);
    im.dptr.assign(NK, nullptr);
    im.aptr.assign(NK, nullptr);
    im.slot_gib = cfg.slot_gib;
    // larry: STRATA_XE_POOL_WORKERS = the CPU pool's workers (cores 1..n; default --threads, else every core but
    // core 0) - with fewer, the last cores take the helpers (STRATA_XE_HELPER_CPUS, xe_cpus.hpp)
    (void) xe::process_cpus();   // the mask helpers inherit, saved before any thread pins itself
    const int pw = env_i("STRATA_XE_POOL_WORKERS", cfg.threads > 0 ? cfg.threads : 0);
    im.pool.reset(new cpu::ExpertPool(pw > 0 ? pw : 0));
    im.cjobs.resize(cpu::ExpertPool::kMaxSplitMulti);
    im.actq.resize(kMaxJobTok);
    for (const cpu::NativeFmt& f : im.fl)
        if (f.bytes && !cpu::q2_native_kernels(f.gu_type) &&
            std::find(im.nact_fmt.begin(), im.nact_fmt.end(), f.gu_act) == im.nact_fmt.end())
            im.nact_fmt.push_back(f.gu_act);
    im.nact.assign((size_t) kMaxJobTok * std::max<size_t>(1, im.nact_fmt.size()) * cpu::kNativeActBytes, 0);
    if (!im.hsum) im.hsum = sycl::malloc_host<float>((size_t) kMaxJobTok * H, im.ctx);
    std::memset(im.hsum, 0, (size_t) kMaxJobTok * H * sizeof(float));
    // STRATA_XE_HSLOTS (default 0): the expert cards write their decode / verify parts into pinned host slots that the
    // dense card's sum reads over PCIe instead of card-to-card copies - measured slower (decode 33 -> 31 tok/s, verify
    // 52 -> 62 ms/pass: kernels writing host memory crawl), kept for the record
    im.host_slots = env_i("STRATA_XE_HSLOTS", 0) != 0;
    if (im.host_slots && !im.hslots) im.hslots = sycl::malloc_host<float>(im.cards.size() * kMaxJobTok * H, im.ctx);
    if (!im.hslots) im.host_slots = false;
    std::fprintf(stderr, "xe tier: %zu card(s), dense half on card %d; %lld classes, largest blob %.2f MiB, swiglu limit %g\n",
                 im.cards.size(), im.dense, (long long) NCL, im.blob_max / 1048576.0, (double) im.L);
    return true;
}

namespace {

/// Reads `list` (layer, expert) blobs from the source, `nth` threads; `sink(i, host_ptr)` places blob i (the host
/// pointer is a reader-owned pinned buffer, valid until sink returns).
template <class Sink>
void read_all(Ds41XeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& list, int nth, Sink sink) {
    // larry, STRATA_XE_LOAD_DIRECT (default 1; patch-loaddirect.py): through the source's O_DIRECT twin (read_staged,
    // a per-thread aligned scratch) - the buffered read() put ~150 GB through the page cache at every start, which
    // swapped the arena out and had the xe shrinker back up the engine's buffers (abwT138)
    static const bool direct = env_i("STRATA_XE_LOAD_DIRECT", 1) != 0;
    const size_t sc = direct ? (im.blobs->read_staged_scratch() + 4095) & ~(size_t) 4095 : 0;
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < nth; ++i)
        th.emplace_back([&]() {
            uint8_t* buf = sycl::malloc_host<uint8_t>((size_t) im.blob_max, im.ctx);
            uint8_t* scr = sc ? (uint8_t*) std::aligned_alloc(4096, sc) : nullptr;
            for (;;) {
                const size_t k = next.fetch_add(1);
                if (k >= list.size()) break;
                if (scr) im.blobs->read_staged(list[k].first, list[k].second, buf, scr, sc);
                else im.blobs->read(list[k].first, list[k].second, buf);
                sink(k, buf);
            }
            std::free(scr);
            sycl::free(buf, im.ctx);
        });
    for (auto& t : th) t.join();
}

}  // namespace

namespace {
// larry, engine1 (STRATA_XE_REPLICA_GIB): the dense card's copies of the expert cards' hottest experts (below)
void rep_prepare(Ds41XeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& order);
int64_t rep_upload(Ds41XeImpl& im);
void rep_drop(Ds41XeImpl& im);
}  // namespace

namespace {
bool hot_on();
size_t hot_chunk();
int hot_slots();
void rt_put(Ds41XeImpl& im, size_t key);
void rep_drop(Ds41XeImpl& im);
/// larry, engine1, STRATA_XE_TP_EVICT (default 0): see tp_make_room / build_arena_ranked
bool tp_evict_on() { static const bool on = env_i("STRATA_XE_TP_EVICT", 0) != 0; return on; }
int64_t tp_defer_flex() {
    static const int64_t b = (int64_t) (env_d("STRATA_XE_TP_DEFER_FLEX_GIB", 0.8) * 1073741824.0);
    return b;
}
/// the card's evictable last slabs back to the arena until `need` bytes are free; their experts become arena experts.
/// Between TP passes only: no kernel reads them (the card's queues are idle; promotions write flex slots).
int64_t evict_slabs(Ds41XeImpl& im, XeCard& c, int64_t need) {
    const size_t ci = (size_t) c.idx;
    if (ci >= im.card_set.size()) return 0;
    auto& set = im.card_set[ci];
    const size_t from = im.evict_from[ci];
    if (set.size() <= from || c.slabs.size() < 2) return 0;
    const double t0 = now_ms();
    rep_drop(im);   // the dense card's copies of expert-card experts first (route tables: those entries change too)
    c.q.wait();
    int64_t n = 0, bytes = 0;
    while (c.free_vram() < need && c.slabs.size() > 1 && set.size() > from) {
        uint8_t* last = c.slabs.back();
        // larry, patch-pcrace: the slab's experts wherever they are in the set (not only its suffix) - moved to its end
        auto in_slab = [&](const std::pair<int32_t, int32_t>& le) {
            const size_t key = (size_t) im.key(le.first, le.second);
            const uint8_t* p = im.dptr[key];
            return im.card_of[key] == (int8_t) c.idx && p >= last && p < last + kSlab;
        };
        bool fixed_in = false;
        for (size_t i = 0; i < from && i < set.size(); ++i) fixed_in = fixed_in || in_slab(set[i]);
        if (fixed_in) break;     // the slab holds experts that are not evictable
        const size_t k = (size_t) (std::stable_partition(set.begin() + (std::ptrdiff_t) from, set.end(),
                                                         [&](const std::pair<int32_t, int32_t>& le) { return !in_slab(le); }) -
                                   set.begin());
        // their host copies first, from the model file (pageable; the arena's own blobs stay where they are)
        std::vector<std::pair<int32_t, int32_t>> list(set.begin() + (std::ptrdiff_t) k, set.end());
        std::vector<size_t> off(list.size());
        size_t tot = 0;
        for (size_t i = 0; i < list.size(); ++i) { off[i] = tot; tot += (im.bbytes(list[i].first, list[i].second) + 4095) & ~(size_t) 4095; }
        const int64_t mem_keep = (int64_t) (env_d("STRATA_XE_TP_EVICT_MEM_GIB", 4.0) * 1073741824.0);
        const int64_t avail = mem_available_bytes();
        if (avail > 0 && avail - (int64_t) tot < mem_keep) {
            std::fprintf(stderr, "xe tier: card %d: no eviction - %.2f GiB of host memory available, %.2f GiB needed\n", c.idx,
                         avail / 1073741824.0, tot / 1073741824.0);
            break;
        }
        uint8_t* hb = (uint8_t*) std::aligned_alloc(4096, tot);
        if (!hb) break;
        read_all(im, list, 8, [&](size_t i, const uint8_t* buf) { std::memcpy(hb + off[i], buf, im.bbytes(list[i].first, list[i].second)); });
        im.evict_bufs.push_back(hb);
        for (size_t i = 0; i < list.size(); ++i) {
            const size_t key = (size_t) im.key(list[i].first, list[i].second);
            im.aptr[key] = hb + off[i];
            im.apin[key] = 0;
        }
        im.arena_bytes += (int64_t) tot;
        im.arena_n += (int64_t) list.size();
        for (size_t i = k; i < set.size(); ++i) {
            const auto& le = set[i];
            const size_t key = (size_t) im.key(le.first, le.second);
            im.card_of[key] = -1;
            im.dptr[key] = nullptr;
            rt_put(im, key);
            c.used -= (int64_t) im.bbytes(le.first, le.second);
            bytes += (int64_t) im.bbytes(le.first, le.second);
            --c.n_experts;
            ++n;
        }
        set.resize(k);
        const int64_t fr0 = c.free_vram();
        local_free(c.q, last);
        c.slabs.pop_back();
        c.slab_used = kSlab;   // the slabs left are full: the next expert opens a new one
        // the xe driver shows freed VRAM a little later: wait until the driver shows this slab (or the reading stops
        // rising) before the next test - else every evictable slab went at once
        for (int i = 0, same = 0; i < 50 && c.free_vram() < fr0 + (int64_t) kSlab * 9 / 10 && same < 5; ++i) {
            const int64_t f0 = c.free_vram();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            same = c.free_vram() == f0 ? same + 1 : 0;
        }
    }
    if (!n) return 0;
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
    im.evicted_n += n;
    std::fprintf(stderr, "xe tier: card %d gave %lld experts (%.2f GiB) back to the arena for the TP core's KV rows (%.0f ms; "
                 "%.2f GiB free)\n", c.idx, (long long) n, bytes / 1073741824.0, now_ms() - t0, c.free_vram() / 1073741824.0);
    return n;
}
}  // namespace

bool Ds41XeTier::build_arena_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked_in, std::string& err) {
    Ds41XeImpl& im = *im_;
    if (im.placed) return true;
    const double t0 = now_ms();
    const int64_t E = im.g.n_experts, NL = im.g.n_layers;
    // the full order: the ranking, then every routed (layer, expert) it did not name
    std::vector<std::pair<int32_t, int32_t>> order;
    std::vector<char> seen((size_t) (NL * E), 0);
    for (const auto& le : ranked_in) {
        if (le.first < 0 || le.first >= NL || le.second < 0 || le.second >= E || !im.g.routed(le.first)) continue;
        char& s = seen[(size_t) im.key(le.first, le.second)];
        if (s) continue;
        s = 1;
        order.push_back(le);
    }
    for (int64_t l = 0; l < NL; ++l)
        for (int64_t e = 0; e < E; ++e)
            if (im.g.routed(l) && !seen[(size_t) im.key(l, e)]) order.push_back({(int32_t) l, (int32_t) e});
    if (!im.cfg.exclude.empty()) {
        std::vector<std::pair<int32_t, int32_t>> kept;
        for (const auto& le : order)
            if (!im.cfg.exclude[(size_t) im.key(le.first, le.second)]) kept.push_back(le);
        order.swap(kept);
    }

    // ---- budgets of the permanent cards (every card but the dense one): free VRAM - reserve
    const double reserve = env_d("STRATA_XE_RESERVE_GIB", 3.0);
    const double cap_gib = env_d("STRATA_XE_CARD_GIB", 0.0);   // >0: at most this many GiB of experts per card
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense) { c.budget = 0; continue; }
        int64_t fr = c.free_vram();
        if (fr < 0) fr = (int64_t) c.dev.get_info<sycl::info::device::global_mem_size>();
        // larry, engine1: VRAM the TP core will map later (its KV replicas) is lent to the flex region, not to experts
        // (STRATA_XE_TP_EVICT: only STRATA_XE_TP_DEFER_FLEX_GIB of it - the rest holds evictable experts, below)
        const auto td = im.tp_defer.find(c.gpu);
        if (td != im.tp_defer.end()) fr -= tp_evict_on() ? std::min<int64_t>((int64_t) td->second, tp_defer_flex()) : (int64_t) td->second;
        double gib = fr / kGiB - reserve;
        if (cap_gib > 0) gib = std::min(gib, cap_gib);
        c.budget = (int64_t) (std::max(0.0, gib) * kGiB);
        std::fprintf(stderr, "xe tier: card %d (%s): %.2f GiB free -> %.2f GiB of experts\n", c.idx, c.name.c_str(),
                     fr / kGiB, c.budget / kGiB);
    }
    // ---- deal the ranked experts over the permanent cards: each to the card with the largest free share
    std::vector<std::vector<std::pair<int32_t, int32_t>>> on((size_t) im.cards.size());
    std::vector<int64_t> left((size_t) im.cards.size(), 0);
    for (auto& cp : im.cards) left[(size_t) cp->idx] = cp->budget;
    std::vector<std::pair<int32_t, int32_t>> rest;
    // larry: past its budget a card still takes the experts its last slab holds (STRATA_XE_FILL_SLAB, default 1) - the
    // 1 GiB slab is allocated anyway, and its tail sat idle (0.3-0.9 GiB a card); sl[] replays alloc_expert's bump
    // (a new slab when used + bytes > kSlab, then used += the 256-aligned size) in the order the experts are uploaded
    static const bool fill = env_i("STRATA_XE_FILL_SLAB", 1) != 0;
    std::vector<int64_t> sl((size_t) im.cards.size(), -1);   // bytes used in the card's open slab (-1: none yet)
    int64_t filled = 0;
    for (const auto& le : order) {
        const int64_t b = (int64_t) im.bbytes(le.first, le.second);
        int best = -1;
        double share = 0.0;
        for (auto& cp : im.cards) {
            const XeCard& c = *cp;
            if (c.budget <= 0 || left[(size_t) c.idx] < b) continue;
            const double s = (double) left[(size_t) c.idx] / (double) c.budget;
            if (best < 0 || s > share) { best = c.idx; share = s; }
        }
        bool extra = false;
        if (best < 0 && fill)
            for (auto& cp : im.cards) {
                const XeCard& c = *cp;
                const int64_t u = sl[(size_t) c.idx];
                if (c.budget > 0 && u >= 0 && u + b <= (int64_t) kSlab && (best < 0 || u < sl[(size_t) best])) { best = c.idx; extra = true; }
            }
        if (best < 0) { rest.push_back(le); continue; }
        left[(size_t) best] -= b;
        int64_t& u = sl[(size_t) best];
        if (u < 0 || u + b > (int64_t) kSlab) u = 0;
        u += (b + 255) & ~(int64_t) 255;
        if (extra) filled += b;
        on[(size_t) best].push_back(le);
    }
    if (filled) std::fprintf(stderr, "xe tier: %.2f GiB more experts in the cards' last slabs (allocated anyway)\n", filled / kGiB);
    // larry, engine1, STRATA_XE_TP_EVICT: the cards' last slabs over the deferred VRAM beyond the flex's share are
    // evictable - their host copies are read from the model file when they are evicted (evict_slabs)
    im.card_set.assign(im.cards.size(), {});
    im.evict_from.assign(im.cards.size(), 0);
    for (auto& cp : im.cards) {
        const XeCard& c = *cp;
        const size_t ci = (size_t) c.idx;
        im.card_set[ci] = on[ci];
        im.evict_from[ci] = on[ci].size();
        const auto td = im.tp_defer.find(c.gpu);
        if (!tp_evict_on() || c.dense || td == im.tp_defer.end() || (int64_t) td->second <= tp_defer_flex()) continue;
        const int64_t want = (int64_t) td->second - tp_defer_flex();
        const size_t nslab = (size_t) ((want + (int64_t) kSlab - 1) / (int64_t) kSlab);
        std::vector<size_t> start;   // the index of each slab's first expert (alloc_expert's bump replayed)
        int64_t u = -1;
        for (size_t i = 0; i < on[ci].size(); ++i) {
            const int64_t b = (int64_t) im.bbytes(on[ci][i].first, on[ci][i].second);
            if (u < 0 || u + b > (int64_t) kSlab) { start.push_back(i); u = 0; }
            u += (b + 255) & ~(int64_t) 255;
        }
        if (start.size() <= nslab) continue;   // keep at least one slab
        im.evict_from[ci] = start[start.size() - nslab];
    }
    // larry, STRATA_XE_PIN_STREAM (see patch-pinstream.py): the arena's head re-ordered for the pinned budget - the
    // dense card's predicted cache (the ranked arena's first STRATA_XE_PIN_STREAM_CACHE_GIB, default its seed budget)
    // below layer L = STRATA_XE_SEED_TAIL_LAYERS, then the next-ranked experts of layers >= L while the budget lasts
    if (env_i("STRATA_XE_PIN_STREAM", 0) && im.dense >= 0) {
        const int tail_l = env_i("STRATA_XE_SEED_TAIL_LAYERS", 0);
        XeCard& dc = *im.cards[(size_t) im.dense];
        double cgib = env_d("STRATA_XE_PIN_STREAM_CACHE_GIB", 0.0);
        if (cgib <= 0) {   // dense_budget's caps as the seed will see them (the slot budget is set later, at the seed)
            cgib = env_d("STRATA_XE_DENSE_GIB", 10.0) + im.dense_bonus_gib;
            const int64_t fr = dc.free_vram();
            if (fr > 0) cgib = std::min(cgib, (fr + dc.used) / kGiB - env_d("STRATA_XE_DENSE_RESERVE_GIB", 8.0));
        }
        if (tail_l > 0 && tail_l < NL && cgib > 0) {
            const int64_t cbud = (int64_t) (cgib * kGiB), pbud = (int64_t) (env_d("STRATA_XE_PINNED_GIB", 16.0) * kGiB);
            std::vector<char> in_cache(rest.size(), 0), taken(rest.size(), 0);
            int64_t cu = 0;
            for (size_t i = 0; i < rest.size(); ++i) {   // seed_dense's walk: the next expert that still fits
                const int64_t b = (int64_t) im.bbytes(rest[i].first, rest[i].second);
                if (cu + b > cbud) continue;
                cu += b;
                in_cache[i] = 1;
            }
            std::vector<std::pair<int32_t, int32_t>> ord;
            ord.reserve(rest.size());
            int64_t pu = 0, n_lo = 0, n_hi = 0, b_hi = 0, n_kept = 0, b_kept = 0;
            for (size_t i = 0; i < rest.size(); ++i)
                if (in_cache[i] && rest[i].first < tail_l) {
                    ord.push_back(rest[i]);
                    taken[i] = 1;
                    pu += (int64_t) im.bbytes(rest[i].first, rest[i].second);
                    ++n_lo;
                }
            const int64_t b_lo = pu;
            for (size_t i = 0; i < rest.size() && pu < pbud; ++i)
                if (!in_cache[i] && rest[i].first >= tail_l) {
                    const int64_t b = (int64_t) im.bbytes(rest[i].first, rest[i].second);
                    ord.push_back(rest[i]);
                    taken[i] = 1;
                    pu += b;
                    b_hi += b;
                    ++n_hi;
                }
            for (size_t i = 0; i < rest.size(); ++i) {
                if (taken[i]) continue;
                if (in_cache[i]) { ++n_kept; b_kept += (int64_t) im.bbytes(rest[i].first, rest[i].second); }
                ord.push_back(rest[i]);
            }
            rest.swap(ord);
            std::fprintf(stderr, "xe tier: STRATA_XE_PIN_STREAM: the pinned arena = the dense card's cache below layer %d (%lld experts, "
                         "%.2f GiB), then %lld next-ranked experts of layers %d.. (%.2f GiB); the cache's %lld of layers %d.. "
                         "(%.2f GiB) pageable (cache predicted %.2f GiB)\n", tail_l, (long long) n_lo, b_lo / kGiB, (long long) n_hi,
                         tail_l, b_hi / kGiB, (long long) n_kept, tail_l, b_kept / kGiB, cu / kGiB);
        }
    }
    // ---- the arena: the rest, in ranked order, up to the RAM budget, in pinned chunks (no blob straddles one)
    const char* aenv = std::getenv("STRATA_XE_ARENA_GIB");
    const double arena_cap_gib = aenv && *aenv ? std::atof(aenv) : 1e9;   // default: every expert not on a card
    const int64_t avail = mem_available_bytes();
    const double floor_gib = env_d("STRATA_XE_MEM_FLOOR_GIB", 12.0);
    int64_t cap = (int64_t) (arena_cap_gib * kGiB);
    if (avail > 0) cap = std::min<int64_t>(cap, avail - (int64_t) (floor_gib * kGiB));
    const size_t chunk = (size_t) (env_d("STRATA_XE_ARENA_CHUNK_GIB", 2.0) * kGiB);
    std::vector<std::pair<int32_t, int32_t>> in_arena;
    std::vector<size_t> where;   // per in_arena entry: chunk index << 40 | offset
    int64_t abytes = 0;
    size_t cur = 0, nchunks = 0, cur_used = chunk;
    std::vector<size_t> chunk_size;
    for (const auto& le : rest) {
        const int64_t b = (int64_t) im.bbytes(le.first, le.second);
        if (abytes + b > cap) { ++im.file_n; continue; }
        if (cur_used + (size_t) b > chunk) { cur = nchunks++; cur_used = 0; chunk_size.push_back(0); }
        where.push_back((cur << 40) | cur_used);
        cur_used += ((size_t) b + 4095) & ~(size_t) 4095;
        chunk_size[cur] = cur_used;
        abytes += b;
        in_arena.push_back(le);
    }
    im.arena_bytes = abytes;
    im.arena_n = (int64_t) in_arena.size();
    // pinned (SYCL host USM) only up to STRATA_XE_PINNED_GIB: GPU-driver host memory is capped by TTM (larry: ~60.7 GiB
    // for the whole box) and a pinned arena past that makes Level Zero refuse kernels (and can livelock the host).
    // The hottest arena experts (ranked first) get the pinned chunks; the rest is ordinary pageable memory.
    int64_t pin_cap = (int64_t) (env_d("STRATA_XE_PINNED_GIB", 16.0) * kGiB);
    // prompt prefetch: two pinned stage buffers, each the largest layer's pageable arena bytes up to
    // STRATA_XE_PREFETCH_STAGE_GIB (1.5), taken out of the same pinned budget (TTM caps the box's pinned memory)
    if (env_i("STRATA_XE_PREFETCH", 1)) {
        std::vector<int64_t> lb((size_t) im.g.n_layers, 0);
        for (const auto& le : in_arena) lb[(size_t) le.first] += (int64_t) ((im.bbytes(le.first, le.second) + 255) & ~(size_t) 255);
        const int64_t mx = lb.empty() ? 0 : *std::max_element(lb.begin(), lb.end());
        const size_t want = (size_t) std::min<int64_t>(mx, (int64_t) (env_d("STRATA_XE_PREFETCH_STAGE_GIB", 1.5) * kGiB));
        // larry, STRATA_XE_HOTRING with STRATA_XE_HOTRING_NOSTAGE (default 1): the stagers' rings carry every copy, so no
        // stage - its pinned budget goes to the arena; `want` stays the planner's cap on a slot's pageable bytes.  The
        // rings now, all or none (none: no prefetch at all - nothing could stage it)
        if (hot_on() && env_i("STRATA_XE_HOTRING_NOSTAGE", 1) && want) {
            const int ns = std::max(1, env_i("STRATA_XE_STAGERS", bar_stage_on() ? 1 : 4));   // larry, patch-barstage2: 1 with BAR
            bool all = true;
            for (int i = 0; i < ns; ++i) {
                im.pf_rings.push_back(sycl::malloc_host<uint8_t>((size_t) hot_slots() * hot_chunk(), im.ctx));
                all = all && im.pf_rings.back();
            }
            if (!all) {
                for (uint8_t* p : im.pf_rings) if (p) sycl::free(p, im.ctx);
                im.pf_rings.clear();
            }
            im.pf_stage_cap = all ? want : 0;
            std::fprintf(stderr, "xe tier: prefetch through %d hot rings of %d x %zu KiB (no stage: %.2f GiB more pinned arena; "
                         "largest layer's arena share %.2f GiB)%s\n", ns, hot_slots(), hot_chunk() >> 10, 2 * want / kGiB, mx / kGiB,
                         all ? "" : " - RING ALLOCATION FAILED: no prefetch");
        } else {
            for (auto& sl : im.pf) {
                sl.stage = want ? sycl::malloc_host<uint8_t>(want, im.ctx) : nullptr;
                if (!sl.stage) { im.pf_stage_cap = 0; break; }
                im.pf_stage_cap = want;
            }
            if (!im.pf_stage_cap) for (auto& sl : im.pf) if (sl.stage) { sycl::free(sl.stage, im.ctx); sl.stage = nullptr; }
            pin_cap -= 2 * (int64_t) im.pf_stage_cap;
            std::fprintf(stderr, "xe tier: prefetch staging 2 x %.2f GiB pinned (largest layer's arena share %.2f GiB)\n",
                         im.pf_stage_cap / kGiB, mx / kGiB);
        }
    }
    int64_t pinned = 0;
    std::vector<uint8_t> chunk_pinned(nchunks, 0);
    // larry, STRATA_XE_PIN_SKIP_GIB (default 0): the first GiB of the ranked arena - what card 0's expert cache holds
    // anyway - stay pageable; the pinned budget goes to the experts prompts stream and decode promotes
    const int64_t pin_skip = (int64_t) (env_d("STRATA_XE_PIN_SKIP_GIB", 0.0) * kGiB);
    int64_t ranked_bytes = 0;
    for (size_t c = 0; c < nchunks; ++c) {
        uint8_t* p = nullptr;
        const bool skip = ranked_bytes < pin_skip;
        ranked_bytes += (int64_t) chunk_size[c];
        if (!skip && pinned + (int64_t) chunk_size[c] <= pin_cap) {
            p = sycl::malloc_host<uint8_t>(chunk_size[c], im.ctx);
            if (p) { pinned += (int64_t) chunk_size[c]; chunk_pinned[c] = 1; }
        }
        if (!p) {
            void* q = nullptr;
            if (posix_memalign(&q, 2u << 20, chunk_size[c]) != 0 || !q) { err = "xe tier: arena allocation failed"; return false; }
            p = (uint8_t*) q;
            // larry, STRATA_XE_ARENA_THP: 2 MiB pages for the staging's streaming reads (faulted in when the arena fills)
            static const int arena_thp = env_i("STRATA_XE_ARENA_THP", 0);
            if (arena_thp && madvise(q, chunk_size[c], MADV_HUGEPAGE) != 0)
                std::fprintf(stderr, "xe tier: madvise(MADV_HUGEPAGE) on an arena chunk failed (errno %d)\n", errno);
        }
        im.achunks.push_back(p);
    }
    im.arena_usm = pinned > 0;
    im.arena_pinned = pinned;
    im.apin.assign(im.aptr.size(), 0);
    for (size_t i = 0; i < in_arena.size(); ++i) {
        const size_t key = (size_t) im.key(in_arena[i].first, in_arena[i].second);
        im.aptr[key] = im.achunks[where[i] >> 40] + (where[i] & (((size_t) 1 << 40) - 1));
        im.apin[key] = chunk_pinned[where[i] >> 40];
    }
    std::fprintf(stderr, "xe tier: arena %.2f GiB in %zu chunks (%.2f GiB pinned), %lld experts; file tier %lld\n", abytes / kGiB,
                 nchunks, pinned / kGiB, (long long) im.arena_n, (long long) im.file_n);
    for (auto& cp : im.cards) {   // STRATA_XE_TP_EVICT
        const size_t ci = (size_t) cp->idx;
        if (ci >= im.card_set.size() || im.evict_from[ci] >= im.card_set[ci].size()) continue;
        int64_t eb = 0;
        for (size_t i = im.evict_from[ci]; i < im.card_set[ci].size(); ++i)
            eb += (int64_t) im.bbytes(im.card_set[ci][i].first, im.card_set[ci][i].second);
        std::fprintf(stderr, "xe tier: card %d: %zu experts (%.2f GiB, its last slabs) go back to the arena when the TP core's KV "
                     "rows need the room (STRATA_XE_TP_EVICT)\n", cp->idx, im.card_set[ci].size() - im.evict_from[ci], eb / kGiB);
    }
    // ---- load: the cards' experts (device copies out of pinned reader buffers), then the arena
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        for (const auto& le : on[(size_t) c.idx]) {
            uint8_t* p = c.alloc_expert(im.bbytes(le.first, le.second));
            if (!p) { err = "xe tier: device allocation failed on card " + std::to_string(c.idx); return false; }
            im.card_of[(size_t) im.key(le.first, le.second)] = (int8_t) c.idx;
            im.dptr[(size_t) im.key(le.first, le.second)] = p;
            rt_put(im, (size_t) im.key(le.first, le.second));
        }
    }
    std::vector<std::pair<int32_t, int32_t>> dev_list;
    for (auto& v : on) dev_list.insert(dev_list.end(), v.begin(), v.end());
    const int nth = env_i("STRATA_XE_READERS", 12);
    read_all(im, dev_list, nth, [&](size_t k, const uint8_t* buf) {
        const auto& le = dev_list[k];
        const size_t key = (size_t) im.key(le.first, le.second);
        XeCard& c = *im.cards[(size_t) im.card_of[key]];
        upload_planar(im, c, le.first, le.second, buf, (uint8_t*) im.dptr[key]).wait();
    });
    const double t1 = now_ms();
    read_all(im, in_arena, nth, [&](size_t k, const uint8_t* buf) {
        const auto& le = in_arena[k];
        std::memcpy(im.aptr[(size_t) im.key(le.first, le.second)], buf, im.bbytes(le.first, le.second));
    });
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
    for (int64_t k = 0; k < (int64_t) im.card_of.size(); ++k) {   // STRATA_XE_ARM: each card's null job
        const int c = im.card_of[(size_t) k];
        if (c < 0 || im.cards[(size_t) c]->arm_null_ok) continue;
        XeCard& cd = *im.cards[(size_t) c];
        cd.arm_null = make_job(im, k / im.g.n_experts, k % im.g.n_experts, im.dptr[(size_t) k]);
        cd.arm_null.nt = 0;
        cd.arm_null_ok = true;
    }
    im.arm_on = env_i("STRATA_XE_ARM", 1) != 0 && im.planar && im.x_direct;
    im.arm_multi = im.arm_on && env_i("STRATA_XE_ARM_MULTI", 1) != 0;
    // larry, STRATA_XE_GROUTE (default 0): the route table (pinned, every key) and the route kernel's lists / output
    im.groute_on = env_i("STRATA_XE_GROUTE", 0) != 0 && im.arm_on && im.dense >= 0 && !im.host_slots && im.cards.size() <= 8;
    if (im.groute_on && !im.rt) {
        XeCard& dcd = *im.cards[(size_t) im.dense];
        const size_t NK = im.card_of.size();
        im.rt = sycl::malloc_host<RtEnt>(NK, im.ctx);
        im.rt_hdr = sycl::malloc_host<RtHdr>(1, im.ctx);
        im.rt_jobs = sycl::malloc_device<XeJob>(im.cards.size() * kMaxDecodeJobs, dcd.q);
        im.rt_null = sycl::malloc_device<XeJob>(im.cards.size(), dcd.q);
        if (!im.rt || !im.rt_hdr || !im.rt_jobs || !im.rt_null) {
            std::fprintf(stderr, "xe tier: route buffers: allocation failed - STRATA_XE_GROUTE off\n");
            im.groute_on = false;
        } else {
            for (size_t k = 0; k < NK; ++k) { im.rt[k] = RtEnt{}; im.rt[k].card = -1; rt_put(im, k); }
            std::vector<XeJob> nulls(im.cards.size());   // an expert-less card's: blob null (a null job's blob is never read)
            for (auto& cp : im.cards) { nulls[(size_t) cp->idx] = cp->arm_null; nulls[(size_t) cp->idx].nt = 0; }
            dcd.q.memcpy(im.rt_null, nulls.data(), nulls.size() * sizeof(XeJob)).wait();
            std::memset((void*) im.rt_hdr, 0, sizeof(RtHdr));
            std::fprintf(stderr, "xe tier: STRATA_XE_GROUTE: the dense card routes decode / verify calls\n");
        }
    }
    if (im.arm_on && !im.arm_err) {
        im.arm_err = sycl::malloc_host<int>(1, im.ctx);
        if (im.arm_err) *im.arm_err = 0; else im.arm_on = im.arm_multi = false;
    }
    if (im.arm_on && !im.relq) {
        if (im.dense < 0) im.arm_on = im.arm_multi = false;
        else {
            im.relq = std::make_unique<sycl::queue>(im.ctx, im.cards[(size_t) im.dense]->dev, sycl::property::queue::in_order());
            for (auto& cp : im.cards)   // warm-up: the first cross-card copy into each card is set up lazily (slow)
                if (cp->arm_flag && cp->arm_src) im.relq->memcpy(cp->arm_flag, cp->arm_src, sizeof(int)).wait();
        }
    }
    {   // larry: each expert card's flex region (prompt: prefetch buffers; decode: promotion slots), from the reserve
        const double fg = env_d("STRATA_XE_FLEX_GIB", 1.0), fm = env_d("STRATA_XE_FLEX_MARGIN_GIB", 1.5);
        for (auto& cp : im.cards) {
            XeCard& c = *cp;
            if (c.dense || c.flex) continue;
            const int64_t fr = c.free_vram();
            // larry, engine1: + what the TP core's KV replicas have not mapped yet (given back as the context grows)
            const auto td = im.tp_defer.find(c.gpu);
            int64_t extra = td == im.tp_defer.end() ? 0 : (int64_t) td->second;
            if (tp_evict_on()) extra = std::min<int64_t>(extra, tp_defer_flex());   // STRATA_XE_TP_EVICT: the rest holds experts
            const int64_t want = std::min<int64_t>((int64_t) (fg * kGiB) + extra, fr > 0 ? fr - (int64_t) (fm * kGiB) : 0) & ~(int64_t) 255;
            if (want < 8 * im.blob_max) continue;
            c.flex = card_alloc(c.q, (size_t) want);   // larry, patch-barstage
            c.flex_cap = c.flex ? (size_t) want : 0;
        }
        if (env_i("STRATA_XE_PROMOTE", 1) && !im.pc_bounce) {   // the promotions' pinned bounces, now too
            im.pc_slot = ((size_t) im.blob_max + 255) & ~(size_t) 255;
            // larry, engine1 (STRATA_TP): promotions driven between TP passes, several a card (STRATA_TP_PROMOTE_PAR)
            if (env_i("STRATA_TP", 0) > 0) im.pc_nb = std::max(kPcBounce, std::max(1, env_i("STRATA_TP_PROMOTE_PAR", 8)) * (int) im.cards.size());
            im.pc_bounce = sycl::malloc_host<uint8_t>((size_t) im.pc_nb * im.pc_slot, im.ctx);
        }
    }
    rep_prepare(im, order);
    im.placed = true;
    std::fprintf(stderr, "xe tier: placed in %.1f s (cards %.1f s, arena %.1f s)\n%s", (now_ms() - t0) / 1000.0,
                 (t1 - t0) / 1000.0, (now_ms() - t1) / 1000.0, report().c_str());
    return true;
}

namespace {
/// The dense card's next experts in ranked order that live in the arena only, while c.budget allows (resident ones are
/// skipped: a regrow continues where the cache ends).  Pageable arena experts go through a pinned bounce: a DMA straight
/// from pageable memory makes the driver pin the source pages, and 15 GiB queued at once pushed larry's driver-held RAM
/// past the watchdog (58 GiB, 2026-10-19; TTM's cap for all of it is ~60.7 GiB).  Returns the experts added, -1 on error.
int64_t seed_dense(Ds41XeImpl& im, XeCard& c, const std::vector<std::pair<int32_t, int32_t>>& ranked, const char* what, std::string& err,
                   bool defer = false) {   // larry, patch-regrowasync: defer = the uploads queued, published by regrow_poll
    const double t0 = now_ms();
    const size_t half = std::max<size_t>((size_t) im.blob_max, (size_t) 64 << 20);
    if (c.bounce_cap < 2 * half) {
        if (c.bounce) sycl::free(c.bounce, c.q);
        c.bounce = sycl::malloc_host<uint8_t>(2 * half, c.q);
        c.bounce_cap = c.bounce ? 2 * half : 0;
        if (!c.bounce) { err = "xe tier: bounce allocation failed"; return -1; }
    }
    sycl::event hev[2];
    int hcur = 0;
    size_t hused = 0;
    double t_alloc = 0;
    int64_t n_pin = 0, n_page = 0;
    for (const auto& le : ranked) {
        if (le.first < 0 || le.first >= im.g.n_layers || le.second < 0 || le.second >= im.g.n_experts) continue;
        const size_t key = (size_t) im.key(le.first, le.second);
        if (im.card_of[key] >= 0 || !im.aptr[key]) continue;
        // larry, patch-pcrace: not an expert a promotion holds (queued / filling / done): its completion would move it
        if (key < im.pc_where.size() && im.pc_where[key] != -1) continue;
        const size_t b = im.bbytes(le.first, le.second);
        if (c.used + (int64_t) b > c.budget) continue;
        const double ta = now_ms();
        uint8_t* p = c.alloc_expert(b);
        t_alloc += now_ms() - ta;
        if (!p) break;
        if (im.apin[key]) {
            upload_planar(im, c, le.first, le.second, im.aptr[key], p);
            ++n_pin;
        } else {
            if (hused + b > half) { hcur ^= 1; hused = 0; hev[hcur].wait(); }
            std::memcpy(c.bounce + (size_t) hcur * half + hused, im.aptr[key], b);
            hev[hcur] = upload_planar(im, c, le.first, le.second, c.bounce + (size_t) hcur * half + hused, p);
            hused += (b + 255) & ~(size_t) 255;
            ++n_page;
        }
        if (defer) {   // larry, patch-regrowasync
            im.rg_list.push_back({key, p, le});
            if (im.rg_mark.size() != im.card_of.size()) im.rg_mark.assign(im.card_of.size(), 0);
            im.rg_mark[key] = 1;
            continue;
        }
        im.card_of[key] = (int8_t) c.idx;
        im.dptr[key] = p;
        rt_put(im, key);
        im.dense_set.push_back(le);
    }
    if (defer) {   // larry, patch-regrowasync: the bounce's copies done (the next prompt's streams use it), the rest queued
        hev[0].wait();
        hev[1].wait();
        im.rg_ev = c.q.ext_oneapi_submit_barrier();
        im.rg_pending = !im.rg_list.empty();
        im.rg_t0 = t0;
        std::fprintf(stderr, "xe tier: dense card %d %s: %lld experts (%lld pinned / %lld pageable) queued in %.2f s (slabs %.0f ms)\n",
                     c.idx, what, (long long) (n_pin + n_page), (long long) n_pin, (long long) n_page, (now_ms() - t0) / 1000.0, t_alloc);
        return n_pin + n_page;
    }
    c.q.wait();
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
    std::fprintf(stderr, "xe tier: dense card %d %s: %lld experts (%lld pinned / %lld pageable) in %.2f s (slabs %.0f ms); %zu experts "
                 "(%.2f GiB) on it\n", c.idx, what, (long long) (n_pin + n_page), (long long) n_pin, (long long) n_page,
                 (now_ms() - t0) / 1000.0, t_alloc, im.dense_set.size(), c.used / kGiB);
    return n_pin + n_page;
}

/// The dense card's expert budget now: STRATA_XE_DENSE_GIB (and the engine's slot budget), with
/// STRATA_XE_DENSE_RESERVE_GIB of what is free NOW left to the dense half's graphs.  The dense card's share stays small:
/// ggml-sycl's pools are virtual (VMM) and commit while decode runs, so "free VRAM" right after a prompt overstates what
/// is left (2026-10-19: card 0 at 98 % with 15 GiB of experts, 12 GiB spilled into host RAM).
int64_t dense_budget(Ds41XeImpl& im, XeCard& c) {
    const double cap = env_d("STRATA_XE_DENSE_GIB", 10.0);
    double gib = im.slot_gib;
    if (cap >= 0) gib = std::min(gib, cap + im.dense_bonus_gib);
    const int64_t fr = c.free_vram();
    if (fr > 0) gib = std::min(gib, (fr + c.used) / kGiB - env_d("STRATA_XE_DENSE_RESERVE_GIB", 8.0));
    return gib < 0.5 ? 0 : (int64_t) (gib * kGiB);
}
}  // namespace

bool Ds41XeTier::seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync
    if (!im.placed && !build_arena_ranked(ranked, err)) return false;
    if (im.dense < 0 || !im.cache_enabled || !im.dense_set.empty()) return true;
    XeCard& c = *im.cards[(size_t) im.dense];
    im.seed_ranked = ranked;
    c.budget = dense_budget(im, c);
    if (c.budget <= 0) return true;
    // larry, STRATA_XE_SEED_TAIL_LAYERS (see patch-seedtail.py): the cache's set as the ranked seed takes it, laid out
    // with the layers >= L first; then everything else in rank order (a regrow into more room continues as before)
    static const int tail_l = env_i("STRATA_XE_SEED_TAIL_LAYERS", 0);
    if (tail_l > 0 && tail_l < im.g.n_layers) {
        std::vector<std::pair<int32_t, int32_t>> hi, lo, rest;
        int64_t used = c.used;
        for (const auto& le : ranked) {
            const bool ok = le.first >= 0 && le.first < im.g.n_layers && le.second >= 0 && le.second < im.g.n_experts;
            const size_t key = ok ? (size_t) im.key(le.first, le.second) : 0;
            if (!ok || im.card_of[key] >= 0 || !im.aptr[key]) { rest.push_back(le); continue; }
            const int64_t b = (int64_t) im.bbytes(le.first, le.second);
            if (used + b > c.budget) { rest.push_back(le); continue; }
            used += b;
            (le.first >= tail_l ? hi : lo).push_back(le);
        }
        std::vector<std::pair<int32_t, int32_t>> order;
        order.reserve(ranked.size());
        order.insert(order.end(), hi.begin(), hi.end());
        order.insert(order.end(), lo.begin(), lo.end());
        order.insert(order.end(), rest.begin(), rest.end());
        std::fprintf(stderr, "xe tier: dense card cache laid out layers %d.. first: %zu experts, then %zu below layer %d\n", tail_l,
                     hi.size(), lo.size(), tail_l);
        im.seed_ranked = order;
    }
    if (seed_dense(im, c, im.seed_ranked, "seeded", err) < 0) return false;
    rep_upload(im);   // larry, engine1: the replicas in what the cache left (STRATA_XE_REPLICA_MARGIN_GIB kept free)
    return true;
}

/// larry (serve keep-cache): a prompt batch of a few thousand tokens needs `need_bytes` free on the dense card - the
/// cache's tail (its lowest-ranked experts, the last slabs) goes back to the arena slab by slab until that is free.
/// Returns the experts released, -1 when even an empty cache would not make the room (the caller releases it all).
int64_t Ds41XeTier::shrink_cache_for(int64_t need_bytes, std::string& err) {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync
    if (im.dense < 0 || im.dense_set.empty()) return -1;
    XeCard& c = *im.cards[(size_t) im.dense];
    c.q.wait();
    if (im.groute_on) { im.rt_dense_ev.wait(); im.last_sum.wait(); }   // STRATA_XE_GROUTE: its share on the dense queue
    rep_drop(im);   // larry, engine1: the replicas first (copies: the expert cards still hold them)
    if (c.free_vram() + c.used < need_bytes) { err = "the dense card has no room for the batch even without its experts"; return -1; }
    int64_t released = 0;
    while (c.free_vram() < need_bytes && !c.slabs.empty()) {
        uint8_t* last = c.slabs.back();
        // larry, patch-pcrace: EVERY cached expert in the slab goes - not just dense_set's suffix (an entry a late
        // promotion had moved stopped that walk, and the slab was freed under live experts: card-0 CAT errors) - and
        // entries no longer on this card leave the list
        size_t w = 0;
        for (size_t i = 0; i < im.dense_set.size(); ++i) {
            const auto le = im.dense_set[i];
            const size_t key = (size_t) im.key(le.first, le.second);
            if (im.card_of[key] != (int8_t) c.idx) {   // stale: on another card / in the arena since
                c.used -= (int64_t) im.bbytes(le.first, le.second);
                --c.n_experts;
                continue;
            }
            const uint8_t* p = im.dptr[key];
            if (p >= last && p < last + kSlab) {
                im.card_of[key] = -1;
                im.dptr[key] = nullptr;
                rt_put(im, key);
                c.used -= (int64_t) im.bbytes(le.first, le.second);
                --c.n_experts;
                ++released;
                continue;
            }
            im.dense_set[w++] = le;
        }
        im.dense_set.resize(w);
        local_free(c.q, last);
        c.slabs.pop_back();
        c.slab_used = kSlab;   // the slabs left are full: the next expert opens a new one
    }
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
    return released;
}

/// larry, patch-regrowasync: regrow_cache's copies queued, published by regrow_poll (see the header)
int64_t Ds41XeTier::regrow_cache_async(std::string& err) {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);
    if (im.dense < 0 || !im.cache_enabled || im.seed_ranked.empty()) return 0;
    XeCard& c = *im.cards[(size_t) im.dense];
    c.budget = dense_budget(im, c);
    return c.used >= c.budget ? 0 : seed_dense(im, c, im.seed_ranked, "regrow", err, true);   // (the replicas: at publish)
}

bool Ds41XeTier::regrow_poll(bool wait) {
    Ds41XeImpl& im = *im_;
    if (!im.rg_pending) return false;
    if (!wait && im.rg_ev.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete)
        return false;
    const double tw = now_ms();
    im.rg_ev.wait();
    const double tp = now_ms();
    for (const auto& r : im.rg_list) {
        im.card_of[r.key] = (int8_t) im.dense;
        im.dptr[r.key] = r.p;
        rt_put(im, r.key);
        im.dense_set.push_back(r.le);
        im.rg_mark[r.key] = 0;
    }
    const size_t n = im.rg_list.size();
    im.rg_list.clear();
    im.rg_pending = false;
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
    rep_upload(im);   // larry, engine1: the replicas after the cache
    std::fprintf(stderr, "xe tier: dense card regrow published: %zu experts, %.2f s after it was queued (%s %.0f ms); %zu experts "
                 "(%.2f GiB) on it\n", n, (tp - im.rg_t0) / 1000.0, wait ? "waited" : "done,", tp - tw, im.dense_set.size(),
                 im.cards[(size_t) im.dense]->used / kGiB);
    return true;
}

/// larry (serve keep-cache): after the batch, the tail back - the seed continued within what is free now.
int64_t Ds41XeTier::regrow_cache(std::string& err) {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync
    if (im.dense < 0 || !im.cache_enabled || im.seed_ranked.empty()) return 0;
    XeCard& c = *im.cards[(size_t) im.dense];
    c.budget = dense_budget(im, c);
    const int64_t n = c.used >= c.budget ? 0 : seed_dense(im, c, im.seed_ranked, "regrown", err);
    if (n >= 0) rep_upload(im);   // larry, engine1: the replicas after the cache
    return n;
}

void Ds41XeTier::set_slot_gib(double gib) { im_->slot_gib = gib; }
void Ds41XeTier::set_dense_queue(void* q) { im_->dense_q = (sycl::queue*) q; }

namespace {
Ds41XeImpl::HeadSplit head_init(Ds41XeImpl& im, const void* w, int64_t K, int64_t N) {
    Ds41XeImpl::HeadSplit hs;
    std::vector<XeCard*> parts;
    for (auto& cp : im.cards) if (!cp->dense) parts.push_back(cp.get());
    const int np = (int) std::min<size_t>(parts.size(), 7) + 1;
    if (np < 2) return hs;
    for (int i = 0; i <= np; ++i) hs.r[i] = N * i / np;
    const int64_t bpr = K / 256, nb = N * bpr;
    const size_t po[4] = {0, 128, 192, 208}, pb[4] = {128, 64, 16, 2};   // R8 planes: offset / bytes per block
    const auto gv = im.head_given.find(w);
    for (int i = 1; i < np; ++i) {
        XeCard& c = *parts[(size_t) i - 1];
        const int64_t rows = hs.r[i + 1] - hs.r[i], b0 = hs.r[i] * bpr, bn = rows * bpr;
        const uint8_t* given = nullptr;   // larry, engine1: the TP core's shard of exactly these rows on this card
        if (gv != im.head_given.end())
            for (const auto& g : gv->second)
                if (g.gpu == c.gpu && g.r0 == hs.r[i] && g.r1 == hs.r[i + 1]) given = (const uint8_t*) g.p;
        const size_t need = (given ? 0 : (size_t) bn * 210) + (size_t) 8 * K * 2 + (size_t) 8 * rows * 4;
        // (with the TP core's slice only the scratch is new: a small margin then)
        if (c.free_vram() > 0 && c.free_vram() < (int64_t) (need + ((size_t) (given ? 64 : 768) << 20))) {
            std::fprintf(stderr, "xe tier: no room on card %d for the head split (%zu MB) - not split\n", c.idx, need >> 20);
            return Ds41XeImpl::HeadSplit();
        }
        uint8_t* wk = given ? const_cast<uint8_t*>(given) : sycl::malloc_device<uint8_t>((size_t) bn * 210, c.q);
        // larry, patch-headstage: xq and xd one buffer (staged: xd right after the used int8; direct: at 8 * K)
        int8_t* xq = sycl::malloc_device<int8_t>((size_t) 8 * K + (size_t) 8 * K / 32 * sizeof(float), c.q);
        float* xd = xq ? reinterpret_cast<float*>(xq + (size_t) 8 * K) : nullptr;
        float* y = sycl::malloc_device<float>((size_t) 8 * rows, c.q);
        if (!wk || !xq || !y) { std::fprintf(stderr, "xe tier: head split allocation failed - not split\n"); return Ds41XeImpl::HeadSplit(); }
        if (!given) for (int p = 0; p < 4; ++p) c.q.memcpy(wk + po[p] * bn, (const uint8_t*) w + po[p] * nb + pb[p] * b0, pb[p] * bn);
        c.q.wait();
        hs.c.push_back(&c); hs.w.push_back(wk); hs.own.push_back(given ? 0 : 1); hs.xq.push_back(xq); hs.xd.push_back(xd); hs.y.push_back(y);
    }
    hs.ok = true;
    int shared = 0;
    for (char o : hs.own) shared += o ? 0 : 1;
    std::fprintf(stderr, "xe tier: %lld x %lld LM head split over %d cards%s\n", (long long) N, (long long) K, np,
                 shared ? (" (" + std::to_string(shared) + " slices the TP core's own rows)").c_str() : "");
    return hs;
}
}  // namespace

bool Ds41XeTier::mv_hook(void* queue, const void* w, int64_t K, int64_t N, int ncols, const int8_t* xq, const float* xd,
                         float* y, int64_t y_col_stride, void* user) {
    Ds41XeImpl& im = *static_cast<Ds41XeTier*>(user)->im_;
    if (!queue || K % 256 || ncols < 1 || ncols > 8 || im.cards.empty()) return false;
    try {
        auto it = im.heads.find(w);
        if (it == im.heads.end()) it = im.heads.emplace(w, head_init(im, w, K, N)).first;
        Ds41XeImpl::HeadSplit& hs = it->second;
        if (!hs.ok) return false;
        sycl::queue& q0 = *static_cast<sycl::queue*>(queue);
        const int64_t bpr = K / 256, R1 = hs.r[1];
        auto own = [&](float* yo, int64_t ys) {   // part 0, on the dense card
            fastmv::Args a0;
            a0.w = w; a0.K = K; a0.N = R1; a0.G = 1; a0.g_div = 1; a0.n_blocks_total = N * bpr;
            a0.x_col_stride = K; a0.x_batch_stride = K * ncols;
            a0.y = yo; a0.y_col_stride = ys; a0.y_batch_stride = ys * ncols; a0.ncols = ncols;
            a0.xq = xq; a0.xd = xd;
            fastmv::launch(q0, fastmv::WType::Q6K_R8, a0);
        };
        auto direct = [&](float* yo, int64_t ys) {   // the cards copy from xq / xd and into yo themselves
            const sycl::event e0 = q0.ext_oneapi_submit_barrier();
            std::vector<sycl::event> ev;
            for (size_t i = 0; i < hs.c.size(); ++i) {
                XeCard& c = *hs.c[i];
                const int64_t r0 = hs.r[i + 1], rows = hs.r[i + 2] - r0;
                c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.memcpy(hs.xq[i], xq, (size_t) ncols * K); });
                c.q.memcpy(hs.xd[i], xd, (size_t) ncols * K / 32 * sizeof(float));
                fastmv::Args a;
                a.w = hs.w[i]; a.K = K; a.N = rows; a.G = 1; a.g_div = 1; a.n_blocks_total = rows * bpr;
                a.x_col_stride = K; a.x_batch_stride = K * ncols;
                a.y = hs.y[i]; a.y_col_stride = rows; a.y_batch_stride = rows * ncols; a.ncols = ncols;
                a.xq = hs.xq[i]; a.xd = hs.xd[i];
                fastmv::launch(c.q, fastmv::WType::Q6K_R8, a);
                for (int t = 0; t < ncols; ++t) c.q.memcpy(yo + t * ys + r0, hs.y[i] + t * rows, (size_t) rows * 4);
                ev.push_back(c.q.ext_oneapi_submit_barrier());
            }
            own(yo, ys);
            q0.ext_oneapi_submit_barrier(ev);
        };
        // larry, STRATA_XE_HEAD_STAGE (patch-headstage.py, default 1): the peer copies through card-0 staging
        static const bool stage_on = env_i("STRATA_XE_HEAD_STAGE", 1) != 0;
        if (stage_on && !hs.sq) {
            hs.sq = &q0;
            hs.sx = sycl::malloc_device<uint8_t>((size_t) 8 * K + (size_t) 8 * K / 32 * sizeof(float), q0);
            hs.sy = sycl::malloc_device<float>((size_t) 8 * (N - R1), q0);
            if (!hs.sx || !hs.sy) {
                if (hs.sx) sycl::free(hs.sx, q0);
                if (hs.sy) sycl::free(hs.sy, q0);
                hs.sx = nullptr; hs.sy = nullptr;
                std::fprintf(stderr, "xe tier: head split staging allocation failed - direct copies\n");
            }
        }
        if (!stage_on || !hs.sx || hs.sq != &q0) {
            direct(y, y_col_stride);
            return true;
        }
        const size_t nq = (size_t) ncols * K, nd = nq / 32;   // int8 bytes, scales
        uint8_t* sx = hs.sx;
        float* sy = hs.sy;
        const sycl::event e0 = q0.parallel_for(sycl::range<1>(nq / 4 + nd), [=](sycl::id<1> id) {
            const size_t j = id[0];
            if (j < nq / 4) reinterpret_cast<uint32_t*>(sx)[j] = reinterpret_cast<const uint32_t*>(xq)[j];
            else reinterpret_cast<float*>(sx + nq)[j - nq / 4] = xd[j - nq / 4];
        });
        std::vector<sycl::event> ev;
        for (size_t i = 0; i < hs.c.size(); ++i) {
            XeCard& c = *hs.c[i];
            const int64_t r0 = hs.r[i + 1], rows = hs.r[i + 2] - r0;
            int8_t* cx = hs.xq[i];
            c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.memcpy(cx, sx, nq + nd * sizeof(float)); });
            fastmv::Args a;
            a.w = hs.w[i]; a.K = K; a.N = rows; a.G = 1; a.g_div = 1; a.n_blocks_total = rows * bpr;
            a.x_col_stride = K; a.x_batch_stride = K * ncols;
            a.y = hs.y[i]; a.y_col_stride = rows; a.y_batch_stride = rows * ncols; a.ncols = ncols;
            a.xq = cx; a.xd = reinterpret_cast<const float*>(cx + nq);
            fastmv::launch(c.q, fastmv::WType::Q6K_R8, a);
            ev.push_back(c.q.memcpy(sy + (r0 - R1) * 8, hs.y[i], (size_t) rows * ncols * sizeof(float)));
        }
        own(y, y_col_stride);
        q0.ext_oneapi_submit_barrier(ev);
        int64_t rr[9];
        for (int i = 0; i < 9; ++i) rr[i] = hs.r[i];
        const int np = (int) hs.c.size() + 1;
        const int64_t NR = N - R1, ys = y_col_stride;
        q0.parallel_for(sycl::range<1>((size_t) (NR * ncols)), [=](sycl::id<1> id) {
            const int64_t t = (int64_t) id[0] / NR, row = R1 + (int64_t) id[0] % NR;
            int p = 1;
            while (p + 1 < np && row >= rr[p + 1]) ++p;   // part p: rows [rr[p], rr[p + 1])
            const int64_t p0 = rr[p], rows = rr[p + 1] - p0;
            y[t * ys + row] = sy[(p0 - R1) * 8 + t * rows + (row - p0)];
        });
        // STRATA_XE_HEAD_STAGE_CHECK=1: the first 4 staged calls of each width the direct way too, into a scratch, compared
        static const bool check = env_i("STRATA_XE_HEAD_STAGE_CHECK", 0) != 0;
        static int checked = 0, by_n[9] = {};
        if (check && by_n[ncols] < 4) {
            ++checked; ++by_n[ncols];
            float* yc = sycl::malloc_device<float>((size_t) N * ncols, q0);
            if (yc) {
                direct(yc, N);
                std::vector<float> ha((size_t) N * ncols), hb((size_t) N * ncols);
                for (int t = 0; t < ncols; ++t) q0.memcpy(ha.data() + (size_t) t * N, y + t * y_col_stride, (size_t) N * 4);
                q0.memcpy(hb.data(), yc, (size_t) N * ncols * 4);
                q0.wait();
                size_t bad = 0;
                for (size_t j = 0; j < ha.size(); ++j) bad += std::memcmp(&ha[j], &hb[j], 4) != 0;
                std::fprintf(stderr, "xe tier: head stage check %d (%d columns): %zu of %zu logits differ\n", checked, ncols,
                             bad, ha.size());
                sycl::free(yc, q0);
            }
        }
        return true;
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xe tier: head split: %s\n", e.what());
        return false;
    }
}

bool Ds41XeTier::tl_on() { static const bool on = env_i("STRATA_GPU_TL", 0) != 0; return on; }

// larry, STRATA_GPU_PTL=1 (diagnostic, patch-ptl.py): a prompt batch's GPU timeline.  Ring 0 = the dense queue,
// 1 + k = card k's tier queue, 9 + k = card k's prefetch copy queue (in-order queues: a ring's flags flip in order),
// 17 = host-only steps.  A stamp is a one-lane kernel writing its flag; the monitor thread stamps what it sees.
bool Ds41XeTier::ptl_on() { static const bool on = env_i("STRATA_GPU_PTL", 0) != 0; return on; }

namespace {
struct PtlRing {   // larry, patch-ptl2: a flag a 64-byte line, written only by the card (the record's sequence number)
    static constexpr uint32_t kCap = 1u << 14, kStride = 16;
    uint32_t* flag = nullptr;   // pinned host memory, kCap * kStride; nullptr: the host ring
    struct Rec { int tag, l, a; double th, td; };
    std::vector<Rec> rec = std::vector<Rec>(kCap);
    std::atomic<uint32_t> issued{0}, seen{0};
    uint32_t done = 0;          // dumped so far
};
constexpr int kPtlRings = 18, kPtlHost = 17;
std::array<std::atomic<PtlRing*>, kPtlRings> g_ptl{};
std::mutex g_ptl_mu;
std::thread g_ptl_mon;

void ptl_monitor() {
    for (;;) {
        bool any = false;
        for (auto& ap : g_ptl) {
            PtlRing* r = ap.load(std::memory_order_acquire);
            if (!r || !r->flag) continue;
            volatile const uint32_t* f = r->flag;
            uint32_t sn = r->seen.load(std::memory_order_relaxed);
            const uint32_t is = r->issued.load(std::memory_order_acquire);
            auto got = [&](uint32_t i) { return f[(size_t) (i % PtlRing::kCap) * PtlRing::kStride] == i + 1; };
            while (sn < is) {
                if (got(sn)) {
                    r->rec[sn % PtlRing::kCap].td = now_ms();
                } else {   // a later record of this in-order queue seen: this one ran before it (time inferred)
                    uint32_t j = sn + 1;
                    while (j < is && j < sn + 256 && !got(j)) ++j;
                    if (j >= is || j >= sn + 256) break;
                    r->rec[sn % PtlRing::kCap].td = -now_ms();
                }
                ++sn;
                any = true;
            }
            r->seen.store(sn, std::memory_order_release);
        }
        if (!any) std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

PtlRing* ptl_ring(int ring, const sycl::context& ctx) {
    if (ring < 0 || ring >= kPtlRings) return nullptr;
    PtlRing* r = g_ptl[(size_t) ring].load(std::memory_order_acquire);
    if (r) return r;
    std::lock_guard<std::mutex> lk(g_ptl_mu);
    r = g_ptl[(size_t) ring].load(std::memory_order_acquire);
    if (r) return r;
    r = new PtlRing();
    if (ring != kPtlHost) {
        r->flag = sycl::malloc_host<uint32_t>((size_t) PtlRing::kCap * PtlRing::kStride, ctx);
        if (!r->flag) { delete r; return nullptr; }
        std::memset(r->flag, 0, (size_t) PtlRing::kCap * PtlRing::kStride * sizeof(uint32_t));
    }
    g_ptl[(size_t) ring].store(r, std::memory_order_release);
    if (!g_ptl_mon.joinable()) { g_ptl_mon = std::thread(ptl_monitor); g_ptl_mon.detach(); }
    return r;
}

/// a stamp on `q` (the caller holds the submit lock like every enqueue), `th` = when the host began the step
void ptl_stamp(const sycl::context& ctx, int ring, sycl::queue& q, int tag, int l, int a, double th) {
    if (!Ds41XeTier::ptl_on()) return;
    PtlRing* r = ptl_ring(ring, ctx);
    if (!r) return;
    const uint32_t i = r->issued.load(std::memory_order_relaxed);
    if (i - r->done >= PtlRing::kCap) return;   // full until the next dump
    const uint32_t k = i % PtlRing::kCap;
    r->rec[k] = {tag, l, a, th, 0.0};
    if (r->flag) {
        volatile uint32_t* f = r->flag + (size_t) k * PtlRing::kStride;
        const uint32_t seq = i + 1;
        q.single_task([=] { *f = seq; });
    } else {
        r->rec[k].td = th;
    }
    r->issued.store(i + 1, std::memory_order_release);
    if (!r->flag) r->seen.store(i + 1, std::memory_order_release);
}
}  // namespace

void Ds41XeTier::ptl(int tag, int l, int a, double th) {
    Ds41XeImpl& im = *im_;
    if (!ptl_on() || !im.dense_q) return;
    ptl_stamp(im.ctx, 0, *im.dense_q, tag, l, a, th);
}

void Ds41XeTier::ptl_host(int tag, int l, int a) {
    if (!ptl_on()) return;
    PtlRing* r = ptl_ring(kPtlHost, im_->ctx);
    if (!r) return;
    const uint32_t i = r->issued.load(std::memory_order_relaxed);
    if (i - r->done >= PtlRing::kCap) return;
    const double t = now_ms();
    r->rec[i % PtlRing::kCap] = {tag, l, a, t, t};
    r->issued.store(i + 1, std::memory_order_release);
    r->seen.store(i + 1, std::memory_order_release);
}

void Ds41XeTier::ptl_dump(const char* label) {
    if (!ptl_on()) return;
    const double t0 = now_ms();
    for (auto& ap : g_ptl) {   // everything stamped has run or will in a moment (the batch's logits were read)
        PtlRing* r = ap.load(std::memory_order_acquire);
        while (r && r->seen.load(std::memory_order_acquire) < r->issued.load(std::memory_order_acquire) && now_ms() - t0 < 200)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const char* dir = std::getenv("STRATA_GPU_PTL_DIR");
    char path[512];
    std::snprintf(path, sizeof path, "%s/ptl.%d.tsv", dir ? dir : "/opt/ai/dsv41/logs", (int) getpid());
    std::FILE* f = std::fopen(path, "a");
    if (f) std::fprintf(f, "# %s %.3f\n", label ? label : "batch", t0);
    for (int ring = 0; ring < kPtlRings; ++ring) {
        PtlRing* r = g_ptl[(size_t) ring].load(std::memory_order_acquire);
        if (!r) continue;
        const uint32_t sn = r->seen.load(std::memory_order_acquire);
        for (uint32_t k = r->done; f && k < sn; ++k) {
            const PtlRing::Rec& e = r->rec[k % PtlRing::kCap];
            std::fprintf(f, "%d\t%d\t%d\t%d\t%.3f\t%.3f\n", ring, e.tag, e.l, e.a, e.th, e.td);
        }
        r->done = sn;
        const uint32_t is = r->issued.load(std::memory_order_acquire);
        if (f && is > sn) std::fprintf(f, "# ring %d: %u records not seen\n", ring, is - sn);
    }
    if (f) std::fclose(f);
}

namespace {
double tl_now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void tl_start(Ds41XeImpl& im) {
    if (!im.tl_rings.empty()) return;
    for (size_t r = 0; r <= im.cards.size(); ++r) {
        auto ring = std::make_unique<Ds41XeImpl::TlRing>();
        ring->flag = sycl::malloc_host<uint32_t>(Ds41XeImpl::kTlCap, im.ctx);
        for (uint32_t k = 0; k < Ds41XeImpl::kTlCap; ++k) ring->flag[k] = 0;
        ring->id.assign(Ds41XeImpl::kTlCap, -1);
        ring->t.assign(Ds41XeImpl::kTlCap, 0.0);
        im.tl_rings.push_back(std::move(ring));
    }
    im.tl_mon = std::thread([&im] {
        while (!im.tl_stop.load(std::memory_order_relaxed)) {
            for (auto& rp : im.tl_rings) {
                Ds41XeImpl::TlRing& r = *rp;
                uint32_t sn = r.seen.load(std::memory_order_relaxed);
                while (sn < r.issued.load(std::memory_order_acquire) && r.flag[sn % Ds41XeImpl::kTlCap]) {
                    r.t[sn % Ds41XeImpl::kTlCap] = tl_now_us();
                    r.flag[sn % Ds41XeImpl::kTlCap] = 0;
                    r.seen.store(++sn, std::memory_order_release);
                }
            }
        }
    });
}

void tl_print(Ds41XeImpl& im);

void tl_stamp(Ds41XeImpl& im, size_t ring, sycl::queue& q, int id) {
    tl_start(im);
    Ds41XeImpl::TlRing& r = *im.tl_rings[ring];
    const uint32_t k = r.issued.load(std::memory_order_relaxed) % Ds41XeImpl::kTlCap;
    r.id[k] = id;
    volatile uint32_t* f = r.flag;
    q.single_task([=] { f[k] = 1u; });
    r.issued.fetch_add(1, std::memory_order_release);
}
}  // namespace

void Ds41XeTier::tl_tag(int id) {
    Ds41XeImpl& im = *im_;
    if (!tl_on() || !im.dense_q) return;
    tl_stamp(im, 0, *im.dense_q, id);
}

void Ds41XeTier::tl_flush() {
    Ds41XeImpl& im = *im_;
    if (!tl_on() || im.tl_rings.empty()) return;
    const double t0 = tl_now_us();
    for (auto& rp : im.tl_rings)   // everything stamped has run (the logits were read); give the monitor a moment
        while (rp->seen.load(std::memory_order_acquire) < rp->issued.load(std::memory_order_acquire) && tl_now_us() - t0 < 2e5) {}
    const uint32_t C = Ds41XeImpl::kTlCap;
    Ds41XeImpl::TlRing& d = *im.tl_rings[0];
    const uint32_t dn = d.seen.load(std::memory_order_acquire);
    // dense transitions; and the times of this token's dense stamps 1 and 3 in order (one per layer)
    std::vector<double> t1, t3, t5;
    for (uint32_t k = d.done; k < dn; ++k) {
        const int b = d.id[k % C];
        if (b == 5) t5.push_back(d.t[k % C]);
        if (b == 1 || b == 9) t1.push_back(d.t[k % C]);
        if (b == 3 || b == 11) t3.push_back(d.t[k % C]);
        if (k == d.done) continue;
        const int a = d.id[(k - 1) % C];
        if (a < 0 || a > 16 || b < 0 || b > 16) continue;
        im.tl_sum[a][b] += d.t[k % C] - d.t[(k - 1) % C];
        ++im.tl_n[a][b];
    }
    d.done = dn;
    for (const auto& hm : im.tl_hmark) {   // the readback stamp (1) each decode call follows
        auto lb = std::upper_bound(t1.begin(), t1.end(), hm.first);
        if (lb == t1.begin()) continue;
        im.tl_h_entry += hm.first - *(lb - 1); im.tl_h_sum += hm.second - *(lb - 1); ++im.tl_h_n;
    }
    im.tl_hmark.clear();
    const size_t NC = im.tl_rings.size() - 1;
    im.tl_cbusy.resize(NC, 0.0); im.tl_clag.resize(NC, 0.0); im.tl_csum_lag.resize(NC, 0.0); im.tl_cn.resize(NC, 0);
    im.tl_chunk.resize(NC, 0.0); im.tl_chunk_max.resize(NC, 0.0); im.tl_chunk_n.resize(NC, 0); im.tl_chunk_late.resize(NC, 0);
    im.tl_copy_lag.resize(NC, 0.0); im.tl_copy_sum.resize(NC, 0.0); im.tl_copy_n.resize(NC, 0);
    im.tl_op_busy.resize(NC, 0.0); im.tl_op_lag.resize(NC, 0.0); im.tl_op_n.resize(NC, 0);
    for (size_t c = 0; c < NC; ++c) {
        Ds41XeImpl::TlRing& r = *im.tl_rings[1 + c];
        const uint32_t rn = r.seen.load(std::memory_order_acquire);
        for (uint32_t k = r.done; k + 1 < rn; ++k) {
            if (r.id[k % C] == 12 && r.id[(k + 1) % C] == 13) {   // a promotion chunk
                const double ts = r.t[k % C], te = r.t[(k + 1) % C];
                im.tl_chunk[c] += te - ts;
                im.tl_chunk_max[c] = std::max(im.tl_chunk_max[c], te - ts);
                ++im.tl_chunk_n[c];
                auto nb = std::upper_bound(t1.begin(), t1.end(), ts);   // the next readback after its start
                if (nb != t1.end() && te > *nb) ++im.tl_chunk_late[c];
                continue;
            }
            if (r.id[k % C] == 20 && r.id[(k + 1) % C] == 21) {   // an oproj part (STRATA_TP_OPROJ)
                const double ts = r.t[k % C], te = r.t[(k + 1) % C];
                auto lb = std::upper_bound(t5.begin(), t5.end(), te);   // the dense card's start of that projection
                if (lb != t5.begin()) { im.tl_op_busy[c] += te - ts; im.tl_op_lag[c] += te - *(lb - 1); ++im.tl_op_n[c]; }
                continue;
            }
            if (r.id[k % C] == 11 && r.id[(k + 1) % C] == 14) {   // end -> copy done; copy done -> the next sum
                const double te = r.t[k % C], tc = r.t[(k + 1) % C];
                auto ub = std::lower_bound(t3.begin(), t3.end(), tc);
                if (ub != t3.end()) { im.tl_copy_lag[c] += tc - te; im.tl_copy_sum[c] += *ub - tc; ++im.tl_copy_n[c]; }
                continue;
            }
            if (r.id[k % C] != 10 || r.id[(k + 1) % C] != 11) continue;
            const double ts = r.t[k % C], te = r.t[(k + 1) % C];
            // the dense stamps around it: the last readback-done before its start, the first sum-done after its end
            auto lb = std::upper_bound(t1.begin(), t1.end(), ts);
            auto ub = std::lower_bound(t3.begin(), t3.end(), te);
            if (lb == t1.begin() || ub == t3.end()) continue;
            im.tl_cbusy[c] += te - ts; im.tl_clag[c] += ts - *(lb - 1); im.tl_csum_lag[c] += *ub - te; ++im.tl_cn[c];
        }
        r.done = rn;
    }
    if (++im.tl_tokens % 64 == 0) tl_print(im);
}

namespace {
void tl_print(Ds41XeImpl& im) {
    const size_t NC = im.tl_cn.size();
    std::fprintf(stderr, "gpu timeline (%ld flushes, host-clock stamps) dense us:", im.tl_tokens);
    for (int a = 0; a < 17; ++a)
        for (int b = 0; b < 17; ++b)
            if (im.tl_n[a][b]) std::fprintf(stderr, " %d->%d %.1f (x%ld)", a, b, im.tl_sum[a][b] / im.tl_n[a][b], im.tl_n[a][b]);
    if (im.tl_h_n) std::fprintf(stderr, "; host after readback: tier entry %.1f us, sum submitted %.1f us (x%ld)",
                                im.tl_h_entry / im.tl_h_n, im.tl_h_sum / im.tl_h_n, im.tl_h_n);
    std::fprintf(stderr, "; cards [readback->start, start->end, end->sum]:");
    for (size_t c = 0; c < NC; ++c)
        if (im.tl_cn[c]) std::fprintf(stderr, " %zu: %.1f %.1f %.1f (x%ld)", c, im.tl_clag[c] / im.tl_cn[c], im.tl_cbusy[c] / im.tl_cn[c],
                                      im.tl_csum_lag[c] / im.tl_cn[c], im.tl_cn[c]);
    for (size_t c = 0; c < NC; ++c)
        if (c < im.tl_copy_n.size() && im.tl_copy_n[c])
            std::fprintf(stderr, "; card %zu end->copy done %.1f us, copy done->sum %.1f us (x%ld)", c,
                         im.tl_copy_lag[c] / im.tl_copy_n[c], im.tl_copy_sum[c] / im.tl_copy_n[c], im.tl_copy_n[c]);
    for (size_t c = 0; c < NC; ++c)
        if (c < im.tl_op_n.size() && im.tl_op_n[c])
            std::fprintf(stderr, "; card %zu oproj part ready->done %.1f us, dense start->part done %.1f us (x%ld)", c,
                         im.tl_op_busy[c] / im.tl_op_n[c], im.tl_op_lag[c] / im.tl_op_n[c], im.tl_op_n[c]);
    for (size_t c = 0; c < NC; ++c)
        if (c < im.tl_chunk_n.size() && im.tl_chunk_n[c])
            std::fprintf(stderr, "; card %zu promotion chunks %.1f us (max %.0f, x%ld, %ld past the next readback)", c,
                         im.tl_chunk[c] / im.tl_chunk_n[c], im.tl_chunk_max[c], im.tl_chunk_n[c], im.tl_chunk_late[c]);
    std::fprintf(stderr, "\n");
}
}  // namespace
void Ds41XeTier::set_cache_enabled(bool on) { im_->cache_enabled = on; }

bool Ds41XeTier::reclaim(int gpu, int64_t bytes) {
    Ds41XeImpl& im = *im_;
    if (im.dense < 0 || im.rep_slabs.empty()) return false;
    XeCard& c = *im.cards[(size_t) im.dense];
    if (c.gpu != gpu) return false;
    const int64_t before = c.free_vram(), freed = (int64_t) (im.rep_slabs.size() * ((size_t) 256 << 20));
    rep_drop(im);
    // the driver may show the freed pages late: up to 2 s for most of them
    for (int i = 0; i < 200; ++i) {
        const int64_t fr = c.free_vram();
        if (fr < 0 || before < 0 || fr >= before + freed * 9 / 10) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::fprintf(stderr, "xe tier: replicas given back for a %.0f MiB allocation on the dense card (%.2f GiB free now)\n",
                 bytes / 1048576.0, c.free_vram() / kGiB);
    return true;
}

void Ds41XeTier::release_cache() {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync
    rep_drop(im);   // larry, engine1: the replicas go first (copies: the expert cards still hold them)
    if (im.dense < 0 || im.dense_set.empty()) return;
    if (im.groute_on) { im.rt_dense_ev.wait(); im.last_sum.wait(); }   // STRATA_XE_GROUTE: its share on the dense queue
    for (const auto& le : im.dense_set) {
        const size_t key = (size_t) im.key(le.first, le.second);
        im.card_of[key] = -1;
        im.dptr[key] = nullptr;
        rt_put(im, key);
    }
    im.dense_set.clear();
    im.cards[(size_t) im.dense]->free_experts();
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
}


// ================================ decode / verify ================================

namespace {

/// The CPU pool's share of a call: `ents` (all in the arena / file tier), outputs weighted into `hs` [8][n_embd].
void cpu_ents(Ds41XeImpl& im, int64_t l, const std::vector<const Ent*>& ents, const float* x, float* hs) {
    const int64_t H = im.g.n_embd;
    static const bool skip_dbg = env_i("STRATA_XE_CPU_SKIP", 0) != 0;   // DEBUG: timing bound only (wrong output)
    if (ents.empty() || skip_dbg) return;
    const double t0 = now_ms();
    if (im.cpart.size() < (size_t) ents.size() * kMaxJobTok * H) im.cpart.resize((size_t) ents.size() * kMaxJobTok * H);
    // the activation of every token the CPU experts need, once per format
    bool tok_q2[kMaxJobTok] = {}, tok_nat[kMaxJobTok][8] = {};
    std::vector<std::vector<uint8_t>> tmp(ents.size());
    size_t done = 0;
    while (done < ents.size()) {
        const size_t nb = std::min<size_t>(ents.size() - done, (size_t) cpu::ExpertPool::kMaxSplitMulti);
        for (size_t i = 0; i < nb; ++i) {
            const Ent& en = *ents[done + i];
            const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, en.e)];
            int fs = -1;
            if (!cpu::q2_native_kernels(f.gu_type))
                for (size_t b = 0; b < im.nact_fmt.size(); ++b) if (im.nact_fmt[b] == f.gu_act) fs = (int) b;
            cpu::ExpertJobMulti& j = im.cjobs[i];
            j = cpu::ExpertJobMulti();
            j.blob = host_blob(im, l, en.e, tmp[done + i]);
            j.fmt = &f;
            j.nt = en.nt;
            for (int s = 0; s < en.nt; ++s) {
                const int t = en.tok[s];
                if (fs < 0) {
                    if (!tok_q2[t]) { cpu::act_quant_any(x + (size_t) t * H, (int) H, im.actq[(size_t) t]); tok_q2[t] = true; }
                } else if (!tok_nat[t][fs]) {
                    cpu::native_quant_act(f, x + (size_t) t * H,
                                          im.nact.data() + ((size_t) t * im.nact_fmt.size() + (size_t) fs) * cpu::kNativeActBytes);
                    tok_nat[t][fs] = true;
                }
                j.act[s] = &im.actq[(size_t) t];
                j.nact[s] = fs < 0 ? nullptr : im.nact.data() + ((size_t) t * im.nact_fmt.size() + (size_t) fs) * cpu::kNativeActBytes;
                j.out[s] = im.cpart.data() + ((done + i) * kMaxJobTok + (size_t) s) * H;
            }
        }
        im.pool->run_split_multi_native(*im.cjobs[0].fmt, im.cjobs.data(), (int) nb);
        done += nb;
    }
    for (size_t i = 0; i < ents.size(); ++i) {
        const Ent& en = *ents[i];
        for (int s = 0; s < en.nt; ++s) {
            float* o = hs + (size_t) en.tok[s] * H;
            const float* p = im.cpart.data() + (i * kMaxJobTok + (size_t) s) * H;
            const float w = en.w[s];
            for (int64_t h = 0; h < H; ++h) o[h] += w * p[h];
        }
    }
    im.st.cpu += (int64_t) ents.size();
    im.st.cpu_ms += now_ms() - t0;
}

// the CPU-pool helper (STRATA_XE_CPU_ASYNC): spins ~100 us for a task, then sleeps on the condition variable
void cpu_worker(Ds41XeImpl& im) {
    // STRATA_XE_CPU_SPIN: pause iterations before the helper sleeps (20000 ~ 100 us; a CPU-pool task comes every few
    // ms in decode, so with the default nearly every task first pays the condition variable's wake-up)
    static const int spin = env_i("STRATA_XE_CPU_SPIN", 20000);
    for (;;) {
        int st = im.cpu_state.load(std::memory_order_acquire);
        for (int i = 0; st == 0 && i < spin; ++i) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
            st = im.cpu_state.load(std::memory_order_acquire);
        }
        if (st == 0) {
            std::unique_lock<std::mutex> lk(im.cpu_mu);
            im.cpu_cv.wait(lk, [&] { return im.cpu_state.load(std::memory_order_acquire) != 0; });
            st = im.cpu_state.load(std::memory_order_acquire);
        }
        if (st == 3) return;
        if (st != 1) continue;
        const Ds41XeImpl::CpuTask t = im.cpu_task;
        std::memset(t.hs, 0, t.n_out * 4);
        cpu_ents(im, t.l, *static_cast<const std::vector<const Ent*>*>(t.ents), t.x, t.hs);
        im.cpu_state.store(2, std::memory_order_release);
    }
}

void cpu_post(Ds41XeImpl& im, const Ds41XeImpl::CpuTask& t) {
    if (!im.cpu_thr.joinable()) im.cpu_thr = std::thread(cpu_worker, std::ref(im));
    im.cpu_task = t;
    {
        std::lock_guard<std::mutex> lk(im.cpu_mu);
        im.cpu_state.store(1, std::memory_order_release);
    }
    im.cpu_cv.notify_one();
}

void cpu_join(Ds41XeImpl& im) {
    while (im.cpu_state.load(std::memory_order_acquire) != 2) {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
    im.cpu_state.store(0, std::memory_order_release);
}

/// The planar kernels of a card's share on queue q: n jobs (device-readable) on X [nt][n_embd] into dst, c's scratch.
sycl::event card_experts_at(Ds41XeImpl& im, XeCard& c, sycl::queue& q, const XeJob* jobs, int n, int nt, int max_nt,
                            const float* X, float* dst) {
    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
    static const bool xmx = env_i("STRATA_XE_XMX", 0) != 0;   // larry: the XMX kernels (xe_xmx.cpp)
    if (xmx) {
        xe::xe_xt(q, X, nt, H, c.xt);
        return xe::xe_experts_x(q, jobs, n, nt, c.xt, c.ht, dst, H, F, im.L, false);
    }
    xe::xe_quant_rows(q, X, nt, H, c.xq);
    return xe::xe_experts_q8(q, jobs, n, nt, c.xq, c.hq, dst, H, F, im.L, false, max_nt);
}

/// A card's share of a decode / verify call: its n jobs (staged in c.stage_dev, X at xoff) into dst [nt][n_embd].
sycl::event card_experts(Ds41XeImpl& im, XeCard& c, int n, int nt, int max_nt, size_t xoff, float* dst, const float* xsrc = nullptr) {
    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
    const XeJob* jobs = (const XeJob*) c.stage_dev;
    const float* X = xsrc ? xsrc : (const float*) (c.stage_dev + xoff);   // xsrc: pinned host rows, read over PCIe
    if (!im.planar) return xe::xe_experts(c.q, jobs, n, nt, X, c.H, dst, H, F, im.L, false);
    return card_experts_at(im, c, c.q, jobs, n, nt, max_nt, X, dst);
}

/// STRATA_XE_ARM: one release of card c - the next count into a ring slot, copied into its flag by the dense card
void arm_release(Ds41XeImpl& im, XeCard& c) {
    int* src = c.arm_src + 16 * (c.arm_src_i++ & 63);
    *src = ++c.arm_rel;
    im.relq->memcpy(c.arm_flag, src, sizeof(int));   // decode: the only submitting thread (no submit lock needed)
}

/// STRATA_XE_ARM: releases every armed sequence with null jobs and waits for them (a call they were not armed for, a
/// prompt chunk, close) - a spin kernel never waits for a release that is not coming.
void disarm(Ds41XeImpl& im) {
    bool any = false;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.armed) continue;
        if (!c.arm_routed) {   // STRATA_XE_GROUTE: a routed one was released by the dense card (route) with its jobs
            for (int k = 0; k < c.arm_ng; ++k) std::memcpy(c.stage_host + (size_t) k * sizeof(XeJob), &c.arm_null, sizeof(XeJob));
            arm_release(im, c);
        }
        c.armed = false;
        c.arm_routed = false;
        any = true;
    }
    if (im.rt_dense_armed) { im.rt_dense_ev.wait(); im.rt_dense_armed = false; any = true; }
    im.routed_l = -1;
    if (!any) return;
    for (auto& cp : im.cards) if (cp->arm_flag) cp->arm_ev.wait();
    im.armed_l = -1;
    ++im.arm_disarmed;
}

bool ev_done(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}

// larry, engine1: the host repack 32 blocks at a time - each plane's run of them (whole 64-byte lines: 64 | 2 x 32,
// 12 x 32, ...) gathered in a small buffer with fixed-size copies, then streamed out (no read-for-ownership, nothing
// left in the caches the CPU pool uses; half-line runs made the write-combining flush partial lines: 1.6 GB/s).  Under
// STRATA_TP the promotions' fill writes the planes straight into the pinned bounce, so a card only copies them in: no
// repack kernel beside the TP core's passes and its draft (those delayed the draft's first cross-card sum by up to
// 5.5 ms a step).  The per-field memcpy form took 10-20 ms an expert on a busy host.
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
constexpr int kRpG = 32;
template <int BB, int SO, int SZ>
__attribute__((target("avx2"))) inline void plane32(uint8_t* run, const uint8_t* sb) {
    alignas(64) uint8_t buf[kRpG * SZ];
    for (int b = 0; b < kRpG; ++b) std::memcpy(buf + b * SZ, sb + b * BB + SO, SZ);
    for (int k = 0; k < kRpG * SZ; k += 32) _mm256_stream_si256((__m256i*) (run + k), _mm256_load_si256((const __m256i*) (buf + k)));
}
__attribute__((target("avx2"))) bool cpu_repack_mat16(int type, const uint8_t* src, uint8_t* dst, size_t NB) {
    if (NB % kRpG || ((uintptr_t) dst & 63)) return false;
    if (type == xe::kQ3K) {
        if ((64 * NB) % 64 || (96 * NB) % 64 || (108 * NB) % 64) return false;
        for (size_t i = 0; i < NB; i += kRpG) {
            const uint8_t* sb = src + i * 110;
            plane32<110, 32, 64>(dst + i * 64, sb);
            plane32<110, 0, 32>(dst + 64 * NB + i * 32, sb);
            plane32<110, 96, 12>(dst + 96 * NB + i * 12, sb);
            plane32<110, 108, 2>(dst + 108 * NB + i * 2, sb);
        }
    } else if (type == xe::kQ20) {
        if ((16 * NB) % 64) return false;
        for (size_t i = 0; i < NB; i += kRpG) {
            const uint8_t* sb = src + i * 18;
            alignas(64) uint8_t buf[kRpG * 16];
            for (int b = 0; b < kRpG; ++b)
                for (int u = 0; u < 4; ++u) {
                    // a unit's 4 bytes as one word: code (byte m, field k) at bit 8m + 2k goes to bit 8k + 2m - a 4 x 4
                    // transpose of 2-bit fields in two delta swaps (the anti-diagonals of the 2 x 2 quads, then the quads)
                    uint32_t x;
                    std::memcpy(&x, sb + b * 18 + 2 + 4 * u, 4);
                    uint32_t t = (x ^ (x >> 6)) & 0x00CC00CCu;
                    x ^= t ^ (t << 6);
                    t = (x ^ (x >> 12)) & 0x0000F0F0u;
                    x ^= t ^ (t << 12);
                    std::memcpy(buf + b * 16 + 4 * u, &x, 4);
                }
            for (int k = 0; k < kRpG * 16; k += 32) _mm256_stream_si256((__m256i*) (dst + i * 16 + k), _mm256_load_si256((const __m256i*) (buf + k)));
            plane32<18, 0, 2>(dst + 16 * NB + i * 2, sb);
        }
    } else if (type == xe::kPtq1) {
        if ((16 * NB) % 64 || (24 * NB) % 64) return false;
        for (size_t i = 0; i < NB; i += kRpG) {
            const uint8_t* sb = src + i * 28;
            plane32<28, 0, 16>(dst + i * 16, sb);
            plane32<28, 16, 8>(dst + 16 * NB + i * 8, sb);
            plane32<28, 24, 4>(dst + 24 * NB + i * 4, sb);
        }
    } else {
        return false;
    }
    _mm_sfence();   // weakly ordered: visible before the fill is flagged done
    return true;
}
#endif

/// Host twin of xe_planar.cpp's repack_mat (the same bytes): an [nrows x n] matrix of NB blocks, file layout -> planes.
void cpu_repack_mat(int type, const uint8_t* src, uint8_t* dst, size_t NB) {
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
    static const bool fast = env_i("STRATA_XE_CPU_REPACK16", 1) != 0;
    if (fast && cpu_repack_mat16(type, src, dst, NB)) return;
#endif
    if (type == xe::kQ3K) {
        for (size_t i = 0; i < NB; ++i) {
            const uint8_t* sb = src + i * 110;
            std::memcpy(dst + i * 64, sb + 32, 64);
            std::memcpy(dst + 64 * NB + i * 32, sb, 32);
            std::memcpy(dst + 96 * NB + i * 12, sb + 96, 12);
            std::memcpy(dst + 108 * NB + i * 2, sb + 108, 2);
        }
    } else if (type == xe::kQ20) {
        // 16-value unit: file value j' at byte j'/4 bits 2(j'%4) -> byte j'%4 bits 2(j'/4), a 4 x 4 transpose of 2-bit
        // codes: input byte b's code i goes to output byte i at bits 2b.  spread[x] puts x's codes at bytes 0..3.
        static const auto spread = [] {
            std::array<uint32_t, 256> t{};
            for (uint32_t x = 0; x < 256; ++x)
                for (int i = 0; i < 4; ++i) t[x] |= ((x >> (2 * i)) & 3u) << (8 * i);
            return t;
        }();
        for (size_t i = 0; i < NB; ++i) {
            const uint8_t* sb = src + i * 18;
            for (int u = 0; u < 4; ++u) {
                const uint8_t* in = sb + 2 + 4 * u;
                const uint32_t o = spread[in[0]] | (spread[in[1]] << 2) | (spread[in[2]] << 4) | (spread[in[3]] << 6);
                std::memcpy(dst + i * 16 + 4 * u, &o, 4);   // little-endian: byte k = o >> 8k
            }
            std::memcpy(dst + 16 * NB + i * 2, sb, 2);
        }
    } else {
        for (size_t i = 0; i < NB; ++i) {
            const uint8_t* sb = src + i * 28;
            std::memcpy(dst + i * 16, sb, 16);
            std::memcpy(dst + 16 * NB + i * 8, sb + 16, 8);
            std::memcpy(dst + 24 * NB + i * 4, sb + 24, 4);
        }
    }
}

// larry, engine1 (STRATA_XE_REPLICA_GIB, default 0 = none): which expert-card experts the dense card copies - each
// layer's hottest there (the ranked order), a round over the layers at a time, at most STRATA_XE_REPLICA_PER_LAYER (16)
// a layer, up to the GiB - and their planar bytes read once into pinned host memory (~2 GiB of the driver-held RAM),
// so a seed uploads them with plain copies.  Only with STRATA_TP (the TP core's route_jobs balances them).
void rep_prepare(Ds41XeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& order) {
    const double gib = env_d("STRATA_XE_REPLICA_GIB", 0.0);
    if (gib <= 0.0 || im.dense < 0 || env_i("STRATA_TP", 0) <= 0 || !im.planar || im.cards.size() < 2) return;
    const double t0 = now_ms();
    const int per_layer = std::max(1, env_i("STRATA_XE_REPLICA_PER_LAYER", 16));
    const int64_t NL = im.g.n_layers;
    // the candidates' order: the ranking, or STRATA_XE_REPLICA_USAGE=<file> (STRATA_XE_USAGE_OUT's: the most jobs first)
    std::vector<std::pair<int32_t, int32_t>> cand = order;
    if (const char* uf = std::getenv("STRATA_XE_REPLICA_USAGE"); uf && *uf) {
        std::vector<std::pair<uint32_t, size_t>> use;   // (jobs, key)
        if (std::FILE* f = std::fopen(uf, "r")) {
            char tag;
            long long l, e, jobs, picks, card, bytes;
            char line[256];
            while (std::fgets(line, sizeof line, f))
                if (std::sscanf(line, "%c %lld %lld %lld %lld %lld %lld", &tag, &l, &e, &jobs, &picks, &card, &bytes) == 7 && tag == 'U' &&
                    l >= 0 && l < NL && e >= 0 && e < im.g.n_experts)
                    use.emplace_back((uint32_t) jobs, (size_t) im.key(l, e));
            std::fclose(f);
        }
        std::stable_sort(use.begin(), use.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        cand.clear();
        for (const auto& u : use) cand.emplace_back((int32_t) (u.second / (size_t) im.g.n_experts), (int32_t) (u.second % (size_t) im.g.n_experts));
        std::fprintf(stderr, "xe tier: replicas: candidates by the uses in %s (%zu experts)\n", uf, cand.size());
    }
    std::vector<std::vector<std::pair<int32_t, int32_t>>> by_layer((size_t) NL);
    for (const auto& le : cand) {
        const int c = im.card_of[(size_t) im.key(le.first, le.second)];
        auto& v = by_layer[(size_t) le.first];
        if (c >= 0 && c != im.dense && (int) v.size() < per_layer) v.push_back(le);
    }
    const size_t cap = (size_t) (gib * kGiB);
    size_t total = 0;
    for (int r = 0; r < per_layer; ++r) {
        bool any = false;
        for (int64_t l = 0; l < NL; ++l) {
            const auto& v = by_layer[(size_t) l];
            if (r >= (int) v.size()) continue;
            const size_t b = (im.bbytes(v[(size_t) r].first, v[(size_t) r].second) + 255) & ~(size_t) 255;
            if (total + b > cap) continue;
            im.rep_set.push_back(v[(size_t) r]);
            im.rep_off.push_back(total);
            total += b;
            any = true;
        }
        if (!any) break;
    }
    if (im.rep_set.empty()) return;
    im.rep_host = sycl::malloc_host<uint8_t>(total, im.ctx);
    if (!im.rep_host) {
        std::fprintf(stderr, "xe tier: replicas: %.2f GiB of pinned memory refused - none\n", total / kGiB);
        im.rep_set.clear();
        im.rep_off.clear();
        return;
    }
    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
    auto qk = [](int t) { return t == xe::kQ3K ? 256 : t == xe::kQ20 ? 64 : 128; };
    read_all(im, im.rep_set, env_i("STRATA_XE_READERS", 12), [&](size_t k, const uint8_t* buf) {
        const auto& le = im.rep_set[k];
        const cpu::NativeFmt& f = im.fl[(size_t) im.cls(le.first, le.second)];
        uint8_t* dst = im.rep_host + im.rep_off[k];
        const size_t nbg = (size_t) F * (H / qk(f.gu_type)), nbd = (size_t) H * (F / qk(f.d_type));
        cpu_repack_mat(f.gu_type, buf, dst, nbg);
        cpu_repack_mat(f.gu_type, buf + f.up_off, dst + f.up_off, nbg);
        cpu_repack_mat(f.d_type, buf + f.down_off, dst + f.down_off, nbd);
    });
    im.rep_ptr.assign(im.card_of.size(), nullptr);
    std::fprintf(stderr, "xe tier: replicas: %zu expert-card experts (%.2f GiB, up to %d a layer) kept pinned for the dense card "
                 "(%.1f s)\n", im.rep_set.size(), total / kGiB, per_layer, (now_ms() - t0) / 1000.0);
}

/// The replicas onto the dense card (after its cache: seed / regrow), slab by slab while STRATA_XE_REPLICA_MARGIN_GIB
/// (1.0: in TP decode the dense card's use barely moves after the seed - 1.97 -> 1.95 GiB free) stays free; published
/// in the route table once their copies landed.  Returns the replicas uploaded.
int64_t rep_upload(Ds41XeImpl& im) {
    if (im.rep_set.empty() || im.dense < 0 || !im.rep_slabs.empty() || !im.rt) return 0;
    XeCard& c = *im.cards[(size_t) im.dense];
    const double t0 = now_ms();
    const int64_t margin = (int64_t) (env_d("STRATA_XE_REPLICA_MARGIN_GIB", 1.0) * kGiB);
    constexpr size_t kRepSlab = (size_t) 256 << 20;
    std::vector<size_t> keys;
    int64_t bytes = 0;
    for (size_t i = 0; i < im.rep_set.size(); ++i) {
        const auto& le = im.rep_set[i];
        const size_t key = (size_t) im.key(le.first, le.second);
        if (im.card_of[key] < 0 || im.card_of[key] == im.dense) continue;   // (no longer on an expert card)
        const size_t b = im.bbytes(le.first, le.second), ba = (b + 255) & ~(size_t) 255;
        if (im.rep_slabs.empty() || im.rep_slab_used + ba > kRepSlab) {
            const int64_t fr = c.free_vram();
            if (fr >= 0 && fr - (int64_t) kRepSlab < margin) break;
            uint8_t* p = local_alloc(c.q, kRepSlab);
            if (!p) break;
            im.rep_slabs.push_back(p);
            im.rep_slab_used = 0;
        }
        uint8_t* dst = im.rep_slabs.back() + im.rep_slab_used;
        im.rep_slab_used += ba;
        c.q.memcpy(dst, im.rep_host + im.rep_off[i], b);
        im.rep_ptr[key] = dst;
        keys.push_back(key);
        bytes += (int64_t) b;
    }
    c.q.wait();
    // STRATA_XE_REPLICA_CHECK=1: every copy against its expert card's bytes (the host repack vs the card's)
    static const bool check = env_i("STRATA_XE_REPLICA_CHECK", 0) != 0;
    if (check && !keys.empty()) {
        int64_t bad = 0;
        uint8_t* ha = sycl::malloc_host<uint8_t>((size_t) im.blob_max * 2, im.ctx);
        for (size_t key : keys) {
            const int64_t l = (int64_t) key / im.g.n_experts, e = (int64_t) key % im.g.n_experts;
            const size_t b = im.bbytes(l, e);
            if (!ha) break;
            im.cards[(size_t) im.card_of[key]]->q.memcpy(ha, im.dptr[key], b).wait();
            c.q.memcpy(ha + im.blob_max, im.rep_ptr[key], b).wait();
            if (std::memcmp(ha, ha + im.blob_max, b) != 0) ++bad;
        }
        if (ha) sycl::free(ha, im.ctx);
        std::fprintf(stderr, "xe tier: replicas: checked %zu against their expert cards, %lld differ\n", keys.size(), (long long) bad);
    }
    for (size_t key : keys) rt_put(im, key);
    im.rep_n = (int64_t) keys.size();
    im.rep_bytes = bytes;
    std::fprintf(stderr, "xe tier: replicas: %lld on the dense card (%.2f GiB in %zu slabs) in %.2f s; %.2f GiB free there\n",
                 (long long) im.rep_n, bytes / kGiB, im.rep_slabs.size(), (now_ms() - t0) / 1000.0, c.free_vram() / kGiB);
    return im.rep_n;
}

/// The replicas off the dense card (before its cache gives VRAM back; no TP pass runs then - each one waits for its
/// cards, and the next refreshes its table first).
void rep_drop(Ds41XeImpl& im) {
    if (im.rep_slabs.empty()) return;
    XeCard& c = *im.cards[(size_t) im.dense];
    for (const auto& le : im.rep_set) {
        const size_t key = (size_t) im.key(le.first, le.second);
        if (im.rep_ptr[key]) {
            im.rep_ptr[key] = nullptr;
            rt_put(im, key);
        }
    }
    c.q.wait();
    for (uint8_t* p : im.rep_slabs) local_free(c.q, p);
    im.rep_slabs.clear();
    im.rep_slab_used = 0;
    im.rep_n = 0;
    im.rep_bytes = 0;
}

/// larry, STRATA_XE_HOTRING (default 0): pageable experts onto a card through a small pinned ring that stays in the
/// CPU's caches - each chunk DMA'd as soon as it is copied (plain stores, so its lines are still in the L3 when the
/// card reads them: one DRAM trip a byte instead of three).  bench/dqb/stage_ring: 13.0 -> 21.7 GB/s onto cards 1-3.
bool hot_on() { static const bool on = env_i("STRATA_XE_HOTRING", 0) != 0; return on; }
size_t hot_chunk() {
    static const size_t c = (size_t) std::max(64, std::min(env_i("STRATA_XE_HOTRING_KB", 1024), 16384)) << 10;
    return c;
}
int hot_slots() { static const int n = std::max(2, std::min(env_i("STRATA_XE_HOTRING_N", 4), 64)); return n; }
bool hot_check_on() { static const bool on = env_i("STRATA_XE_HOTRING_CHECK", 0) != 0; return on; }

/// a copy that leaves the destination's lines in the caches (glibc's memcpy streams copies this large past them)
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
__attribute__((target("avx2")))
#endif
void hot_copy(uint8_t* dst, const uint8_t* src, size_t n) {
    size_t i = 0;
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
    // larry, STRATA_XE_HOTCOPY_PF: software prefetch D bytes ahead (across the 4 KiB pages)
    // larry, STRATA_XE_HOTCOPY_NTA (patch-hotnta.py): the same prefetch with the non-temporal hint
    static const bool pf_nta = env_i("STRATA_XE_HOTCOPY_NTA", 0) != 0;
    static const size_t pf_d = (size_t) std::max(0, env_i("STRATA_XE_HOTCOPY_PF", pf_nta ? 1024 : 0));
    if (pf_d && ((uintptr_t) dst & 31) == 0) {
        for (; i + 128 <= n; i += 128) {
            if (i + pf_d < n && pf_nta) {
                _mm_prefetch((const char*) (src + i + pf_d), _MM_HINT_NTA);
                _mm_prefetch((const char*) (src + i + pf_d + 64), _MM_HINT_NTA);
            } else if (i + pf_d < n) {
                _mm_prefetch((const char*) (src + i + pf_d), _MM_HINT_T0);
                _mm_prefetch((const char*) (src + i + pf_d + 64), _MM_HINT_T0);
            }
            const __m256i a = _mm256_loadu_si256((const __m256i*) (src + i));
            const __m256i b = _mm256_loadu_si256((const __m256i*) (src + i + 32));
            const __m256i c = _mm256_loadu_si256((const __m256i*) (src + i + 64));
            const __m256i d = _mm256_loadu_si256((const __m256i*) (src + i + 96));
            _mm256_store_si256((__m256i*) (dst + i), a);
            _mm256_store_si256((__m256i*) (dst + i + 32), b);
            _mm256_store_si256((__m256i*) (dst + i + 64), c);
            _mm256_store_si256((__m256i*) (dst + i + 96), d);
        }
    }
    if (((uintptr_t) dst & 31) == 0)
        for (; i + 128 <= n; i += 128) {
            const __m256i a = _mm256_loadu_si256((const __m256i*) (src + i));
            const __m256i b = _mm256_loadu_si256((const __m256i*) (src + i + 32));
            const __m256i c = _mm256_loadu_si256((const __m256i*) (src + i + 64));
            const __m256i d = _mm256_loadu_si256((const __m256i*) (src + i + 96));
            _mm256_store_si256((__m256i*) (dst + i), a);
            _mm256_store_si256((__m256i*) (dst + i + 32), b);
            _mm256_store_si256((__m256i*) (dst + i + 64), c);
            _mm256_store_si256((__m256i*) (dst + i + 96), d);
        }
#endif
    if (i < n) std::memcpy(dst + i, src + i, n - i);
}

/// larry, engine1 (STRATA_XE_PC_STREAM, default 1): an arena blob into a pinned bounce with streaming stores - no
/// read-for-ownership of the bounce's lines and none of them left in the caches.  The CPU pool beside the helpers is
/// DRAM-bound: its one-expert layers took ~400 us with no fills beside them, ~500 beside plain-store fills and ~475
/// beside these.  dst 32-byte aligned, n a multiple of 16.
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
__attribute__((target("avx2")))
#endif
void stream_copy(uint8_t* dst, const uint8_t* src, size_t n) {
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
    static const bool on = env_i("STRATA_XE_PC_STREAM", 1) != 0;
    if (on && ((uintptr_t) dst & 31) == 0 && n % 16 == 0) {
        size_t i = 0;
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
        for (; i < n; i += 16) _mm_stream_si128((__m128i*) (dst + i), _mm_loadu_si128((const __m128i*) (src + i)));
        _mm_sfence();   // weakly ordered: globally visible before the fill is flagged done
        return;
    }
#endif
    std::memcpy(dst, src, n);
}

/// The promotion cache's slots (each expert card's flex region, STRATA_XE_FLEX_GIB), the bounces and the helper thread.
void pc_setup(Ds41XeImpl& im) {
    im.pc_slot = ((size_t) im.blob_max + 255) & ~(size_t) 255;
    // (larry, engine1: default 0 since the host repack streams the planes into the bounce - a card repack ran
    // kernels beside the TP core's passes and its draft; STRATA_TP_PC_DEVREPACK=1: the card repacks)
    im.pc_devrepack = env_i("STRATA_TP", 0) > 0 && env_i("STRATA_TP_PC_DEVREPACK", 0) != 0;
    int total = 0;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || c.pc_mem || !c.flex || c.pbuf_cap) continue;   // the slots: the card's flex region (no prompt running)
        int n = (int) std::min<size_t>(c.flex_cap / im.pc_slot, 65535);
        if (n < 4 + (im.pc_devrepack ? 1 : 0)) continue;
        if (im.pc_devrepack) { --n; c.pc_tmp = c.flex + (size_t) n * im.pc_slot; }   // the last slot's bytes: the temp
        c.pc_mem = c.flex;
        c.pc_n = n;
        c.pc_key.assign((size_t) n, -1);
        c.pc_tick.assign((size_t) n, 0);
        c.pc_busy = 0;
        total += n;
    }
    if (!im.pc_bounce) im.pc_bounce = sycl::malloc_host<uint8_t>((size_t) im.pc_nb * im.pc_slot, im.ctx);
    im.pc_bfree.clear();
    for (int b = 0; b < im.pc_nb; ++b) im.pc_bfree.push_back(b);
    im.pc_where.assign(im.card_of.size(), -1);
    if (!im.pc_thr.joinable()) {
        im.pc_stop = false;
        auto helper = [&im] {
            // background work: it must not take a core from the decode thread's latency-bound path.  larry, engine1:
            // under STRATA_TP the host thread spins on the TP passes' mailboxes and the CPU pool runs beside it - an idle-
            // class helper then never ran (5018 promotions skipped): normal priority there (STRATA_TP_PC_IDLE=1: idle)
            if (env_i("STRATA_TP", 0) <= 0 || env_i("STRATA_TP_PC_IDLE", 0) != 0) {
                sched_param sp{};
                sp.sched_priority = 0;
                pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp);
            }
            for (;;) {
                Ds41XeImpl::PcJob* j = nullptr;
                {
                    std::unique_lock<std::mutex> lk(im.pc_mu);
                    im.pc_cv.wait(lk, [&] { return im.pc_stop || !im.pc_fill.empty(); });
                    if (im.pc_stop) return;
                    j = im.pc_fill.front();
                    im.pc_fill.pop_front();
                }
                const double tf0 = now_ms();
                im.pc_wait_us += (int64_t) ((tf0 - j->t_queued) * 1000.0);
                const int64_t l = j->key / im.g.n_experts, e = j->key % im.g.n_experts;
                const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
                const uint8_t* src = im.aptr[(size_t) j->key];
                uint8_t* dst = im.pc_bounce + (size_t) j->bounce * im.pc_slot;
                if (!im.planar || im.pc_devrepack) {   // (the card repacks)
                    // larry, engine1 (STRATA_TP_PC_PACE, default 0): 1 MB at a time, none while a TP pass's CPU pool
                    // computes - beside it a fill takes the pool's cores' SMT siblings and DRAM (one-expert layers
                    // ~400 -> ~460 us).  Paced they were ~440 us, but later fills missed more next passes: DSpark
                    // cycles 30.2 / 30.5 ms against 29.9 / 30.1 unpaced (bench/engine1/tcpu4.sh)
                    static const bool pace = env_i("STRATA_TP_PC_PACE", 0) != 0;
                    const size_t b = im.bbytes(l, e), piece = pace ? (size_t) 1 << 20 : b;
                    for (size_t off = 0; off < b; off += piece) {
                        while (pace && im.pool_busy.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(25));
                        stream_copy(dst + off, src + off, std::min(piece, b - off));
                    }
                }
                else {
                    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
                    auto qk = [](int t) { return t == xe::kQ3K ? 256 : t == xe::kQ20 ? 64 : 128; };
                    const size_t nbg = (size_t) F * (H / qk(f.gu_type)), nbd = (size_t) H * (F / qk(f.d_type));
                    cpu_repack_mat(f.gu_type, src, dst, nbg);
                    cpu_repack_mat(f.gu_type, src + f.up_off, dst + f.up_off, nbg);
                    cpu_repack_mat(f.d_type, src + f.down_off, dst + f.down_off, nbd);
                }
                im.pc_fill_us += (int64_t) ((now_ms() - tf0) * 1000.0);
                ++im.pc_fill_n;
                j->filled.store(1, std::memory_order_release);
            }
        };
        im.pc_thr = std::thread(helper);
        xe::place_helper(im.pc_thr.native_handle());   // (STRATA_XE_HELPER_CPUS; else off the pass thread's pin)
        // larry, engine1: under STRATA_TP the repacks (~10-20 ms an expert on a busy host) were the promotions' limit
        // (with the card's repack (pc_devrepack) a helper only copies pageable blobs: 2)
        const int more = env_i("STRATA_TP", 0) > 0 ? std::max(0, env_i("STRATA_TP_PC_THREADS", 2) - 1) : 0;
        for (int k = 0; k < more; ++k) {
            im.pc_thr_more.emplace_back(helper);
            xe::place_helper(im.pc_thr_more.back().native_handle());
        }
    }
    im.pc_ready = im.pc_bounce && total > 0;
    static bool logged = false;
    if (!logged || env_i("STRATA_XE_PROF", 0)) {
        logged = true;
        std::fprintf(stderr, "xe tier: promotion cache: %d slots (%.2f GiB) on the expert cards\n", total, total * (double) im.pc_slot / kGiB);
    }
}

/// STRATA_XE_PROMOTE_CHECK=1 (debug): a published slot against the arena blob repacked afresh on the same card
bool pc_check(Ds41XeImpl& im, XeCard& c, int64_t key, const uint8_t* slot) {
    const int64_t l = key / im.g.n_experts, e = key % im.g.n_experts;
    const size_t b = im.bbytes(l, e);
    uint8_t* ref = sycl::malloc_device<uint8_t>(b, c.q);
    std::vector<uint8_t> a(b), r(b);
    upload_planar(im, c, l, e, im.aptr[(size_t) key], ref);
    c.q.memcpy(r.data(), ref, b);
    c.q.memcpy(a.data(), slot, b).wait();
    sycl::free(ref, c.q);
    return std::memcmp(a.data(), r.data(), b) == 0;
}

/// Promotions in flight, a step each: bounce filled -> one chunk of the copy a call on the card's queue (push = true:
/// the end of a call, behind its card work) -> the repack behind the last chunk -> published once the repack is done
/// (push = false: the start of a call)
void pc_poll(Ds41XeImpl& im, bool push) {
    static const bool check = env_i("STRATA_XE_PROMOTE_CHECK", 0) != 0;
    // larry, engine1: 4 MB under STRATA_TP (the copies run on their own queues during the draft; fewer submissions
    // between passes - 64 a pass are ~20 experts)
    static const size_t chunk = (size_t) (env_d("STRATA_XE_PROMOTE_CHUNK_MB", env_i("STRATA_TP", 0) > 0 ? 4.0 : 1.0) * 1048576.0) & ~(size_t) 255;
    for (auto it = im.pc_jobs.begin(); it != im.pc_jobs.end();) {
        Ds41XeImpl::PcJob& j = *it;
        if (j.card < 0) { ++it; continue; }   // larry, engine1: its bounce filling early (pc_early), no card / slot yet
        XeCard& c = *im.cards[(size_t) j.card];
        const int64_t l = j.key / im.g.n_experts, e = j.key % im.g.n_experts;
        uint8_t* slot = c.pc_mem + (size_t) j.slot * im.pc_slot;
        if (j.stage == 0 && j.filled.load(std::memory_order_acquire)) {
            j.src = im.pc_bounce + (size_t) j.bounce * im.pc_slot;
            j.stage = 1;
        }
        if (j.stage == 1 && push) {
            const size_t b = im.bbytes(l, e), n = std::min(chunk ? chunk : b, b - j.sent);
            // a kernel reads the pinned bounce (zero-copy, as the experts read X): a queue memcpy would put the copy
            // engine in the card's queue, and the next kernel after an H2D memcpy starts ~300 us late when it comes to
            // an idle queue (bench/dqb/defer_q.cpp) - the next call's spin / experts.  Blobs are multiples of 16 bytes.
            using u4v = sycl::vec<uint32_t, 4>;
            const u4v* sv = reinterpret_cast<const u4v*>(j.src + j.sent);
            u4v* dv = reinterpret_cast<u4v*>(slot + j.sent);
            if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 12);
            sycl::queue& pq = (size_t) c.idx < im.pc_q.size() && im.pc_q[(size_t) c.idx] ? *im.pc_q[(size_t) c.idx] : c.q;
            // larry, engine1 (STRATA_TP_PC_DMA, default 1 under STRATA_TP): the copy engine instead - the copy kernels
            // (~270 us a 4 MB chunk of PCIe reads) held the compute units the TP core's draft runs on, every card at
            // once (card 0's first draft phase 0.1 -> 3.5 ms); the kernel after the memcpy on this (promotion-only)
            // queue starting late does not matter
            static const bool dma = env_i("STRATA_TP", 0) > 0 && env_i("STRATA_TP_PC_DMA", 1) != 0;
            const sycl::event kev = dma ? pq.memcpy(slot + j.sent, j.src + j.sent, n)
                                        : pq.parallel_for(sycl::range<1>(n / 16), [=](sycl::id<1> i) { dv[i] = sv[i]; });
            if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 13);
            j.sent += n;
            ++im.pc_chunks;
            if (j.sent == b) {   // in queue order: once this kernel is done the whole copy is
                j.ev = kev;
                if (im.pc_devrepack && im.planar) {   // the file layout -> planes: slot -> temp, temp -> slot (in order)
                    const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
                    xe::xe_repack_blob(pq, slot, c.pc_tmp, f.gu_type, f.d_type, f.up_off, f.down_off, (int) im.g.n_embd, (int) im.g.n_ff);
                    const u4v* tv = reinterpret_cast<const u4v*>(c.pc_tmp);
                    u4v* ov = reinterpret_cast<u4v*>(slot);
                    j.ev = pq.parallel_for(sycl::range<1>(b / 16), [=](sycl::id<1> i) { ov[i] = tv[i]; });
                }
                j.stage = 3;
            }
        } else if (j.stage == 3 && !push && ev_done(j.ev)) {
            if (j.bounce >= 0) { im.pc_bfree.push_back(j.bounce); j.bounce = -1; }
            c.pc_busy = std::max(0, c.pc_busy - 1);
            if (check && !pc_check(im, c, j.key, slot)) {   // debug: keep it on the CPU
                ++im.pc_bad;
                c.pc_key[(size_t) j.slot] = -1;
                im.pc_where[(size_t) j.key] = -1;
                std::fprintf(stderr, "xe tier: promotion of layer %lld expert %lld on card %d: slot differs from a fresh repack\n",
                             (long long) l, (long long) e, c.idx);
            } else if (im.card_of[(size_t) j.key] >= 0) {   // larry, patch-pcrace: placed meanwhile - it stays there
                c.pc_key[(size_t) j.slot] = -1;
                im.pc_where[(size_t) j.key] = -1;
            } else {
                im.card_of[(size_t) j.key] = (int8_t) c.idx;
                im.dptr[(size_t) j.key] = slot;
                rt_put(im, (size_t) j.key);
                c.pc_key[(size_t) j.slot] = j.key;
                c.pc_tick[(size_t) j.slot] = im.pc_clock;
                im.pc_where[(size_t) j.key] = (c.idx << 16) | j.slot;
                ++im.pc_done;
                if (im.cpu_log) std::fprintf(im.cpu_log, "P %lld %lld %d\n", (long long) im.tp_pass, (long long) j.key, c.idx);
            }
            it = im.pc_jobs.erase(it);
            continue;
        }
        ++it;
    }
}

/// A CPU-pool expert of layer l: its promotion starts if an expert card has its stage free.  The slot: a free one, else
/// the least recently used of another layer (no queued job reads it: only layer l's are in flight now, every earlier
/// layer's cards were done before this layer's attention; its repack is queued after them anyway).
void pc_want(Ds41XeImpl& im, int64_t l, size_t key) {
    // larry, engine1: a pageable expert's bounce may be filling already (pc_early, at its CPU use) - it only waits for
    // its card and slot (and keeps its bounce until it gets them)
    Ds41XeImpl::PcJob* early = nullptr;
    if (im.pc_where[key] == -3)
        for (auto& j : im.pc_jobs)
            if (j.key == (int64_t) key && j.card < 0) { early = &j; break; }
    if (!early && (im.pc_where[key] != -1 || !im.aptr[key])) return;
    static const int pmin = std::max(1, env_i("STRATA_XE_PROMOTE_MIN", 2));
    if (im.pc_uses.size() != im.card_of.size()) im.pc_uses.assign(im.card_of.size(), 0);
    if (im.pc_uses[key] < 255) ++im.pc_uses[key];
    if (!early && im.pc_uses[key] < pmin) return;
    const int NC = (int) im.cards.size();
    XeCard* cd = nullptr;
    for (int k = 0; k < NC && !cd; ++k) {
        XeCard& c = *im.cards[(size_t) ((im.pc_rr + k) % NC)];
        if (!c.dense && c.pc_n > 0 && c.pc_busy < im.pc_par) cd = &c;
    }
    if (!cd || (!early && im.pc_bfree.empty() && !(im.pc_devrepack && im.apin[key]))) {
        ++im.pc_busy_skip;
        if (im.cpu_log) std::fprintf(im.cpu_log, "S %lld %lld %d\n", (long long) im.tp_pass, (long long) key, cd ? 1 : 0);
        return;
    }
    int slot = -1;
    uint64_t best = UINT64_MAX;
    for (int sl = 0; sl < cd->pc_n; ++sl) {
        const int64_t k = cd->pc_key[(size_t) sl];
        if (k == -1) { slot = sl; break; }
        if (k >= 0 && k / im.g.n_experts != l && cd->pc_tick[(size_t) sl] < best) { best = cd->pc_tick[(size_t) sl]; slot = sl; }
    }
    if (slot < 0) return;
    im.pc_rr = (cd->idx + 1) % NC;
    const int64_t old = cd->pc_key[(size_t) slot];
    if (old >= 0) {   // evicted: the CPU pool runs it again
        im.card_of[(size_t) old] = -1;
        im.dptr[(size_t) old] = nullptr;
        rt_put(im, (size_t) old);
        im.pc_where[(size_t) old] = -1;
        ++im.pc_evicted;
        if (im.cpu_log) std::fprintf(im.cpu_log, "V %lld %lld\n", (long long) im.tp_pass, (long long) old);
    }
    if (im.cpu_log) std::fprintf(im.cpu_log, "Q %lld %lld %d\n", (long long) im.tp_pass, (long long) key, cd->idx);
    cd->pc_key[(size_t) slot] = -2;
    cd->pc_busy += 1;
    im.pc_where[key] = -2;
    if (early) {   // pc_poll takes it from here (filled -> its copies)
        early->card = cd->idx;
        early->slot = slot;
        return;
    }
    im.pc_jobs.emplace_back();
    Ds41XeImpl::PcJob& j = im.pc_jobs.back();
    j.key = (int64_t) key;
    j.card = cd->idx;
    j.slot = slot;
    j.t_queued = now_ms();
    if (im.pc_devrepack && im.apin[key]) {   // the card reads the pinned arena blob itself and repacks it
        j.src = im.aptr[key];
        j.stage = 1;
        return;
    }
    j.bounce = im.pc_bfree.back();   // the helper repacks it into a bounce (pinned arena copies too)
    im.pc_bfree.pop_back();
    { std::lock_guard<std::mutex> lk(im.pc_mu); im.pc_fill.push_back(&j); }
    im.pc_cv.notify_one();
}

/// larry, engine1 (STRATA_TP_PC_EARLY): the pageable experts a TP pass's CPU pool just ran that the next tp_between_passes
/// will want get their bounces filled now, while the pass goes on (their bytes just went through the caches) - there
/// pc_want only places them and their copies run during the draft: the next pass finds them on a card, not two to
/// five passes later
void pc_early(Ds41XeImpl& im, int64_t l, const std::vector<const Ent*>& ents) {
    static const int pmin = std::max(1, env_i("STRATA_XE_PROMOTE_MIN", 2));
    if (im.pc_uses.size() != im.card_of.size()) im.pc_uses.assign(im.card_of.size(), 0);
    for (const Ent* en : ents) {
        const size_t key = (size_t) im.key(l, en->e);
        if (im.pc_where[key] != -1 || !im.aptr[key] || (im.pc_devrepack && im.apin[key])) continue;   // pinned: read in place
        if (im.pc_uses[key] + 1 < pmin || im.pc_bfree.empty()) continue;
        im.pc_jobs.emplace_back();
        Ds41XeImpl::PcJob& j = im.pc_jobs.back();
        j.key = (int64_t) key;
        j.t_queued = now_ms();
        j.bounce = im.pc_bfree.back();
        im.pc_bfree.pop_back();
        im.pc_where[key] = -3;
        { std::lock_guard<std::mutex> lk(im.pc_mu); im.pc_fill.push_back(&j); }
        im.pc_cv.notify_one();
    }
}

/// The cache gone (a prompt's prefetch wants the VRAM; close): in-flight promotions finished, every promoted expert
/// back to the CPU pool, the slots freed.
void pc_drop(Ds41XeImpl& im) {
    im.pc_calls = 0;
    im.pc_tried = false;
    if (!im.pc_ready && im.pc_jobs.empty()) return;
    for (;;) {   // the helper's fills in flight (their jobs are freed below)
        bool busy = false;
        for (auto& j : im.pc_jobs) if (j.stage == 0 && !j.filled.load(std::memory_order_acquire)) busy = true;
        if (!busy) break;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.pc_mem) continue;
        c.q.wait();
        if ((size_t) c.idx < im.pc_q.size() && im.pc_q[(size_t) c.idx]) im.pc_q[(size_t) c.idx]->wait();   // engine1's copies
        for (int sl = 0; sl < c.pc_n; ++sl) {
            const int64_t k = c.pc_key[(size_t) sl];
            if (k >= 0) { im.card_of[(size_t) k] = -1; im.dptr[(size_t) k] = nullptr; rt_put(im, (size_t) k); }
        }
        c.pc_mem = nullptr;   // the flex region (prefetch buffers next)
        c.pc_tmp = nullptr;
        c.pc_n = 0;
        c.pc_key.clear();
        c.pc_tick.clear();
        c.pc_busy = 0;
    }
    im.pc_jobs.clear();
    im.pc_bfree.clear();
    for (int b = 0; b < im.pc_nb; ++b) im.pc_bfree.push_back(b);
    std::fill(im.pc_where.begin(), im.pc_where.end(), -1);
    im.pc_ready = false;
}

/// STRATA_XE_GROUTE_CHECK=1 (debug): the route kernel's job lists and CPU entries of this call against the host's own
/// grouping and card assignment (run_small's) - counts mismatches into groute_bad, prints the first few
void groute_check(Ds41XeImpl& im, int64_t l, int nt, const int32_t* ids, const float* w) {
    const int K = (int) im.g.top_k;
    const size_t NC = im.cards.size();
    std::vector<Ent> ents;
    std::vector<int32_t> first;
    group(ids, w, nt, K, im.g.n_experts, ents, first);
    std::vector<XeJob> got(NC * kMaxDecodeJobs);
    im.dense_q->memcpy(got.data(), im.rt_jobs, got.size() * sizeof(XeJob)).wait();
    std::vector<int> nj(NC, 0);
    int ncpu = 0;
    bool bad = false;
    for (const Ent& en : ents) {
        const size_t key = (size_t) im.key(l, en.e);
        const int c = im.card_of[key];
        if (c < 0 || !((im.routed_mask >> c) & 1u)) {
            const RtCpu& rc = im.rt_hdr->cpu[ncpu++];
            bad = bad || rc.e != en.e || rc.nt != en.nt;
            for (int s = 0; s < en.nt && !bad; ++s) bad = rc.tok[s] != en.tok[s] || rc.k[s] != en.k[s] || rc.w[s] != en.w[s];
            continue;
        }
        const XeJob& j = got[(size_t) c * kMaxDecodeJobs + (size_t) nj[(size_t) c]++];
        XeJob r = make_job(im, l, en.e, im.dptr[key]);
        r.nt = (uint8_t) en.nt;
        for (int s = 0; s < en.nt; ++s) { r.tok[s] = en.tok[s]; r.w[s] = en.w[s]; }
        bad = bad || j.blob != r.blob || j.up_off != r.up_off || j.down_off != r.down_off || j.gu_type != r.gu_type ||
              j.d_type != r.d_type || j.gu_row != r.gu_row || j.d_row != r.d_row || j.nt != r.nt;
        for (int s = 0; s < kMaxJobTok && !bad; ++s) bad = j.tok[s] != r.tok[s] || j.w[s] != r.w[s];
    }
    bad = bad || ncpu != im.rt_hdr->ncpu;
    for (size_t c = 0; c < NC && !bad; ++c)
        if ((im.routed_mask >> c) & 1u)
            for (int k = nj[c]; k < im.routed_ng && !bad; ++k) bad = got[c * kMaxDecodeJobs + (size_t) k].nt != 0;
    if (bad && im.groute_bad++ < 8)
        std::fprintf(stderr, "xe tier: STRATA_XE_GROUTE_CHECK: layer %lld (%d tokens): the route kernel's lists differ from the host's\n",
                     (long long) l, nt);
}

/// STRATA_XE_GROUTE: a call route() dispatched - every card has its jobs (the dense card's route kernel); here the CPU
/// pool's entries (the kernel's list) and the sum, its parts in run_small's order (armed cards, the dense card, the
/// CPU pool; an armed card without jobs adds its zeros)
bool run_routed(Ds41XeImpl& im, int64_t l, int nt, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    const double tl_h0 = Ds41XeTier::tl_on() && nt == 1 ? tl_now_us() : 0;
    const double t0 = now_ms();
    const int64_t H = im.g.n_embd;
    const int K = (int) im.g.top_k;
    const size_t NC = im.cards.size();
    XeCard* dc = im.cards[(size_t) im.dense].get();
    sycl::queue& qd = *im.dense_q;
    const size_t n_out = (size_t) nt * H, stride = (size_t) kMaxJobTok * H;
    float* slots = dc->dslots;
    static const bool check = env_i("STRATA_XE_GROUTE_CHECK", 0) != 0;
    if (check) groute_check(im, l, nt, ids, w);
    if (im.pc_mode < 0) im.pc_mode = env_i("STRATA_XE_PROMOTE", 1) != 0 && im.planar && im.dense >= 0 ? 1 : 0;
    if (im.pc_mode > 0) {   // as run_small (the route kernel ran on the table as it was before this)
        ++im.pc_clock;
        if (!im.pc_ready && !im.pc_tried && !im.pf_on && ++im.pc_calls >= env_i("STRATA_XE_PROMOTE_AFTER", 1)) {
            im.pc_tried = true;
            pc_setup(im);
        }
        if (!im.pc_jobs.empty()) {
            const double tq = now_ms();
            pc_poll(im, false);
            im.pc_poll_us += (now_ms() - tq) * 1000.0;
        }
        if (im.pc_ready)   // the promoted experts this call ran: their slots used now (the LRU)
            for (int i = 0; i < nt * K; ++i) {
                const int32_t e = ids[i];
                if (e < 0 || e >= im.g.n_experts) continue;
                const int32_t pw = im.pc_where[(size_t) im.key(l, e)];
                if (pw >= 0) { im.cards[(size_t) (pw >> 16)]->pc_tick[(size_t) (pw & 0xFFFF)] = im.pc_clock; ++im.pc_hit; }
            }
    }
    // the CPU pool's entries (pinned, complete: the host waited for the readbacks queued behind the route kernel)
    static thread_local std::vector<Ent> cents;
    static thread_local std::vector<const Ent*> cpu_list;
    const int ncpu = std::max(0, std::min(im.rt_hdr->ncpu, kMaxDecodeJobs));
    cents.resize((size_t) ncpu);
    cpu_list.clear();
    for (int i = 0; i < ncpu; ++i) {
        const RtCpu& rc = im.rt_hdr->cpu[i];
        Ent& en = cents[(size_t) i];
        en.e = rc.e;
        en.nt = std::min(rc.nt, kMaxJobTok);
        for (int s = 0; s < kMaxJobTok; ++s) { en.tok[s] = rc.tok[s]; en.k[s] = rc.k[s]; en.w[s] = rc.w[s]; }
        cpu_list.push_back(&en);
    }
    for (const Ent* en : cpu_list) if (im.pc_ready) pc_want(im, l, (size_t) im.key(l, en->e));
    im.st.hits += im.rt_hdr->ncard;
    if (!im.rt_dense_armed) {   // arm() did not queue the dense card's share: now (behind what the dense queue has)
        im.rt_dense_ev = card_experts_at(im, *dc, qd, im.rt_jobs + (size_t) dc->idx * kMaxDecodeJobs, im.routed_ng, nt,
                                         im.routed_mv, im.rt_x, slots + (size_t) dc->idx * stride);
        im.rt_dense_armed = true;
    }
    std::vector<sycl::event> deps;
    struct Src { const float* p[10]; } src{};   // <= 7 expert cards, the dense card, the CPU pool
    int ns = 0;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.armed || !c.arm_routed) continue;
        deps.push_back(c.arm_ev);
        c.stage_ev = c.arm_ev;
        c.armed = false;
        c.arm_routed = false;
        ++im.arm_used;
        src.p[ns++] = slots + (size_t) c.idx * stride;
    }
    deps.push_back(im.rt_dense_ev);
    im.rt_dense_armed = false;
    src.p[ns++] = slots + (size_t) dc->idx * stride;
    if (!cpu_list.empty()) {
        im.hsum_ev.wait();   // the previous call's copy of hsum has been made
        std::memset(im.hsum, 0, n_out * 4);
        cpu_ents(im, l, cpu_list, x, im.hsum);
        im.hsum_ev = qd.memcpy(slots + NC * stride, im.hsum, n_out * 4);
        deps.push_back(im.hsum_ev);
        src.p[ns++] = slots + NC * stride;
    }
    float* o = (float*) out_dev;
    if (Ds41XeTier::tl_on() && nt == 1) tl_stamp(im, 0, qd, 15);
    im.last_sum = qd.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n_out), [=](sycl::id<1> i) {
            float sum = 0.0f;
            for (int k = 0; k < ns; ++k) sum += src.p[k][i];
            o[i] = sum;
        });
    });
    if (Ds41XeTier::tl_on()) tl_stamp(im, 0, qd, nt == 1 ? 3 : 11);
    if (tl_h0 > 0) im.tl_hmark.emplace_back(tl_h0, tl_now_us());
    im.routed_l = -1;
    im.armed_l = -1;
    ++im.groute_used;
    if (im.pc_mode > 0 && !im.pc_jobs.empty()) {   // promotions: behind this call's card work
        const double tq = now_ms();
        pc_poll(im, true);
        im.pc_push_us += (now_ms() - tq) * 1000.0;
    }
    im.st.wall_ms += now_ms() - t0;
    return true;
}

/// One decode / verify call: nt <= 8 tokens.  out_host (nt x n_embd) or out_dev (the dense card) gets the sum.
bool run_small(Ds41XeImpl& im, int64_t l, int nt, const int32_t* ids, const float* w, const float* x, float* out_host,
               void* out_dev) {
    const double tl_h0 = Ds41XeTier::tl_on() && nt == 1 ? tl_now_us() : 0;
    const int64_t H = im.g.n_embd;
    const int K = (int) im.g.top_k;
    im.groute_want = false;   // STRATA_XE_GROUTE: a groute_next() whose attn_router did not route ends with this call
    if (im.routed_l >= 0) {   // the dense card dispatched this call (route)
        if (im.routed_l == l && im.routed_nt == nt && out_dev && im.dense_q && x == im.arm_x) return run_routed(im, l, nt, ids, w, x, out_dev);
        ++im.groute_miss;     // another call: the routed sequences run their jobs, this one goes the host's way behind them
        disarm(im);
    }
    const double t0 = now_ms();
    static thread_local std::vector<Ent> ents;
    static thread_local std::vector<int32_t> first;
    group(ids, w, nt, K, im.g.n_experts, ents, first);
    if (im.pc_mode < 0) im.pc_mode = env_i("STRATA_XE_PROMOTE", 1) != 0 && im.planar && im.dense >= 0 ? 1 : 0;
    if (im.pc_mode > 0) {
        ++im.pc_clock;
        // the slots (the expert cards' flex regions: nothing allocated here - that cost 555 ms mid-decode once)
        if (!im.pc_ready && !im.pc_tried && !im.pf_on && ++im.pc_calls >= env_i("STRATA_XE_PROMOTE_AFTER", 1)) {
            im.pc_tried = true;   // once a drop (no prompt prefetch running: the flex regions are its buffers then)
            pc_setup(im);
        }
        if (!im.pc_jobs.empty()) {
            const double tq = now_ms();
            pc_poll(im, false);
            im.pc_poll_us += (now_ms() - tq) * 1000.0;
        }
    }
    static const bool cpu_stats = env_i("STRATA_XE_CPU_STATS", 0) != 0;
    struct CpuStats { std::vector<uint8_t> seen; int64_t uses = 0, repeats = 0, distinct = 0, calls = 0, tok_uses = 0; };
    static CpuStats cst;
    if (cpu_stats && ++cst.calls % 4000 == 0)
        std::fprintf(stderr, "xe cpu stats (%lld calls): %lld CPU expert uses (%lld token-uses), %lld repeats of an expert already run on the "
                     "CPU, %lld distinct; promoted %lld (hits %lld, evicted %lld, skipped busy %lld, bad %lld)\n", (long long) cst.calls,
                     (long long) cst.uses, (long long) cst.tok_uses, (long long) cst.repeats, (long long) cst.distinct,
                     (long long) im.pc_done, (long long) im.pc_hit, (long long) im.pc_evicted, (long long) im.pc_busy_skip,
                     (long long) im.pc_bad);
    // per card: its jobs; the rest: the CPU pool
    std::vector<const Ent*> cpu_list;
    const size_t NC = im.cards.size();
    std::vector<int> nj(NC, 0), mnt(NC, 0);
    // STRATA_XE_ARM: this call takes the armed sequences only if they were armed for it (layer, width, X, the device
    // output path); anything else releases them with null jobs first
    const bool take_armed = im.armed_l == l && im.armed_nt == nt && out_dev && im.dense_q && x == im.arm_x;
    if (!take_armed) disarm(im);
    if (x && im.x_direct) im.arm_x = x;   // the X the next arm's sequences read
    static double pw_us = 0, pl_us = 0;
    static int64_t pw_n = 0;
    const double tw0 = now_ms();
    for (auto& cp : im.cards) {
        cp->busy = false;
        if (im.dense_q && !cp->armed) cp->stage_ev.wait();   // a back-to-back call (no dense sync between): its stage copy is done
    }
    const double tw1 = now_ms();
    for (const Ent& en : ents) {
        const size_t key = (size_t) im.key(l, en.e);
        const int c = im.card_of[key];
        if (c < 0 && cpu_stats) {
            if (cst.seen.size() < im.card_of.size()) cst.seen.resize(im.card_of.size(), 0);
            ++cst.uses;
            cst.tok_uses += en.nt;
            if (cst.seen[key]) ++cst.repeats;
            else { cst.seen[key] = 1; ++cst.distinct; }
        }
        if (c < 0) {
            if (im.pc_ready) pc_want(im, l, key);
            cpu_list.push_back(&en);
            continue;
        }
        static const bool skip_dense_dbg = env_i("STRATA_XE_DEBUG_SKIP_DENSE", 0) != 0;   // DEBUG: timing bound only (wrong output)
        if (skip_dense_dbg && c == im.dense) continue;   // the dense card's own expert jobs left out
        if (im.pc_ready && im.pc_where[key] >= 0) {   // a promoted expert: its slot used now
            im.cards[(size_t) (im.pc_where[key] >> 16)]->pc_tick[(size_t) (im.pc_where[key] & 0xFFFF)] = im.pc_clock;
            ++im.pc_hit;
        }
        XeCard& cd = *im.cards[(size_t) c];
        if (nj[(size_t) c] >= (cd.armed ? cd.arm_ng : kMaxDecodeJobs)) { cpu_list.push_back(&en); continue; }
        XeJob j = make_job(im, l, en.e, im.dptr[key]);
        j.nt = (uint8_t) en.nt;
        for (int s = 0; s < en.nt; ++s) { j.tok[s] = en.tok[s]; j.w[s] = en.w[s]; }
        std::memcpy(cd.stage_host + (size_t) nj[(size_t) c] * sizeof(XeJob), &j, sizeof(XeJob));
        ++nj[(size_t) c];
        mnt[(size_t) c] = std::max(mnt[(size_t) c], en.nt);
        im.st.hits += 1;
    }
    pw_us += (tw1 - tw0) * 1000.0;
    pl_us += (now_ms() - tw1) * 1000.0;
    if (++pw_n % 15000 == 0 && env_i("STRATA_XE_PROF", 0))
        std::fprintf(stderr, "xe decode prep split (%lld calls): unarmed cards' stage waits %.1f us, routing loop %.1f us a call\n",
                     (long long) pw_n, pw_us / pw_n, pl_us / pw_n);
    const size_t xoff = (kMaxDecodeJobs * sizeof(XeJob) + kStageAlign - 1) / kStageAlign * kStageAlign;
    const size_t n_out = (size_t) nt * H;
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    static const bool sprof = env_i("STRATA_XE_PROF", 0) != 0;
    static double spa[2][5] = {};   // [decode, verify]
    static int64_t spa_n[2] = {0, 0}, spa_jobs[2] = {0, 0};
    const int spk = nt > 1;
    double* sp = spa[spk];
    double ts0 = sprof ? now_ms() : 0;
    if (out_dev && dc) {
        // DEVICE OUTPUT (every decode / verify step): each card's part goes straight into the dense card's slot
        // (card-to-card copy), the CPU pool's part into the last slot, and one sum kernel on the dense card - which
        // waits for the other cards' copies through their events, so the host waits once per layer, not per card.
        float* slots = dc->dslots;   // [NC + 1][8][H] on the dense card
        const bool cpu_bg = im.cpu_async && !cpu_list.empty();
        if (cpu_bg) {   // the CPU part starts now, under the cards' submissions
            im.hsum_ev.wait();   // the previous call's copy of hsum has been made
            cpu_post(im, Ds41XeImpl::CpuTask{l, &cpu_list, x, im.hsum, n_out});
        }
        // direct X: the call's rows are pinned host memory (the dense half's readback mirror) - no staging copy
        static const float* xd_last = nullptr;
        static bool xd_ok = false;
        if (x != xd_last) { xd_last = x; xd_ok = sycl::get_pointer_type(x, im.ctx) == sycl::usm::alloc::host; }
        const bool xd = im.x_direct && xd_ok;
        std::vector<sycl::event> deps;
        std::vector<int> used;
        auto dispatch = [&](XeCard& c) -> sycl::event {   // one card's share: thread-safe across cards
            const int n = nj[(size_t) c.idx];
            c.busy = true;
            if (!xd) std::memcpy(c.stage_host + xoff, x, n_out * 4);
            // replayed graph: decode as is; a verify call's jobs padded with null jobs (nt 0: the kernels skip them)
            // to a bucket, so a few recordings per card cover every pass
            const bool graph = im.dense_q && im.planar && im.tier_graphs && (nt == 1 || im.tier_graphs_multi);
            int ng = n;
            if (graph && nt > 1) {
                static const int kb[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, kMaxDecodeJobs};
                for (int b : kb) if (b >= n) { ng = b; break; }
                XeJob z;
                std::memcpy(&z, c.stage_host, sizeof(XeJob));   // job 0's blob and types: valid, never read
                z.nt = 0;
                for (int k = n; k < ng; ++k) std::memcpy(c.stage_host + (size_t) k * sizeof(XeJob), &z, sizeof(XeJob));
            }
            float* dst = c.dense ? slots + (size_t) c.idx * kMaxJobTok * H : im.host_slots ? im.hslots + (size_t) c.idx * kMaxJobTok * H : c.OUT;
            const int mv = mnt[(size_t) c.idx] <= 1 ? 1 : mnt[(size_t) c.idx] <= 2 ? 2 : mnt[(size_t) c.idx] <= 4 ? 4 : 8;
            // the whole sequence; its first command waits for `prev` (the previous call's sum reads the slots this
            // call's cards write) unless it is being recorded (a replay carries that dependency instead)
            auto enqueue = [&](const sycl::event* prev) {
                const size_t cpb = xd ? (size_t) ng * sizeof(XeJob) : xoff + n_out * 4;   // the jobs (and X unless direct)
                if (prev) {
                    c.q.submit([&](sycl::handler& h) {
                        h.depends_on(*prev);
                        h.memcpy(c.stage_dev, c.stage_host, cpb);
                    });
                } else {
                    c.q.memcpy(c.stage_dev, c.stage_host, cpb);
                }
                sycl::event ev = card_experts(im, c, ng, nt, mv, xoff, dst, xd ? x : nullptr);
                if (!c.dense && !im.host_slots) ev = c.q.memcpy(slots + (size_t) c.idx * kMaxJobTok * H, c.OUT, n_out * 4);
                return ev;
            };
            sycl::event ev;
            if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 10);
            if (graph) {
                namespace sx = sycl::ext::oneapi::experimental;
                const uint32_t key = (uint32_t) ng | ((uint32_t) nt << 8) | ((uint32_t) mv << 12);
                auto it = c.dgraphs.find(key);
                if (it != c.dgraphs.end() && c.dgraph_x[key] != (xd ? x : nullptr)) { c.dgraphs.erase(it); it = c.dgraphs.end(); }
                if (it == c.dgraphs.end()) {
                    c.dgraph_x[key] = xd ? x : nullptr;   // the X a recording reads (direct X: baked into its quant kernel)
                    sx::command_graph<sx::graph_state::modifiable> g(c.q.get_context(), c.q.get_device());
                    g.begin_recording(c.q);
                    enqueue(nullptr);
                    g.end_recording();
                    it = c.dgraphs.emplace(key, std::make_unique<sx::command_graph<sx::graph_state::executable>>(g.finalize())).first;
                }
                const sycl::event prev = im.last_sum;
                sx::command_graph<sx::graph_state::executable>& eg = *it->second;
                ev = c.q.submit([&](sycl::handler& h) {
                    h.depends_on(prev);
                    h.ext_oneapi_graph(eg);
                });
                c.stage_ev = ev;   // the stage copy is inside: the whole replay must be done before stage_host is refilled
            } else if (im.dense_q) {
                const sycl::event prev = im.last_sum;
                ev = enqueue(&prev);
                c.stage_ev = ev;
            } else {
                ev = enqueue(nullptr);
            }
            if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 11);
            return ev;
        };
        // the armed cards: their jobs (null-padded to the armed count) are in stage_host - release them; their
        // sequences are in every sum's dependencies (an unused one must still be done before stage_host is refilled)
        for (auto& cp : im.cards) {
            XeCard& c = *cp;
            if (!c.armed) continue;
            for (int k = nj[(size_t) c.idx]; k < c.arm_ng; ++k)
                std::memcpy(c.stage_host + (size_t) k * sizeof(XeJob), &c.arm_null, sizeof(XeJob));
            arm_release(im, c);
            c.armed = false;
            c.stage_ev = c.arm_ev;
            deps.push_back(c.arm_ev);
            if (nj[(size_t) c.idx] > 0) { used.push_back(c.idx); c.busy = true; nj[(size_t) c.idx] = 0; ++im.arm_used; }
        }
        im.armed_l = -1;
        std::vector<XeCard*> todo;
        for (auto& cp : im.cards) if (nj[(size_t) cp->idx] > 0) todo.push_back(cp.get());
        // one card after another: submitting from a thread per card (2026-10-20) faulted card 0's copy engine once
        // ("Engine memory CAT error class=bcs") - not worth ~2 %
        std::vector<sycl::event> evs(todo.size());
        for (size_t k = 0; k < todo.size(); ++k) evs[k] = dispatch(*todo[k]);
        for (size_t k = 0; k < todo.size(); ++k) { deps.push_back(evs[k]); used.push_back(todo[k]->idx); }
        double ts1 = sprof ? now_ms() : 0;
        sycl::queue& qd = im.dense_q ? *im.dense_q : dc->q;
        if (!cpu_list.empty()) {
            float* hs = im.hsum;
            if (cpu_bg) {
                cpu_join(im);
            } else {
                im.hsum_ev.wait();   // the previous call's copy of hsum has been made
                std::memset(hs, 0, n_out * 4);
                cpu_ents(im, l, cpu_list, x, hs);
            }
            if (!im.host_slots) {
                im.hsum_ev = qd.memcpy(slots + NC * kMaxJobTok * H, hs, n_out * 4);
                deps.push_back(im.hsum_ev);
            }
            used.push_back((int) NC);
        }
        double ts2 = sprof ? now_ms() : 0;
        float* o = (float*) out_dev;
        if (used.empty()) {
            sycl::event ev = qd.memset(o, 0, n_out * 4);
            if (im.dense_q) im.last_sum = ev; else ev.wait();
        } else {
            struct Src { const float* p[8]; } src{};
            const int ns = std::min((int) used.size(), 8);
            const size_t stride = (size_t) kMaxJobTok * H;
            for (int k = 0; k < ns; ++k) {   // each part where it was written: the dense card's slots, host slots, hsum
                const int u = used[(size_t) k];
                const bool on_dense = (u < (int) NC && im.cards[(size_t) u]->dense) || !im.host_slots;
                src.p[k] = on_dense ? slots + (size_t) u * stride : u == (int) NC ? im.hsum : im.hslots + (size_t) u * stride;
            }
            if (Ds41XeTier::tl_on() && nt == 1 && im.dense_q) tl_stamp(im, 0, qd, 15);   // the dense queue is free
            if (Ds41XeTier::tl_on() && im.dense_q) {   // stamp 16: the sum's dependencies are met
                qd.submit([&](sycl::handler& h) { h.depends_on(deps); h.single_task([] {}); });
                tl_stamp(im, 0, qd, 16);
            }
            sycl::event ev = qd.submit([&](sycl::handler& h) {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(n_out), [=](sycl::id<1> i) {
                    float sum = 0.0f;
                    for (int k = 0; k < ns; ++k) sum += src.p[k][i];
                    o[i] = sum;
                });
            });
            if (im.dense_q) im.last_sum = ev; else ev.wait();
            if (im.host_slots && !cpu_list.empty()) im.hsum_ev = ev;   // hsum is read in place: its next write waits for this
        }
        if (Ds41XeTier::tl_on() && im.dense_q) tl_stamp(im, 0, *im.dense_q, nt == 1 ? 3 : 11);
        if (tl_h0 > 0) im.tl_hmark.emplace_back(tl_h0, tl_now_us());
        if (im.pc_mode > 0 && !im.pc_jobs.empty()) {   // promotions: behind this call's card work
            const double tq = now_ms();
            pc_poll(im, true);
            im.pc_push_us += (now_ms() - tq) * 1000.0;
        }
        if (sprof) {
            const double ts3 = now_ms();
            sp[0] += ts0 - t0;      // grouping + job building (before the cards' submissions)
            sp[1] += ts1 - ts0;     // the cards' submissions
            sp[2] += ts2 - ts1;     // CPU-pool experts
            sp[3] += ts3 - ts2;     // sum submission + wait for everything
            int64_t& sp_n = spa_n[spk];
            int64_t& sp_jobs = spa_jobs[spk];
            for (int c = 0; c < (int) NC; ++c) sp_jobs += nj[(size_t) c];
            sp[4] += cpu_list.size();
            static int64_t cpu_hist[2][5] = {};   // calls with 0, 1, 2, 3, 4+ CPU-pool experts
            ++cpu_hist[spk][std::min<size_t>(cpu_list.size(), 4)];
            if (++sp_n % 1000 == 0) {
                std::fprintf(stderr, "xe %s prof (%lld calls, %.2f card jobs/call, %.2f cpu experts/call): per call prep %.1f us, "
                             "submit %.1f us, cpu %.1f us, sum+wait %.1f us = %.1f us\n", spk ? "verify" : "decode", (long long) sp_n,
                             (double) sp_jobs / sp_n, sp[4] / sp_n, 1e3 * sp[0] / sp_n, 1e3 * sp[1] / sp_n, 1e3 * sp[2] / sp_n,
                             1e3 * sp[3] / sp_n, 1e3 * (sp[0] + sp[1] + sp[2] + sp[3]) / sp_n);
                const int64_t* ch = cpu_hist[spk];
                std::fprintf(stderr, "xe %s cpu experts a call: 0: %lld, 1: %lld, 2: %lld, 3: %lld, 4+: %lld calls\n", spk ? "verify" : "decode",
                             (long long) ch[0], (long long) ch[1], (long long) ch[2], (long long) ch[3], (long long) ch[4]);
                if (im.pool && im.st.cpu > 0) {   // the CPU pool: its phases (kernels) vs the CPU part's wall time
                    const cpu::ExpertPool& pl = *im.pool;
                    double wp = 0, dr = 0, rp = 0;
                    pl.phase_ms(wp, dr, rp);
                    std::fprintf(stderr, "xe cpu pool: %lld experts, %.0f us an expert (cpu_ents wall); pool gate/up %.0f + quant %.0f + down "
                                 "%.0f us an expert, %.1f GB/s of expert bytes in the products; %d workers\n", (long long) im.st.cpu,
                                 1e3 * im.st.cpu_ms / im.st.cpu, 1e3 * pl.ms_multi_gu / im.st.cpu, 1e3 * pl.ms_multi_q / im.st.cpu,
                                 1e3 * pl.ms_multi_down / im.st.cpu,
                                 (double) pl.multi_bytes / 1e6 / std::max(1e-9, pl.ms_multi_gu + pl.ms_multi_down), pl.workers());
                }
            }
        }
        im.st.wall_ms += now_ms() - t0;
        return true;
    }
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        const int n = nj[(size_t) c.idx];
        if (n == 0) continue;
        c.busy = true;
        std::memcpy(c.stage_host + xoff, x, n_out * 4);
        c.q.memcpy(c.stage_dev, c.stage_host, xoff + n_out * 4);
        card_experts(im, c, n, nt, mnt[(size_t) c.idx], xoff, c.OUT);
        c.q.memcpy(c.part_host, c.OUT, n_out * 4);
    }
    float* hs = im.hsum;
    std::memset(hs, 0, n_out * 4);
    cpu_ents(im, l, cpu_list, x, hs);
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.busy) continue;
        c.q.wait();
        for (size_t i = 0; i < n_out; ++i) hs[i] += c.part_host[i];
    }
    if (out_dev) {   // no dense card known: through the host
        std::fprintf(stderr, "xe tier: device output without a dense card\n");
        return false;
    }
    std::memcpy(out_host, hs, n_out * 4);
    im.st.wall_ms += now_ms() - t0;
    return true;
}

}  // namespace

bool Ds41XeTier::run(int64_t l, const int32_t* ids6, const float* w6, const float* x, float* out) {
    try {
        return run_small(*im_, l, 1, ids6, w6, x, out, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xe tier: run: %s\n", e.what());
        return false;
    }
}
bool Ds41XeTier::run_to_dev(int64_t l, const int32_t* ids6, const float* w6, const float* x, void* out_dev) {
    try {
        return run_small(*im_, l, 1, ids6, w6, x, nullptr, out_dev);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xe tier: run_to_dev: %s\n", e.what());
        return false;
    }
}
bool Ds41XeTier::run_multi(int64_t l, int nt, const int32_t* ids, const float* w, const float* x, float* out) {
    for (int t0 = 0; t0 < nt; t0 += kMaxJobTok) {
        const int m = std::min(kMaxJobTok, nt - t0);
        const size_t K = (size_t) im_->g.top_k, H = (size_t) im_->g.n_embd;
        if (!run_small(*im_, l, m, ids + t0 * K, w + t0 * K, x + t0 * H, out + t0 * H, nullptr)) return false;
    }
    return true;
}
bool Ds41XeTier::run_multi_to_dev(int64_t l, int nt, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    for (int t0 = 0; t0 < nt; t0 += kMaxJobTok) {
        const int m = std::min(kMaxJobTok, nt - t0);
        const size_t K = (size_t) im_->g.top_k, H = (size_t) im_->g.n_embd;
        if (!run_small(*im_, l, m, ids + t0 * K, w + t0 * K, x + t0 * H, nullptr, (float*) out_dev + t0 * H)) return false;
    }
    return true;
}

// ================================ prompt chunks ================================

namespace {

// larry, STRATA_XE_ALLOC_LOG (patch-alloclog.py): a growth over 30 ms (or every one with =1) on stderr
void alloc_note(const char* what, size_t bytes, double t0, double t1) {
    static const bool all = env_i("STRATA_XE_ALLOC_LOG", 0) != 0;
    const double t2 = now_ms();
    if (all || t2 - t0 > 30.0)
        std::fprintf(stderr, "xe tier: alloc %s %.1f MiB: wait %.1f ms, free + alloc %.1f ms\n", what, bytes / 1048576.0, t1 - t0, t2 - t1);
}
template <class T>
bool grow_dev(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    const double t0 = now_ms();
    if (p) q.wait();
    const double t1 = now_ms();
    if (p) { addr_log("tier usm free", q, p, 0); sycl::free(p, q); }
    p = sycl::malloc_device<T>(need, q);
    cap = p ? need : 0;
    if (p) addr_log("tier usm alloc", q, p, need * sizeof(T));
    alloc_note("device", need * sizeof(T), t0, t1);
    return p != nullptr;
}
template <class T>
bool grow_host(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    const double t0 = now_ms();
    if (p) q.wait();
    const double t1 = now_ms();
    if (p) sycl::free(p, q);
    p = sycl::malloc_host<T>(need, q);
    cap = p ? need : 0;
    alloc_note("host (pinned)", need * sizeof(T), t0, t1);
    return p != nullptr;
}
/// larry: grow_dev in card-local memory (local_alloc / local_free): buffers no other card's queue touches
template <class T>
bool grow_loc(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    const double t0 = now_ms();
    if (p) q.wait();
    const double t1 = now_ms();
    if (p) local_free(q, p);
    p = reinterpret_cast<T*>(local_alloc(q, need * sizeof(T)));
    cap = p ? need : 0;
    alloc_note("card-local", need * sizeof(T), t0, t1);
    return p != nullptr;
}
/// larry: the card's stream buffer, card-local (local_alloc) and grown by half its size at least (to `cap`; whole 32 MiB)
bool grow_stream(XeCard& c, size_t need, size_t cap) {
    if (c.stream_bytes >= need) return true;
    const size_t step = (size_t) 32 << 20;
    size_t want = std::max(need, c.stream_bytes + c.stream_bytes / 2);
    want = std::min((want + step - 1) / step * step, std::max(need, cap));
    const double t0 = now_ms();
    if (c.stream) c.q.wait();
    const double t1 = now_ms();
    if (c.stream) { local_free(c.q, c.stream); c.stream = nullptr; c.stream_bytes = 0; }
    c.stream = card_alloc(c.q, want);   // larry, patch-barstage
    c.stream_bytes = c.stream ? want : 0;
    alloc_note("stream", want, t0, t1);
    return c.stream != nullptr;
}

/// The Level Zero driver (NEO 26.35) deadlocks when several host threads enqueue to different devices at once while
/// one of them copies from PAGEABLE host memory (seen 2026-10-19: three appendUSMMemcpy and one appendKernelLaunch
/// blocked on two driver mutexes).  So the card threads take this lock around every enqueue (never around a wait),
/// and pageable sources go through a pinned bounce buffer.
/// larry: a priority waiter (the pipelined batch's dense half: Ds41XeTier::submit_priority) makes the tier's threads
/// hold off their next acquisition while it waits.
// larry, STRATA_XE_LOCK_PROF (patch-lockprof.py): per call site (return address) takes, wait and hold times
bool lock_prof_on() { static const bool on = env_i("STRATA_XE_LOCK_PROF", 0) != 0; return on; }
struct LockSite { const void* site; int64_t n; double wait, hold, hold_max; };
std::vector<LockSite> g_lk_sites;            // touched only with g_submit held
thread_local double g_lk_t0 = 0;
thread_local const void* g_lk_site = nullptr;
LockSite& lock_site(const void* site) {
    for (auto& e : g_lk_sites) if (e.site == site) return e;
    g_lk_sites.push_back({site, 0, 0, 0, 0});
    return g_lk_sites.back();
}
struct PrioMutex {
    std::mutex m;
    std::atomic<int> hi{0};
    __attribute__((noinline)) void lock() {
        const bool lp = lock_prof_on();
        const double tw = lp ? now_ms() : 0;
        while (hi.load(std::memory_order_acquire) > 0) std::this_thread::yield();
        m.lock();
        if (lp) {
            g_lk_t0 = now_ms();
            g_lk_site = __builtin_return_address(0);
            LockSite& e = lock_site(g_lk_site);
            ++e.n;
            e.wait += g_lk_t0 - tw;
        }
    }
    void unlock() {
        if (lock_prof_on() && g_lk_site) {
            const double h = now_ms() - g_lk_t0;
            LockSite& e = lock_site(g_lk_site);
            e.hold += h;
            e.hold_max = std::max(e.hold_max, h);
            g_lk_site = nullptr;
        }
        m.unlock();
    }
};
PrioMutex g_submit;
/// the top sites by hold time (call with g_submit held), then zeroed
void lock_prof_dump(const char* what) {
    std::vector<LockSite> v = g_lk_sites;
    std::sort(v.begin(), v.end(), [](const LockSite& a, const LockSite& b) { return a.hold + a.wait > b.hold + b.wait; });
    std::fprintf(stderr, "xe lock prof (%s): site: takes, waited ms, held ms (max)\n", what);
    for (size_t i = 0; i < v.size() && i < 12; ++i) {
        Dl_info di;
        const char* nm = "?";
        uintptr_t off = 0;
        if (dladdr(v[i].site, &di)) { off = (uintptr_t) v[i].site - (uintptr_t) di.dli_fbase; nm = di.dli_fname ? di.dli_fname : "?"; }   // the offset in its object (addr2line -f)
        std::fprintf(stderr, "  %-60.60s +0x%-6lx %7lld  wait %8.1f  hold %8.1f (%6.2f)\n", nm, (unsigned long) off, (long long) v[i].n,
                     v[i].wait, v[i].hold, v[i].hold_max);
    }
    g_lk_sites.clear();
}

/// larry, STRATA_XE_ZCOPY (patch-zcopy.py): `bytes` (a multiple of 4) of PINNED host memory at `src` to `dst` by a kernel
/// on `q` - the compute engine reads the host memory (zero-copy); the card's one copy engine stays with the bulk DMA
bool zcopy_on() { static const bool on = env_i("STRATA_XE_ZCOPY", 0) != 0; return on; }
sycl::event zc_copy(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return {};
    if (bytes % 16 == 0 && (((uintptr_t) dst | (uintptr_t) src) % 16) == 0) {
        using v4 = sycl::vec<uint32_t, 4>;
        const v4* sv = (const v4*) src;
        v4* dv = (v4*) dst;
        return q.parallel_for(sycl::range<1>(bytes / 16), [=](sycl::id<1> i) { dv[i] = sv[i]; });
    }
    const uint32_t* sv = (const uint32_t*) src;
    uint32_t* dv = (uint32_t*) dst;
    return q.parallel_for(sycl::range<1>(bytes / 4), [=](sycl::id<1> i) { dv[i] = sv[i]; });
}
/// larry: a small upload from pinned host memory - a kernel (STRATA_XE_ZCOPY) or the queue's memcpy
sycl::event up_pinned(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    return zcopy_on() ? zc_copy(q, dst, src, bytes) : q.memcpy(dst, src, bytes);
}

/// STRATA_XE_HOTRING: `bytes` of host memory at `src` to `dst` on queue `q` in ring chunks: a slot whose last copy is
/// still running is waited for (outside the submit lock), then filled and its DMA enqueued at once.  `pinned`: the
/// source is DMA-able as it is (the chunks go straight, in the same order).  Returns the last copy's event.
// larry, STRATA_XE_HOT_PROF: hot_put's parts in us, summed over threads (slot wait, copy, lock wait, submit), bytes
std::atomic<int64_t> g_hp_wait{0}, g_hp_copy{0}, g_hp_lock{0}, g_hp_sub{0}, g_hp_bytes{0}, g_hp_pin{0};
bool hot_prof() { static const bool on = env_i("STRATA_XE_HOT_PROF", 0) != 0; return on; }
sycl::event hot_put(sycl::queue& q, uint8_t* dst, const uint8_t* src, size_t bytes, bool pinned, uint8_t* ring,
                    std::vector<sycl::event>& ev, int& next) {
    const size_t C = hot_chunk();
    const bool hp = hot_prof();
    auto us = [] { return (int64_t) (now_ms() * 1000.0); };
    const int R = hot_slots();
    if ((int) ev.size() != R) { ev.assign((size_t) R, sycl::event()); next = 0; }
    sycl::event last;
    for (size_t off = 0; off < bytes; off += C) {
        const size_t nb = std::min(C, bytes - off);
        if (pinned) {
            // larry, STRATA_XE_HOT_PINWAIT (patch-hotpinwait.py): the chunk takes a slot (its previous DMA done first)
            static const bool pin_wait = env_i("STRATA_XE_HOT_PINWAIT", 0) != 0;
            int pk = -1;
            if (pin_wait) {
                pk = next;
                next = (pk + 1) % R;
                const int64_t tw = hp ? us() : 0;
                ev[(size_t) pk].wait();
                if (hp) g_hp_wait += us() - tw;
            }
            const int64_t t0 = hp ? us() : 0;
            std::lock_guard<PrioMutex> lk(g_submit);
            const int64_t t1 = hp ? us() : 0;
            last = q.memcpy(dst + off, src + off, nb);
            if (pk >= 0) ev[(size_t) pk] = last;
            if (hp) { g_hp_lock += t1 - t0; g_hp_sub += us() - t1; g_hp_pin += (int64_t) nb; }
            continue;
        }
        const int k = next;
        next = (k + 1) % R;
        // larry, STRATA_XE_HOTRING_SPIN: poll first (a blocking wait may sleep and wake late, every chunk)
        static const int hot_spin = env_i("STRATA_XE_HOTRING_SPIN", 0);
        if (hot_spin > 0)
            for (int it = 0; it < (1 << 20) && ev[(size_t) k].get_info<sycl::info::event::command_execution_status>() !=
                                                   sycl::info::event_command_status::complete; ++it)
                for (int sp = 0; sp < hot_spin; ++sp) _mm_pause();
        const int64_t t0 = hp ? us() : 0;
        ev[(size_t) k].wait();   // a default-constructed event is complete
        const int64_t t1 = hp ? us() : 0;
        uint8_t* slot = ring + (size_t) k * C;
        hot_copy(slot, src + off, nb);
        const int64_t t2 = hp ? us() : 0;
        std::lock_guard<PrioMutex> lk(g_submit);
        const int64_t t3 = hp ? us() : 0;
        last = ev[(size_t) k] = q.memcpy(dst + off, slot, nb);
        if (hp) { g_hp_wait += t1 - t0; g_hp_copy += t2 - t1; g_hp_lock += t3 - t2; g_hp_sub += us() - t3; g_hp_bytes += (int64_t) nb; }
    }
    return last;
}

/// STRATA_XE_HOTRING_CHECK: an expert's bytes on its card (queue q, after its copies) against its host source
void hot_check(Ds41XeImpl& im, XeCard& c, sycl::queue& q, const uint8_t* dev, const uint8_t* src, size_t bytes,
               int64_t l, int64_t e, const char* what) {
    if (!c.hchk) {
        std::lock_guard<PrioMutex> lk(g_submit);
        c.hchk = sycl::malloc_host<uint8_t>((size_t) im.blob_max, im.ctx);
    }
    if (!c.hchk || bytes > (size_t) im.blob_max) return;
    sycl::event ev;
    {
        std::lock_guard<PrioMutex> lk(g_submit);
        ev = q.memcpy(c.hchk, dev, bytes);
    }
    ev.wait();
    const int64_t n = ++im.hc_n;
    if (std::memcmp(c.hchk, src, bytes) != 0) {
        size_t d = 0;
        for (size_t i = 0; i < bytes; ++i) d += c.hchk[i] != src[i];
        const int64_t bad = ++im.hc_bad;
        if (bad <= 20)
            std::fprintf(stderr, "xe tier: HOTRING_CHECK %s: layer %lld expert %lld on card %d: %zu of %zu bytes differ\n", what,
                         (long long) l, (long long) e, c.idx, d, bytes);
    }
    if (n % 20000 == 0)
        std::fprintf(stderr, "xe tier: HOTRING_CHECK: %lld experts compared, %lld differ\n", (long long) n, (long long) im.hc_bad.load());
}

/// STRATA_XE_PROF=1: run_big's phases (wall ms of the whole call / per-card busy ms summed over the cards), printed
/// every 40 calls.
struct BigProf {
    double prep = 0, cards = 0, combine = 0, total = 0;
    double prep_pf = 0, prep_plan = 0, prep_lock = 0;   ///< prep's prefetch waits, planning, submit-lock wait
    double prep_grow = 0, prep_x = 0, prep_w = 0, prep_z = 0;   ///< prep's buffer checks, X / W copy and memset enqueues
    std::atomic<int64_t> lk_gemm_us{0}, lk_jobs_us{0}, lk_n{0};   ///< submit-lock hold time of the GEMM / job sites
    std::atomic<int64_t> st_us{0}, st_bytes{0}, pfw_us{0}, dma_us{0}, dma_wait_us{0}, dma_bytes{0}, dma_n{0};   ///< prefetch
    std::atomic<int64_t> min_free[8] = {{-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}};   ///< per card, bytes
    std::atomic<int64_t> stream_us{0}, gemm_us{0}, small_us{0};
    std::atomic<int64_t> streamed{0}, gemm_e{0}, small_j{0}, kept{0};
    int64_t calls = 0;
};
BigProf g_bp;
bool xe_prof() { static const bool on = env_i("STRATA_XE_PROF", 0) != 0; return on; }

/// Prompt prefetch: plans layer `l` into slot `s` - every arena expert of the layer (pinned ones copied as they are,
/// pageable ones staged first, while the stage has room) onto the expert card with the fewest planned bytes that has
/// room in its buffer - and asks the stager thread for the pageable copies.
void pf_plan(Ds41XeImpl& im, int s, int64_t l) {
    Ds41XeImpl::PfSlot& ps = im.pf[s];
    const int64_t E = im.g.n_experts;
    ps.layer = l;
    ps.enqueued = false;
    ps.rec = false;   // larry, STRATA_XE_PF_NOBLOCK
    ps.timed = false;
    ps.t_plan = now_ms();   // larry, STRATA_XE_PF_TRACE
    ps.t_staged = 0;
    ps.traced = false;
    ps.card.assign((size_t) E, -1);
    ps.off.assign((size_t) E, 0);
    ps.soff.assign((size_t) E, SIZE_MAX);
    ps.n = 0;
    ps.bytes = 0;
    std::vector<size_t> used(im.cards.size(), 0);
    size_t sused = 0;
    for (int64_t e = 0; e < E; ++e) {
        const size_t key = (size_t) im.key(l, e);
        if (im.card_of[key] >= 0 || !im.aptr[key]) continue;
        const size_t b = (im.bbytes(l, e) + 255) & ~(size_t) 255;
        const bool pin = im.apin[key] != 0;
        if (!pin && sused + b > im.pf_stage_cap) continue;
        int best = -1;
        for (int pass = 0; pass < 2 && best < 0; ++pass)   // the expert cards first; the dense card takes the overflow
            for (auto& cp : im.cards) {
                const XeCard& c = *cp;
                if (c.dense != (pass == 1) || !c.pbuf[s] || used[(size_t) c.idx] + b > c.pbuf_cap) continue;
                if (best < 0 || used[(size_t) c.idx] < used[(size_t) best]) best = c.idx;
            }
        if (best < 0) continue;
        ps.card[(size_t) e] = (int8_t) best;
        ps.off[(size_t) e] = used[(size_t) best];
        used[(size_t) best] += b;
        if (!pin) { ps.soff[(size_t) e] = sused; sused += b; }
        ++ps.n;
        ps.bytes += (int64_t) b;
    }
    im.pf_planned += ps.n;
    im.pf_bytes += ps.bytes;
    std::lock_guard<std::mutex> lk(im.pf_mu);
    ps.next = 0;
    ps.active = 0;
    ps.staged = (hot_on() && !im.pf_rings.empty() ? ps.n : (int64_t) sused) > 0 ? 1 : 0;
    if (ps.staged) im.pf_cv.notify_all();
}

/// A stager thread (STRATA_XE_STAGERS of them): copies planned slots' pageable experts into their pinned stage, an
/// expert at a time, the older request (lower layer) first; whoever finishes a slot's last expert marks it staged.
/// No SYCL calls here: the Level Zero driver must not see another submitting thread.
void pf_stager(Ds41XeImpl& im, int idx) {
    const int64_t E = im.g.n_experts;
    // larry, STRATA_XE_HOTRING: this stager's ring - it puts each planned expert of a slot on its card itself
    const bool hot = hot_on() && idx < (int) im.pf_rings.size() && im.pf_rings[(size_t) idx];
    std::vector<sycl::event> hev;
    int hnext = 0;
    auto has_work = [&](int s) { return im.pf[s].staged == 1 && im.pf[s].next < E; };
    for (;;) {
        int s = -1;
        int64_t e = -1;
        {
            std::unique_lock<std::mutex> lk(im.pf_mu);
            im.pf_cv.wait(lk, [&] { return im.pf_stop || has_work(0) || has_work(1); });
            if (im.pf_stop) return;
            if (has_work(0) && has_work(1)) s = im.pf[0].layer <= im.pf[1].layer ? 0 : 1;
            else s = has_work(0) ? 0 : 1;
            Ds41XeImpl::PfSlot& ps = im.pf[s];
            while (ps.next < E && !(ps.card[(size_t) ps.next] >= 0 && (hot || ps.soff[(size_t) ps.next] != SIZE_MAX))) ++ps.next;
            if (ps.next < E) { e = ps.next++; ++ps.active; }
            if (e < 0 && ps.active == 0) { ps.staged = 0; im.pf_cv.notify_all(); continue; }
            if (e < 0) continue;
        }
        Ds41XeImpl::PfSlot& ps = im.pf[s];
        const int64_t l = ps.layer;
        const double ts = xe_prof() ? now_ms() : 0;
        uint8_t* bc = nullptr;   // larry, STRATA_XE_BAR_STAGE: a pageable expert straight into the slot through the BAR
        if (hot && ps.soff[(size_t) e] != SIZE_MAX) bc = bar_cpu(im.cards[(size_t) ps.card[(size_t) e]]->pbuf[s] + ps.off[(size_t) e]);
        if (bc) {
            bar_put(bc, im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));
        } else if (hot) {
            XeCard& cd = *im.cards[(size_t) ps.card[(size_t) e]];
            hot_put(cd.pq, cd.pbuf[s] + ps.off[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e),
                    ps.soff[(size_t) e] == SIZE_MAX, im.pf_rings[(size_t) idx], hev, hnext);
        } else {
            std::memcpy(ps.stage + ps.soff[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));
        }
        if (xe_prof()) { g_bp.st_us += (int64_t) ((now_ms() - ts) * 1000.0); g_bp.st_bytes += (int64_t) im.bbytes(l, e); }
        // larry, STRATA_XE_PF_NOBLOCK (patch-pfnb.py): the stager that queued the slot's last copy records its barriers
        static const int pf_nb = env_i("STRATA_XE_PF_NOBLOCK", 0);
        bool last = false;
        {
            std::lock_guard<std::mutex> lk(im.pf_mu);
            --ps.active;
            last = ps.active == 0 && ps.next >= E;
            if (last && !(hot && pf_nb)) { ps.staged = 0; ps.t_staged = now_ms(); im.pf_cv.notify_all(); }
        }
        if (last && hot && pf_nb) {
            const double tr = now_ms();
            {
                std::lock_guard<PrioMutex> lk(g_submit);
                for (auto& cp : im.cards)
                    if (cp->pbuf_cap && cp->pbuf[s]) {
                        cp->pev[s] = cp->pq.ext_oneapi_submit_barrier();
                        ptl_stamp(im.ctx, 9 + cp->idx, cp->pq, 30, (int) l, cp->idx, tr);   // larry, STRATA_GPU_PTL
                    }
            }
            std::lock_guard<std::mutex> lk(im.pf_mu);
            ps.rec = true;
            ps.staged = 0;
            ps.t_staged = now_ms();
            im.pf_cv.notify_all();
        }
    }
}

/// Puts slot `s`'s copies on the cards' copy queues: waits for its staging when `block`, else returns if it is not
/// staged yet.
void pf_enqueue(Ds41XeImpl& im, int s, bool block) {
    Ds41XeImpl::PfSlot& ps = im.pf[s];
    if (ps.layer < 0 || ps.enqueued) return;
    {
        const double tw = xe_prof() ? now_ms() : 0;
        std::unique_lock<std::mutex> lk(im.pf_mu);
        if (ps.staged && !block) return;
        im.pf_cv.wait(lk, [&] { return ps.staged == 0; });
        if (xe_prof()) g_bp.pfw_us += (int64_t) ((now_ms() - tw) * 1000.0);
    }
    ps.t_enq = now_ms();
    const int64_t l = ps.layer;
    if (hot_on() && !im.pf_rings.empty()) {   // STRATA_XE_HOTRING: the stagers put every copy on the queues already
        std::lock_guard<PrioMutex> lk(g_submit);
        for (auto& cp : im.cards)
            if (!ps.rec && cp->pbuf_cap && cp->pbuf[s]) {   // larry, PF_NOBLOCK: rec = its last stager recorded them
                cp->pev[s] = cp->pq.ext_oneapi_submit_barrier();
                ptl_stamp(im.ctx, 9 + cp->idx, cp->pq, 30, (int) l, cp->idx, ps.t_enq);   // larry, STRATA_GPU_PTL
            }
        ps.enqueued = true;
        return;
    }
    std::unique_lock<PrioMutex> lk(g_submit, std::defer_lock);
    int held = 0;
    for (int64_t e = 0; e < im.g.n_experts; ++e) {
        const int c = ps.card[(size_t) e];
        if (c < 0) continue;
        if (!lk.owns_lock()) lk.lock();
        XeCard& cd = *im.cards[(size_t) c];
        const uint8_t* src = ps.soff[(size_t) e] != SIZE_MAX ? ps.stage + ps.soff[(size_t) e] : im.aptr[(size_t) im.key(l, e)];
        cd.pev[s] = cd.pq.memcpy(cd.pbuf[s] + ps.off[(size_t) e], src, im.bbytes(l, e));
        if (++held == 16) { lk.unlock(); held = 0; }   // the other threads' enqueues get in between
    }
    ps.enqueued = true;
}

/// n tokens of one layer, expert-major.  out_host (n x n_embd) or out_dev (the dense card).
///
/// Per expert the call routes to: an expert with >= STRATA_XE_GEMM_MIN (8) tokens runs as two fp16 GEMMs on oneMKL
/// (gather its tokens, dequantise it once, gate|up GEMM, SwiGLU, down GEMM: XMX, each weight read once); a smaller one
/// as mat-vec jobs of <= 8 tokens.  Experts a card does not hold (arena / file tier) are streamed onto the card with
/// the fewest streamed bytes, in rounds that fit its stream buffer.  Every card runs its part on its own host thread;
/// each writes its experts' rows of its Y, a deterministic gather sums them into the card's part, and the parts are
/// copied card-to-card (P2P) onto the dense card and added there.
bool run_big(Ds41XeImpl& im, int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out_host,
             void* out_dev, bool last_call = true, const float* x_dev = nullptr) {
    namespace mkl = oneapi::mkl;
    const int64_t H = im.g.n_embd, F = im.g.n_ff;
    const int K = (int) im.g.top_k;
    const double t0 = now_ms();
    const size_t NC = im.cards.size();
    const int64_t E = im.g.n_experts;
    const int gemm_min = env_i("STRATA_XE_GEMM_MIN", 8);
    const bool prof = xe_prof();
    // larry, STRATA_XE_DENSE_LITE / STRATA_XE_GATHER_INIT: see patch-denselite.py (the dense card's part of a pipelined
    // call: no streamed experts, X read in place, no host waits; the first gather writes the part)
    static const int dense_lite = env_i("STRATA_XE_DENSE_LITE", 0);
    static const int gather_init = env_i("STRATA_XE_GATHER_INIT", 0);
    const bool lite = dense_lite && x_dev && out_dev && im.dense >= 0 && NC > 1;
    // larry, STRATA_XE_DENSE_LITE_MIN: the streamed experts kept off the dense card only in calls this long (see
    // patch-litemin.py); shorter calls stream on every card as before
    static const int lite_min = env_i("STRATA_XE_DENSE_LITE_MIN", 0);   // larry, patch-litemin0 (was 2048)
    const bool lite_nostream = lite && n >= lite_min;
    // larry, STRATA_XE_DENSE_OFFLOAD: the dense card's cached experts to the expert cards (see patch-denseoff.py)
    static const int dense_offload = env_i("STRATA_XE_DENSE_OFFLOAD", 0);
    const bool offl_on = lite && dense_offload > 0 && n >= dense_offload && !im.apin.empty();
    auto offl = [&](int64_t e) -> bool {
        if (!offl_on) return false;
        const size_t key = (size_t) im.key(l, e);
        return im.card_of[key] == (int8_t) im.dense && im.aptr[key] && im.apin[key];
    };
    // larry, STRATA_XE_DENSE_INLINE: the dense card's part on the dense half's queue for this call (see patch-denseinline.py)
    static const int dense_inline = env_i("STRATA_XE_DENSE_INLINE", 0);
    struct QSwap {
        XeCard* c = nullptr;
        std::optional<sycl::queue> saved;
        ~QSwap() { if (c && saved) c->q = *saved; }
    } qswap;
    // larry, STRATA_XE_CPU_LIGHT=T (0 = off, at most 8): an arena expert this call would stream (neither prefetched
    // nor kept in a stream buffer) that serves <= T tokens runs on the CPU pool beside the cards instead.  A short
    // prompt's expert phase is bound by host memory: a pageable expert streamed is read and written by the bounce copy
    // and read again by the DMA; the pool reads it once.  The pool needs the call's X on the host: a pipelined batch
    // (> 512 tokens) passes x_dev and leaves the host copy stale - read back once a call (n x 20 KB on the dense card's
    // prefetch queue, idle there).  Prompt calls only run here: the pool is idle.  STRATA_XE_CPU_LIGHT_PINNED=0:
    // pinned arena experts (the DMA's one read) stream as before.  STRATA_XE_CPU_LIGHT_MAX_N: calls of more tokens
    // stream everything (default 4096).
    static const int light_t = std::max(0, std::min(env_i("STRATA_XE_CPU_LIGHT", 0), kMaxJobTok));
    static const int light_pin = env_i("STRATA_XE_CPU_LIGHT_PINNED", 1);
    static const int light_max_n = env_i("STRATA_XE_CPU_LIGHT_MAX_N", 4096);
    const bool light_on = light_t > 0 && n <= light_max_n && (x_dev || x) && im.pool && (!x_dev || im.dense >= 0);
    std::vector<int32_t> light;   // this call's CPU-pool experts
    // per expert: its (token, slot) entries in routing order
    std::vector<std::vector<int32_t>> by_e((size_t) E);   // slot index t*K + k
    for (int t = 0; t < n; ++t)
        for (int k = 0; k < K; ++k) {
            const int32_t e = ids[(size_t) t * K + k];
            if (e >= 0 && e < E) by_e[(size_t) e].push_back(t * K + k);
        }
    // prompt prefetch: this layer's slot (its copies enqueued now if they were not yet), and the next layer's copies
    // if they are staged already (else at the end of this call)
    Ds41XeImpl::PfSlot* ps = nullptr;
    int pslot = -1;
    if (im.pf_on) {
        for (int s2 = 0; s2 < 2; ++s2) if (im.pf[s2].layer == l) { pslot = s2; ps = &im.pf[s2]; }
        // larry, STRATA_XE_PF_TRACE: the slot's timeline at its first call (the host waits for its DMAs here)
        static const bool pf_trace = env_i("STRATA_XE_PF_TRACE", 0) != 0;
        const double t_call = now_ms();
        const double t_st0 = ps ? ps->t_staged : 0;
        if (ps) pf_enqueue(im, pslot, true);
        if (ps && pf_trace && !ps->traced) {
            ps->traced = true;
            const double t_enq_done = now_ms();
            for (auto& cp : im.cards) if (cp->pbuf_cap && cp->pbuf[pslot]) cp->pev[pslot].wait();
            const double t_dma = now_ms();
            const double tst = ps->t_staged > 0 ? ps->t_staged : t_enq_done;
            std::fprintf(stderr, "pf trace: layer %lld (call n %d): planned %.0f ms before, staged %s%.0f ms, enqueue wait %.0f ms, "
                         "DMAs done +%.0f ms; %.2f GB, %lld experts\n", (long long) l, n, t_call - ps->t_plan,
                         t_st0 > 0 ? "-" : "+", t_st0 > 0 ? t_call - t_st0 : tst - t_call, t_enq_done - t_call, t_dma - t_call,
                         ps->bytes / 1e9, (long long) ps->n);
        }
        if (ps && prof && !ps->timed) {   // the layer's copies: done how long after their enqueue (this waits for them)
            ps->timed = true;
            const double tw = now_ms();
            for (auto& cp : im.cards) if (!cp->dense && cp->pbuf_cap) cp->pev[pslot].wait();
            const double td = now_ms();
            g_bp.dma_wait_us += (int64_t) ((td - tw) * 1000.0);
            g_bp.dma_us += (int64_t) ((td - ps->t_enq) * 1000.0);
            g_bp.dma_bytes += ps->bytes;
            ++g_bp.dma_n;
        }
        if (ps && im.pf[pslot ^ 1].layer == l + 1) pf_enqueue(im, pslot ^ 1, false);
    }
    const double t_pf = now_ms();
    // which card computes each expert; streamed experts go to the card with the fewest streamed bytes (a prefetched
    // one is computed where it was prefetched to)
    std::vector<int64_t> sbytes(NC, 0);
    int64_t n_pf = 0;
    std::vector<std::vector<int32_t>> cl(NC);   // experts per card, resident first then streamed
    std::vector<std::vector<int32_t>> streamed(NC);
    // larry, STRATA_XE_KEEP_STREAM (default 1): overflow experts the layer's previous call (its other column half) left
    // in a card's stream buffer are computed there again, from where they are - no second copy
    static const int keep_on = env_i("STRATA_XE_KEEP_STREAM", 1);
    std::vector<int32_t> kept_card;   // expert -> the card holding it, -1
    std::vector<int64_t> kept_off;    // its offset in that card's stream buffer
    std::vector<std::vector<int32_t>> kept(NC);
    if (keep_on) {
        kept_card.assign((size_t) E, -1);
        kept_off.assign((size_t) E, -1);
        for (auto& cp : im.cards)
            if (cp->st_layer == l && cp->stream)
                for (const auto& pr : cp->st_have) { kept_card[(size_t) pr.first] = cp->idx; kept_off[(size_t) pr.first] = pr.second; }
    }
    for (int64_t e = 0; e < E; ++e) {
        if (by_e[(size_t) e].empty()) continue;
        const int c = im.card_of[(size_t) im.key(l, e)];
        if (c >= 0 && !offl(e)) cl[(size_t) c].push_back((int32_t) e);   // larry, STRATA_XE_DENSE_OFFLOAD: offloaded ones below
    }
    for (int64_t e = 0; e < E; ++e) {
        if (by_e[(size_t) e].empty() || (im.card_of[(size_t) im.key(l, e)] >= 0 && !offl(e))) continue;
        if (prof) {   // STRATA_XE_PROF: the arena experts by token count (would the CPU beat streaming them?)
            static int64_t hist[8] = {}, hbytes[8] = {}, hcalls = 0;
            const size_t nt_e = by_e[(size_t) e].size();
            const int b = nt_e <= 1 ? 0 : nt_e <= 2 ? 1 : nt_e <= 4 ? 2 : nt_e <= 8 ? 3 : nt_e <= 16 ? 4 : nt_e <= 32 ? 5 : nt_e <= 64 ? 6 : 7;
            ++hist[b];
            hbytes[b] += (int64_t) im.bbytes(l, e);
            if (++hcalls % 2000 == 0)
                std::fprintf(stderr, "xe prof: arena experts by tokens (1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65+): %lld %lld %lld %lld %lld %lld %lld %lld"
                             " (GB %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f)\n", (long long) hist[0], (long long) hist[1], (long long) hist[2],
                             (long long) hist[3], (long long) hist[4], (long long) hist[5], (long long) hist[6], (long long) hist[7],
                             hbytes[0] / 1e9, hbytes[1] / 1e9, hbytes[2] / 1e9, hbytes[3] / 1e9, hbytes[4] / 1e9, hbytes[5] / 1e9,
                             hbytes[6] / 1e9, hbytes[7] / 1e9);
        }
        if (ps && ps->card[(size_t) e] >= 0) { cl[(size_t) ps->card[(size_t) e]].push_back((int32_t) e); ++n_pf; continue; }
        if (keep_on && kept_card[(size_t) e] >= 0) {
            const int kc = kept_card[(size_t) e];
            cl[(size_t) kc].push_back((int32_t) e);
            kept[(size_t) kc].push_back((int32_t) e);
            continue;
        }
        if (light_on && !offl(e) && (int) by_e[(size_t) e].size() <= light_t && im.aptr[(size_t) im.key(l, e)] &&
            (light_pin || !im.apin[(size_t) im.key(l, e)])) {
            light.push_back((int32_t) e);
            continue;
        }
        size_t best = SIZE_MAX;
        for (size_t k = 0; k < NC; ++k) {
            if (lite_nostream && im.cards[k]->dense) continue;   // larry, STRATA_XE_DENSE_LITE (_MIN)
            if (best == SIZE_MAX || sbytes[k] < sbytes[best]) best = k;
        }
        sbytes[best] += (int64_t) ((im.bbytes(l, e) + 255) & ~(size_t) 255);
        streamed[best].push_back((int32_t) e);
        cl[best].push_back((int32_t) e);
    }
    size_t stream_cap = (size_t) (env_d("STRATA_XE_STREAM_GIB", 1.0) * kGiB);
    if (im.pf_on) stream_cap = std::min(stream_cap, (size_t) (env_d("STRATA_XE_PREFETCH_STREAM_GIB", 0.25) * kGiB));   // the overflow
    const int BJ = env_i("STRATA_XE_CHUNK_JOBS", 512);
    // ---- per card: buffers, X / W upload (under the submit lock: a pipelined batch's dense half enqueues concurrently)
    const double t_plan = now_ms();
    std::unique_lock<PrioMutex> prep_lk(g_submit);
    const double t_lk = now_ms();
    if (lite && dense_inline && im.dense_q) {   // larry, STRATA_XE_DENSE_INLINE
        XeCard& dcq = *im.cards[(size_t) im.dense];
        qswap.saved = dcq.q;
        qswap.c = &dcq;
        dcq.q = *im.dense_q;
    }
    // larry, STRATA_XE_FN16: the call's X once as fp16 on the dense card (the dense half's pass_sync made it final);
    // the expert cards copy half the bytes and widen it
    static const int fn16_env = env_i("STRATA_XE_FN16", 0);
    XeCard* cd16 = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    static const int fn16_min = env_i("STRATA_XE_FN16_MIN", 0);   // larry, patch-fn16min.py: the big calls only
    bool fn16 = fn16_env && x_dev && cd16 && n > kMaxJobTok && n >= fn16_min && ((size_t) n * H) % 4 == 0;
    sycl::event fn16_ev;
    // larry, patch-fn16q: on the dense card's own tier queue (DENSE_INLINE swapped its queue for the dense half's)
    sycl::queue& q16 = qswap.c && qswap.saved ? *qswap.saved : (cd16 ? cd16->q : im.cards[0]->q);
    if (fn16 && grow_dev(im.fn16, im.fn16_cap, (size_t) n * H, q16)) {
        const float* xs = x_dev;
        sycl::half* hd = im.fn16;
        fn16_ev = q16.parallel_for(sycl::range<1>((size_t) n * H / 4), [=](sycl::id<1> i) {
            const sycl::float4 v = *reinterpret_cast<const sycl::float4*>(xs + 4 * i[0]);
            *reinterpret_cast<sycl::vec<sycl::half, 4>*>(hd + 4 * i[0]) =
                sycl::vec<sycl::half, 4>((sycl::half) v.x(), (sycl::half) v.y(), (sycl::half) v.z(), (sycl::half) v.w());
        });
    } else {
        fn16 = false;
    }
    // larry, STRATA_SHEXP_TIER: the shared expert on the expert cards holding its slices (they run even with no routed
    // expert of their own)
    const bool shx = im.sh_call && (size_t) l < im.sh_slices.size() && !im.sh_slices[(size_t) l].empty();
    im.sh_call = false;
    auto sh_has = [&](int idx) {
        if (!shx) return false;
        for (const auto& sl : im.sh_slices[(size_t) l]) if (sl.card == idx) return true;
        return false;
    };
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.busy = !cl[(size_t) c.idx].empty() || (!c.dense && sh_has(c.idx));
        if (!c.busy) continue;
        const double tg0 = prof ? now_ms() : 0;
        uint8_t* const st0 = c.stream;   // a card keeping experts keeps its buffer (no regrow)
        // larry, STRATA_XE_STREAM_FIT (patch-streamfit.py): the buffer grows only into what the card has free beyond the
        // call's other buffers - a long context's KV rows or the promotion cache must not leave it overcommitted
        size_t cap_c = stream_cap;
        static const int stream_fit = env_i("STRATA_XE_STREAM_FIT", 1);
        static const double fit_margin = env_d("STRATA_XE_STREAM_FIT_MARGIN_GIB", 0.5);
        if (stream_fit && !c.dense) {
            const int64_t fr = c.free_vram();
            if (fr >= 0) {
                const int64_t room = fr + (int64_t) c.stream_bytes - (int64_t) (fit_margin * kGiB) - (int64_t) n * 100 * 1024;
                const size_t fit = (size_t) std::max<int64_t>(room, (int64_t) 64 << 20);
                if (fit < cap_c) {
                    static int64_t nfit = 0;
                    if (nfit++ < 20 || nfit % 1000 == 0)
                        std::fprintf(stderr, "xe tier: card %d: stream buffer capped at %.0f MiB (%.2f GiB free, %d tokens) - more rounds\n",
                                     c.idx, fit / 1048576.0, fr / kGiB, n);
                    cap_c = fit;
                }
            }
        }
        const size_t sneed = std::min<size_t>(cap_c, std::max<size_t>((size_t) sbytes[(size_t) c.idx], 1));
        // with the prefetch on, a regrowth goes straight to the cap (planned into the prefetch's reserve)
        const size_t sreq = !kept[(size_t) c.idx].empty() ? c.stream_bytes : im.pf_on && sneed > c.stream_bytes ? cap_c : sneed;
        if (!grow_stream(c, sreq, cap_c)) {
            std::fprintf(stderr, "xe tier: card %d: stream buffer allocation failed (%.2f GiB free)\n", c.idx, c.free_vram() / kGiB);
            return false;
        }
        if (c.stream != st0 && !kept[(size_t) c.idx].empty()) {
            std::fprintf(stderr, "xe tier: card %d: stream buffer moved with experts kept in it\n", c.idx);
            return false;
        }
        if (!grow_loc(c.cX, c.cX_cap, (size_t) n * H, c.q) || !grow_dev(c.cOUT, c.cOUT_cap, (size_t) n * H, c.q)) {
            std::fprintf(stderr, "xe tier: card %d: chunk X/OUT allocation failed (n %d, %.2f GiB free)\n", c.idx, n, c.free_vram() / kGiB);
            return false;
        }
        if (!out_dev && !grow_host(c.cOUT_host, c.cOUT_host_cap, (size_t) n * H, c.q)) {
            std::fprintf(stderr, "xe tier: card %d: chunk host buffer allocation failed (n %d)\n", c.idx, n);
            return false;
        }
        if (c.cS_cap < (size_t) n * K) {
            size_t scap = 0, wcap = 0;
            if (c.cS) { c.q.wait(); sycl::free(c.cS, c.q); sycl::free(c.cW, c.q); c.cS = nullptr; c.cW = nullptr; }
            if (!grow_dev(c.cS, scap, (size_t) n * K, c.q) || !grow_dev(c.cW, wcap, (size_t) n * K, c.q)) {
                std::fprintf(stderr, "xe tier: card %d: chunk map allocation failed\n", c.idx);
                return false;
            }
            c.cS_cap = (size_t) n * K;
        }
        if (c.cjobs_cap < BJ) {
            c.q.wait();
            if (c.cjobs) sycl::free(c.cjobs, c.q);
            if (c.cH) sycl::free(c.cH, c.q);
            if (c.cY) sycl::free(c.cY, c.q);
            if (c.cjobs_host) sycl::free(c.cjobs_host, c.q);
            c.cjobs = sycl::malloc_device<XeJob>((size_t) BJ, c.q);
            c.cjobs_host = sycl::malloc_host<XeJob>((size_t) BJ, c.q);
            c.cH = sycl::malloc_device<float>((size_t) BJ * kMaxJobTok * F, c.q);
            c.cY = sycl::malloc_device<float>((size_t) BJ * kMaxJobTok * H, c.q);
            if (!c.cjobs || !c.cjobs_host || !c.cH || !c.cY) {
                std::fprintf(stderr, "xe tier: card %d: chunk job buffers allocation failed (%.2f GiB free)\n", c.idx, c.free_vram() / kGiB);
                return false;
            }
            c.cjobs_cap = BJ;
        }
        const double tq0 = prof ? now_ms() : 0;
        if (fn16 && !c.dense && fn16_env >= 2 && g_fn16_peer && (((size_t) n * H) % 8) == 0) {
            // larry, STRATA_XE_FN16=2 (patch-fn16k.py): the dense card's fp16 X by peer loads, widened here - the copy
            // engine stays with the prefetch's staging
            const sycl::half* hs = im.fn16;
            float* xd = c.cX;
            c.q.submit([&](sycl::handler& h) {
                h.depends_on(fn16_ev);
                h.parallel_for(sycl::range<1>((size_t) n * H / 8), [=](sycl::id<1> i) {
                    const sycl::vec<sycl::half, 8> v = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(hs + 8 * i[0]);
                    *reinterpret_cast<sycl::float4*>(xd + 8 * i[0]) = sycl::float4((float) v[0], (float) v[1], (float) v[2], (float) v[3]);
                    *reinterpret_cast<sycl::float4*>(xd + 8 * i[0] + 4) = sycl::float4((float) v[4], (float) v[5], (float) v[6], (float) v[7]);
                });
            });
        } else if (fn16 && !c.dense && grow_loc(c.cXh, c.cXh_cap, (size_t) n * H, c.q)) {   // larry, STRATA_XE_FN16
            c.q.memcpy(c.cXh, im.fn16, (size_t) n * H * 2, fn16_ev);   // card-to-card from the dense card, this card's queue
            const sycl::half* hs = c.cXh;
            float* xd = c.cX;
            c.q.parallel_for(sycl::range<1>((size_t) n * H / 4), [=](sycl::id<1> i) {
                const sycl::vec<sycl::half, 4> v = *reinterpret_cast<const sycl::vec<sycl::half, 4>*>(hs + 4 * i[0]);
                *reinterpret_cast<sycl::float4*>(xd + 4 * i[0]) = sycl::float4((float) v[0], (float) v[1], (float) v[2], (float) v[3]);
            });
        } else if (!(lite && c.dense)) {   // larry, STRATA_XE_DENSE_LITE: the dense card reads x_dev where it is
            c.q.memcpy(c.cX, x_dev ? x_dev : x, (size_t) n * H * 4);   // x_dev: card-to-card from the dense card
            if (!c.dense) ptl_stamp(im.ctx, 1 + c.idx, c.q, 10, (int) l, n, tq0 > 0 ? tq0 : now_ms());   // larry, STRATA_GPU_PTL
        }
        const double tq1 = prof ? now_ms() : 0;
        c.q.memcpy(c.cW, w, (size_t) n * K * 4);
        const double tq2 = prof ? now_ms() : 0;
        if (!gather_init) c.q.memset(c.cOUT, 0, (size_t) n * H * 4);   // larry, STRATA_XE_GATHER_INIT: the first gather writes
        if (prof) {
            const double tq3 = now_ms();
            if (tq0 - tg0 > 5.0)
                std::fprintf(stderr, "xe prof: call %lld layer %lld card %d: buffers %.1f ms (n %d, stream %.0f MiB, cX %.0f MiB, cjobs %d)\n",
                             (long long) g_bp.calls, (long long) l, c.idx, tq0 - tg0, n, c.stream_bytes / 1048576.0,
                             c.cX_cap * 4 / 1048576.0, c.cjobs_cap);
            g_bp.prep_grow += tq0 - tg0;
            g_bp.prep_x += tq1 - tq0;
            g_bp.prep_w += tq2 - tq1;
            g_bp.prep_z += tq3 - tq2;
        }
    }
    prep_lk.unlock();
    const double t_prep = now_ms();
    // ---- every card's part on its own host thread
    std::vector<char> ok(NC, 1);
    std::atomic<int64_t> hits{0}, pcie{0};
    // STRATA_XE_COMBINE (see the combine below): the cards push their parts themselves; the dense card's landing
    // buffer (a slot per card) must exist before their threads start
    static const int cmb_env = env_i("STRATA_XE_COMBINE", 0);
    XeCard* cdc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    int cmb = out_dev && cdc ? cmb_env : 0;
    if (cmb > 0) {
        std::lock_guard<PrioMutex> lk(g_submit);
        const size_t need = im.cards.size() * (size_t) n * H;
        const bool okb = cmb == 2 ? grow_dev(cdc->cX2h, cdc->cX2h_cap, need, cdc->q) : grow_dev(cdc->cX2, cdc->cX2_cap, need, cdc->q);
        if (!okb) { std::fprintf(stderr, "xe tier: dense card: combine slots allocation failed - the copy-after combine\n"); cmb = 0; }
    }
    // larry, STRATA_XE_CALL_TL: per card, ms after the call's start (0 = not reached)
    static const bool call_tl = env_i("STRATA_XE_CALL_TL", 0) != 0;
    struct CallTl { double s = 0, st = 0, a = 0, g = 0, b = 0, e = 0; int64_t nst = 0, npin = 0, stb = 0, ng = 0, rows = 0, nj = 0; };
    std::vector<CallTl> ctl(NC);
    double tl_light = 0;
    auto card_work = [&](XeCard& c) -> bool {
        CallTl& tl = ctl[(size_t) c.idx];
        if (call_tl) tl.s = now_ms() - t0;
        // larry, STRATA_XE_DENSE_LITE: X where the dense half wrote it; the part queued without host waits (with the
        // combine on this queue after it).  The staging: GEMM rows + slot map, each job batch's jobs + slot map.
        const float* xin = lite && c.dense ? x_dev : c.cX;
        const bool nowait = lite && c.dense && cmb > 0 && streamed[(size_t) c.idx].empty() && kept[(size_t) c.idx].empty();
        // larry, STRATA_XE_ZCOPY (patch-zcopy.py): an expert card's uploads from pinned per-call copies too (no host waits);
        // single-round calls only (the staging below is sized for one pass: nothing streamed in the call)
        // larry, STRATA_XE_ZCOPY=2 (patch-zcopy2.py): every call (rounds included), the staging sized for them
        static const int zcopy_mode = env_i("STRATA_XE_ZCOPY", 0);
        const bool zc = zcopy_on() && !c.dense &&
                        (zcopy_mode >= 2 || (streamed[(size_t) c.idx].empty() && kept[(size_t) c.idx].empty()));
        const size_t zc_rounds = zc ? 2 + std::min<size_t>(streamed[(size_t) c.idx].size(), 64) : 1;
        const bool stg = nowait || zc;
        size_t nw_off = 0;
        bool out_init = !gather_init;   // larry, STRATA_XE_GATHER_INIT: false = the next gather writes the part
        if (stg) {
            int64_t rg = 0, njp = 0, njr = 0, ng_all = 0, nt_all = 0;   // larry, patch-zcopy3: GEMM experts, their tiles
            for (int32_t e : cl[(size_t) c.idx]) {
                const int64_t m = (int64_t) by_e[(size_t) e].size();
                if (m >= gemm_min) { rg += (m + 7) & ~(int64_t) 7; ++ng_all; nt_all += (m + 127) / 128 * (2 * F / 128 + H / 128); }
                else if (im.planar && im.card_of[(size_t) im.key(l, e)] >= 0) njp += (m + kMaxJobTok - 1) / kMaxJobTok;
                else njr += (m + kMaxJobTok - 1) / kMaxJobTok;
            }
            const int64_t nbat = (njp + BJ - 1) / BJ + (njr + BJ - 1) / BJ;
            const size_t need = 4 * 256 + (size_t) rg * 4 + zc_rounds * ((size_t) n * K * 4 + 512) +
                                ((size_t) nbat + zc_rounds) * ((size_t) BJ * sizeof(XeJob) + (size_t) n * K * 4 + 512) +
                                (zc ? (size_t) 3 * ng_all * sizeof(xe::DqJob) + (size_t) nt_all * sizeof(xe::DqTile) + zc_rounds * 1024 : 0);
            if (c.nw_cap < need) {
                std::lock_guard<PrioMutex> lk(g_submit);
                const double ta0 = now_ms();   // larry, STRATA_XE_ALLOC_LOG
                c.q.wait();   // nothing of this call reads the old staging yet; the previous call's combine waited
                const double ta1 = now_ms();
                if (c.nw_host) sycl::free(c.nw_host, c.q);
                c.nw_host = sycl::malloc_host<uint8_t>(need + need / 4, c.q);
                c.nw_cap = c.nw_host ? need + need / 4 : 0;
                alloc_note("call staging (pinned)", need + need / 4, ta0, ta1);
            }
            if (!c.nw_host) { std::fprintf(stderr, "xe tier: card %d: staging allocation failed\n", c.idx); return false; }
        }
        auto nw_take = [&](size_t bytes) -> uint8_t* {
            uint8_t* p = c.nw_host + nw_off;
            nw_off += (bytes + 255) & ~(size_t) 255;
            if (zc && nw_off > c.nw_cap) {   // larry, patch-zcopy: never past the staging (sized for one pass)
                std::fprintf(stderr, "xe tier: card %d: per-call staging overrun (%zu of %zu bytes) - aborting\n", c.idx, nw_off, c.nw_cap);
                std::abort();
            }
            return p;
        };
        std::vector<int32_t> S((size_t) n * K);
        std::vector<int64_t> soff((size_t) E, -1);
        std::vector<int32_t> round_of((size_t) E, 0);
        std::vector<int32_t> ptok;
        std::vector<uint8_t> ftmp;
        // larry, STRATA_GPU_PTL: this card's ring (card 0's part on the dense queue: ring 0)
        const int ptl_r = im.dense_q && c.q == *im.dense_q ? 0 : 1 + c.idx;
        const double ptl_t0 = Ds41XeTier::ptl_on() ? now_ms() : 0;
        if (c.dense && Ds41XeTier::ptl_on()) { std::lock_guard<PrioMutex> lk(g_submit); ptl_stamp(im.ctx, ptl_r, c.q, 20, (int) l, n, ptl_t0); }
        if (ps && ps->enqueued) {   // the prefetched experts' copies land before any kernel of this call
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.ext_oneapi_submit_barrier({c.pev[pslot]});
            ptl_stamp(im.ctx, ptl_r, c.q, 11, (int) l, n, ptl_t0);
        }
        if (ps && ps->enqueued && hot_check_on() && last_call)   // STRATA_XE_HOTRING_CHECK: this card's prefetched experts
            for (int64_t e = 0; e < E; ++e)
                if (ps->card[(size_t) e] == c.idx)
                    hot_check(im, c, c.q, c.pbuf[pslot] + ps->off[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e), l, e,
                              "prefetch");
        double tp = now_ms();
        auto lap = [&](std::atomic<int64_t>& acc) {
            if (!prof) return;
            c.q.wait();
            const double t = now_ms();
            acc += (int64_t) ((t - tp) * 1000.0);
            tp = t;
        };
        // larry, STRATA_XE_OVERLAP: round 0's streamed experts arrive while this thread computes the resident and
        // prefetched ones (round -1); a round's pageable bytes then also fit the bounce, so the helper never waits on it
        static const int ovl_on = env_i("STRATA_XE_OVERLAP", 0);   // measured neutral (8.5K 12.5-13.2 vs 12.8 s)
        const size_t bhalf = std::max<size_t>((size_t) im.blob_max, (size_t) 64 << 20);
        bool overlap = ovl_on && !streamed[(size_t) c.idx].empty();
        for (int32_t e : streamed[(size_t) c.idx]) if (!im.aptr[(size_t) im.key(l, e)]) overlap = false;   // file tier: old path
        // larry, patch-ovlring: the helper puts pageable experts through the card's hot ring (no bounce-sized rounds)
        if (overlap && hot_on() && !c.hring) c.hring = sycl::malloc_host<uint8_t>((size_t) hot_slots() * hot_chunk(), im.ctx);
        const bool ring_ovl = overlap && hot_on() && c.hring;
        int nr = 1;
        size_t used = 0, pused = 0;
        const std::vector<int32_t>& kp = kept[(size_t) c.idx];   // round 0, where they are; new copies go after them
        for (int32_t e : kp) soff[(size_t) e] = kept_off[(size_t) e];
        if (!kp.empty()) used = c.st_end;
        g_bp.kept += (int64_t) kp.size();
        c.st_layer = -1;   // set again once this call's copies are known to be in place
        for (int32_t e : streamed[(size_t) c.idx]) {
            const size_t b = (im.bbytes(l, e) + 255) & ~(size_t) 255;
            const size_t pg = im.apin[(size_t) im.key(l, e)] || ring_ovl ? 0 : b;
            if ((used + b > c.stream_bytes || (overlap && pused + pg > 2 * bhalf)) && used > 0) { ++nr; used = 0; pused = 0; }
            round_of[(size_t) e] = nr - 1;
            soff[(size_t) e] = (int64_t) used;
            used += b;
            pused += pg;
        }
        std::thread ohelp;
        struct Joiner { std::thread& t; ~Joiner() { if (t.joinable()) t.join(); } } ojoin{ohelp};
        sycl::event oev;
        if (overlap) {
            bool okb = true;
            if (c.bounce_cap < 2 * bhalf) {
                std::lock_guard<PrioMutex> lk(g_submit);
                if (c.bounce) { c.q.wait(); sycl::free(c.bounce, c.q); }
                c.bounce = sycl::malloc_host<uint8_t>(2 * bhalf, c.q);
                c.bounce_cap = c.bounce ? 2 * bhalf : 0;
                okb = c.bounce != nullptr;
            }
            if (!okb) { std::fprintf(stderr, "xe tier: card %d: bounce allocation failed\n", c.idx); return false; }
            ohelp = std::thread([&]() {   // CPU copies, then the DMAs (under the submit lock like every enqueue)
                size_t bo = 0;
                for (int32_t e : streamed[(size_t) c.idx]) {
                    if (round_of[(size_t) e] != 0) continue;
                    const size_t key = (size_t) im.key(l, e);
                    const size_t bytes = im.bbytes(l, e);
                    const uint8_t* src = im.aptr[key];
                    if (!im.apin[key]) {   // larry, STRATA_XE_BAR_STAGE: through the stream buffer's BAR mapping
                        if (uint8_t* bc = bar_cpu(c.stream + soff[(size_t) e])) {
                            bar_put(bc, src, bytes);
                            ++pcie;
                            ++g_bp.streamed;
                            continue;
                        }
                    }
                    if (!im.apin[key] && ring_ovl) {   // larry, patch-ovlring
                        oev = hot_put(c.oq, c.stream + soff[(size_t) e], src, bytes, false, c.hring, c.hring_ev, c.hring_next);
                        ++pcie;
                        ++g_bp.streamed;
                        continue;
                    }
                    if (!im.apin[key]) {
                        std::memcpy(c.bounce + bo, src, bytes);
                        src = c.bounce + bo;
                        bo += (bytes + 255) & ~(size_t) 255;
                    }
                    std::lock_guard<PrioMutex> lk(g_submit);
                    oev = c.oq.memcpy(c.stream + soff[(size_t) e], src, bytes);
                    ++pcie;
                    ++g_bp.streamed;
                }
            });
        }
        for (int r = overlap ? -1 : 0; r < nr; ++r) {
            if (overlap && r == 0) {   // round 0 came in on the overflow queue
                ohelp.join();
                std::lock_guard<PrioMutex> lk(g_submit);
                c.q.ext_oneapi_submit_barrier({oev});
            } else if (r >= 0 && !streamed[(size_t) c.idx].empty()) {
                if (r > 0) c.q.wait();   // the previous round's kernels read the stream buffer
                // pageable sources: copied (CPU) into one half of the pinned bounce, DMA'd from there; the other half
                // fills meanwhile.  Pinned (host USM) arena chunks DMA directly.
                const size_t half = std::max<size_t>((size_t) im.blob_max, (size_t) 64 << 20);
                if (c.bounce_cap < 2 * half) {
                    if (c.bounce) sycl::free(c.bounce, c.q);
                    c.bounce = sycl::malloc_host<uint8_t>(2 * half, c.q);
                    c.bounce_cap = c.bounce ? 2 * half : 0;
                    if (!c.bounce) { std::fprintf(stderr, "xe tier: card %d: bounce allocation failed\n", c.idx); return false; }
                }
                sycl::event hev[2];
                int hcur = 0;
                size_t hused = 0;
                struct Pend { size_t dst, src, bytes; };
                std::vector<Pend> pend;
                auto flush = [&]() {
                    if (pend.empty()) return;
                    std::lock_guard<PrioMutex> lk(g_submit);
                    for (const Pend& pd : pend) hev[hcur] = c.q.memcpy(c.stream + pd.dst, c.bounce + pd.src, pd.bytes);
                    pend.clear();
                    hcur ^= 1;
                    hused = 0;
                };
                for (int32_t e : streamed[(size_t) c.idx]) {
                    if (round_of[(size_t) e] != r) continue;
                    const size_t bytes = im.bbytes(l, e);
                    const uint8_t* hb = host_blob(im, l, e, ftmp);
                    const bool pinned = hb != ftmp.data() && sycl::get_pointer_type(hb, im.ctx) == sycl::usm::alloc::host;
                    if (call_tl) { ++tl.nst; tl.stb += (int64_t) bytes; tl.npin += pinned ? 1 : 0; }
                    uint8_t* bc = pinned ? nullptr : bar_cpu(c.stream + soff[(size_t) e]);   // larry, STRATA_XE_BAR_STAGE
                    if (pinned) {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.stream + soff[(size_t) e], hb, bytes);
                    } else if (bc) {
                        bar_put(bc, hb, bytes);
                        if (hot_check_on())
                            hot_check(im, c, c.q, c.stream + soff[(size_t) e], hb, bytes, l, e, "stream (BAR)");
                    } else if (hot_on() && (c.hring || (c.hring = sycl::malloc_host<uint8_t>((size_t) hot_slots() * hot_chunk(), im.ctx)))) {
                        // STRATA_XE_HOTRING: through the card thread's ring, a DMA a chunk as it is copied
                        hot_put(c.q, c.stream + soff[(size_t) e], hb, bytes, false, c.hring, c.hring_ev, c.hring_next);
                        if (hot_check_on())
                            hot_check(im, c, c.q, c.stream + soff[(size_t) e], hb, bytes, l, e, "stream");
                    } else {
                        if (hused + bytes > half) flush();
                        if (hused == 0) hev[hcur].wait();   // this half's previous DMAs are done
                        std::memcpy(c.bounce + (size_t) hcur * half + hused, hb, bytes);
                        pend.push_back({(size_t) soff[(size_t) e], (size_t) hcur * half + hused, bytes});
                        hused += (bytes + 255) & ~(size_t) 255;
                    }
                    ++pcie;
                    ++g_bp.streamed;
                }
                flush();
                if (call_tl) tl.st = now_ms() - t0;
                lap(g_bp.stream_us);
            }
            auto dptr_of = [&](int32_t e) -> const uint8_t* {
                const size_t key = (size_t) im.key(l, e);
                if (im.card_of[key] >= 0 && !offl(e)) return im.dptr[key];
                if (ps && ps->card[(size_t) e] >= 0) return c.pbuf[pslot] + ps->off[(size_t) e];
                return c.stream + soff[(size_t) e];
            };
            auto in_round = [&](int32_t e) {
                return (im.card_of[(size_t) im.key(l, e)] >= 0 && !offl(e)) || (ps && ps->card[(size_t) e] >= 0)
                           ? r == (overlap ? -1 : 0) : round_of[(size_t) e] == r;
            };
            // resident experts are stored planar (int8 kernels); streamed ones arrive in the file layout (F32 kernels)
            auto is_planar = [&](int32_t e) { return im.planar && im.card_of[(size_t) im.key(l, e)] >= 0 && !offl(e); };
            // ---- the GEMM experts of this round
            std::vector<int32_t> ge;
            int64_t gm_max = 0, gp = 0;
            for (int32_t e : cl[(size_t) c.idx])
                if (in_round(e) && (int) by_e[(size_t) e].size() >= gemm_min) {
                    ge.push_back(e);
                    gm_max = std::max<int64_t>(gm_max, (int64_t) by_e[(size_t) e].size());
                    gp += (int64_t) by_e[(size_t) e].size();
                }
            // larry, STRATA_XE_DQGEMM (default 1): the GEMM experts through xe_dqgemm - the weights decoded inside the
            // XMX GEMM (no fp16 copy of the matrices), every expert of a batch in a few launches (one per quant type
            // and layout).  Each expert's rows padded to 8 (whole 8-row tiles); batches of <= RB rows for the
            // activation / gate|up / SwiGLU buffers, Yg for all of them (the gather reads it at the end).
            static const int dq_on = env_i("STRATA_XE_DQGEMM", 1);
            const bool dq = dq_on && F % 128 == 0 && H % 128 == 0;
            if (!ge.empty() && dq) {
                const size_t NG = ge.size();
                std::vector<int64_t> ebase(NG + 1, 0);
                for (size_t i = 0; i < NG; ++i) ebase[i + 1] = ebase[i] + (((int64_t) by_e[(size_t) ge[i]].size() + 7) & ~(int64_t) 7);
                const int64_t R = ebase[NG];
                static const int64_t rb_env = std::max(256, env_i("STRATA_XE_GEMM_RB", 1024));   // larry, patch-gemmrb.py
                const int64_t RB = std::max<int64_t>(rb_env, (gm_max + 7) & ~(int64_t) 7);
                int64_t ntile_all = 0;
                for (size_t i = 0; i < NG; ++i) {
                    const int64_t mt = ((int64_t) by_e[(size_t) ge[i]].size() + 127) / 128;
                    ntile_all += mt * (2 * F / 128 + H / 128);
                }
                const size_t jbytes = (size_t) 3 * NG * sizeof(xe::DqJob), tbytes = (size_t) ntile_all * sizeof(xe::DqTile);
                const size_t need = (size_t) RB * H * 2 + (size_t) RB * 2 * F * 4 + (size_t) RB * F * 2 + (size_t) R * H * 4 +
                                    (size_t) R * 4 + jbytes + tbytes + 8 * 256;
                if (c.gscratch_cap < need) c.q.wait();
                bool grown;
                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    grown = grow_loc(c.gscratch, c.gscratch_cap, c.gscratch_cap < need ? need + need / 4 : need, c.q);   // headroom: few regrowths
                    if (grown && c.dq_host_cap < jbytes + tbytes) {
                        const size_t hb = (jbytes + tbytes) + (jbytes + tbytes) / 4;
                        if (c.dq_host) sycl::free(c.dq_host, c.q);
                        c.dq_host = sycl::malloc_host<uint8_t>(hb, c.q);
                        c.dq_host_cap = c.dq_host ? hb : 0;
                        grown = c.dq_host != nullptr;
                    }
                }
                if (!grown) {
                    std::fprintf(stderr, "xe tier: card %d: GEMM scratch allocation failed (%.2f GiB, %.2f GiB free)\n", c.idx,
                                 need / kGiB, c.free_vram() / kGiB);
                    return false;
                }
                uint8_t* p = c.gscratch;
                auto take = [&](size_t bytes) { uint8_t* r0 = p; p += (bytes + 255) & ~(size_t) 255; return r0; };
                sycl::half* Xg = (sycl::half*) take((size_t) RB * H * 2);
                float* G = (float*) take((size_t) RB * 2 * F * 4);
                sycl::half* Hh = (sycl::half*) take((size_t) RB * F * 2);
                float* Yg = (float*) take((size_t) R * H * 4);
                int32_t* dtok = (int32_t*) take((size_t) R * 4);
                xe::DqJob* djobs = (xe::DqJob*) take(jbytes);
                xe::DqTile* dtiles = (xe::DqTile*) take(tbytes);
                // larry, patch-zcopy3: under ZCOPY each round's tables in fresh per-call staging (round 0 must not
                // overwrite what round -1's table copies have yet to read: no host wait between them any more)
                xe::DqJob* hjobs = zc ? (xe::DqJob*) nw_take(jbytes) : (xe::DqJob*) c.dq_host;
                xe::DqTile* htiles = zc ? (xe::DqTile*) nw_take(tbytes) : (xe::DqTile*) (c.dq_host + jbytes);
                ptok.assign((size_t) R, 0);
                std::fill(S.begin(), S.end(), -1);
                for (size_t i = 0; i < NG; ++i) {
                    const auto& v = by_e[(size_t) ge[i]];
                    for (size_t j = 0; j < v.size(); ++j) { S[(size_t) v[j]] = (int32_t) (ebase[i] + (int64_t) j); ptok[(size_t) (ebase[i] + (int64_t) j)] = v[j] / K; }
                    for (int64_t j = (int64_t) v.size(); j < ebase[i + 1] - ebase[i]; ++j) ptok[(size_t) (ebase[i] + j)] = v[0] / K;   // padding rows: any token
                }
                if (stg) {   // larry, STRATA_XE_DENSE_LITE / STRATA_XE_ZCOPY: from pinned per-call copies, no wait
                    uint8_t* hp = nw_take(ptok.size() * 4);
                    uint8_t* hs = nw_take(S.size() * 4);
                    std::memcpy(hp, ptok.data(), ptok.size() * 4);
                    std::memcpy(hs, S.data(), S.size() * 4);
                    std::lock_guard<PrioMutex> lk(g_submit);
                    up_pinned(c.q, dtok, hp, ptok.size() * 4);
                    up_pinned(c.q, c.cS, hs, S.size() * 4);
                } else {
                    {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(dtok, ptok.data(), ptok.size() * 4);
                        c.q.memcpy(c.cS, S.data(), S.size() * 4);
                    }
                    c.q.wait();   // S and ptok are reused below
                }
                if (call_tl && tl.a == 0) tl.a = now_ms() - t0;
                size_t jo = 0, to = 0;   // the call's jobs / tiles so far (host staging and device arrays)
                // groups: quant type x layout
                auto gidx = [](int type, bool pl) { return (type == xe::kQ20 ? 1 : type == xe::kPtq1 ? 2 : 0) * 2 + (pl ? 1 : 0); };
                const int gtype[6] = {xe::kQ3K, xe::kQ3K, xe::kQ20, xe::kQ20, xe::kPtq1, xe::kPtq1};
                for (size_t i0 = 0; i0 < NG; ) {
                    size_t i1 = i0;
                    while (i1 < NG && (i1 == i0 || ebase[i1 + 1] - ebase[i0] <= RB)) ++i1;
                    const int64_t r0 = ebase[i0], rows = ebase[i1] - ebase[i0];
                    std::lock_guard<PrioMutex> lk(g_submit);
                    const double tlk = prof ? now_ms() : 0;
                    struct LkT { bool on; double t0; ~LkT() { if (on) { g_bp.lk_gemm_us += (int64_t) ((now_ms() - t0) * 1000.0); ++g_bp.lk_n; } } } lkt{prof, tlk};
                    xe::xe_gather_rows_f16(c.q, xin, dtok + r0, (int) rows, (int) H, Xg);
                    // gate | up: two jobs an expert (their own planes / rows), tiles of 128 output rows x 128 tokens
                    for (int phase = 0; phase < 2; ++phase) {
                        std::vector<xe::DqTile> gt[6];
                        const size_t jo0 = jo;
                        for (size_t i = i0; i < i1; ++i) {
                            const int32_t e = ge[i];
                            const int64_t m = (int64_t) by_e[(size_t) e].size();
                            const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
                            const uint8_t* blob = dptr_of(e);
                            const bool pl = is_planar(e);
                            if (phase == 0) {
                                const int qk = f.gu_type == xe::kQ3K ? 256 : f.gu_type == xe::kQ20 ? 64 : 128;
                                for (int u = 0; u < 2; ++u) {
                                    xe::DqJob jb;
                                    jb.W = blob + (u ? f.up_off : 0);
                                    jb.A = Xg + (size_t) (ebase[i] - r0) * H;
                                    jb.C = G + (size_t) (ebase[i] - r0) * 2 * F + (size_t) u * F;
                                    jb.NB = (int64_t) F * (H / qk);
                                    jb.m = (int32_t) m; jb.lda = (int32_t) H; jb.ldc = (int32_t) (2 * F); jb.row_bytes = (int32_t) f.gu_row;
                                    hjobs[jo] = jb;
                                    for (int64_t m0 = 0; m0 < m; m0 += 128)
                                        for (int64_t n0 = 0; n0 < F; n0 += 128) gt[gidx(f.gu_type, pl)].push_back({(int32_t) (jo - jo0), (int32_t) m0, (int32_t) n0, 0});
                                    ++jo;
                                }
                            } else {
                                const int qk = f.d_type == xe::kQ3K ? 256 : f.d_type == xe::kQ20 ? 64 : 128;
                                xe::DqJob jb;
                                jb.W = blob + f.down_off;
                                jb.A = Hh + (size_t) (ebase[i] - r0) * F;
                                jb.C = Yg + (size_t) ebase[i] * H;
                                jb.NB = (int64_t) H * (F / qk);
                                jb.m = (int32_t) m; jb.lda = (int32_t) F; jb.ldc = (int32_t) H; jb.row_bytes = (int32_t) f.d_row;
                                hjobs[jo] = jb;
                                for (int64_t m0 = 0; m0 < m; m0 += 128)
                                    for (int64_t n0 = 0; n0 < H; n0 += 128) gt[gidx(f.d_type, pl)].push_back({(int32_t) (jo - jo0), (int32_t) m0, (int32_t) n0, 0});
                                ++jo;
                            }
                        }
                        up_pinned(c.q, djobs + jo0, hjobs + jo0, (jo - jo0) * sizeof(xe::DqJob));   // larry, ZCOPY
                        for (int g = 0; g < 6; ++g) {
                            if (gt[g].empty()) continue;
                            std::memcpy(htiles + to, gt[g].data(), gt[g].size() * sizeof(xe::DqTile));
                            up_pinned(c.q, dtiles + to, htiles + to, gt[g].size() * sizeof(xe::DqTile));   // larry, ZCOPY
                            xe::xe_dqgemm(c.q, gtype[g], (g & 1) != 0, djobs + jo0, dtiles + to, (int) gt[g].size(), (int) (phase == 0 ? H : F));
                            to += gt[g].size();
                        }
                        if (phase == 0) xe::xe_swiglu_f16(c.q, G, (int) rows, (int) F, im.L, Hh);
                    }
                    for (size_t i = i0; i < i1; ++i) if (im.card_of[(size_t) im.key(l, ge[i])] >= 0) ++hits;
                    i0 = i1;
                }
                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    xe::xe_gather(c.q, Yg, c.cS, c.cW, n, K, (int) H, c.cOUT, out_init);
                    out_init = true;
                    ptl_stamp(im.ctx, ptl_r, c.q, 12, (int) l, r, now_ms());   // larry, STRATA_GPU_PTL
                }
                g_bp.gemm_e += (int64_t) ge.size();
                if (call_tl) { tl.g = now_ms() - t0; tl.ng += (int64_t) ge.size(); tl.rows += gp; }
                lap(g_bp.gemm_us);
            } else if (!ge.empty()) {
                const size_t need = (size_t) gm_max * H * 2 + (size_t) 2 * F * H * 2 + (size_t) H * F * 2 +
                                    (size_t) gm_max * 2 * F * 4 + (size_t) gm_max * F * 2 + (size_t) gp * H * 4 + (size_t) gp * 4 + 8192;
                if (c.gscratch_cap < need) c.q.wait();
                bool grown;
                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    grown = grow_loc(c.gscratch, c.gscratch_cap, need, c.q);
                }
                if (!grown) {
                    std::fprintf(stderr, "xe tier: card %d: GEMM scratch allocation failed (%.2f GiB, %.2f GiB free)\n", c.idx,
                                 need / kGiB, c.free_vram() / kGiB);
                    return false;
                }
                uint8_t* p = c.gscratch;
                auto take = [&](size_t bytes) { uint8_t* r0 = p; p += (bytes + 255) & ~(size_t) 255; return r0; };
                sycl::half* Xg = (sycl::half*) take((size_t) gm_max * H * 2);
                sycl::half* Wgu = (sycl::half*) take((size_t) 2 * F * H * 2);
                sycl::half* Wd = (sycl::half*) take((size_t) H * F * 2);
                float* G = (float*) take((size_t) gm_max * 2 * F * 4);
                sycl::half* Hh = (sycl::half*) take((size_t) gm_max * F * 2);
                float* Yg = (float*) take((size_t) gp * H * 4);
                int32_t* dtok = (int32_t*) take((size_t) gp * 4);
                ptok.clear();
                std::fill(S.begin(), S.end(), -1);
                for (int32_t e : ge)
                    for (int32_t sk : by_e[(size_t) e]) { S[(size_t) sk] = (int32_t) ptok.size(); ptok.push_back(sk / K); }
                if (stg) {   // larry, STRATA_XE_DENSE_LITE / STRATA_XE_ZCOPY: from pinned per-call copies, no wait
                    uint8_t* hp = nw_take(ptok.size() * 4);
                    uint8_t* hs = nw_take(S.size() * 4);
                    std::memcpy(hp, ptok.data(), ptok.size() * 4);
                    std::memcpy(hs, S.data(), S.size() * 4);
                    std::lock_guard<PrioMutex> lk(g_submit);
                    up_pinned(c.q, dtok, hp, ptok.size() * 4);
                    up_pinned(c.q, c.cS, hs, S.size() * 4);
                } else {
                    {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(dtok, ptok.data(), ptok.size() * 4);
                        c.q.memcpy(c.cS, S.data(), S.size() * 4);
                    }
                    c.q.wait();   // S and ptok are reused below
                }
                if (call_tl && tl.a == 0) tl.a = now_ms() - t0;
                int64_t base = 0;
                for (int32_t e : ge) {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    const double tlk = prof ? now_ms() : 0;
                    struct LkT { bool on; double t0; ~LkT() { if (on) { g_bp.lk_gemm_us += (int64_t) ((now_ms() - t0) * 1000.0); ++g_bp.lk_n; } } } lkt{prof, tlk};
                    const int64_t m = (int64_t) by_e[(size_t) e].size();
                    const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
                    const uint8_t* blob = dptr_of(e);
                    const bool pl = is_planar(e);
                    auto deq = [&](int type, const uint8_t* mat, int64_t rows, size_t row_b, int64_t nv, sycl::half* out) {
                        if (pl) xe::xe_dequant_planar_f16(c.q, type, mat, (int) rows, (int) nv, out);
                        else xe::xe_dequant_rows_f16(c.q, type, mat, (int) rows, (int) row_b, (int) nv, out);
                    };
                    xe::xe_gather_rows_f16(c.q, xin, dtok + base, (int) m, (int) H, Xg);
                    deq(f.gu_type, blob, F, f.gu_row, H, Wgu);
                    deq(f.gu_type, blob + f.up_off, F, f.gu_row, H, Wgu + (size_t) F * H);
                    mkl::blas::row_major::gemm(c.q, mkl::transpose::nontrans, mkl::transpose::trans, m, 2 * F, H, 1.0f, Xg, H,
                                               Wgu, H, 0.0f, G, 2 * F);
                    xe::xe_swiglu_f16(c.q, G, (int) m, (int) F, im.L, Hh);
                    deq(f.d_type, blob + f.down_off, H, f.d_row, F, Wd);
                    mkl::blas::row_major::gemm(c.q, mkl::transpose::nontrans, mkl::transpose::trans, m, H, F, 1.0f, Hh, F,
                                               Wd, F, 0.0f, Yg + (size_t) base * H, H);
                    base += m;
                    if (im.card_of[(size_t) im.key(l, e)] >= 0) ++hits;
                }
                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    xe::xe_gather(c.q, Yg, c.cS, c.cW, n, K, (int) H, c.cOUT, out_init);
                    out_init = true;
                    ptl_stamp(im.ctx, ptl_r, c.q, 12, (int) l, r, now_ms());   // larry, STRATA_GPU_PTL
                }
                g_bp.gemm_e += (int64_t) ge.size();
                if (call_tl) { tl.g = now_ms() - t0; tl.ng += (int64_t) ge.size(); tl.rows += gp; }
                lap(g_bp.gemm_us);
            }
            // ---- the small experts of this round: mat-vec jobs of <= 8 tokens, in batches of BJ; planar and file-layout
            // jobs in separate batches (their kernels differ)
            std::vector<XeJob> jobs, jobs_raw;
            std::vector<std::vector<int32_t>> jslots, jslots_raw;
            for (int32_t e : cl[(size_t) c.idx]) {
                if (!in_round(e) || (int) by_e[(size_t) e].size() >= gemm_min) continue;
                const auto& v = by_e[(size_t) e];
                const bool pl = is_planar(e);
                for (size_t i0 = 0; i0 < v.size(); i0 += kMaxJobTok) {
                    XeJob j = make_job(im, l, e, dptr_of(e));
                    j.nt = (uint8_t) std::min<size_t>(kMaxJobTok, v.size() - i0);
                    std::vector<int32_t> sl;
                    for (int s = 0; s < j.nt; ++s) { j.tok[s] = v[i0 + s] / K; j.w[s] = 1.0f; sl.push_back(v[i0 + s]); }
                    (pl ? jobs : jobs_raw).push_back(j);
                    (pl ? jslots : jslots_raw).push_back(sl);
                }
                if (im.card_of[(size_t) im.key(l, e)] >= 0) ++hits;
            }
            bool xq_done = false;
            auto run_jobs = [&](const std::vector<XeJob>& js, const std::vector<std::vector<int32_t>>& jss, bool pl) -> bool {
                for (size_t b0 = 0; b0 < js.size(); b0 += (size_t) BJ) {
                    const size_t nb = std::min<size_t>((size_t) BJ, js.size() - b0);
                    if (!stg) c.q.wait();   // cjobs / cjobs_host and cS are read by the previous batch / the GEMM gather
                    if (call_tl) { if (tl.b == 0) tl.b = now_ms() - t0; tl.nj += (int64_t) nb; }
                    XeJob* jh = stg ? (XeJob*) nw_take(nb * sizeof(XeJob)) : c.cjobs_host;   // larry, STRATA_XE_DENSE_LITE / ZCOPY
                    std::fill(S.begin(), S.end(), -1);
                    int max_nt = 1;
                    for (size_t q = 0; q < nb; ++q) {
                        jh[q] = js[b0 + q];
                        max_nt = std::max(max_nt, (int) js[b0 + q].nt);
                        const auto& sl = jss[b0 + q];
                        for (size_t s = 0; s < sl.size(); ++s) S[(size_t) sl[s]] = (int32_t) (q * kMaxJobTok + s);
                    }
                    if (pl && !xq_done) {   // this chunk's X in int8, once per card per call
                        if (c.cxq_rows < (size_t) n) {
                            bool okq;
                            {
                                std::lock_guard<PrioMutex> lk(g_submit);
                                okq = alloc_q8(c.cxq, (size_t) n, (size_t) H, c.q);
                            }
                            if (!okq) { std::fprintf(stderr, "xe tier: card %d: int8 X allocation failed (n %d)\n", c.idx, n); return false; }
                            c.cxq_rows = (size_t) n;
                        }
                        if (c.chq_jobs < BJ) {
                            bool okh;
                            {
                                std::lock_guard<PrioMutex> lk(g_submit);
                                okh = alloc_q8(c.chq, (size_t) BJ * kMaxJobTok, (size_t) F, c.q);
                            }
                            if (!okh) { std::fprintf(stderr, "xe tier: card %d: int8 H allocation failed\n", c.idx); return false; }
                            c.chq_jobs = BJ;
                        }
                        std::lock_guard<PrioMutex> lk(g_submit);
                        xe::xe_quant_rows(c.q, xin, n, (int) H, c.cxq);
                        xq_done = true;
                    }
                    {
                        const int32_t* sp = S.data();
                        if (stg) {   // larry, STRATA_XE_DENSE_LITE / ZCOPY: a per-batch pinned copy, no wait
                            uint8_t* hs = nw_take(S.size() * 4);
                            std::memcpy(hs, S.data(), S.size() * 4);
                            sp = (const int32_t*) hs;
                        }
                        std::lock_guard<PrioMutex> lk(g_submit);
                        if (stg) {   // pinned sources: a kernel under ZCOPY
                            up_pinned(c.q, c.cjobs, jh, nb * sizeof(XeJob));
                            up_pinned(c.q, c.cS, sp, S.size() * 4);
                        } else {
                            c.q.memcpy(c.cjobs, jh, nb * sizeof(XeJob));
                            c.q.memcpy(c.cS, sp, S.size() * 4);
                        }
                    }
                    if (!stg) c.q.wait();
                    std::lock_guard<PrioMutex> lk(g_submit);
                    const double tlk = prof ? now_ms() : 0;
                    struct LkT { bool on; double t0; ~LkT() { if (on) g_bp.lk_jobs_us += (int64_t) ((now_ms() - t0) * 1000.0); } } lkt{prof, tlk};
                    if (pl) xe::xe_experts_y_q8(c.q, c.cjobs, (int) nb, c.cxq, c.chq, c.cY, (int) H, (int) F, im.L, max_nt);
                    else xe::xe_experts_y(c.q, c.cjobs, (int) nb, xin, c.cH, c.cY, (int) H, (int) F, im.L);
                    xe::xe_gather(c.q, c.cY, c.cS, c.cW, n, K, (int) H, c.cOUT, out_init);
                    out_init = true;
                }
                return true;
            };
            if (!run_jobs(jobs, jslots, true) || !run_jobs(jobs_raw, jslots_raw, false)) return false;
            if (Ds41XeTier::ptl_on() && !(jobs.empty() && jobs_raw.empty())) {   // larry, STRATA_GPU_PTL
                std::lock_guard<PrioMutex> lk(g_submit);
                ptl_stamp(im.ctx, ptl_r, c.q, 13, (int) l, r, now_ms());
            }
            jobs.insert(jobs.end(), jobs_raw.begin(), jobs_raw.end());
            g_bp.small_j += (int64_t) jobs.size();
            if (!jobs.empty()) lap(g_bp.small_us);
        }
        {   // what the stream buffer holds now: the last round's copies (with one round, the kept experts too)
            std::vector<std::pair<int32_t, int64_t>> have;
            size_t end = 0;
            auto add = [&](int32_t e) {
                have.emplace_back(e, soff[(size_t) e]);
                end = std::max(end, (size_t) soff[(size_t) e] + ((im.bbytes(l, e) + 255) & ~(size_t) 255));
            };
            if (nr == 1) for (int32_t e : kp) add(e);
            for (int32_t e : streamed[(size_t) c.idx]) if (round_of[(size_t) e] == nr - 1) add(e);
            c.st_have.swap(have);
            c.st_end = end;
            c.st_layer = l;
        }
        // larry, STRATA_SHEXP_TIER (patch-shexptier.py): this card's shared-expert slices for every token of the call,
        // added into its part: per slice its layer's fp16 copy (at the layer's first call), then X rows -> f16, the
        // gate | up GEMM, the clamped SwiGLU, the down GEMM into the part (beta 1 once anything wrote it)
        if (shx && !c.dense && sh_has(c.idx)) {
            const auto& sls = im.sh_slices[(size_t) l];
            size_t wneed = 0, nfm = 0;
            for (const auto& sl : sls) if (sl.card == c.idx) { wneed += (size_t) (3 * sl.nf * H); nfm = std::max(nfm, (size_t) sl.nf); }
            const int64_t RBs = 1024;
            bool ok_sh;
            {
                std::lock_guard<PrioMutex> lk(g_submit);
                const sycl::half* w_was = c.sh_w;
                ok_sh = grow_loc(c.sh_w, c.sh_w_cap, wneed, c.q);
                if (c.sh_w != w_was) c.sh_layer = -1;   // a new buffer: its layer's copy is gone
                if (ok_sh && (c.sh_rows < (size_t) RBs || c.sh_nf < nfm)) {
                    if (c.sh_buf) { c.q.wait(); local_free(c.q, c.sh_buf); c.sh_buf = nullptr; c.sh_rows = c.sh_nf = 0; }
                    const size_t bx = (size_t) RBs * H * 2, bg = (size_t) RBs * 2 * nfm * 4, bh = (size_t) RBs * nfm * 2;
                    c.sh_buf = local_alloc(c.q, bx + bg + bh);
                    ok_sh = c.sh_buf != nullptr;
                    if (ok_sh) {
                        c.sh_x = (sycl::half*) c.sh_buf;
                        c.sh_g = (float*) (c.sh_buf + bx);
                        c.sh_h = (sycl::half*) (c.sh_buf + bx + bg);
                        c.sh_rows = (size_t) RBs;
                        c.sh_nf = nfm;
                    }
                }
                if (ok_sh && c.sh_iota_n < (size_t) n) {
                    if (c.sh_iota) { c.q.wait(); local_free(c.q, c.sh_iota); c.sh_iota = nullptr; c.sh_iota_n = 0; }
                    const size_t ni = std::max<size_t>((size_t) n, 16384);
                    c.sh_iota = (int32_t*) local_alloc(c.q, ni * 4);
                    ok_sh = c.sh_iota != nullptr;
                    if (ok_sh) {
                        int32_t* io = c.sh_iota;
                        c.q.parallel_for(sycl::range<1>(ni), [=](sycl::id<1> i) { io[i] = (int32_t) i[0]; });
                        c.sh_iota_n = ni;
                    }
                }
                if (ok_sh && c.sh_layer != l) {
                    size_t off = 0;
                    for (const auto& sl : sls) {
                        if (sl.card != c.idx) continue;
                        fastmv::dequant_q6k_planar_f16(c.q, sl.gu, 2 * sl.nf, H, c.sh_w + off);
                        fastmv::dequant_q6k_planar_f16(c.q, sl.dn, H, sl.nf, c.sh_w + off + (size_t) (2 * sl.nf * H));
                        off += (size_t) (3 * sl.nf * H);
                    }
                    c.sh_layer = l;
                }
            }
            if (!ok_sh) { std::fprintf(stderr, "xe tier: card %d: shared-expert buffers failed\n", c.idx); return false; }
            size_t off = 0;
            for (const auto& sl : sls) {
                if (sl.card != c.idx) continue;
                const int64_t nf = sl.nf;
                const sycl::half* Wgu = c.sh_w + off;
                const sycl::half* Wdn = c.sh_w + off + (size_t) (2 * nf * H);
                for (int64_t r0 = 0; r0 < (int64_t) n; r0 += RBs) {
                    const int64_t m = std::min<int64_t>(RBs, (int64_t) n - r0);
                    std::lock_guard<PrioMutex> lk(g_submit);
                    xe::xe_gather_rows_f16(c.q, xin, c.sh_iota + r0, (int) m, (int) H, c.sh_x);
                    mkl::blas::row_major::gemm(c.q, mkl::transpose::nontrans, mkl::transpose::trans, m, 2 * nf, H, 1.0f, c.sh_x, H,
                                               Wgu, H, 0.0f, c.sh_g, 2 * nf);
                    xe::xe_swiglu_f16(c.q, c.sh_g, (int) m, (int) nf, im.L, c.sh_h);
                    mkl::blas::row_major::gemm(c.q, mkl::transpose::nontrans, mkl::transpose::trans, m, H, nf, 1.0f, c.sh_h, nf,
                                               Wdn, nf, out_init ? 1.0f : 0.0f, c.cOUT + (size_t) r0 * H, H);
                }
                out_init = true;
                off += (size_t) (3 * nf * H);
            }
        }
        if (!out_dev) {
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.memcpy(c.cOUT_host, c.cOUT, (size_t) n * H * 4);
        }
        if (!out_init) {   // larry, STRATA_XE_GATHER_INIT: no gather ran - the part is zero
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.memset(c.cOUT, 0, (size_t) n * H * 4);
            out_init = true;
        }
        if (out_dev && !c.dense && cmb > 0 && cdc) {   // STRATA_XE_COMBINE: this card's part onto the dense card now
            const size_t ne = (size_t) n * H;
            std::lock_guard<PrioMutex> lk(g_submit);
            if (cmb == 2) {   // fp16 in the call's X buffer (no longer read), half the bytes over the dense card's link
                sycl::half* h16 = reinterpret_cast<sycl::half*>(c.cX);
                const float* src = c.cOUT;
                c.q.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> i) { h16[i] = (sycl::half) src[i]; });
                c.q.memcpy(cdc->cX2h + (size_t) c.idx * ne, h16, ne * 2);
            } else {
                c.q.memcpy(cdc->cX2 + (size_t) c.idx * ne, c.cOUT, ne * 4);
            }
        }
        if (Ds41XeTier::ptl_on()) {   // larry, STRATA_GPU_PTL
            std::lock_guard<PrioMutex> lk(g_submit);
            ptl_stamp(im.ctx, ptl_r, c.q, c.dense ? 21 : 14, (int) l, n, now_ms());
        }
        if (!nowait) c.q.wait();   // larry, STRATA_XE_DENSE_LITE: the combine on this queue waits for the part
        if (call_tl) tl.e = now_ms() - t0;
        if (prof) {   // the lowest free VRAM at the end of a call (every buffer of the call allocated)
            const int64_t fr = c.free_vram();
            std::atomic<int64_t>& mf = g_bp.min_free[c.idx & 7];
            int64_t cur = mf.load();
            while (fr >= 0 && (cur < 0 || fr < cur) && !mf.compare_exchange_weak(cur, fr)) {}
        }
        return true;
    };
    const size_t NE = (size_t) n * H;
    bool light_ok = true;
    {
        std::vector<std::thread> th;
        for (auto& cp : im.cards) {
            XeCard* c = cp.get();
            if (!c->busy) continue;
            th.emplace_back([&, c]() {
                try {
                    ok[(size_t) c->idx] = card_work(*c) ? 1 : 0;
                } catch (const sycl::exception& e) {
                    std::fprintf(stderr, "xe tier: card %d chunk: %s\n", c->idx, e.what());
                    ok[(size_t) c->idx] = 0;
                }
            });
        }
        // STRATA_XE_CPU_LIGHT: the light experts on the pool while the cards run theirs (their weighted sum -> lt_hs).
        // Nothing here may throw past the card threads' join (a joinable std::thread's destructor terminates).
        if (!light.empty()) try {
            const double tq = now_ms();
            // larry, STRATA_XE_LT_PROF (patch-ltprof.py): the light part's steps, ms summed over calls
            static const bool lt_prof = env_i("STRATA_XE_LT_PROF", 0) != 0;
            static double lp[6] = {};   // setup, readback, jobs, pool, sum, calls
            double lpt = lt_prof ? now_ms() : 0;
            auto lpl = [&](int i) { if (lt_prof) { const double t = now_ms(); lp[i] += t - lpt; lpt = t; } };
            const size_t nf = im.nact_fmt.size();
            if (im.lt_actq.size() < (size_t) n) im.lt_actq.resize((size_t) n);
            if (im.lt_nact.size() < (size_t) n * std::max<size_t>(1, nf) * cpu::kNativeActBytes)
                im.lt_nact.resize((size_t) n * std::max<size_t>(1, nf) * cpu::kNativeActBytes);
            im.lt_have.assign((size_t) n * (1 + nf), 0);
            if (im.lt_jobs.size() < (size_t) cpu::ExpertPool::kMaxSplitMulti) im.lt_jobs.resize(cpu::ExpertPool::kMaxSplitMulti);
            if (im.lt_cpart.size() < light.size() * kMaxJobTok * (size_t) H) im.lt_cpart.resize(light.size() * kMaxJobTok * (size_t) H);
            if (im.lt_hs_cap < NE) {
                if (im.lt_hs) sycl::free(im.lt_hs, im.ctx);
                im.lt_hs = sycl::malloc_host<float>(NE, im.ctx);
                im.lt_hs_cap = im.lt_hs ? NE : 0;
            }
            if (!im.lt_hs) { std::fprintf(stderr, "xe tier: CPU-light sum buffer allocation failed\n"); light_ok = false; }
            // larry, STRATA_XE_LT_COMPACT: the tokens the light experts serve, a compact row each (first-seen order)
            static const int lt_compact = env_i("STRATA_XE_LT_COMPACT", 1);
            int nrows = 0;
            im.lt_nrows = -1;
            if (light_ok && lt_compact) {
                im.lt_slot.assign((size_t) n, -1);
                for (int32_t e : light)
                    for (int32_t sk : by_e[(size_t) e]) {
                        const int t = sk / K;
                        if (im.lt_slot[(size_t) t] < 0) im.lt_slot[(size_t) t] = nrows++;
                    }
                if (im.lt_rows_cap < (size_t) nrows) {
                    if (im.lt_rows) sycl::free(im.lt_rows, im.ctx);
                    const size_t cap = std::max<size_t>((size_t) nrows, 1024);
                    im.lt_rows = sycl::malloc_host<int32_t>(cap, im.ctx);
                    im.lt_rows_cap = im.lt_rows ? cap : 0;
                }
                if (!im.lt_rows) { std::fprintf(stderr, "xe tier: CPU-light row list allocation failed\n"); light_ok = false; }
                else for (int t = 0; t < n; ++t) if (im.lt_slot[(size_t) t] >= 0) im.lt_rows[im.lt_slot[(size_t) t]] = t;
            }
            const bool xl_compact = light_ok && lt_compact && x_dev;
            lpl(0);   // larry, STRATA_XE_LT_PROF: buffers + row list
            const float* xl = x;
            if (light_ok && x_dev) {   // a pipelined call: X off the dense card (its host copy is stale)
                const size_t xn = xl_compact ? (size_t) nrows * H : NE;
                if (im.lt_x_cap < xn) {
                    if (im.lt_x) sycl::free(im.lt_x, im.ctx);
                    im.lt_x = sycl::malloc_host<float>(xn, im.ctx);
                    im.lt_x_cap = im.lt_x ? xn : 0;
                }
                XeCard& dcx = *im.cards[(size_t) im.dense];
                if (xl_compact && im.lt_xg_cap < xn) {   // the dense card's gather buffer (grown, never on the decode path)
                    std::lock_guard<PrioMutex> lk(g_submit);
                    if (im.lt_xg) { dcx.pq.wait(); sycl::free(im.lt_xg, dcx.pq); }
                    const size_t cap = std::max<size_t>(xn, (size_t) 256 * H);
                    im.lt_xg = sycl::malloc_device<float>(cap, dcx.pq);
                    im.lt_xg_cap = im.lt_xg ? cap : 0;
                }
                if (!im.lt_x || (xl_compact && !im.lt_xg)) { std::fprintf(stderr, "xe tier: CPU-light X buffer allocation failed\n"); light_ok = false; }
                else {
                    sycl::event xev;
                    {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        if (xl_compact) {   // larry, STRATA_XE_LT_COMPACT: just the light rows, gathered on the dense card
                            const int32_t* rr = im.lt_rows;
                            float* xg = im.lt_xg;
                            const float* xs = x_dev;
                            const size_t h4 = (size_t) H / 4;
                            dcx.pq.parallel_for(sycl::range<1>((size_t) nrows * h4), [=](sycl::id<1> i) {
                                const size_t j = i[0] / h4, c = i[0] % h4;
                                reinterpret_cast<sycl::float4*>(xg + j * h4 * 4)[c] =
                                    reinterpret_cast<const sycl::float4*>(xs + (size_t) rr[j] * h4 * 4)[c];
                            });
                            xev = dcx.pq.memcpy(im.lt_x, xg, xn * 4);
                        } else {
                            xev = dcx.pq.memcpy(im.lt_x, x_dev, NE * 4);
                        }
                    }
                    xev.wait();
                    xl = im.lt_x;
                }
            }
            auto xrow = [&](int t) -> const float* {   // larry, STRATA_XE_LT_COMPACT
                return xl_compact ? xl + (size_t) im.lt_slot[(size_t) t] * H : xl + (size_t) t * H;
            };
            lpl(1);   // larry, STRATA_XE_LT_PROF: X read back
            double t_run = 0, t_sum = 0;
            size_t done = 0;
            while (light_ok && done < light.size()) {
                const size_t nb = std::min<size_t>(light.size() - done, (size_t) cpu::ExpertPool::kMaxSplitMulti);
                // larry, STRATA_XE_LT_QUANT_THREADS (patch-ltquant.py): this batch's activations quantised first, on
                // helper threads (each (token, format) once; the job setup below then finds them done)
                static const int lq_thr = std::max(0, std::min(16, env_i("STRATA_XE_LT_QUANT_THREADS", 0)));
                if (lq_thr > 0) {
                    struct LQ { int t; int fs; const cpu::NativeFmt* f; };
                    std::vector<LQ> lq;
                    for (size_t i = 0; i < nb; ++i) {
                        const int32_t e = light[done + i];
                        const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
                        int fs = -1;
                        if (!cpu::q2_native_kernels(f.gu_type))
                            for (size_t b = 0; b < nf; ++b) if (im.nact_fmt[b] == f.gu_act) fs = (int) b;
                        for (int32_t sk : by_e[(size_t) e]) {
                            const int t = sk / K;
                            uint8_t& hv = im.lt_have[(size_t) t * (1 + nf) + (size_t) (fs + 1)];
                            if (!hv) { hv = 1; lq.push_back({t, fs, &f}); }
                        }
                    }
                    auto work = [&](size_t k0, size_t step) {
                        for (size_t k = k0; k < lq.size(); k += step) {
                            const LQ& q = lq[k];
                            if (q.fs < 0) cpu::act_quant_any(xrow(q.t), (int) H, im.lt_actq[(size_t) q.t]);
                            else cpu::native_quant_act(*q.f, xrow(q.t), im.lt_nact.data() + ((size_t) q.t * nf + (size_t) q.fs) * cpu::kNativeActBytes);
                        }
                    };
                    const size_t nh = std::min<size_t>((size_t) lq_thr, lq.size() / 16);
                    std::vector<std::thread> th;
                    for (size_t k = 1; k <= nh; ++k) th.emplace_back(work, k, nh + 1);
                    work(0, nh + 1);
                    for (auto& t : th) t.join();
                }
                for (size_t i = 0; i < nb; ++i) {
                    const int32_t e = light[done + i];
                    const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
                    int fs = -1;
                    if (!cpu::q2_native_kernels(f.gu_type))
                        for (size_t b = 0; b < nf; ++b) if (im.nact_fmt[b] == f.gu_act) fs = (int) b;
                    cpu::ExpertJobMulti& j = im.lt_jobs[i];
                    j = cpu::ExpertJobMulti();
                    j.blob = im.aptr[(size_t) im.key(l, e)];
                    j.fmt = &f;
                    const std::vector<int32_t>& sl = by_e[(size_t) e];
                    j.nt = (int) sl.size();
                    for (int si = 0; si < j.nt; ++si) {
                        const int t = sl[(size_t) si] / K;
                        uint8_t& hv = im.lt_have[(size_t) t * (1 + nf) + (size_t) (fs + 1)];
                        uint8_t* na = fs < 0 ? nullptr : im.lt_nact.data() + ((size_t) t * nf + (size_t) fs) * cpu::kNativeActBytes;
                        if (!hv) {
                            if (fs < 0) cpu::act_quant_any(xrow(t), (int) H, im.lt_actq[(size_t) t]);
                            else cpu::native_quant_act(f, xrow(t), na);
                            hv = 1;
                        }
                        j.act[si] = &im.lt_actq[(size_t) t];
                        j.nact[si] = na;
                        j.out[si] = im.lt_cpart.data() + ((done + i) * kMaxJobTok + (size_t) si) * H;
                    }
                    im.lt_bytes += (double) im.bbytes(l, e);
                }
                lpl(2);   // larry, STRATA_XE_LT_PROF: the jobs' setup (quantisation)
                const double tr = now_ms();
                im.pool->run_split_multi_native(*im.lt_jobs[0].fmt, im.lt_jobs.data(), (int) nb);
                t_run += now_ms() - tr;
                lpl(3);   // the pool
                done += nb;
            }
            if (light_ok) {   // weighted into the call's rows (routing order within a token: slot k ascending per expert)
                const double ts = now_ms();
                const bool sc = lt_compact != 0;   // larry, STRATA_XE_LT_COMPACT: a compact row a token
                std::memset(im.lt_hs, 0, (sc ? (size_t) nrows * H : NE) * sizeof(float));
                for (size_t i = 0; i < light.size(); ++i) {
                    const std::vector<int32_t>& sl = by_e[(size_t) light[i]];
                    for (size_t si = 0; si < sl.size(); ++si) {
                        const int t = sl[si] / K;
                        const float wt = w[(size_t) sl[si]];
                        float* o = im.lt_hs + (size_t) (sc ? im.lt_slot[(size_t) t] : t) * H;
                        const float* p = im.lt_cpart.data() + (i * kMaxJobTok + si) * H;
                        for (int64_t h = 0; h < H; ++h) o[h] += wt * p[h];
                    }
                }
                t_sum = now_ms() - ts;
                if (sc) im.lt_nrows = nrows;
            }
            ++im.lt_calls;
            im.lt_experts += (int64_t) light.size();
            im.lt_q_ms += now_ms() - tq - t_run - t_sum;
            lpl(4);   // larry, STRATA_XE_LT_PROF: the sum
            if (lt_prof && ++lp[5] >= 0 && (int64_t) lp[5] % 100 == 0)
                std::fprintf(stderr, "xe tier: CPU-light prof (%d calls): setup %.0f, readback %.0f, jobs %.0f, pool %.0f, sum %.0f ms\n",
                             (int) lp[5], lp[0], lp[1], lp[2], lp[3], lp[4]);
            im.lt_run_ms += t_run;
            im.lt_sum_ms += t_sum;
            im.st.cpu += (int64_t) light.size();
            im.st.cpu_ms += now_ms() - tq;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xe tier: CPU-light (layer %lld, %d tokens): %s\n", (long long) l, n, e.what());
            light_ok = false;
        }
        const double tw = now_ms();
        if (call_tl && !light.empty()) tl_light = tw - t0;
        for (auto& t : th) t.join();
        if (!light.empty()) im.lt_wait_ms += now_ms() - tw;
        for (size_t k = 0; k < NC; ++k) im.lt_streamed += (int64_t) streamed[k].size();
        if (!light.empty() && im.lt_calls % (prof ? 40 : 400) == 0)
            std::fprintf(stderr, "xe tier: CPU-light (T %d): %lld calls, %lld experts on the pool (%.1f GB; %lld streamed), "
                         "quantise %.0f ms, pool %.0f ms (%.1f GB/s), sum %.0f ms, then the cards %.0f ms more\n", light_t,
                         (long long) im.lt_calls, (long long) im.lt_experts, im.lt_bytes / 1e9, (long long) im.lt_streamed,
                         im.lt_q_ms, im.lt_run_ms, im.lt_run_ms > 0 ? im.lt_bytes / 1e6 / im.lt_run_ms : 0.0, im.lt_sum_ms,
                         im.lt_wait_ms);
    }
    for (size_t k = 0; k < NC; ++k) if (!ok[k]) return false;
    if (!light_ok) return false;
    im.st.hits += hits;
    im.st.pcie += pcie + n_pf;
    if (ps) {   // the layer's last call: the next layer's copies (if not on the queues yet), this slot takes l + 2
        im.pf_used += n_pf;
        if (last_call) {
            // larry, STRATA_XE_PF_NOBLOCK: with the hot rings the next layer's copies are queued by the stagers (and its
            // barriers recorded by them); its first call waits for its staging - this call's combine does not
            static const int pf_nb = env_i("STRATA_XE_PF_NOBLOCK", 0);
            const bool pf_block = !(pf_nb && hot_on() && !im.pf_rings.empty());
            if (im.pf[pslot ^ 1].layer == l + 1) pf_enqueue(im, pslot ^ 1, pf_block);
            ps->layer = -1;
            if (l + 2 <= im.pf_last) pf_plan(im, pslot, l + 2);
        }
    }
    const double t_cards = now_ms();
    // ---- combine: the other cards' parts copied card-to-card onto the dense card and added there
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    if (out_dev) {
        if (!dc) { std::fprintf(stderr, "xe tier: device output without a dense card\n"); return false; }
        std::vector<const XeCard*> others;
        for (auto& cp : im.cards) if (cp->busy && !cp->dense) others.push_back(cp.get());
        sycl::event cev;
        // STRATA_XE_CPU_LIGHT: the pool's sum, read by the combine straight from pinned host memory - a copy in a
        // dense-card buffer (~70 MB after a 6.7K prompt) put DSpark's long prompts under ggml-sycl's fp16 weight-cache
        // floor (3 GB free on card 0): their dense phase +13-20 %
        const float* cpu_part = light.empty() || im.lt_nrows >= 0 ? nullptr : im.lt_hs;   // larry, LT_COMPACT: rows after
        if (cmb > 0) {   // STRATA_XE_COMBINE: every busy card's part is in its slot (its thread waited for the push)
            std::lock_guard<PrioMutex> lk(g_submit);
            const int ptl_rc = im.dense_q && dc->q == *im.dense_q ? 0 : 1 + dc->idx;   // larry, STRATA_GPU_PTL
            const double ptl_tc = Ds41XeTier::ptl_on() ? now_ms() : 0;
            ptl_stamp(im.ctx, ptl_rc, dc->q, 22, (int) l, n, ptl_tc);
            struct PtlEnd { const sycl::context& cx; int r; sycl::queue& q; int l, n; double t;
                            ~PtlEnd() { ptl_stamp(cx, r, q, 23, l, n, t); } } ptl_end{im.ctx, ptl_rc, dc->q, (int) l, n, ptl_tc};
            int idx[8] = {};
            const int np = (int) std::min<size_t>(others.size(), 8);
            for (int k = 0; k < np; ++k) idx[k] = others[(size_t) k]->idx;
            const float* own = dc->busy ? dc->cOUT : nullptr;
            float* o = (float*) out_dev;
            if (cmb == 2) {
                const sycl::half* parts = dc->cX2h;
                cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                    float sm = own ? own[i] : 0.0f;
                    for (int k = 0; k < np; ++k) sm += (float) parts[(size_t) idx[k] * NE + i];
                    if (cpu_part) sm += cpu_part[i];
                    o[i] = sm;
                });
            } else {
                const float* parts = dc->cX2;
                cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                    float sm = own ? own[i] : 0.0f;
                    for (int k = 0; k < np; ++k) sm += parts[(size_t) idx[k] * NE + i];
                    if (cpu_part) sm += cpu_part[i];
                    o[i] = sm;
                });
            }
        } else {
            std::lock_guard<PrioMutex> lk(g_submit);
            if (!others.empty() && !grow_dev(dc->cX2, dc->cX2_cap, NE * others.size(), dc->q)) {
                std::fprintf(stderr, "xe tier: dense card: combine buffer allocation failed\n");
                return false;
            }
            for (size_t k = 0; k < others.size(); ++k) dc->q.memcpy(dc->cX2 + k * NE, others[k]->cOUT, NE * 4);
            const int np = (int) others.size();
            const float* parts = dc->cX2;
            const float* own = dc->busy ? dc->cOUT : nullptr;
            float* o = (float*) out_dev;
            cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                float s = own ? own[i] : 0.0f;
                for (int k = 0; k < np; ++k) s += parts[(size_t) k * NE + i];
                if (cpu_part) s += cpu_part[i];
                o[i] = s;
            });
        }
        if (!light.empty() && im.lt_nrows > 0) {   // larry, STRATA_XE_LT_COMPACT: the CPU part's rows (the sum's last add)
            std::lock_guard<PrioMutex> lk(g_submit);
            const int32_t* rr = im.lt_rows;
            const float* cp = im.lt_hs;
            float* o = (float*) out_dev;
            const size_t hh = (size_t) H;
            cev = dc->q.parallel_for(sycl::range<1>((size_t) im.lt_nrows * hh), [=](sycl::id<1> i) {
                const size_t j = i[0] / hh, h = i[0] % hh;
                o[(size_t) rr[j] * hh + h] += cp[i[0]];
            });
        }
        cev.wait();
    } else {
        bool first = true;
        for (auto& cp : im.cards) {
            const XeCard& c = *cp;
            if (!c.busy) continue;
            if (first) { std::memcpy(out_host, c.cOUT_host, NE * 4); first = false; }
            else for (size_t i = 0; i < NE; ++i) out_host[i] += c.cOUT_host[i];
        }
        if (first) std::memset(out_host, 0, NE * 4);
        if (!light.empty() && im.lt_nrows >= 0) {   // larry, STRATA_XE_LT_COMPACT
            for (int j = 0; j < im.lt_nrows; ++j) {
                float* o = out_host + (size_t) im.lt_rows[j] * H;
                const float* p = im.lt_hs + (size_t) j * H;
                for (int64_t h = 0; h < H; ++h) o[h] += p[h];
            }
        } else if (!light.empty()) for (size_t i = 0; i < NE; ++i) out_host[i] += im.lt_hs[i];
    }
    if (const char* dp = std::getenv("STRATA_XE_DUMP_CHUNK")) {   // debug: every chunk call's output (per card too)
        static std::FILE* df = std::fopen(dp, "wb");
        if (df) {
            std::vector<float> hv(NE);
            for (auto& cp : im.cards) {
                if (!cp->busy) continue;
                cp->q.memcpy(hv.data(), cp->cOUT, NE * 4).wait();
                const int32_t hdr[3] = {(int32_t) l, cp->idx, (int32_t) n};
                std::fwrite(hdr, 4, 3, df);
                std::fwrite(hv.data(), 4, NE, df);
            }
            std::fflush(df);
        }
    }
    {   // larry, STRATA_XE_PF_TRACE: the call's wall split
        static const bool ct = env_i("STRATA_XE_PF_TRACE", 0) != 0;
        if (ct)
            std::fprintf(stderr, "call trace: layer %lld n %d: %.1f ms = prefetch %.1f + plan %.1f + lock %.1f + prep %.1f + cards %.1f + "
                         "combine %.1f\n", (long long) l, n, now_ms() - t0, t_pf - t0, t_plan - t_pf, t_lk - t_plan, t_prep - t_lk,
                         t_cards - t_prep, now_ms() - t_cards);
    }
    if (call_tl) {   // larry, STRATA_XE_CALL_TL
        std::string ln;
        char b[320];
        std::snprintf(b, sizeof b, "call tl: layer %lld n %d: prep %.1f cards %.1f end %.1f light %zu %.1f |", (long long) l, n,
                      t_prep - t0, t_cards - t0, now_ms() - t0, light.size(), tl_light);
        ln += b;
        for (size_t k = 0; k < NC; ++k) {
            const CallTl& q = ctl[k];
            if (q.s == 0 && q.e == 0) continue;
            std::snprintf(b, sizeof b, " c%zu[s %.1f st %.1f (%lld e %lld pin %.0f MB) A %.1f G %.1f B %.1f E %.1f; g %lld r %lld j %lld]", k,
                          q.s, q.st, (long long) q.nst, (long long) q.npin, q.stb / 1e6, q.a, q.g, q.b, q.e, (long long) q.ng,
                          (long long) q.rows, (long long) q.nj);
            ln += b;
        }
        std::fprintf(stderr, "%s\n", ln.c_str());
    }
    if (prof) {
        g_bp.prep += t_prep - t0;
        g_bp.prep_pf += t_pf - t0;
        g_bp.prep_plan += t_plan - t_pf;
        g_bp.prep_lock += t_lk - t_plan;
        g_bp.cards += t_cards - t_prep;
        g_bp.combine += now_ms() - t_cards;
        g_bp.total += now_ms() - t0;
        if (++g_bp.calls % 40 == 0)
            std::fprintf(stderr, "xe prof (%lld chunk calls, last n %d): wall %.0f ms = prep %.0f (prefetch %.0f, plan %.0f, "
                         "lock %.0f) + cards %.0f + combine %.0f; "
                         "card-busy sums: stream %.0f (%lld experts, %lld kept), gemm %.0f (%lld experts), small %.0f (%lld jobs)\n",
                         (long long) g_bp.calls, n, g_bp.total, g_bp.prep, g_bp.prep_pf, g_bp.prep_plan, g_bp.prep_lock, g_bp.cards,
                         g_bp.combine, g_bp.stream_us / 1000.0,
                         (long long) g_bp.streamed.load(), (long long) g_bp.kept.load(), g_bp.gemm_us / 1000.0, (long long) g_bp.gemm_e.load(),
                         g_bp.small_us / 1000.0, (long long) g_bp.small_j.load());
        if (prof && g_bp.calls % 40 == 0) {
            std::fprintf(stderr, "xe prof: prep enqueues: buffers %.0f ms, X copy %.0f, W copy %.0f, memset %.0f\n", g_bp.prep_grow,
                         g_bp.prep_x, g_bp.prep_w, g_bp.prep_z);
            std::fprintf(stderr, "xe prof: lowest free VRAM after a call:");
            for (int k = 0; k < 4; ++k) std::fprintf(stderr, " %.2f", g_bp.min_free[k].load() / kGiB);
            std::fprintf(stderr, " GiB\n");
            std::fprintf(stderr, "xe prof: submit-lock holds: GEMM experts %.0f ms (%lld), job batches %.0f ms\n",
                         g_bp.lk_gemm_us / 1000.0, (long long) g_bp.lk_n.load(), g_bp.lk_jobs_us / 1000.0);
            std::fprintf(stderr, "xe prof: prefetch: staging %.1f GB in %.0f thread-ms (%.1f GB/s a thread), enqueue waits %.0f ms; "
                         "%lld layers' copies %.1f GB, enqueue -> done %.0f ms total (%.1f GB/s), run_big waited %.0f ms\n",
                         g_bp.st_bytes / 1e9, g_bp.st_us / 1000.0, g_bp.st_us ? g_bp.st_bytes / 1e3 / g_bp.st_us : 0.0,
                         g_bp.pfw_us / 1000.0, (long long) g_bp.dma_n.load(), g_bp.dma_bytes / 1e9, g_bp.dma_us / 1000.0,
                         g_bp.dma_us ? g_bp.dma_bytes / 1e3 / g_bp.dma_us : 0.0, g_bp.dma_wait_us / 1000.0);
        }
    }
    im.st.wall_ms += now_ms() - t0;
    if (lock_prof_on()) {   // larry, STRATA_XE_LOCK_PROF
        static int64_t lpn = 0;
        if (++lpn % 400 == 0) { std::lock_guard<PrioMutex> lk(g_submit); lock_prof_dump("400 prompt calls"); }
    }
    return true;
}

}  // namespace

namespace {
// STRATA_TP_OPROJ: one layer's parts (lazily, at its first decode call: no card is armed while an attention graph queues)
Ds41XeImpl::OSplit osplit_init(Ds41XeImpl& im, sycl::queue& q0, const void* wo_a, size_t sb, const void* wo_b, int64_t OGD,
                               int64_t OL, int64_t OG, int64_t D) {
    Ds41XeImpl::OSplit s;
    std::vector<XeCard*> parts;
    for (auto& cp : im.cards) if (!cp->dense) parts.push_back(cp.get());
    const int np = (int) std::min<size_t>(parts.size(), 7) + 1;
    if (np < 2 || OG % np) return s;
    const int64_t gp = OG / np, KB = OL * gp, KA = OL * OG;
    if (KB % 256 || OGD % 256) return s;
    s.np = np;
    for (int i = 0; i <= np; ++i) s.g0[i] = gp * i;
    if (!im.os_xq) {
        im.os_xq = sycl::malloc_device<int8_t>((size_t) 8 * OGD * OG, q0);
        im.os_xd = sycl::malloc_device<float>((size_t) 8 * OGD * OG / 32, q0);
        im.os_oa = sycl::malloc_device<float>((size_t) 8 * KB, q0);
        im.os_aq = sycl::malloc_device<int8_t>((size_t) 8 * KB, q0);
        im.os_ad = sycl::malloc_device<float>((size_t) 8 * KB / 32, q0);
        im.os_slots = sycl::malloc_device<float>((size_t) 8 * (np - 1) * D, q0);
        if (!im.os_xq || !im.os_xd || !im.os_oa || !im.os_aq || !im.os_ad || !im.os_slots) {
            std::fprintf(stderr, "xe tier: oproj split buffers failed - not split\n");
            return Ds41XeImpl::OSplit();
        }
    }
    // wo_b's planar planes hold block (row r, b) at r * bpr + b: part i's columns are each row's blocks [i bpp, +bpp)
    const int64_t bpr = KA / 256, nb = D * bpr, bpp = KB / 256, nbp = D * bpp;
    const size_t po[4] = {0, 128, 192, 208}, pb[4] = {128, 64, 16, 2};
    const size_t wb_bytes = (size_t) nbp * 210, wa_bytes = (size_t) gp * sb;
    auto gather = [&](int i, uint8_t* out) {   // on the dense card: part i's columns of the four planes into out
        for (int p = 0; p < 4; ++p) {
            const uint8_t* sp = (const uint8_t*) wo_b + po[p] * (size_t) nb + pb[p] * (size_t) (i * bpp);
            uint8_t* dp = out + po[p] * (size_t) nbp;
            const size_t w = pb[p] * (size_t) bpp, spitch = pb[p] * (size_t) bpr;
            q0.parallel_for(sycl::range<1>((size_t) D * w), [=](sycl::id<1> id) {
                const size_t r = id[0] / w, b = id[0] % w;
                dp[r * w + b] = sp[r * spitch + b];
            });
        }
    };
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    if (!dc) return Ds41XeImpl::OSplit();
    uint8_t* wb0 = local_alloc(q0, wb_bytes);   // the dense card's columns: read by its own kernels only
    uint8_t* stage = sycl::malloc_device<uint8_t>(wb_bytes, q0);   // the expert cards' columns, one card at a time
    if (!wb0 || !stage) {
        if (wb0) local_free(q0, wb0);
        if (stage) sycl::free(stage, q0);
        std::fprintf(stderr, "xe tier: oproj split: no room on the dense card - not split\n");
        return Ds41XeImpl::OSplit();
    }
    gather(0, wb0);
    s.wa.push_back((uint8_t*) wo_a);
    s.wb.push_back(wb0);
    s.xq.push_back(nullptr); s.aq.push_back(nullptr);
    s.xd.push_back(nullptr); s.oa.push_back(nullptr); s.ad.push_back(nullptr); s.y.push_back(nullptr);
    bool ok = true;
    for (int i = 1; i < np && ok; ++i) {
        XeCard& c = *parts[(size_t) i - 1];
        const size_t scratch = (size_t) 8 * gp * OGD * 5 / 4 + (size_t) 8 * KB * 6 + (size_t) 8 * D * 4 + ((size_t) 1 << 20);
        if (c.free_vram() > 0 && c.free_vram() < (int64_t) (wa_bytes + wb_bytes + scratch + ((size_t) 768 << 20))) {
            std::fprintf(stderr, "xe tier: no room on card %d for the oproj split (%zu MB) - not split\n", c.idx,
                         (wa_bytes + wb_bytes) >> 20);
            ok = false;
            break;
        }
        // its weights only its kernels read (no UR peer residency); its scratch is touched by its own engines only
        uint8_t* wa = local_alloc(c.q, wa_bytes);
        uint8_t* wb = local_alloc(c.q, wb_bytes);
        int8_t* xq = sycl::malloc_device<int8_t>((size_t) 8 * gp * OGD, c.q);
        float* xd = sycl::malloc_device<float>((size_t) 8 * gp * OGD / 32, c.q);
        float* oa = sycl::malloc_device<float>((size_t) 8 * KB, c.q);
        int8_t* aq = sycl::malloc_device<int8_t>((size_t) 8 * KB, c.q);
        float* ad = sycl::malloc_device<float>((size_t) 8 * KB / 32, c.q);
        float* y = sycl::malloc_device<float>((size_t) 8 * D, c.q);
        if (!wa || !wb || !xq || !xd || !oa || !aq || !ad || !y) {
            std::fprintf(stderr, "xe tier: oproj split allocation on card %d failed - not split\n", c.idx);
            ok = false;
            break;
        }
        c.q.memcpy(wa, (const uint8_t*) wo_a + (size_t) (i * gp) * sb, wa_bytes);   // its slices: whole plane sets
        gather(i, stage);
        q0.wait();
        c.q.memcpy(wb, stage, wb_bytes).wait();
        s.c.push_back(&c);
        s.wa.push_back(wa); s.wb.push_back(wb);
        s.xq.push_back(xq); s.aq.push_back(aq);
        s.xd.push_back(xd); s.oa.push_back(oa); s.ad.push_back(ad); s.y.push_back(y);
    }
    q0.wait();
    sycl::free(stage, q0);
    if (!ok) return Ds41XeImpl::OSplit();   // (what was allocated stays: a few layers' worth at most, then nothing splits)
    s.g.resize((size_t) np);
    s.ok = true;
    static int told = 0;
    if (!told++) std::fprintf(stderr, "xe tier: attention output split over %d cards (%lld groups each; %.0f MB a card a layer)\n", np,
                              (long long) gp, (double) (wa_bytes + wb_bytes) / 1e6);
    return s;
}

// a Q6K_R8 product of G slices / one matrix for ncols columns (quant_x's int8 rows: batch-major, g * ncols + c)
void os_mv(sycl::queue& q, const void* w, int64_t K, int64_t N, int64_t G, size_t batch_bytes, int n, const int8_t* xq,
           const float* xd, float* y, int64_t y_col_stride, int64_t y_batch_stride) {
    fastmv::Args a;
    a.w = w; a.K = K; a.N = N; a.G = G; a.g_div = 1; a.n_blocks_total = N * (K / 256); a.w_batch_bytes = batch_bytes;
    a.x_col_stride = K; a.x_batch_stride = K * n;
    a.y = y; a.y_col_stride = y_col_stride; a.y_batch_stride = y_batch_stride; a.ncols = n;
    a.xq = xq; a.xd = xd;
    if (!fastmv::launch(q, fastmv::WType::Q6K_R8, a)) throw std::runtime_error("oproj split: fastmv shape not covered");
}
}  // namespace

bool Ds41XeTier::oproj_hook(void* queue, const void* wo_a, size_t sb, const void* wo_b, int64_t OGD, int64_t OL, int64_t OG,
                            int64_t D, int ncols, const float* o, int64_t ocs, float* dst, int64_t dcs, void* user) {
    Ds41XeImpl& im = *static_cast<Ds41XeTier*>(user)->im_;
    if (!queue || ncols < 1 || ncols > 8 || im.cards.empty()) return false;
    sycl::queue& q0 = *static_cast<sycl::queue*>(queue);
    // a SYCL graph recording (GGML_SYCL_GRAPH_CACHE) would capture this queue's part only: the dense card computes it all
    if (q0.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording) { ++im.os_declined; return false; }
    try {
        auto it = im.osplits.find(wo_a);
        if (it == im.osplits.end()) it = im.osplits.emplace(wo_a, osplit_init(im, q0, wo_a, sb, wo_b, OGD, OL, OG, D)).first;
        Ds41XeImpl::OSplit& s = it->second;
        if (!s.ok) { ++im.os_declined; return false; }
        const int n = ncols;
        const int64_t gp = s.g0[1], KB = OL * gp;
        static const bool tp_prof = env_i("STRATA_TP_PROF", 0) != 0;
        static double tpp[6] = {};   // lock, quant_x, cards (sum over cards), own quarter, sum, calls
        double tq = tp_prof ? now_ms() : 0;
        auto tick = [&](int k) { if (tp_prof) { const double t = now_ms(); tpp[k] += t - tq; tq = t; } };
        std::lock_guard<PrioMutex> lk(g_submit);
        tick(0);
        const bool tl = Ds41XeTier::tl_on();
        if (tl) tl_stamp(im, 0, q0, 5);
        const sycl::event e0 = fastmv::quant_x(q0, o, ocs, OGD, n, OG, OGD, im.os_xq, im.os_xd);   // all groups
        tick(1);
        std::vector<sycl::event> ev;
        static const bool rec = env_i("STRATA_TP_GRAPH", 1) != 0;
        for (int i = 1; i < s.np; ++i) {
            XeCard& c = *s.c[(size_t) i - 1];
            const size_t r0 = (size_t) s.g0[i] * n;   // its groups' rows of the quantised o
            int8_t* xq = s.xq[(size_t) i];
            const int8_t* xq_src = im.os_xq + r0 * OGD;
            // its copies of its groups' int8 o, both products, its partial sum to the dense card (the first copy waits
            // for the dense card's quant_x unless recorded: a replay carries that dependency instead)
            auto enqueue = [&](const sycl::event* dep) {
                if (dep) c.q.submit([&](sycl::handler& h) { h.depends_on(*dep); h.memcpy(xq, xq_src, (size_t) gp * n * OGD); });
                else c.q.memcpy(xq, xq_src, (size_t) gp * n * OGD);
                c.q.memcpy(s.xd[(size_t) i], im.os_xd + r0 * (OGD / 32), (size_t) gp * n * (OGD / 32) * sizeof(float));
                os_mv(c.q, s.wa[(size_t) i], OGD, OL, gp, sb, n, xq, s.xd[(size_t) i], s.oa[(size_t) i], KB, OL);
                fastmv::quant_x(c.q, s.oa[(size_t) i], KB, KB, n, 1, KB, s.aq[(size_t) i], s.ad[(size_t) i]);
                os_mv(c.q, s.wb[(size_t) i], KB, D, 1, 0, n, s.aq[(size_t) i], s.ad[(size_t) i], s.y[(size_t) i], D, D * n);
                return c.q.memcpy(im.os_slots + (size_t) (i - 1) * 8 * D, s.y[(size_t) i], (size_t) n * D * sizeof(float));
            };
            if (rec) {
                namespace sx = sycl::ext::oneapi::experimental;
                auto& eg = s.g[(size_t) i][(size_t) n];
                if (!eg) {
                    sx::command_graph<sx::graph_state::modifiable> g(c.q.get_context(), c.q.get_device());
                    g.begin_recording(c.q);
                    enqueue(nullptr);
                    g.end_recording();
                    eg = std::make_shared<sx::command_graph<sx::graph_state::executable>>(g.finalize());
                }
                sx::command_graph<sx::graph_state::executable>& gr = *eg;
                if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 20);
                ev.push_back(c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.ext_oneapi_graph(gr); }));
                if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 21);
            } else {
                if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 20);
                ev.push_back(enqueue(&e0));
                if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 21);
            }
        }
        tick(2);
        os_mv(q0, s.wa[0], OGD, OL, gp, sb, n, im.os_xq, im.os_xd, im.os_oa, KB, OL);   // its groups: o's first rows
        fastmv::quant_x(q0, im.os_oa, KB, KB, n, 1, KB, im.os_aq, im.os_ad);
        os_mv(q0, s.wb[0], KB, D, 1, 0, n, im.os_aq, im.os_ad, dst, dcs, dcs * n);
        if (tl) tl_stamp(im, 0, q0, 6);
        tick(3);
        const float* sl = im.os_slots;
        const int nparts = s.np - 1;
        q0.submit([&](sycl::handler& h) {   // dst += the expert cards' partial sums
            h.depends_on(ev);
            h.parallel_for(sycl::range<1>((size_t) n * D), [=](sycl::id<1> id) {
                const int64_t c = (int64_t) id[0] / D, r = (int64_t) id[0] % D;
                float v = dst[c * dcs + r];
                for (int k = 0; k < nparts; ++k) v += sl[(size_t) k * 8 * D + c * D + r];
                dst[c * dcs + r] = v;
            });
        });
        if (tl) tl_stamp(im, 0, q0, 7);
        tick(4);
        if (tp_prof && ++tpp[5] >= 1 && (long) tpp[5] % 4000 == 0)
            std::fprintf(stderr, "xe tp prof (%.0f calls): host us a call - lock %.1f, quant_x %.1f, cards %.1f, own quarter %.1f, sum %.1f\n",
                         tpp[5], 1e3 * tpp[0] / tpp[5], 1e3 * tpp[1] / tpp[5], 1e3 * tpp[2] / tpp[5], 1e3 * tpp[3] / tpp[5], 1e3 * tpp[4] / tpp[5]);
        ++im.os_calls;
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xe tier: oproj split: %s\n", e.what());
        return false;
    }
}

namespace {
/// STRATA_XE_ARM: card c's share of a coming call, queued ahead of its routing - a spin until its release (bounded),
/// its ng jobs (from the host's stage, or STRATA_XE_GROUTE: pulled from the dense card's route list), the experts on
/// the call's X (arm_x: pinned, read over PCIe), the part's copy into its slot on the dense card.  Under g_submit.
void arm_card(Ds41XeImpl& im, XeCard& c, int ng, int nt, int mv, const sycl::event& prev, const XeJob* jobs_src) {
    namespace sx = sycl::ext::oneapi::experimental;
    const int64_t H = im.g.n_embd;
    const size_t xoff = (kMaxDecodeJobs * sizeof(XeJob) + kStageAlign - 1) / kStageAlign * kStageAlign;
    float* slots = im.cards[(size_t) im.dense]->dslots;
    int* errp = im.arm_err;
    int* flag = c.arm_flag;
    int* seen = c.arm_seen;
    const bool tl = Ds41XeTier::tl_on();
    // STRATA_XE_GROUTE (jobs_src): the release comes from the dense card behind its route kernel, which its queue runs
    // after the previous call's sum - the slot this writes is free by then, no wait on `prev` (a cross-card dependency
    // costs the host ~10 us a submission)
    auto seq = [&](bool dep) {
        c.q.submit([&](sycl::handler& h) {
            if (dep) h.depends_on(prev);
            h.single_task([=] {   // until the host's release count passes the consumed one (bounded: a few seconds)
                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space> f(*flag);
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> sn(*seen);
                const int want = sn.load() + 1;
                int k = 0;
                for (; k < 0x7fffffff && f.load(sycl::memory_order::acquire) < want; ++k) {}
                if (k == 0x7fffffff)
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                     sycl::access::address_space::global_space>(*errp).store(1, sycl::memory_order::release);
                sn.store(want);
            });
        });
        if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 10);   // released: the card starts
        if (jobs_src) c.q.memcpy(c.stage_dev, jobs_src, (size_t) ng * sizeof(XeJob));   // card-to-card (the dense card's list)
        else c.q.memcpy(c.stage_dev, c.stage_host, (size_t) ng * sizeof(XeJob));
        card_experts(im, c, ng, nt, mv, xoff, c.OUT, im.arm_x);
        if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 11);
        sycl::event ev = c.q.memcpy(slots + (size_t) c.idx * kMaxJobTok * H, c.OUT, (size_t) nt * H * 4);
        if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 14);   // its copy is done
        return ev;
    };
    // STRATA_XE_ARM_GRAPH=1 (routed, no timeline): the sequence recorded once per shape and replayed - one submission
    static const bool agraph = env_i("STRATA_XE_ARM_GRAPH", 0) != 0;
    if (jobs_src && agraph && !tl && im.arm_x) {
        const uint32_t key = (uint32_t) ng | ((uint32_t) nt << 8) | ((uint32_t) mv << 12);
        auto it = c.agraphs.find(key);
        // X is baked into the recording (the job source and the slot are fixed): a different X - the dense half's
        // readback mirror reallocated by set_columns - records afresh
        if (it != c.agraphs.end() && c.agraph_x[key] != im.arm_x) { c.agraphs.erase(it); it = c.agraphs.end(); }
        if (it == c.agraphs.end()) {
            c.agraph_x[key] = im.arm_x;
            sx::command_graph<sx::graph_state::modifiable> g(c.q.get_context(), c.q.get_device());
            g.begin_recording(c.q);
            seq(false);
            g.end_recording();
            it = c.agraphs.emplace(key, std::make_unique<sx::command_graph<sx::graph_state::executable>>(g.finalize())).first;
        }
        sx::command_graph<sx::graph_state::executable>& eg = *it->second;
        c.arm_ev = c.q.submit([&](sycl::handler& h) { h.ext_oneapi_graph(eg); });
    } else {
        c.arm_ev = seq(jobs_src == nullptr);
    }
    c.armed = true;
    c.arm_routed = jobs_src != nullptr;
    c.arm_ng = ng;
}

/// The dense card's route kernel for a call of nt tokens (ids / wts: its router's [nt][K] on the device): run_small's
/// group() + card assignment.  One work-group, a work-item per routing slot (token-major): an expert's first slot builds
/// its entry (its tokens = its slots in order, one a token) and writes it as a job of its card - index = the earlier
/// entries of that card, the host's order - or (on no card / a card not taking jobs) as a CPU-pool entry (pinned);
/// then every taking card's list is null-padded to ng, and the counts written.
sycl::event route_kernel(sycl::queue& q, const int32_t* ids, const float* wts, int nt, int K, int64_t E, int64_t l,
                         const RtEnt* rt, XeJob* jobs, const XeJob* nulls, RtHdr* hdr, int ng, uint32_t mask, int NC) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> se(sycl::range<1>(64), h);   // slot -> expert (-1 none)
        sycl::local_accessor<int32_t, 1> sc(sycl::range<1>(64), h);   // slot -> entry's card (-1 CPU pool, -2 no entry)
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(64), sycl::range<1>(64)), [=](sycl::nd_item<1> it) {
            const int i = (int) it.get_local_id(0);
            const int n = nt * K;
            int e = -1;
            if (i < n) {
                e = ids[i];
                if (e < 0 || e >= E) e = -1;
            }
            se[i] = e;
            sycl::group_barrier(it.get_group());
            bool first = e >= 0;
            for (int j = 0; j < i && first; ++j) if (se[j] == e) first = false;
            RtEnt re{};
            int card = -2;
            if (first) {
                re = rt[l * E + e];
                card = re.card >= 0 && re.card < NC && ((mask >> re.card) & 1u) ? re.card : -1;
            }
            sc[i] = card;
            sycl::group_barrier(it.get_group());
            if (first) {
                int idx = 0;
                for (int j = 0; j < i; ++j) if (sc[j] == card) ++idx;
                int ntk = 0;
                int32_t tok[kMaxJobTok], kk[kMaxJobTok];
                float w[kMaxJobTok];
                for (int s = 0; s < kMaxJobTok; ++s) { tok[s] = 0; kk[s] = 0; w[s] = 0.f; }
                for (int j = i; j < n && ntk < kMaxJobTok; ++j)
                    if (se[j] == e) { tok[ntk] = j / K; kk[ntk] = j; w[ntk] = wts[j]; ++ntk; }
                if (card >= 0) {
                    XeJob jb;
                    jb.blob = re.blob;
                    jb.up_off = re.up_off;
                    jb.down_off = re.down_off;
                    jb.gu_type = re.gu_type;
                    jb.d_type = re.d_type;
                    jb.gu_row = re.gu_row;
                    jb.d_row = re.d_row;
                    jb.nt = (uint8_t) ntk;
                    for (int s = 0; s < kMaxJobTok; ++s) { jb.tok[s] = tok[s]; jb.w[s] = w[s]; }
                    jobs[(size_t) card * kMaxDecodeJobs + (size_t) idx] = jb;
                } else {
                    RtCpu& c = hdr->cpu[idx];
                    c.e = e;
                    c.nt = ntk;
                    for (int s = 0; s < kMaxJobTok; ++s) { c.tok[s] = tok[s]; c.k[s] = kk[s]; c.w[s] = w[s]; }
                }
            }
            if (i < NC && ((mask >> i) & 1u)) {   // card i's list, null-padded
                int cnt = 0;
                for (int j = 0; j < n; ++j) cnt += sc[j] == i;
                for (int k = cnt; k < ng; ++k) jobs[(size_t) i * kMaxDecodeJobs + (size_t) k] = nulls[i];
            }
            if (i == 0) {
                int ncard = 0, ncpu = 0;
                for (int j = 0; j < n; ++j) { ncard += sc[j] >= 0; ncpu += sc[j] == -1; }
                hdr->ncard = ncard;
                hdr->ncpu = ncpu;
            }
        });
    });
}
}  // namespace

void Ds41XeTier::arm(int64_t l, int nt) {
    Ds41XeImpl& im = *im_;
    if (im.routed_l >= 0 && im.routed_l == l && im.routed_nt == nt) {
        // STRATA_XE_GROUTE: route() armed and released the expert cards; the dense card's own share now, on the dense
        // queue behind the shared expert (its jobs: the route kernel's list)
        XeCard& dc = *im.cards[(size_t) im.dense];
        std::lock_guard<PrioMutex> lk(g_submit);
        im.rt_dense_ev = card_experts_at(im, dc, *im.dense_q, im.rt_jobs + (size_t) dc.idx * kMaxDecodeJobs, im.routed_ng, nt,
                                         im.routed_mv, im.rt_x, dc.dslots + (size_t) dc.idx * kMaxJobTok * im.g.n_embd);
        im.rt_dense_armed = true;
        return;
    }
    disarm(im);   // an earlier arm nobody took
    if (!im.arm_on || !im.relq || !im.dense_q || !im.arm_x || nt < 1 || nt > kMaxJobTok || (nt > 1 && !im.arm_multi)) return;
    if (im.arm_err && *(volatile int*) im.arm_err) {   // a spin kernel timed out once: arming stays off
        std::fprintf(stderr, "xe tier: an armed card waited out its release - STRATA_XE_ARM off\n");
        im.arm_on = im.arm_multi = false;
        return;
    }
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    if (!dc || !dc->dslots) return;
    const int K = (int) im.g.top_k;
    static const int kb[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, kMaxDecodeJobs};
    int ng = kMaxDecodeJobs;
    for (int b : kb) if (b >= K * nt) { ng = b; break; }
    const int mv = nt <= 1 ? 1 : nt <= 2 ? 2 : nt <= 4 ? 4 : 8;
    const sycl::event prev = im.last_sum;   // the previous call's sum reads the slots these sequences write
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.arm_flag || !c.arm_null_ok) continue;
        std::lock_guard<PrioMutex> lk(g_submit);
        arm_card(im, c, ng, nt, mv, prev, nullptr);
    }
    im.armed_l = l;
    im.armed_nt = nt;
}

void Ds41XeTier::groute_next() { im_->groute_want = im_->groute_on; }

void Ds41XeTier::route(int64_t l, int nt, const void* ids_dev, const void* wts_dev, const void* fn_dev, const float* fn_host) {
    Ds41XeImpl& im = *im_;
    const bool want = im.groute_want;
    im.groute_want = false;
    if (!want || !im.groute_on) return;
    if (!im.arm_on || !im.relq || !im.dense_q || !fn_host || !ids_dev || !wts_dev || !fn_dev || nt < 1 || nt > kMaxJobTok ||
        (nt > 1 && !im.arm_multi))
        return;   // the host's dispatch (arm(), run_small)
    {   // the expert cards read this layer's X straight from the fn readback's mirror: pinned host memory only
        static const float* chk_p = nullptr;
        static bool chk_ok = false;
        if (fn_host != chk_p) { chk_p = fn_host; chk_ok = sycl::get_pointer_type(fn_host, im.ctx) == sycl::usm::alloc::host; }
        if (!chk_ok) return;
    }
    if (im.arm_err && *(volatile int*) im.arm_err) return;   // arm() turns arming off (and says so)
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    const int K = (int) im.g.top_k;
    if (!dc || !dc->dslots || nt * K > 64) return;   // (the route kernel: a work-item a routing slot, one work-group)
    disarm(im);   // an earlier arm nobody took
    im.arm_x = fn_host;   // what the armed sequences read, and what run_small's take test compares the call's x with
    static const int kb[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, kMaxDecodeJobs};
    int ng = kMaxDecodeJobs;
    for (int b : kb) if (b >= K * nt) { ng = b; break; }
    const int mv = nt <= 1 ? 1 : nt <= 2 ? 2 : nt <= 4 ? 4 : 8;
    const sycl::event prev = im.last_sum;
    uint32_t mask = 1u << dc->idx;
    std::lock_guard<PrioMutex> lk(g_submit);
    // the expert cards' sequences (each spins until its release below, then pulls its list card-to-card)
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.arm_flag || !c.arm_null_ok) continue;
        arm_card(im, c, ng, nt, mv, prev, im.rt_jobs + (size_t) c.idx * kMaxDecodeJobs);
        mask |= 1u << c.idx;
    }
    // the route kernel on the dense queue (behind the fn readback the expert cards read), then the releases on the
    // release queue once it is done (the dense queue goes on with the readbacks and the shared expert meanwhile)
    sycl::queue& qd = *im.dense_q;
    const sycl::event rev = route_kernel(qd, (const int32_t*) ids_dev, (const float*) wts_dev, nt, K, im.g.n_experts, l, im.rt,
                                         im.rt_jobs, im.rt_null, im.rt_hdr, ng, mask, (int) im.cards.size());
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.armed || !c.arm_routed) continue;
        int* srcp = c.arm_src + 16 * (c.arm_src_i++ & 63);
        *srcp = ++c.arm_rel;
        im.relq->submit([&](sycl::handler& h) {
            h.depends_on(rev);
            h.memcpy(c.arm_flag, srcp, sizeof(int));
        });
    }
    im.routed_l = l;
    im.routed_nt = nt;
    im.routed_ng = ng;
    im.routed_mv = mv;
    im.routed_mask = mask;
    im.rt_x = (const float*) fn_dev;
    im.rt_dense_armed = false;
    im.armed_l = l;
    im.armed_nt = nt;
}

const void* Ds41XeTier::tp_route_table(const volatile uint64_t** version) {
    Ds41XeImpl& im = *im_;
    if (!im.rt) {
        const size_t NK = im.card_of.size();
        im.rt = sycl::malloc_host<RtEnt>(NK, im.ctx);
        if (!im.rt) return nullptr;
        for (size_t k = 0; k < NK; ++k) { im.rt[k] = RtEnt{}; im.rt[k].card = -1; rt_put(im, k); }
    }
    if (version) *version = &im.rt_version;
    return im.rt;
}

void Ds41XeTier::tp_between_passes(int chunks) {
    Ds41XeImpl& im = *im_;
    if (im.cpu_log) std::fprintf(im.cpu_log, "B %lld %.3f\n", (long long) im.tp_pass, now_ms());   // the pass is back
    ++im.tp_pass;
    if (im.pc_mode < 0) im.pc_mode = env_i("STRATA_XE_PROMOTE", 1) != 0 && im.planar && im.dense >= 0 ? 1 : 0;
    if (im.pc_mode <= 0) return;
    ++im.pc_clock;
    // between passes nothing races the copies: several promotions a card in flight (STRATA_TP_PROMOTE_PAR) - set here
    // every time: wider passes (DSpark's 16-token seed) go the tier's own way, whose pc_setup runs first
    static const int par = std::max(1, env_i("STRATA_TP_PROMOTE_PAR", 8));
    im.pc_par = std::min(par, im.pc_nb);
    if (!im.pc_ready && !im.pc_tried && !im.pf_on && ++im.pc_calls >= env_i("STRATA_XE_PROMOTE_AFTER", 1)) {
        im.pc_tried = true;
        pc_setup(im);
    }
    pc_poll(im, false);   // copies done (in order behind the last pass): published - their slots free for new wants
    // larry, patch-regrowasync: no new promotion while a regrow is pending (its experts are arena experts until published)
    for (const auto& lk : im.tp_want) if (im.pc_ready && !im.rg_pending) pc_want(im, lk.first, lk.second);
    im.tp_want.clear();
    // early bounces no card took (every card at its in-flight cap): given back once filled (the helper is done with them)
    for (auto it = im.pc_jobs.begin(); it != im.pc_jobs.end();) {
        if (it->card < 0 && it->filled.load(std::memory_order_acquire)) {
            im.pc_bfree.push_back(it->bounce);
            im.pc_where[(size_t) it->key] = -1;
            it = im.pc_jobs.erase(it);
        } else {
            ++it;
        }
    }
    if (env_i("STRATA_TP_PC_DEBUG", 0) && im.pc_clock % 32 == 0) {   // the pipeline's state
        int st[4] = {0, 0, 0, 0};
        for (auto& j : im.pc_jobs) ++st[std::min(3, std::max(0, j.stage))];
        std::string b;
        for (auto& cp : im.cards) b += " c" + std::to_string(cp->idx) + ":" + std::to_string(cp->pc_busy) + "/" + std::to_string(cp->pc_n);
        std::fprintf(stderr, "tp pc (clock %llu): jobs stage0 %d stage1 %d stage3 %d, bounces free %zu of %d, busy%s, wants %zu, done %lld skipped %lld\n",
                     (unsigned long long) im.pc_clock, st[0], st[1], st[3], im.pc_bfree.size(), im.pc_nb, b.c_str(), im.tp_want.size(),
                     (long long) im.pc_done, (long long) im.pc_busy_skip);
    }
    // copy chunks (1 MB, STRATA_XE_PROMOTE_CHUNK_MB) for the jobs whose bounces are filled, `chunks` in all a pass
    const int64_t c0 = im.pc_chunks;
    for (int k = 0; k < chunks && !im.pc_jobs.empty() && im.pc_chunks - c0 < chunks; ++k) {
        const int64_t b = im.pc_chunks;
        pc_poll(im, true);
        if (im.pc_chunks == b) break;   // nothing ready to push (bounces still being filled)
    }
}

void Ds41XeTier::tp_defer_vram(int gpu, size_t bytes) { im_->tp_defer[gpu] = bytes; }

void Ds41XeTier::dbg_capture(int slot, const void* src, size_t bytes) {
    Ds41XeImpl& im = *im_;
    if (!im.dense_q || !src || !bytes) return;
    auto& e = im.dbg[slot];
    if (e.second < bytes) {
        if (e.first) { im.dense_q->wait(); sycl::free(e.first, im.ctx); }
        e.first = sycl::malloc_host<uint8_t>(bytes, im.ctx);
        e.second = e.first ? bytes : 0;
    }
    if (e.first) im.dense_q->memcpy(e.first, src, bytes);
}

std::vector<uint8_t> Ds41XeTier::dbg_take(int slot) {
    Ds41XeImpl& im = *im_;
    auto it = im.dbg.find(slot);
    if (it == im.dbg.end() || !it->second.first || !im.dense_q) return {};
    im.dense_q->wait();
    return std::vector<uint8_t>(it->second.first, it->second.first + it->second.second);
}

void Ds41XeTier::tp_dense_bonus(size_t bytes) {
    im_->dense_bonus_gib = bytes / kGiB;
    std::fprintf(stderr, "xe tier: dense card's expert cap +%.2f GiB (the dense weights the TP cards hold, paged; engine1's draft)\n", bytes / kGiB);
}

void Ds41XeTier::tp_head_slice(const void* w, int gpu, int64_t r0, int64_t r1, const void* p) {
    if (w && p) im_->head_given[w].push_back({gpu, r0, r1, p});
}

void Ds41XeTier::tp_shexp_slice(int64_t l, int gpu, const void* gu, const void* dn, int64_t nf) {
    Ds41XeImpl& im = *im_;
    const int64_t H = im.g.n_embd;
    if (l < 0 || !gu || !dn || nf <= 0 || nf % 256 || H % 256) return;
    if ((size_t) l >= im.sh_slices.size()) im.sh_slices.resize((size_t) l + 1);
    XeCard* own = nullptr;
    for (auto& cp : im.cards) if (cp->gpu == gpu) own = cp.get();
    if (!own) return;
    if (!own->dense) { im.sh_slices[(size_t) l].push_back({own->idx, (const uint8_t*) gu, (const uint8_t*) dn, nf}); return; }
    // rank 0's slice: onto the first expert card, through the host (once, at load)
    XeCard* dst = nullptr;
    for (auto& cp : im.cards) if (!cp->dense) { dst = cp.get(); break; }
    if (!dst) return;
    const size_t bgu = (size_t) (2 * nf * H / 256) * 210, bdn = (size_t) (H * nf / 256) * 210;
    uint8_t* p = local_alloc(dst->q, bgu + bdn);
    if (!p) {
        std::fprintf(stderr, "xe tier: shared expert: no room on card %d for layer %lld's rank-0 slice\n", dst->idx, (long long) l);
        return;
    }
    std::vector<uint8_t> tmp(bgu + bdn);
    {
        std::lock_guard<PrioMutex> lk(g_submit);
        own->q.memcpy(tmp.data(), gu, bgu);
        own->q.memcpy(tmp.data() + bgu, dn, bdn);
        own->q.wait();
        dst->q.memcpy(p, tmp.data(), bgu + bdn).wait();
    }
    im.sh_own.emplace_back(dst, p);
    im.sh_slices[(size_t) l].push_back({dst->idx, p, p + bgu, nf});
}

bool Ds41XeTier::shexp_ready() const {
    const Ds41XeImpl& im = *im_;
    if ((int64_t) im.sh_slices.size() < im.g.n_layers) return false;
    for (const auto& v : im.sh_slices) if (v.empty()) return false;
    return true;
}

void Ds41XeTier::set_shexp_next(bool on) { im_->sh_next = on; }

bool Ds41XeTier::tp_make_room(int gpu, size_t bytes) {
    Ds41XeImpl& im = *im_;
    XeCard* cp = nullptr;
    for (auto& c : im.cards) if (c->gpu == gpu) cp = c.get();
    if (!cp || cp->dense) return true;   // not an expert card of the tier
    XeCard& c = *cp;
    // the rows fit beside what is free (STRATA_TP_ROOM_KEEP_GIB, 0.25, left over); else the flex region gives the
    // shortfall back - after a long prompt its chunk buffers hold the card's slack, as they did with eager replicas
    const int64_t keep = (int64_t) (env_d("STRATA_TP_ROOM_KEEP_GIB", 0.25) * kGiB);
    const int64_t fr = c.free_vram();
    auto tdl = im.tp_defer.find(gpu);
    if (tdl != im.tp_defer.end()) tdl->second = tdl->second > bytes ? tdl->second - bytes : 0;
    if (fr < 0) return true;
    int64_t give = (int64_t) bytes + keep - fr;
    // larry, STRATA_XE_TP_EVICT: the free VRAM down to STRATA_XE_TP_EVICT_KEEP_GIB (as the flex logic: 0.25 - the
    // run_big buffers kept after a prompt already hold its reserve; 1.5 evicted 570 experts for 36 MiB of rows), then
    // the card's evictable slabs, the flex region last
    if (tp_evict_on() && !c.pbuf_cap) {
        const int64_t ekeep = (int64_t) (env_d("STRATA_XE_TP_EVICT_KEEP_GIB", 0.25) * kGiB);
        if ((int64_t) bytes + ekeep > fr && evict_slabs(im, c, (int64_t) bytes + ekeep) > 0) give = (int64_t) bytes + keep - c.free_vram();
    }
    if (give <= 0) return true;
    if (!c.flex || c.pbuf_cap) return c.free_vram() > (int64_t) bytes;   // no flex region to give back (a prompt holds it)
    give = std::min<int64_t>(give, (int64_t) c.flex_cap);
    pc_drop(im);   // the promoted experts out (every card; the slots come back at the next promotion)
    c.q.wait();
    const size_t was = c.flex_cap;
    local_free(c.q, c.flex);
    const int64_t left = ((int64_t) was - give) & ~(int64_t) 255;
    c.flex = left >= 8 * (int64_t) im.blob_max ? card_alloc(c.q, (size_t) left) : nullptr;   // larry, patch-barstage
    c.flex_cap = c.flex ? (size_t) left : 0;
    std::fprintf(stderr, "xe tier: card %d's flex region %.2f -> %.2f GiB (the TP core's KV rows)\n", c.idx, was / kGiB,
                 c.flex_cap / kGiB);
    return c.free_vram() > (int64_t) bytes;
}

void Ds41XeTier::tp_set_promote_queues(const std::vector<void*>& qs) {
    Ds41XeImpl& im = *im_;
    im.pc_q.assign(im.cards.size(), nullptr);
    for (size_t k = 0; k < qs.size() && k < im.cards.size(); ++k) im.pc_q[k] = (sycl::queue*) qs[k];
}

bool Ds41XeTier::tp_take_dirty(std::vector<uint32_t>& keys) {
    Ds41XeImpl& im = *im_;
    // larry, engine1 (STRATA_TP_PC_PUBLISH, default 1): the TP core asks before every pass - the promotions whose copies
    // finished since (they run during the draft) are published first, so this pass's table has them
    static const bool publish = env_i("STRATA_TP_PC_PUBLISH", 1) != 0;
    if (publish && im.pc_mode > 0 && !im.pc_jobs.empty()) pc_poll(im, false);
    if (im.cpu_log) std::fprintf(im.cpu_log, "A %lld %.3f\n", (long long) im.tp_pass, now_ms());   // a pass starts
    const bool all = im.rt_dirty_all;
    keys.swap(im.rt_dirty);
    im.rt_dirty.clear();
    im.rt_dirty_all = false;
    return all;
}

int Ds41XeTier::tp_cpu_part(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, float* out) {
    Ds41XeImpl& im = *im_;
    static thread_local std::vector<Ent> ents;
    static thread_local std::vector<int32_t> first;
    static thread_local std::vector<const Ent*> cpu_list;
    static const char* log_path = std::getenv("STRATA_XE_CPU_LOG");
    if (log_path && !im.cpu_log) {
        im.cpu_log = std::fopen(log_path, "w");
        if (im.cpu_log) std::setvbuf(im.cpu_log, nullptr, _IOFBF, 1 << 20);
        log_path = nullptr;   // one try
    }
    if (im.cpu_log && layer == 0) std::fprintf(im.cpu_log, "T %lld %d\n", (long long) im.tp_pass, nt);
    group(ids, w, nt, (int) im.g.top_k, im.g.n_experts, ents, first);
    static const bool use_on = std::getenv("STRATA_XE_USAGE_OUT") != nullptr;
    if (use_on) {
        if (im.use_jobs.empty()) {   // the placement as decode begins (before any promotion)
            const size_t ne = (size_t) (im.g.n_layers * im.g.n_experts);
            im.use_jobs.assign(ne, 0);
            im.use_picks.assign(ne, 0);
            im.use_card.assign(im.card_of.begin(), im.card_of.begin() + (std::ptrdiff_t) ne);
        }
        for (const Ent& en : ents) {
            const size_t key = (size_t) im.key(layer, en.e);
            ++im.use_jobs[key];
            im.use_picks[key] += (uint32_t) en.nt;
        }
    }
    cpu_list.clear();
    for (const Ent& en : ents) {
        const size_t key = (size_t) im.key(layer, en.e);
        if (im.card_of[key] < 0) {
            cpu_list.push_back(&en);
            // promoted between passes (tp_between_passes): an eviction now would take an expert off the host's view
            // while the running pass's cards still hold it in their table (a later layer would count it twice)
            if (im.pc_ready) im.tp_want.emplace_back(layer, key);
        } else if (im.pc_ready && im.pc_where[key] >= 0) {   // a promoted expert the cards ran: its slot used now
            im.cards[(size_t) (im.pc_where[key] >> 16)]->pc_tick[(size_t) (im.pc_where[key] & 0xFFFF)] = im.pc_clock;
            ++im.pc_hit;
            if (im.cpu_log) std::fprintf(im.cpu_log, "H %lld %lld %d %d\n", (long long) im.tp_pass, (long long) layer, en.e, en.nt);
        }
    }
    if (cpu_list.empty()) return 0;
    std::memset(out, 0, (size_t) nt * (size_t) im.g.n_embd * sizeof(float));
    // larry, engine1 (STRATA_TP_PC_EARLY, default 1): the bounces of the experts the next tp_between_passes promotes fill
    // from now on (pc_early).  With STRATA_TP_PC_PUBLISH and 4 MB chunks: CPU-pool experts a verify pass 10.2-11.3 ->
    // 7.0-8.0, a DSpark cycle at the same pass widths -6 % (bench/engine1/tcpu3.sh .. tcpu5.sh, engine1/DESIGN.md)
    static const bool early = env_i("STRATA_TP_PC_EARLY", 1) != 0;
    if (!im.cpu_log) {
        im.pool_busy.store(1, std::memory_order_release);
        cpu_ents(im, layer, cpu_list, x, out);
        im.pool_busy.store(0, std::memory_order_release);
        if (early && im.pc_ready) pc_early(im, layer, cpu_list);
        return (int) cpu_list.size();
    }
    size_t bytes = 0;
    for (const Ent* en : cpu_list) {
        const size_t key = (size_t) im.key(layer, en->e);
        const int where = key < im.pc_where.size() ? im.pc_where[key] : -1;
        const int uses = key < im.pc_uses.size() ? im.pc_uses[key] : 0;
        const int pin = !im.aptr[key] ? 2 : key < im.apin.size() && im.apin[key] ? 1 : 0;
        bytes += im.bbytes(layer, en->e);
        std::fprintf(im.cpu_log, "E %lld %lld %d %d %d %d %d %d %zu\n", (long long) im.tp_pass, (long long) layer, en->e, en->nt,
                     im.cls(layer, en->e), pin, uses, where, im.bbytes(layer, en->e));
    }
    cpu::ExpertPool& pl = *im.pool;
    const double g0 = pl.ms_multi_gu, q0 = pl.ms_multi_q, d0 = pl.ms_multi_down, t0 = now_ms();
    im.pool_busy.store(1, std::memory_order_release);
    cpu_ents(im, layer, cpu_list, x, out);
    im.pool_busy.store(0, std::memory_order_release);
    const double t1 = now_ms(), g1 = pl.ms_multi_gu, q1 = pl.ms_multi_q, d1 = pl.ms_multi_down;
    static const bool twice = env_i("STRATA_XE_CPU_TWICE", 0) != 0;
    double hot = 0, hg = 0, hq = 0, hd = 0;
    if (twice) {   // the same experts again, into scratch (their counters left as the first run set them)
        static thread_local std::vector<float> scratch;
        scratch.assign((size_t) nt * (size_t) im.g.n_embd, 0.0f);
        const auto st = im.st;
        const double t2 = now_ms();
        cpu_ents(im, layer, cpu_list, x, scratch.data());
        hot = (now_ms() - t2) * 1e3;
        hg = pl.ms_multi_gu - g1; hq = pl.ms_multi_q - q1; hd = pl.ms_multi_down - d1;
        im.st = st;
    }
    if (early && im.pc_ready) pc_early(im, layer, cpu_list);
    std::fprintf(im.cpu_log, "L %lld %lld %d %zu %zu %.1f %.3f %.3f %.3f %.1f %.3f %.3f %.3f\n", (long long) im.tp_pass, (long long) layer,
                 nt, cpu_list.size(), bytes, (t1 - t0) * 1e3, g1 - g0, q1 - q0, d1 - d0, hot, hg, hq, hd);
    return (int) cpu_list.size();
}

bool Ds41XeTier::run_chunk(int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out) {
    try {
        if (n > kMaxJobTok) disarm(*im_);
        if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, out, nullptr);
        return run_big(*im_, l, n, ids, w, x, out, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xe tier: run_chunk (layer %lld, %d tokens): %s\n", (long long) l, n, e.what());
        return false;
    }
}
bool Ds41XeTier::run_chunk_dev(int64_t l, int n, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    try {
        if (n > kMaxJobTok) disarm(*im_);
        if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, nullptr, out_dev);
        return run_big(*im_, l, n, ids, w, x, nullptr, out_dev);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xe tier: run_chunk_dev (layer %lld, %d tokens): %s\n", (long long) l, n, e.what());
        return false;
    }
}

// ================================ the rest ================================

bool Ds41XeTier::prefetch_begin(int64_t first, int64_t last, int n_call) {
    Ds41XeImpl& im = *im_;
    static const int on = env_i("STRATA_XE_PREFETCH", 1);
    if (!on || !im.pf_stage_cap || im.pf_on || last < first) return false;
    // larry, patch-pfbtrace (STRATA_XE_ALLOC_LOG=1): prefetch_begin's parts
    static const bool pfb_log = env_i("STRATA_XE_ALLOC_LOG", 0) != 0;
    const double pfb0 = now_ms();
    std::string pfb;
    pc_drop(im);   // the promoted experts' VRAM back first (the buffers below are sized from what is free)
    if (pfb_log) pfb += "pc_drop " + std::to_string((int) (now_ms() - pfb0)) + " ms";
    // the buffers: per expert card two of the largest layer's per-card share, within its free VRAM less a reserve
    // for run_big's own buffers (STRATA_XE_PREFETCH_RESERVE_GIB, 1.3; the xe driver spills an over-commit into host
    // RAM instead of failing, so stay well inside)
    int64_t need = 0;
    int ncards = 0;
    for (auto& cp : im.cards) if (!cp->dense) ++ncards;
    if (!ncards) return false;
    for (int64_t l = first; l <= last; ++l) {
        int64_t b = 0;
        for (int64_t e = 0; e < im.g.n_experts; ++e) {
            const size_t key = (size_t) im.key(l, e);
            if (im.card_of[key] < 0 && im.aptr[key]) b += (int64_t) ((im.bbytes(l, e) + 255) & ~(size_t) 255);
        }
        need = std::max(need, b);
    }
    // run_big's buffers for n_call tokens (an upper bound: a card's GEMM rows taken as half of all, its largest expert
    // as half the tokens), less what the card holds already; then the small overflow stream buffer and a margin
    const int64_t H = im.g.n_embd, F = im.g.n_ff, K = im.g.top_k, nc = std::max(1, n_call);
    const int64_t BJ = env_i("STRATA_XE_CHUNK_JOBS", 512), gp = nc * K / 2, gm = nc / 2 + 1;
    const int64_t rb = 2 * nc * H * 4 + 2 * nc * K * 4 + BJ * kMaxJobTok * (F + H) * 4 + nc * H * 2 + BJ * kMaxJobTok * F * 2 +
                       gm * H * 2 + 3 * F * H * 2 + gm * 2 * F * 4 + gm * F * 2 + gp * H * 4 + gp * 4;
    const double margin = env_d("STRATA_XE_PREFETCH_MARGIN_GIB", 0.5) + env_d("STRATA_XE_PREFETCH_STREAM_GIB", 0.25);
    const size_t per = (size_t) (need / ncards + 2 * (int64_t) im.blob_max);
    std::vector<XeCard*> touch;   // larry, patch-barstage2: new BAR flex regions, faulted in below (a thread a card)
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense) continue;
        // larry, patch-barstage2: a BAR stream buffer of the planned size stays (a new one costs its CPU pages)
        const size_t sb0 = (size_t) (env_d("STRATA_XE_PREFETCH_STREAM_GIB", 0.25) * kGiB) & ~(size_t) 255;
        const bool keep_st = c.stream && c.stream_bytes == sb0 && bar_cpu(c.stream);
        if (c.stream && !keep_st) { c.q.wait(); local_free(c.q, c.stream); c.stream = nullptr; c.stream_bytes = 0; }   // regrown on demand
        c.st_layer = -1;
        const int64_t have = (int64_t) (c.cX_cap + c.cOUT_cap) * 4 + (int64_t) c.cS_cap * 8 + (int64_t) c.gscratch_cap +
                             (c.cjobs_cap ? BJ * kMaxJobTok * (F + H) * 4 : 0) + (int64_t) c.cxq_rows * H * 2;
        const int64_t reserve = std::max<int64_t>(0, rb - have) + (int64_t) (margin * kGiB);
        const int64_t fr = c.free_vram() + (int64_t) c.flex_cap + (keep_st ? (int64_t) sb0 : 0);   // the flex region becomes the buffers
        int64_t room = fr > 0 ? (fr - reserve) / 2 : 0;
        const size_t cap = (size_t) std::max<int64_t>(0, std::min<int64_t>((int64_t) per, room)) & ~(size_t) 255;
        if (cap < (size_t) im.blob_max) { c.pbuf_cap = 0; continue; }
        // sized to this prompt: a larger region left over would over-commit the card (the xe driver then spills buffers
        // into host RAM: a 16K prompt after an 8K one pushed larry past the watchdog, 2026-10-21 04:30)
        // larry, patch-barstage3: a BAR region stays unless it is > 256 MiB too small or > 160 MiB too large (its CPU
        // pages cost ~0.25 s a GiB to fault in again); the slots then take what it holds
        const bool bar_f = c.flex && bar_cpu(c.flex);
        const size_t f_lo = bar_f ? ((size_t) 256 << 20) : 0, f_hi = (size_t) (bar_f ? 160 : 64) << 20;
        if (!c.flex || c.flex_cap + f_lo < 2 * cap || c.flex_cap > 2 * cap + f_hi) {
            const double ta = now_ms();
            const size_t was = c.flex_cap;
            if (c.flex) { c.q.wait(); local_free(c.q, c.flex); }
            c.flex = card_alloc(c.q, 2 * cap);   // larry, patch-barstage
            c.flex_cap = c.flex ? 2 * cap : 0;
            if (pfb_log) {
                char b[160];
                std::snprintf(b, sizeof b, ", card %d flex %.2f -> %.2f GiB %s %.0f ms", c.idx, was / kGiB, c.flex_cap / kGiB,
                              c.flex && bar_cpu(c.flex) ? "BAR" : "local", now_ms() - ta);
                pfb += b;
            }
            if (!c.flex) { c.pbuf_cap = 0; continue; }
            if (bar_cpu(c.flex)) touch.push_back(&c);
        } else if (pfb_log) {
            pfb += ", card " + std::to_string(c.idx) + " flex kept";
        }
        if (pfb_log && keep_st) pfb += " (stream kept)";
        const size_t use = std::min(cap, (c.flex_cap / 2) & ~(size_t) 255);   // larry, patch-barstage3
        if (use < (size_t) im.blob_max) { c.pbuf_cap = 0; continue; }
        c.pbuf[0] = c.flex;
        c.pbuf[1] = c.flex + use;
        c.pbuf_cap = use;
    }
    if (!touch.empty()) {   // larry, patch-barstage2: the new BAR regions' CPU pages, the cards side by side
        std::vector<double> ms(touch.size(), 0.0);
        std::vector<std::thread> th;
        for (size_t i = 0; i < touch.size(); ++i) th.emplace_back([&, i] { ms[i] = bar_touch(touch[i]->flex, touch[i]->flex_cap); });
        for (auto& t : th) t.join();
        if (pfb_log) pfb += ", fault-in done at " + std::to_string((int) (now_ms() - pfb0)) + " ms";
        static const bool all = env_i("STRATA_XE_ALLOC_LOG", 0) != 0;
        if (all)
            for (size_t i = 0; i < touch.size(); ++i)
                std::fprintf(stderr, "xe prefetch: card %d's BAR flex region (%.2f GiB) faulted in: %.0f ms\n", touch[i]->idx,
                             touch[i]->flex_cap / kGiB, ms[i]);
    }
    // the calls' buffers now, once (they are in the reserve above): the overflow stream buffer at its planned size, X /
    // OUT / the slot maps for n_call tokens
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.pbuf_cap) continue;
        const size_t sb = (size_t) (env_d("STRATA_XE_PREFETCH_STREAM_GIB", 0.25) * kGiB) & ~(size_t) 255;
        if (!grow_stream(c, sb, sb) || !grow_loc(c.cX, c.cX_cap, (size_t) nc * H, c.q) ||
            !grow_dev(c.cOUT, c.cOUT_cap, (size_t) nc * H, c.q))
            std::fprintf(stderr, "xe prefetch: card %d: call buffers not pre-allocated (%.2f GiB free)\n", c.idx, c.free_vram() / kGiB);
        if (c.cS_cap < (size_t) nc * K) {
            size_t scap = 0, wcap = 0;
            if (c.cS) { c.q.wait(); sycl::free(c.cS, c.q); sycl::free(c.cW, c.q); c.cS = nullptr; c.cW = nullptr; c.cS_cap = 0; }
            if (grow_dev(c.cS, scap, (size_t) nc * K, c.q) && grow_dev(c.cW, wcap, (size_t) nc * K, c.q)) c.cS_cap = (size_t) nc * K;
            else {
                if (c.cS) sycl::free(c.cS, c.q);
                if (c.cW) sycl::free(c.cW, c.q);
                c.cS = nullptr; c.cW = nullptr;
            }
        }
    }
    static const int pf_dense = env_i("STRATA_XE_PREFETCH_DENSE", 0);   // measured no gain (8.5K: 13.4-13.8 vs 12.7-13.1 s)
    if (pf_dense && im.dense >= 0) {
        XeCard& c = *im.cards[(size_t) im.dense];
        size_t ex = 0;
        for (auto& cp : im.cards) if (!cp->dense) ex += cp->pbuf_cap;
        const int64_t want = need + 4 * (int64_t) im.blob_max - (int64_t) ex;
        const int64_t fr = c.free_vram();
        const int64_t room = fr > 0 ? (fr - (int64_t) (env_d("STRATA_XE_PREFETCH_DENSE_RESERVE_GIB", 6.0) * kGiB) - rb) / 2 : 0;
        const size_t cap = (size_t) std::max<int64_t>(0, std::min<int64_t>(want, room)) & ~(size_t) 255;
        if (cap >= (size_t) im.blob_max && !c.pbuf[0]) {
            c.pbuf[0] = sycl::malloc_device<uint8_t>(cap, c.q);
            c.pbuf[1] = c.pbuf[0] ? sycl::malloc_device<uint8_t>(cap, c.q) : nullptr;
            if (c.pbuf[1]) c.pbuf_cap = cap;
            else { if (c.pbuf[0]) sycl::free(c.pbuf[0], c.q); c.pbuf[0] = nullptr; c.pbuf_cap = 0; }
        }
    }
    if (im.pf_thr.empty()) {
        im.pf_stop = false;
        const int ns = std::max(1, env_i("STRATA_XE_STAGERS", bar_stage_on() ? 1 : 4));   // larry, patch-barstage2: 1 with BAR
        if (hot_on() && im.pf_rings.empty()) {   // STRATA_XE_HOTRING: a ring per stager (all or none: pf_enqueue counts on the stagers' copies)
            std::lock_guard<PrioMutex> lk(g_submit);
            for (int i = 0; i < ns; ++i) im.pf_rings.push_back(sycl::malloc_host<uint8_t>((size_t) hot_slots() * hot_chunk(), im.ctx));
            bool all = true;
            for (uint8_t* p : im.pf_rings) all = all && p;
            if (!all) {
                for (uint8_t* p : im.pf_rings) if (p) sycl::free(p, im.ctx);
                im.pf_rings.clear();
            }
            std::fprintf(stderr, "xe prefetch: hot rings %s %d x %d x %zu KiB\n", all ? "on," : "FAILED (old copies),", ns, hot_slots(),
                         hot_chunk() >> 10);
        }
        for (int i = 0; i < ns; ++i) im.pf_thr.emplace_back([&im, i] { pf_stager(im, i); });
    }
    if (pfb_log)
        std::fprintf(stderr, "xe prefetch begin: %s; call buffers etc. done at %.0f ms\n", pfb.c_str(), now_ms() - pfb0);
    static bool logged = false;
    if (!logged || xe_prof()) {
        logged = true;
        std::fprintf(stderr, "xe prefetch: layers %lld..%lld, %d tokens a call, largest layer %.2f GiB; buffers per card 2 x", (long long) first,
                     (long long) last, n_call, need / kGiB);
        for (auto& cp : im.cards) if (!cp->dense) std::fprintf(stderr, " %.2f", cp->pbuf_cap / kGiB);
        for (auto& cp : im.cards) if (cp->dense && cp->pbuf_cap) std::fprintf(stderr, " + dense card %.2f", cp->pbuf_cap / kGiB);
        std::fprintf(stderr, " GiB\n");
    }
    im.pf_on = true;
    im.pf_last = last;
    im.pf_used = im.pf_planned = im.pf_bytes = 0;
    im.pf[0].layer = im.pf[1].layer = -1;
    pf_plan(im, (int) (first & 1), first);
    if (first + 1 <= last) pf_plan(im, (int) ((first + 1) & 1), first + 1);
    return true;
}

std::mutex& Ds41XeTier::submit_mutex() { return g_submit.m; }
std::atomic<int>& Ds41XeTier::submit_priority() { return g_submit.hi; }

bool Ds41XeTier::run_chunk_dev_async(int64_t l, int n, const int32_t* ids, const float* w, const float* x, void* out_dev,
                                     bool last_of_layer, const float* x_dev) {
    Ds41XeImpl& im = *im_;
    if (im.run_thr.joinable()) return false;   // one call in flight
    im.run_ok = true;
    im.sh_call = im.sh_next && n > kMaxJobTok;   // larry, STRATA_SHEXP_TIER: this call's (run_big's) shared expert
    im.sh_next = false;
    im.run_thr = std::thread([&im, l, n, ids, w, x, out_dev, last_of_layer, x_dev] {
        try {
            im.run_ok = n <= kMaxJobTok ? run_small(im, l, n, ids, w, x, nullptr, out_dev)
                                        : run_big(im, l, n, ids, w, x, nullptr, out_dev, last_of_layer, x_dev);
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "xe tier: run_chunk_dev_async (layer %lld, %d tokens): %s\n", (long long) l, n, e.what());
            im.run_ok = false;
        }
    });
    return true;
}

bool Ds41XeTier::run_chunk_wait() {
    Ds41XeImpl& im = *im_;
    if (!im.run_thr.joinable()) return true;
    im.run_thr.join();
    return im.run_ok;
}

void Ds41XeTier::prefetch_end() {
    Ds41XeImpl& im = *im_;
    if (!im.pf_on) return;
    if (hot_prof()) {   // larry, STRATA_XE_HOT_PROF
        std::fprintf(stderr, "xe hot prof: %.2f GB through the rings (%.2f GB pinned direct): slot waits %.0f ms, copies %.0f ms (%.1f GB/s a "
                     "thread), lock waits %.0f ms, submits %.0f ms (thread-ms)\n", g_hp_bytes.load() / 1e9, g_hp_pin.load() / 1e9,
                     g_hp_wait.load() / 1e3, g_hp_copy.load() / 1e3, g_hp_copy.load() > 0 ? g_hp_bytes.load() / 1e3 / g_hp_copy.load() : 0.0,
                     g_hp_lock.load() / 1e3, g_hp_sub.load() / 1e3);
        g_hp_wait = 0; g_hp_copy = 0; g_hp_lock = 0; g_hp_sub = 0; g_hp_bytes = 0; g_hp_pin = 0;
    }
    {
        std::unique_lock<std::mutex> lk(im.pf_mu);   // a staging still running reads the arena into its stage
        im.pf_cv.wait(lk, [&] { return im.pf[0].staged == 0 && im.pf[1].staged == 0; });
    }
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.pbuf_cap) continue;
        c.pq.wait();
        c.q.wait();
        if (c.pbuf[0] == c.flex) c.pbuf[0] = c.pbuf[1] = nullptr;   // the flex region stays (decode: promotion slots)
        for (auto& p : c.pbuf) { if (p) sycl::free(p, c.q); p = nullptr; }
        c.pbuf_cap = 0;
    }
    if (xe_prof())
        std::fprintf(stderr, "xe prefetch: %lld experts (%.2f GB) planned, %lld used; %.2f GB through BARs so far\n",
                     (long long) im.pf_planned, im.pf_bytes / 1e9, (long long) im.pf_used, g_bar_put.load() / 1e9);
    im.pf[0].layer = im.pf[1].layer = -1;
    im.pf_on = false;
}

void Ds41XeTier::close() {
    if (!im_) return;
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync
    disarm(im);
    pc_drop(im);
    if (im.pc_thr.joinable()) {
        { std::lock_guard<std::mutex> lk(im.pc_mu); im.pc_stop = true; }
        im.pc_cv.notify_all();
        im.pc_thr.join();
        for (auto& t : im.pc_thr_more) t.join();
        im.pc_thr_more.clear();
    }
    if (im.pc_bounce) { sycl::free(im.pc_bounce, im.ctx); im.pc_bounce = nullptr; }
    rep_drop(im);
    if (im.rep_host) { sycl::free(im.rep_host, im.ctx); im.rep_host = nullptr; }
    if (im.cpu_log) { std::fclose(im.cpu_log); im.cpu_log = nullptr; }
    if (const char* uo = std::getenv("STRATA_XE_USAGE_OUT"); uo && !im.use_jobs.empty()) {
        if (std::FILE* f = std::fopen(uo, "w")) {   // "U layer expert jobs picks card bytes", after "P passes"
            std::fprintf(f, "P %lld\n", (long long) im.tp_pass);
            for (int64_t l = 0; l < im.g.n_layers; ++l)
                for (int64_t e = 0; e < im.g.n_experts; ++e) {
                    if (!im.g.routed(l)) continue;
                    const size_t key = (size_t) im.key(l, e);
                    std::fprintf(f, "U %lld %lld %u %u %d %zu\n", (long long) l, (long long) e, im.use_jobs[key], im.use_picks[key],
                                 (int) im.use_card[key], im.bbytes(l, e));
                }
            std::fclose(f);
        }
    }
    if (im.pc_done || im.pc_busy_skip)
        std::fprintf(stderr, "xe tier: promotion cache: %lld promoted, %lld hits, %lld evicted, %lld skipped (cards busy), %lld bad; "
                     "%lld chunks, host push %.1f ms, poll %.1f ms\n", (long long) im.pc_done, (long long) im.pc_hit, (long long) im.pc_evicted,
                     (long long) im.pc_busy_skip, (long long) im.pc_bad, (long long) im.pc_chunks, im.pc_push_us / 1000.0, im.pc_poll_us / 1000.0);
    if (im.pc_fill_n > 0)
        std::fprintf(stderr, "xe tier: promotion repacks: %lld, %.1f ms each, %.1f ms queued before a helper took it\n", (long long) im.pc_fill_n.load(),
                     im.pc_fill_us.load() / 1000.0 / im.pc_fill_n.load(), im.pc_wait_us.load() / 1000.0 / im.pc_fill_n.load());
    if (im.relq) { im.relq->wait(); im.relq.reset(); }
    if (im.os_calls || im.os_declined)
        std::fprintf(stderr, "xe tier: attention output split %lld calls (%zu layers), declined %lld\n", (long long) im.os_calls,
                     im.osplits.size(), (long long) im.os_declined);
    if (im.arm_used || im.arm_disarmed)
        std::fprintf(stderr, "xe tier: armed card shares used %lld, disarmed calls %lld\n", (long long) im.arm_used,
                     (long long) im.arm_disarmed);
    if (im.groute_used || im.groute_miss || im.groute_bad)
        std::fprintf(stderr, "xe tier: STRATA_XE_GROUTE: %lld routed calls, %lld others after a route, %lld check mismatches\n",
                     (long long) im.groute_used, (long long) im.groute_miss, (long long) im.groute_bad);
    if (im.run_thr.joinable()) im.run_thr.join();
    prefetch_end();
    if (!im.pf_thr.empty()) {
        { std::lock_guard<std::mutex> lk(im.pf_mu); im.pf_stop = true; }
        im.pf_cv.notify_all();
        for (auto& t : im.pf_thr) t.join();
        im.pf_thr.clear();
    }
    for (auto& sl : im.pf) if (sl.stage) { sycl::free(sl.stage, im.ctx); sl.stage = nullptr; }
    for (uint8_t* p : im.pf_rings) if (p) sycl::free(p, im.ctx);
    im.pf_rings.clear();
    if (im.hc_n > 0)
        std::fprintf(stderr, "xe tier: HOTRING_CHECK: %lld experts compared, %lld differ\n", (long long) im.hc_n.load(),
                     (long long) im.hc_bad.load());
    for (auto& cs : im.sh_own) { cs.first->q.wait(); local_free(cs.first->q, cs.second); }   // larry, STRATA_SHEXP_TIER
    im.sh_own.clear();
    im.sh_slices.clear();
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.sh_w && !c.sh_buf && !c.sh_iota) continue;
        c.q.wait();
        if (c.sh_w) local_free(c.q, c.sh_w);
        if (c.sh_buf) local_free(c.q, c.sh_buf);
        if (c.sh_iota) local_free(c.q, c.sh_iota);
        c.sh_w = nullptr; c.sh_buf = nullptr; c.sh_iota = nullptr;
    }
    for (auto& kv : im.heads)   // the head split's slices and scratch, on their cards (before the cards go)
        for (size_t i = 0; i < kv.second.c.size(); ++i) {
            sycl::queue& q = kv.second.c[i]->q;
            q.wait();
            if (i >= kv.second.own.size() || kv.second.own[i]) sycl::free(kv.second.w[i], q);
            sycl::free(kv.second.xq[i], q); sycl::free(kv.second.y[i], q);   // (xd lives in xq's buffer: patch-headstage)
        }
    for (auto& kv : im.heads) {   // larry, patch-headstage: the card-0 staging (by the context: ggml's queue may be gone)
        if (kv.second.sx) sycl::free(kv.second.sx, im.ctx);
        if (kv.second.sy) sycl::free(kv.second.sy, im.ctx);
    }
    im.heads.clear();
    if (im.lt_dev && im.dense >= 0 && (size_t) im.dense < im.cards.size()) {
        sycl::queue& q = im.cards[(size_t) im.dense]->q;
        q.wait();
        sycl::free(im.lt_dev, q);
    }
    im.lt_dev = nullptr;
    im.lt_dev_cap = 0;
    if (im.fn16 && im.dense >= 0) { im.cards[(size_t) im.dense]->q.wait(); sycl::free(im.fn16, im.cards[(size_t) im.dense]->q); }   // larry
    im.fn16 = nullptr;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.free_experts();
        auto fr = [&](void* p) { if (p) sycl::free(p, c.q); };
        if (c.pbuf[0] == c.flex) c.pbuf[0] = c.pbuf[1] = nullptr;
        if (c.flex) { c.q.wait(); local_free(c.q, c.flex); }
        c.flex = nullptr;
        fr(c.stage_dev); fr(c.stage_host); fr(c.H); fr(c.OUT); fr(c.part_host); fr(c.arm_flag); fr(c.arm_seen); fr(c.arm_src);
        local_free(c.q, c.cX); local_free(c.q, c.cXh); fr(c.cOUT); fr(c.cOUT_host); fr(c.cS); fr(c.cW); fr(c.cjobs); fr(c.cjobs_host); fr(c.dq_host); fr(c.nw_host); fr(c.cH); fr(c.cY);
        local_free(c.q, c.stream);   // larry: card-local (grow_stream)
        local_free(c.q, c.gscratch); fr(c.cX2); fr(c.cX2h); fr(c.bounce); fr(c.dslots); fr(c.rstage); fr(c.xt); fr(c.ht);
        fr(c.hring); fr(c.hchk);
        for (xe::XeQ8* a : {&c.xq, &c.hq, &c.cxq, &c.chq}) { fr(a->q); fr(a->d); fr(a->s); }
    }
    im.cards.clear();
    for (uint8_t* p : im.evict_bufs) std::free(p);   // STRATA_XE_TP_EVICT: the evicted experts' copies
    im.evict_bufs.clear();
    for (uint8_t* p : im.achunks) {
        if (sycl::get_pointer_type(p, im.ctx) == sycl::usm::alloc::host) sycl::free(p, im.ctx);
        else std::free(p);
    }
    im.achunks.clear();
    for (void* p : {(void*) im.rt, (void*) im.rt_hdr, (void*) im.rt_jobs, (void*) im.rt_null}) if (p) sycl::free(p, im.ctx);
    im.rt = nullptr; im.rt_hdr = nullptr; im.rt_jobs = nullptr; im.rt_null = nullptr;
    if (im.hsum) { im.hsum_ev.wait(); sycl::free(im.hsum, im.ctx); im.hsum = nullptr; }
    if (im.lt_hs) { sycl::free(im.lt_hs, im.ctx); im.lt_hs = nullptr; im.lt_hs_cap = 0; }
    if (im.lt_x) { sycl::free(im.lt_x, im.ctx); im.lt_x = nullptr; im.lt_x_cap = 0; }
    if (im.lt_rows) { sycl::free(im.lt_rows, im.ctx); im.lt_rows = nullptr; im.lt_rows_cap = 0; }   // larry, LT_COMPACT
    if (im.lt_xg && im.dense >= 0 && (size_t) im.dense < im.cards.size()) {
        sycl::queue& q = im.cards[(size_t) im.dense]->pq;
        q.wait();
        sycl::free(im.lt_xg, q);
    }
    im.lt_xg = nullptr; im.lt_xg_cap = 0;
    if (im.hslots) { for (auto& cp : im.cards) cp->q.wait(); sycl::free(im.hslots, im.ctx); im.hslots = nullptr; }
    if (im.cpu_thr.joinable()) {
        { std::lock_guard<std::mutex> lk(im.cpu_mu); im.cpu_state.store(3, std::memory_order_release); }
        im.cpu_cv.notify_one();
        im.cpu_thr.join();
    }
    if (im.tl_tokens % 64) tl_print(im);
    if (im.tl_mon.joinable()) { im.tl_stop = true; im.tl_mon.join(); }
    for (auto& rp : im.tl_rings) sycl::free((void*) rp->flag, im.ctx);
    im.tl_rings.clear();
    im.placed = false;
    im.dense_set.clear();
    im.pool.reset();
}

const Ds41MoeStats& Ds41XeTier::stats() const { return im_->st; }
void Ds41XeTier::reset_stats() { im_->st = Ds41MoeStats(); }
void Ds41XeTier::add_stats(const Ds41MoeStats& s) { im_->st.add(s); }
const Ds41MoeGeom& Ds41XeTier::geom() const { return im_->g; }
const Ds41MoeConfig& Ds41XeTier::config() const { return im_->cfg; }
const char* Ds41XeTier::mode() const { return im_->cards.empty() ? "uninitialised" : "xe"; }
int64_t Ds41XeTier::resident() const { return im_->resident_n; }
int64_t Ds41XeTier::arena_experts() const { return im_->arena_n; }
int64_t Ds41XeTier::file_tier() const { return im_->file_n; }
double Ds41XeTier::arena_gib() const { return im_->arena_bytes / kGiB; }
int64_t Ds41XeTier::blob_bytes(int64_t l) const {
    if (!im_->g.routed(l)) return 0;
    return (int64_t) im_->bbytes(l, 0);
}
bool Ds41XeTier::resident(int64_t l, int64_t e) const {
    if (l < 0 || l >= im_->g.n_layers || e < 0 || e >= im_->g.n_experts) return false;
    const size_t key = (size_t) im_->key(l, e);
    // larry, patch-regrowasync: a pending regrow's experts are resident for the route bias
    return im_->card_of[key] >= 0 || (im_->rg_pending && key < im_->rg_mark.size() && im_->rg_mark[key]);
}
bool Ds41XeTier::in_arena(int64_t l, int64_t e) const {
    if (l < 0 || l >= im_->g.n_layers || e < 0 || e >= im_->g.n_experts) return false;
    return im_->aptr[(size_t) im_->key(l, e)] != nullptr;
}
std::string Ds41XeTier::report() const {
    std::string s;
    char b[256];
    for (const auto& cp : im_->cards) {
        const XeCard& c = *cp;
        const int64_t fr = c.free_vram();
        std::snprintf(b, sizeof b, "  card %d%s: %lld experts, %.2f GiB; %.2f GiB VRAM free\n", c.idx, c.dense ? " (dense)" : "",
                      (long long) c.n_experts, c.used / kGiB, fr >= 0 ? fr / kGiB : -1.0);
        s += b;
    }
    std::snprintf(b, sizeof b, "  arena: %lld experts, %.2f GiB; file tier %lld\n", (long long) im_->arena_n,
                  im_->arena_bytes / kGiB, (long long) im_->file_n);
    s += b;
    return s;
}

}  // namespace strata::v41
