// tools/v41/engine/xe/ds41_xe_tier.cpp - see ds41_xe_tier.hpp.
#include "ds41_xe_tier.hpp"
#include "xe_kernels.hpp"
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
#include <thread>

#include <pthread.h>
#include <sched.h>

namespace strata::v41 {

namespace cpu = strata::kernels::cpu;
using xe::XeJob;
using xe::kMaxJobTok;

namespace {

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
    return (uint8_t*) p;
}
inline void local_free(sycl::queue& q, void* p) {
    if (!p) return;
    if (!local_alloc_on()) { sycl::free(p, q); return; }
    zeMemFree(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context()), p);
}
constexpr int kMaxDecodeJobs = 64;               ///< a verify pass of 8 tokens routes at most 48 distinct experts
constexpr size_t kStageAlign = 256;
constexpr int kPcBounce = 3;                     ///< STRATA_XE_PROMOTE: pinned bounces (promotions in flight)

}  // namespace

/// One card.
struct XeCard {
    int idx = 0;
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
    int arm_ng = 0;                 ///< its job count (null-padded)
    sycl::event arm_ev;             ///< its last command (the part's copy to the dense card)
    XeJob arm_null{};               ///< a null job (nt 0) on a blob this card holds
    bool arm_null_ok = false;
    // larry, STRATA_XE_PROMOTE: this card's share of the decode-time cache of promoted experts (Ds41XeImpl::pc_*)
    uint8_t* pc_mem = nullptr;      ///< pc_n slots of Ds41XeImpl::pc_slot bytes (planar experts)
    int pc_n = 0;
    std::vector<int64_t> pc_key;    ///< slot -> expert key; -1 free, -2 being filled
    std::vector<uint64_t> pc_tick;  ///< slot -> the decode call that last used it
    bool pc_busy = false;           ///< a promotion of this card is in flight
    /// decode / verify: the card's command sequence of a call (stage copy, quantise, gate/up, down, copy to the dense
    /// card's slot) recorded once per shape (jobs, tokens, kernel variant) and replayed - one submission, not five
    std::map<uint32_t, std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>>> dgraphs;
    std::map<uint32_t, const float*> dgraph_x;   ///< per recording: the direct X it reads (null: the staged one)
    // prompt-chunk scratch (grown on demand)
    float* cX = nullptr; size_t cX_cap = 0;          ///< floats
    float* cOUT = nullptr; size_t cOUT_cap = 0;      ///< floats
    float* cOUT_host = nullptr; size_t cOUT_host_cap = 0;
    int32_t* cS = nullptr; float* cW = nullptr; size_t cS_cap = 0;   ///< entries
    XeJob* cjobs = nullptr; float* cH = nullptr; float* cY = nullptr; int cjobs_cap = 0;
    XeJob* cjobs_host = nullptr;    ///< pinned: cjobs' host copy (no shared USM: its page-fault handler deadlocked)
    uint8_t* dq_host = nullptr; size_t dq_host_cap = 0;   ///< pinned: xe_dqgemm's job / tile tables of a call
    uint8_t* stream = nullptr; size_t stream_bytes = 0;   ///< streamed (arena / file) experts of one chunk round
    /// the experts the last prompt call left in `stream` (expert, offset), all of layer st_layer, ending at st_end
    int64_t st_layer = -1; size_t st_end = 0;
    std::vector<std::pair<int32_t, int64_t>> st_have;
    uint8_t* gscratch = nullptr; size_t gscratch_cap = 0;  ///< the GEMM path's operands (bytes)
    float* cX2 = nullptr; size_t cX2_cap = 0;              ///< the dense card's copy of the other cards' sum
    sycl::half* cX2h = nullptr; size_t cX2h_cap = 0;       ///< STRATA_XE_COMBINE=2: the parts as fp16 (slot = card)
    uint8_t* bounce = nullptr; size_t bounce_cap = 0;      ///< pinned staging for streamed experts in pageable memory
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
        std::vector<int8_t*> xq;
        std::vector<float*> xd, y;
    };
    std::map<const void*, HeadSplit> heads;
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
    };
    int pc_mode = -1;                  ///< -1 not read yet, 0 off, 1 on
    bool pc_ready = false;             ///< the slots exist (allocated a few calls into a decode)
    int64_t pc_calls = 0;              ///< decode calls since the last drop
    bool pc_tried = false;             ///< pc_setup ran since the last drop
    size_t pc_slot = 0;                ///< bytes a slot
    std::vector<int32_t> pc_where;     ///< key -> card << 16 | slot (published); -1; -2 in flight
    std::vector<uint8_t> pc_uses;      ///< key -> its CPU-pool uses (saturating; STRATA_XE_PROMOTE_MIN)
    uint8_t* pc_bounce = nullptr;      ///< pinned: kPcBounce x pc_slot
    std::vector<int> pc_bfree;
    std::list<PcJob> pc_jobs;          ///< in flight (stable addresses: the helper holds pointers)
    std::thread pc_thr;
    std::mutex pc_mu;
    std::condition_variable pc_cv;
    std::deque<PcJob*> pc_fill;        ///< bounces for the helper to fill (under pc_mu)
    bool pc_stop = false;
    uint64_t pc_clock = 0;
    int pc_rr = 0;
    int64_t pc_done = 0, pc_evicted = 0, pc_hit = 0, pc_busy_skip = 0, pc_bad = 0;
    double pc_push_us = 0, pc_poll_us = 0;
    int64_t pc_chunks = 0;
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
    };
    PfSlot pf[2];
    size_t pf_stage_cap = 0;           ///< bytes per slot's stage
    bool pf_on = false;
    int64_t pf_last = -1;
    int64_t pf_used = 0, pf_planned = 0, pf_bytes = 0;   ///< over a batch (the prof line)
    std::vector<std::thread> pf_thr;
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
        c->dev = devs[(size_t) i];
        c->q = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
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
    im.pool.reset(new cpu::ExpertPool(cfg.threads > 0 ? cfg.threads : 0));
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
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < nth; ++i)
        th.emplace_back([&]() {
            uint8_t* buf = sycl::malloc_host<uint8_t>((size_t) im.blob_max, im.ctx);
            for (;;) {
                const size_t k = next.fetch_add(1);
                if (k >= list.size()) break;
                im.blobs->read(list[k].first, list[k].second, buf);
                sink(k, buf);
            }
            sycl::free(buf, im.ctx);
        });
    for (auto& t : th) t.join();
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
        if (best < 0) { rest.push_back(le); continue; }
        left[(size_t) best] -= b;
        on[(size_t) best].push_back(le);
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
    int64_t pinned = 0;
    std::vector<uint8_t> chunk_pinned(nchunks, 0);
    for (size_t c = 0; c < nchunks; ++c) {
        uint8_t* p = nullptr;
        if (pinned + (int64_t) chunk_size[c] <= pin_cap) {
            p = sycl::malloc_host<uint8_t>(chunk_size[c], im.ctx);
            if (p) { pinned += (int64_t) chunk_size[c]; chunk_pinned[c] = 1; }
        }
        if (!p) {
            void* q = nullptr;
            if (posix_memalign(&q, 2u << 20, chunk_size[c]) != 0 || !q) { err = "xe tier: arena allocation failed"; return false; }
            p = (uint8_t*) q;
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
    // ---- load: the cards' experts (device copies out of pinned reader buffers), then the arena
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        for (const auto& le : on[(size_t) c.idx]) {
            uint8_t* p = c.alloc_expert(im.bbytes(le.first, le.second));
            if (!p) { err = "xe tier: device allocation failed on card " + std::to_string(c.idx); return false; }
            im.card_of[(size_t) im.key(le.first, le.second)] = (int8_t) c.idx;
            im.dptr[(size_t) im.key(le.first, le.second)] = p;
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
            const int64_t want = std::min<int64_t>((int64_t) (fg * kGiB), fr > 0 ? fr - (int64_t) (fm * kGiB) : 0) & ~(int64_t) 255;
            if (want < 8 * im.blob_max) continue;
            c.flex = local_alloc(c.q, (size_t) want);
            c.flex_cap = c.flex ? (size_t) want : 0;
        }
        if (env_i("STRATA_XE_PROMOTE", 1) && !im.pc_bounce) {   // the promotions' pinned bounces, now too
            im.pc_slot = ((size_t) im.blob_max + 255) & ~(size_t) 255;
            im.pc_bounce = sycl::malloc_host<uint8_t>((size_t) kPcBounce * im.pc_slot, im.ctx);
        }
    }
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
int64_t seed_dense(Ds41XeImpl& im, XeCard& c, const std::vector<std::pair<int32_t, int32_t>>& ranked, const char* what, std::string& err) {
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
        im.card_of[key] = (int8_t) c.idx;
        im.dptr[key] = p;
        im.dense_set.push_back(le);
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
    if (cap >= 0) gib = std::min(gib, cap);
    const int64_t fr = c.free_vram();
    if (fr > 0) gib = std::min(gib, (fr + c.used) / kGiB - env_d("STRATA_XE_DENSE_RESERVE_GIB", 8.0));
    return gib < 0.5 ? 0 : (int64_t) (gib * kGiB);
}
}  // namespace

bool Ds41XeTier::seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    Ds41XeImpl& im = *im_;
    if (!im.placed && !build_arena_ranked(ranked, err)) return false;
    if (im.dense < 0 || !im.cache_enabled || !im.dense_set.empty()) return true;
    XeCard& c = *im.cards[(size_t) im.dense];
    im.seed_ranked = ranked;
    c.budget = dense_budget(im, c);
    if (c.budget <= 0) return true;
    return seed_dense(im, c, ranked, "seeded", err) >= 0;
}

/// larry (serve keep-cache): a prompt batch of a few thousand tokens needs `need_bytes` free on the dense card - the
/// cache's tail (its lowest-ranked experts, the last slabs) goes back to the arena slab by slab until that is free.
/// Returns the experts released, -1 when even an empty cache would not make the room (the caller releases it all).
int64_t Ds41XeTier::shrink_cache_for(int64_t need_bytes, std::string& err) {
    Ds41XeImpl& im = *im_;
    if (im.dense < 0 || im.dense_set.empty()) return -1;
    XeCard& c = *im.cards[(size_t) im.dense];
    c.q.wait();
    if (c.free_vram() + c.used < need_bytes) { err = "the dense card has no room for the batch even without its experts"; return -1; }
    int64_t released = 0;
    while (c.free_vram() < need_bytes && !c.slabs.empty()) {
        uint8_t* last = c.slabs.back();
        while (!im.dense_set.empty()) {   // the slab's experts: a suffix of dense_set (slabs fill in seeding order)
            const auto& le = im.dense_set.back();
            const size_t key = (size_t) im.key(le.first, le.second);
            const uint8_t* p = im.dptr[key];
            if (p < last || p >= last + kSlab) break;
            im.card_of[key] = -1;
            im.dptr[key] = nullptr;
            c.used -= (int64_t) im.bbytes(le.first, le.second);
            --c.n_experts;
            im.dense_set.pop_back();
            ++released;
        }
        local_free(c.q, last);
        c.slabs.pop_back();
        c.slab_used = kSlab;   // the slabs left are full: the next expert opens a new one
    }
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
    return released;
}

/// larry (serve keep-cache): after the batch, the tail back - the seed continued within what is free now.
int64_t Ds41XeTier::regrow_cache(std::string& err) {
    Ds41XeImpl& im = *im_;
    if (im.dense < 0 || !im.cache_enabled || im.seed_ranked.empty()) return 0;
    XeCard& c = *im.cards[(size_t) im.dense];
    c.budget = dense_budget(im, c);
    if (c.used >= c.budget) return 0;
    return seed_dense(im, c, im.seed_ranked, "regrown", err);
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
    for (int i = 1; i < np; ++i) {
        XeCard& c = *parts[(size_t) i - 1];
        const int64_t rows = hs.r[i + 1] - hs.r[i], b0 = hs.r[i] * bpr, bn = rows * bpr;
        const size_t need = (size_t) bn * 210 + (size_t) 8 * K * 2 + (size_t) 8 * rows * 4;
        if (c.free_vram() > 0 && c.free_vram() < (int64_t) (need + ((size_t) 768 << 20))) {
            std::fprintf(stderr, "xe tier: no room on card %d for the head split (%zu MB) - not split\n", c.idx, need >> 20);
            return Ds41XeImpl::HeadSplit();
        }
        uint8_t* wk = sycl::malloc_device<uint8_t>((size_t) bn * 210, c.q);
        int8_t* xq = sycl::malloc_device<int8_t>((size_t) 8 * K, c.q);
        float* xd = sycl::malloc_device<float>((size_t) 8 * K / 32, c.q);
        float* y = sycl::malloc_device<float>((size_t) 8 * rows, c.q);
        if (!wk || !xq || !xd || !y) { std::fprintf(stderr, "xe tier: head split allocation failed - not split\n"); return Ds41XeImpl::HeadSplit(); }
        for (int p = 0; p < 4; ++p) c.q.memcpy(wk + po[p] * bn, (const uint8_t*) w + po[p] * nb + pb[p] * b0, pb[p] * bn);
        c.q.wait();
        hs.c.push_back(&c); hs.w.push_back(wk); hs.xq.push_back(xq); hs.xd.push_back(xd); hs.y.push_back(y);
    }
    hs.ok = true;
    std::fprintf(stderr, "xe tier: %lld x %lld LM head split over %d cards\n", (long long) N, (long long) K, np);
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
        const int64_t bpr = K / 256;
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
            for (int t = 0; t < ncols; ++t) c.q.memcpy(y + t * y_col_stride + r0, hs.y[i] + t * rows, (size_t) rows * 4);
            ev.push_back(c.q.ext_oneapi_submit_barrier());
        }
        fastmv::Args a0;
        a0.w = w; a0.K = K; a0.N = hs.r[1]; a0.G = 1; a0.g_div = 1; a0.n_blocks_total = N * bpr;
        a0.x_col_stride = K; a0.x_batch_stride = K * ncols;
        a0.y = y; a0.y_col_stride = y_col_stride; a0.y_batch_stride = y_col_stride * ncols; a0.ncols = ncols;
        a0.xq = xq; a0.xd = xd;
        fastmv::launch(q0, fastmv::WType::Q6K_R8, a0);
        q0.ext_oneapi_submit_barrier(ev);
        return true;
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xe tier: head split: %s\n", e.what());
        return false;
    }
}

bool Ds41XeTier::tl_on() { static const bool on = env_i("STRATA_GPU_TL", 0) != 0; return on; }

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

void Ds41XeTier::release_cache() {
    Ds41XeImpl& im = *im_;
    if (im.dense < 0 || im.dense_set.empty()) return;
    for (const auto& le : im.dense_set) {
        const size_t key = (size_t) im.key(le.first, le.second);
        im.card_of[key] = -1;
        im.dptr[key] = nullptr;
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
    for (;;) {
        int st = im.cpu_state.load(std::memory_order_acquire);
        for (int i = 0; st == 0 && i < 20000; ++i) {
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

/// A card's share of a decode / verify call: its n jobs (staged in c.stage_dev, X at xoff) into dst [nt][n_embd].
sycl::event card_experts(Ds41XeImpl& im, XeCard& c, int n, int nt, int max_nt, size_t xoff, float* dst, const float* xsrc = nullptr) {
    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
    const XeJob* jobs = (const XeJob*) c.stage_dev;
    const float* X = xsrc ? xsrc : (const float*) (c.stage_dev + xoff);   // xsrc: pinned host rows, read over PCIe
    if (!im.planar) return xe::xe_experts(c.q, jobs, n, nt, X, c.H, dst, H, F, im.L, false);
    static const bool xmx = env_i("STRATA_XE_XMX", 0) != 0;   // larry: the XMX kernels (xe_xmx.cpp)
    if (xmx) {
        xe::xe_xt(c.q, X, nt, H, c.xt);
        return xe::xe_experts_x(c.q, jobs, n, nt, c.xt, c.ht, dst, H, F, im.L, false);
    }
    xe::xe_quant_rows(c.q, X, nt, H, c.xq);
    return xe::xe_experts_q8(c.q, jobs, n, nt, c.xq, c.hq, dst, H, F, im.L, false, max_nt);
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
        for (int k = 0; k < c.arm_ng; ++k) std::memcpy(c.stage_host + (size_t) k * sizeof(XeJob), &c.arm_null, sizeof(XeJob));
        arm_release(im, c);
        c.armed = false;
        any = true;
    }
    if (!any) return;
    for (auto& cp : im.cards) if (cp->arm_flag) cp->arm_ev.wait();
    im.armed_l = -1;
    ++im.arm_disarmed;
}

bool ev_done(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}

/// Host twin of xe_planar.cpp's repack_mat (the same bytes): an [nrows x n] matrix of NB blocks, file layout -> planes.
void cpu_repack_mat(int type, const uint8_t* src, uint8_t* dst, size_t NB) {
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

/// The promotion cache's slots (each expert card's flex region, STRATA_XE_FLEX_GIB), the bounces and the helper thread.
void pc_setup(Ds41XeImpl& im) {
    im.pc_slot = ((size_t) im.blob_max + 255) & ~(size_t) 255;
    int total = 0;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || c.pc_mem || !c.flex || c.pbuf_cap) continue;   // the slots: the card's flex region (no prompt running)
        const int n = (int) std::min<size_t>(c.flex_cap / im.pc_slot, 65535);
        if (n < 4) continue;
        c.pc_mem = c.flex;
        c.pc_n = n;
        c.pc_key.assign((size_t) n, -1);
        c.pc_tick.assign((size_t) n, 0);
        c.pc_busy = false;
        total += n;
    }
    if (!im.pc_bounce) im.pc_bounce = sycl::malloc_host<uint8_t>((size_t) kPcBounce * im.pc_slot, im.ctx);
    im.pc_bfree.clear();
    for (int b = 0; b < kPcBounce; ++b) im.pc_bfree.push_back(b);
    im.pc_where.assign(im.card_of.size(), -1);
    if (!im.pc_thr.joinable()) {
        im.pc_stop = false;
        im.pc_thr = std::thread([&im] {
            {   // background work: it must not take a core from the decode thread's latency-bound path
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
                const int64_t l = j->key / im.g.n_experts, e = j->key % im.g.n_experts;
                const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
                const uint8_t* src = im.aptr[(size_t) j->key];
                uint8_t* dst = im.pc_bounce + (size_t) j->bounce * im.pc_slot;
                if (!im.planar) std::memcpy(dst, src, im.bbytes(l, e));
                else {
                    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
                    auto qk = [](int t) { return t == xe::kQ3K ? 256 : t == xe::kQ20 ? 64 : 128; };
                    const size_t nbg = (size_t) F * (H / qk(f.gu_type)), nbd = (size_t) H * (F / qk(f.d_type));
                    cpu_repack_mat(f.gu_type, src, dst, nbg);
                    cpu_repack_mat(f.gu_type, src + f.up_off, dst + f.up_off, nbg);
                    cpu_repack_mat(f.d_type, src + f.down_off, dst + f.down_off, nbd);
                }
                j->filled.store(1, std::memory_order_release);
            }
        });
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
    static const size_t chunk = (size_t) (env_d("STRATA_XE_PROMOTE_CHUNK_MB", 1.0) * 1048576.0) & ~(size_t) 255;
    for (auto it = im.pc_jobs.begin(); it != im.pc_jobs.end();) {
        Ds41XeImpl::PcJob& j = *it;
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
            const sycl::event kev = c.q.parallel_for(sycl::range<1>(n / 16), [=](sycl::id<1> i) { dv[i] = sv[i]; });
            if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 13);
            j.sent += n;
            ++im.pc_chunks;
            if (j.sent == b) {   // in queue order: once this kernel is done the whole copy is
                j.ev = kev;
                j.stage = 3;
            }
        } else if (j.stage == 3 && !push && ev_done(j.ev)) {
            if (j.bounce >= 0) { im.pc_bfree.push_back(j.bounce); j.bounce = -1; }
            c.pc_busy = false;
            if (check && !pc_check(im, c, j.key, slot)) {   // debug: keep it on the CPU
                ++im.pc_bad;
                c.pc_key[(size_t) j.slot] = -1;
                im.pc_where[(size_t) j.key] = -1;
                std::fprintf(stderr, "xe tier: promotion of layer %lld expert %lld on card %d: slot differs from a fresh repack\n",
                             (long long) l, (long long) e, c.idx);
            } else {
                im.card_of[(size_t) j.key] = (int8_t) c.idx;
                im.dptr[(size_t) j.key] = slot;
                c.pc_key[(size_t) j.slot] = j.key;
                c.pc_tick[(size_t) j.slot] = im.pc_clock;
                im.pc_where[(size_t) j.key] = (c.idx << 16) | j.slot;
                ++im.pc_done;
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
    if (im.pc_where[key] != -1 || !im.aptr[key]) return;
    static const int pmin = std::max(1, env_i("STRATA_XE_PROMOTE_MIN", 2));
    if (im.pc_uses.size() != im.card_of.size()) im.pc_uses.assign(im.card_of.size(), 0);
    if (im.pc_uses[key] < 255) ++im.pc_uses[key];
    if (im.pc_uses[key] < pmin) return;
    const int NC = (int) im.cards.size();
    XeCard* cd = nullptr;
    for (int k = 0; k < NC && !cd; ++k) {
        XeCard& c = *im.cards[(size_t) ((im.pc_rr + k) % NC)];
        if (!c.dense && c.pc_n > 0 && !c.pc_busy) cd = &c;
    }
    if (!cd) { ++im.pc_busy_skip; return; }
    if (im.pc_bfree.empty()) { ++im.pc_busy_skip; return; }
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
        im.pc_where[(size_t) old] = -1;
        ++im.pc_evicted;
    }
    cd->pc_key[(size_t) slot] = -2;
    cd->pc_busy = true;
    im.pc_where[key] = -2;
    im.pc_jobs.emplace_back();
    Ds41XeImpl::PcJob& j = im.pc_jobs.back();
    j.key = (int64_t) key;
    j.card = cd->idx;
    j.slot = slot;
    j.bounce = im.pc_bfree.back();   // the helper repacks it into a bounce (pinned arena copies too)
    im.pc_bfree.pop_back();
    { std::lock_guard<std::mutex> lk(im.pc_mu); im.pc_fill.push_back(&j); }
    im.pc_cv.notify_one();
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
        for (int sl = 0; sl < c.pc_n; ++sl) {
            const int64_t k = c.pc_key[(size_t) sl];
            if (k >= 0) { im.card_of[(size_t) k] = -1; im.dptr[(size_t) k] = nullptr; }
        }
        c.pc_mem = nullptr;   // the flex region (prefetch buffers next)
        c.pc_n = 0;
        c.pc_key.clear();
        c.pc_tick.clear();
        c.pc_busy = false;
    }
    im.pc_jobs.clear();
    im.pc_bfree.clear();
    for (int b = 0; b < kPcBounce; ++b) im.pc_bfree.push_back(b);
    std::fill(im.pc_where.begin(), im.pc_where.end(), -1);
    im.pc_ready = false;
}

/// One decode / verify call: nt <= 8 tokens.  out_host (nt x n_embd) or out_dev (the dense card) gets the sum.
bool run_small(Ds41XeImpl& im, int64_t l, int nt, const int32_t* ids, const float* w, const float* x, float* out_host,
               void* out_dev) {
    const double tl_h0 = Ds41XeTier::tl_on() && nt == 1 ? tl_now_us() : 0;
    const int64_t H = im.g.n_embd;
    const int K = (int) im.g.top_k;
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
            if (++sp_n % 1000 == 0)
                std::fprintf(stderr, "xe %s prof (%lld calls, %.2f card jobs/call, %.2f cpu experts/call): per call prep %.1f us, "
                             "submit %.1f us, cpu %.1f us, sum+wait %.1f us = %.1f us\n", spk ? "verify" : "decode", (long long) sp_n,
                             (double) sp_jobs / sp_n, sp[4] / sp_n, 1e3 * sp[0] / sp_n, 1e3 * sp[1] / sp_n, 1e3 * sp[2] / sp_n,
                             1e3 * sp[3] / sp_n, 1e3 * (sp[0] + sp[1] + sp[2] + sp[3]) / sp_n);
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

template <class T>
bool grow_dev(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); sycl::free(p, q); }
    p = sycl::malloc_device<T>(need, q);
    cap = p ? need : 0;
    return p != nullptr;
}
template <class T>
bool grow_host(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); sycl::free(p, q); }
    p = sycl::malloc_host<T>(need, q);
    cap = p ? need : 0;
    return p != nullptr;
}

/// The Level Zero driver (NEO 26.35) deadlocks when several host threads enqueue to different devices at once while
/// one of them copies from PAGEABLE host memory (seen 2026-10-19: three appendUSMMemcpy and one appendKernelLaunch
/// blocked on two driver mutexes).  So the card threads take this lock around every enqueue (never around a wait),
/// and pageable sources go through a pinned bounce buffer.
/// larry: a priority waiter (the pipelined batch's dense half: Ds41XeTier::submit_priority) makes the tier's threads
/// hold off their next acquisition while it waits.
struct PrioMutex {
    std::mutex m;
    std::atomic<int> hi{0};
    void lock() {
        while (hi.load(std::memory_order_acquire) > 0) std::this_thread::yield();
        m.lock();
    }
    void unlock() { m.unlock(); }
};
PrioMutex g_submit;

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
    ps.timed = false;
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
    ps.staged = sused > 0 ? 1 : 0;
    if (ps.staged) im.pf_cv.notify_all();
}

/// A stager thread (STRATA_XE_STAGERS of them): copies planned slots' pageable experts into their pinned stage, an
/// expert at a time, the older request (lower layer) first; whoever finishes a slot's last expert marks it staged.
/// No SYCL calls here: the Level Zero driver must not see another submitting thread.
void pf_stager(Ds41XeImpl& im) {
    const int64_t E = im.g.n_experts;
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
            while (ps.next < E && !(ps.card[(size_t) ps.next] >= 0 && ps.soff[(size_t) ps.next] != SIZE_MAX)) ++ps.next;
            if (ps.next < E) { e = ps.next++; ++ps.active; }
            if (e < 0 && ps.active == 0) { ps.staged = 0; im.pf_cv.notify_all(); continue; }
            if (e < 0) continue;
        }
        Ds41XeImpl::PfSlot& ps = im.pf[s];
        const int64_t l = ps.layer;
        const double ts = xe_prof() ? now_ms() : 0;
        std::memcpy(ps.stage + ps.soff[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));
        if (xe_prof()) { g_bp.st_us += (int64_t) ((now_ms() - ts) * 1000.0); g_bp.st_bytes += (int64_t) im.bbytes(l, e); }
        std::lock_guard<std::mutex> lk(im.pf_mu);
        --ps.active;
        if (ps.active == 0 && ps.next >= E) { ps.staged = 0; im.pf_cv.notify_all(); }
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
        if (ps) pf_enqueue(im, pslot, true);
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
        if (c >= 0) cl[(size_t) c].push_back((int32_t) e);
    }
    for (int64_t e = 0; e < E; ++e) {
        if (by_e[(size_t) e].empty() || im.card_of[(size_t) im.key(l, e)] >= 0) continue;
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
        size_t best = 0;
        for (size_t k = 1; k < NC; ++k) if (sbytes[k] < sbytes[best]) best = k;
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
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.busy = !cl[(size_t) c.idx].empty();
        if (!c.busy) continue;
        const double tg0 = prof ? now_ms() : 0;
        uint8_t* const st0 = c.stream;   // a card keeping experts keeps its buffer (no regrow)
        const size_t sneed = std::min<size_t>(stream_cap, std::max<size_t>((size_t) sbytes[(size_t) c.idx], 1));
        // with the prefetch on, a regrowth goes straight to the cap (planned into the prefetch's reserve)
        const size_t sreq = !kept[(size_t) c.idx].empty() ? c.stream_bytes : im.pf_on && sneed > c.stream_bytes ? stream_cap : sneed;
        if (!grow_dev(c.stream, c.stream_bytes, sreq, c.q)) {
            std::fprintf(stderr, "xe tier: card %d: stream buffer allocation failed (%.2f GiB free)\n", c.idx, c.free_vram() / kGiB);
            return false;
        }
        if (c.stream != st0 && !kept[(size_t) c.idx].empty()) {
            std::fprintf(stderr, "xe tier: card %d: stream buffer moved with experts kept in it\n", c.idx);
            return false;
        }
        if (!grow_dev(c.cX, c.cX_cap, (size_t) n * H, c.q) || !grow_dev(c.cOUT, c.cOUT_cap, (size_t) n * H, c.q)) {
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
        c.q.memcpy(c.cX, x_dev ? x_dev : x, (size_t) n * H * 4);   // x_dev: card-to-card from the dense card
        const double tq1 = prof ? now_ms() : 0;
        c.q.memcpy(c.cW, w, (size_t) n * K * 4);
        const double tq2 = prof ? now_ms() : 0;
        c.q.memset(c.cOUT, 0, (size_t) n * H * 4);
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
    auto card_work = [&](XeCard& c) -> bool {
        std::vector<int32_t> S((size_t) n * K);
        std::vector<int64_t> soff((size_t) E, -1);
        std::vector<int32_t> round_of((size_t) E, 0);
        std::vector<int32_t> ptok;
        std::vector<uint8_t> ftmp;
        if (ps && ps->enqueued) {   // the prefetched experts' copies land before any kernel of this call
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.ext_oneapi_submit_barrier({c.pev[pslot]});
        }
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
        int nr = 1;
        size_t used = 0, pused = 0;
        const std::vector<int32_t>& kp = kept[(size_t) c.idx];   // round 0, where they are; new copies go after them
        for (int32_t e : kp) soff[(size_t) e] = kept_off[(size_t) e];
        if (!kp.empty()) used = c.st_end;
        g_bp.kept += (int64_t) kp.size();
        c.st_layer = -1;   // set again once this call's copies are known to be in place
        for (int32_t e : streamed[(size_t) c.idx]) {
            const size_t b = (im.bbytes(l, e) + 255) & ~(size_t) 255;
            const size_t pg = im.apin[(size_t) im.key(l, e)] ? 0 : b;
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
                    if (pinned) {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.stream + soff[(size_t) e], hb, bytes);
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
                lap(g_bp.stream_us);
            }
            auto dptr_of = [&](int32_t e) -> const uint8_t* {
                const size_t key = (size_t) im.key(l, e);
                if (im.card_of[key] >= 0) return im.dptr[key];
                if (ps && ps->card[(size_t) e] >= 0) return c.pbuf[pslot] + ps->off[(size_t) e];
                return c.stream + soff[(size_t) e];
            };
            auto in_round = [&](int32_t e) {
                return im.card_of[(size_t) im.key(l, e)] >= 0 || (ps && ps->card[(size_t) e] >= 0) ? r == (overlap ? -1 : 0)
                                                                                               : round_of[(size_t) e] == r;
            };
            // resident experts are stored planar (int8 kernels); streamed ones arrive in the file layout (F32 kernels)
            auto is_planar = [&](int32_t e) { return im.planar && im.card_of[(size_t) im.key(l, e)] >= 0; };
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
                const int64_t RB = std::max<int64_t>(1024, (gm_max + 7) & ~(int64_t) 7);
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
                    grown = grow_dev(c.gscratch, c.gscratch_cap, c.gscratch_cap < need ? need + need / 4 : need, c.q);   // headroom: few regrowths
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
                xe::DqJob* hjobs = (xe::DqJob*) c.dq_host;
                xe::DqTile* htiles = (xe::DqTile*) (c.dq_host + jbytes);
                ptok.assign((size_t) R, 0);
                std::fill(S.begin(), S.end(), -1);
                for (size_t i = 0; i < NG; ++i) {
                    const auto& v = by_e[(size_t) ge[i]];
                    for (size_t j = 0; j < v.size(); ++j) { S[(size_t) v[j]] = (int32_t) (ebase[i] + (int64_t) j); ptok[(size_t) (ebase[i] + (int64_t) j)] = v[j] / K; }
                    for (int64_t j = (int64_t) v.size(); j < ebase[i + 1] - ebase[i]; ++j) ptok[(size_t) (ebase[i] + j)] = v[0] / K;   // padding rows: any token
                }
                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    c.q.memcpy(dtok, ptok.data(), ptok.size() * 4);
                    c.q.memcpy(c.cS, S.data(), S.size() * 4);
                }
                c.q.wait();   // S and ptok are reused below
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
                    xe::xe_gather_rows_f16(c.q, c.cX, dtok + r0, (int) rows, (int) H, Xg);
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
                        c.q.memcpy(djobs + jo0, hjobs + jo0, (jo - jo0) * sizeof(xe::DqJob));
                        for (int g = 0; g < 6; ++g) {
                            if (gt[g].empty()) continue;
                            std::memcpy(htiles + to, gt[g].data(), gt[g].size() * sizeof(xe::DqTile));
                            c.q.memcpy(dtiles + to, htiles + to, gt[g].size() * sizeof(xe::DqTile));
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
                    xe::xe_gather(c.q, Yg, c.cS, c.cW, n, K, (int) H, c.cOUT, true);
                }
                g_bp.gemm_e += (int64_t) ge.size();
                lap(g_bp.gemm_us);
            } else if (!ge.empty()) {
                const size_t need = (size_t) gm_max * H * 2 + (size_t) 2 * F * H * 2 + (size_t) H * F * 2 +
                                    (size_t) gm_max * 2 * F * 4 + (size_t) gm_max * F * 2 + (size_t) gp * H * 4 + (size_t) gp * 4 + 8192;
                if (c.gscratch_cap < need) c.q.wait();
                bool grown;
                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    grown = grow_dev(c.gscratch, c.gscratch_cap, need, c.q);
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
                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    c.q.memcpy(dtok, ptok.data(), ptok.size() * 4);
                    c.q.memcpy(c.cS, S.data(), S.size() * 4);
                }
                c.q.wait();   // S and ptok are reused below
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
                    xe::xe_gather_rows_f16(c.q, c.cX, dtok + base, (int) m, (int) H, Xg);
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
                    xe::xe_gather(c.q, Yg, c.cS, c.cW, n, K, (int) H, c.cOUT, true);
                }
                g_bp.gemm_e += (int64_t) ge.size();
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
                    c.q.wait();   // cjobs / cjobs_host and cS are read by the previous batch / the GEMM gather
                    std::fill(S.begin(), S.end(), -1);
                    int max_nt = 1;
                    for (size_t q = 0; q < nb; ++q) {
                        c.cjobs_host[q] = js[b0 + q];
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
                        xe::xe_quant_rows(c.q, c.cX, n, (int) H, c.cxq);
                        xq_done = true;
                    }
                    {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.cjobs, c.cjobs_host, nb * sizeof(XeJob));
                        c.q.memcpy(c.cS, S.data(), S.size() * 4);
                    }
                    c.q.wait();
                    std::lock_guard<PrioMutex> lk(g_submit);
                    const double tlk = prof ? now_ms() : 0;
                    struct LkT { bool on; double t0; ~LkT() { if (on) g_bp.lk_jobs_us += (int64_t) ((now_ms() - t0) * 1000.0); } } lkt{prof, tlk};
                    if (pl) xe::xe_experts_y_q8(c.q, c.cjobs, (int) nb, c.cxq, c.chq, c.cY, (int) H, (int) F, im.L, max_nt);
                    else xe::xe_experts_y(c.q, c.cjobs, (int) nb, c.cX, c.cH, c.cY, (int) H, (int) F, im.L);
                    xe::xe_gather(c.q, c.cY, c.cS, c.cW, n, K, (int) H, c.cOUT, true);
                }
                return true;
            };
            if (!run_jobs(jobs, jslots, true) || !run_jobs(jobs_raw, jslots_raw, false)) return false;
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
        if (!out_dev) {
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.memcpy(c.cOUT_host, c.cOUT, (size_t) n * H * 4);
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
        c.q.wait();
        if (prof) {   // the lowest free VRAM at the end of a call (every buffer of the call allocated)
            const int64_t fr = c.free_vram();
            std::atomic<int64_t>& mf = g_bp.min_free[c.idx & 7];
            int64_t cur = mf.load();
            while (fr >= 0 && (cur < 0 || fr < cur) && !mf.compare_exchange_weak(cur, fr)) {}
        }
        return true;
    };
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
        for (auto& t : th) t.join();
    }
    for (size_t k = 0; k < NC; ++k) if (!ok[k]) return false;
    im.st.hits += hits;
    im.st.pcie += pcie + n_pf;
    if (ps) {   // the layer's last call: the next layer's copies (if not on the queues yet), this slot takes l + 2
        im.pf_used += n_pf;
        if (last_call) {
            if (im.pf[pslot ^ 1].layer == l + 1) pf_enqueue(im, pslot ^ 1, true);
            ps->layer = -1;
            if (l + 2 <= im.pf_last) pf_plan(im, pslot, l + 2);
        }
    }
    const double t_cards = now_ms();
    // ---- combine: the other cards' parts copied card-to-card onto the dense card and added there
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    const size_t NE = (size_t) n * H;
    if (out_dev) {
        if (!dc) { std::fprintf(stderr, "xe tier: device output without a dense card\n"); return false; }
        std::vector<const XeCard*> others;
        for (auto& cp : im.cards) if (cp->busy && !cp->dense) others.push_back(cp.get());
        sycl::event cev;
        if (cmb > 0) {   // STRATA_XE_COMBINE: every busy card's part is in its slot (its thread waited for the push)
            std::lock_guard<PrioMutex> lk(g_submit);
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
                    o[i] = sm;
                });
            } else {
                const float* parts = dc->cX2;
                cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                    float sm = own ? own[i] : 0.0f;
                    for (int k = 0; k < np; ++k) sm += parts[(size_t) idx[k] * NE + i];
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
                o[i] = s;
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

void Ds41XeTier::arm(int64_t l, int nt) {
    Ds41XeImpl& im = *im_;
    disarm(im);   // an earlier arm nobody took
    if (!im.arm_on || !im.relq || !im.dense_q || !im.arm_x || nt < 1 || nt > kMaxJobTok || (nt > 1 && !im.arm_multi)) return;
    if (im.arm_err && *(volatile int*) im.arm_err) {   // a spin kernel timed out once: arming stays off
        std::fprintf(stderr, "xe tier: an armed card waited out its release - STRATA_XE_ARM off\n");
        im.arm_on = im.arm_multi = false;
        return;
    }
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    if (!dc || !dc->dslots) return;
    const int64_t H = im.g.n_embd;
    const int K = (int) im.g.top_k;
    static const int kb[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, kMaxDecodeJobs};
    int ng = kMaxDecodeJobs;
    for (int b : kb) if (b >= K * nt) { ng = b; break; }
    const int mv = nt <= 1 ? 1 : nt <= 2 ? 2 : nt <= 4 ? 4 : 8;
    const size_t xoff = (kMaxDecodeJobs * sizeof(XeJob) + kStageAlign - 1) / kStageAlign * kStageAlign;
    float* slots = dc->dslots;
    const sycl::event prev = im.last_sum;   // the previous call's sum reads the slots these sequences write
    int* errp = im.arm_err;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.arm_flag || !c.arm_null_ok) continue;
        std::lock_guard<PrioMutex> lk(g_submit);
        int* flag = c.arm_flag;
        int* seen = c.arm_seen;
        c.q.submit([&](sycl::handler& h) {
            h.depends_on(prev);
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
        if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 10);   // released: the card starts
        c.q.memcpy(c.stage_dev, c.stage_host, (size_t) ng * sizeof(XeJob));
        card_experts(im, c, ng, nt, mv, xoff, c.OUT, im.arm_x);
        if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 11);
        c.arm_ev = c.q.memcpy(slots + (size_t) c.idx * kMaxJobTok * H, c.OUT, (size_t) nt * H * 4);
        if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 14);   // its copy is done
        c.armed = true;
        c.arm_ng = ng;
    }
    im.armed_l = l;
    im.armed_nt = nt;
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
    pc_drop(im);   // the promoted experts' VRAM back first (the buffers below are sized from what is free)
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
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense) continue;
        if (c.stream) { c.q.wait(); sycl::free(c.stream, c.q); c.stream = nullptr; c.stream_bytes = 0; }   // regrown on demand
        c.st_layer = -1;
        const int64_t have = (int64_t) (c.cX_cap + c.cOUT_cap) * 4 + (int64_t) c.cS_cap * 8 + (int64_t) c.gscratch_cap +
                             (c.cjobs_cap ? BJ * kMaxJobTok * (F + H) * 4 : 0) + (int64_t) c.cxq_rows * H * 2;
        const int64_t reserve = std::max<int64_t>(0, rb - have) + (int64_t) (margin * kGiB);
        const int64_t fr = c.free_vram() + (int64_t) c.flex_cap;   // the flex region becomes the buffers
        int64_t room = fr > 0 ? (fr - reserve) / 2 : 0;
        const size_t cap = (size_t) std::max<int64_t>(0, std::min<int64_t>((int64_t) per, room)) & ~(size_t) 255;
        if (cap < (size_t) im.blob_max) { c.pbuf_cap = 0; continue; }
        // sized to this prompt: a larger region left over would over-commit the card (the xe driver then spills buffers
        // into host RAM: a 16K prompt after an 8K one pushed larry past the watchdog, 2026-10-21 04:30)
        if (!c.flex || c.flex_cap < 2 * cap || c.flex_cap > 2 * cap + ((size_t) 64 << 20)) {
            if (c.flex) { c.q.wait(); local_free(c.q, c.flex); }
            c.flex = local_alloc(c.q, 2 * cap);
            c.flex_cap = c.flex ? 2 * cap : 0;
            if (!c.flex) { c.pbuf_cap = 0; continue; }
        }
        c.pbuf[0] = c.flex;
        c.pbuf[1] = c.flex + cap;
        c.pbuf_cap = cap;
    }
    // the calls' buffers now, once (they are in the reserve above): the overflow stream buffer at its planned size, X /
    // OUT / the slot maps for n_call tokens
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.pbuf_cap) continue;
        const size_t sb = (size_t) (env_d("STRATA_XE_PREFETCH_STREAM_GIB", 0.25) * kGiB) & ~(size_t) 255;
        if (!grow_dev(c.stream, c.stream_bytes, sb, c.q) || !grow_dev(c.cX, c.cX_cap, (size_t) nc * H, c.q) ||
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
        const int ns = std::max(1, env_i("STRATA_XE_STAGERS", 4));
        for (int i = 0; i < ns; ++i) im.pf_thr.emplace_back([&im] { pf_stager(im); });
    }
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
        std::fprintf(stderr, "xe prefetch: %lld experts (%.2f GB) planned, %lld used\n", (long long) im.pf_planned,
                     im.pf_bytes / 1e9, (long long) im.pf_used);
    im.pf[0].layer = im.pf[1].layer = -1;
    im.pf_on = false;
}

void Ds41XeTier::close() {
    if (!im_) return;
    Ds41XeImpl& im = *im_;
    disarm(im);
    pc_drop(im);
    if (im.pc_thr.joinable()) {
        { std::lock_guard<std::mutex> lk(im.pc_mu); im.pc_stop = true; }
        im.pc_cv.notify_all();
        im.pc_thr.join();
    }
    if (im.pc_bounce) { sycl::free(im.pc_bounce, im.ctx); im.pc_bounce = nullptr; }
    if (im.pc_done || im.pc_busy_skip)
        std::fprintf(stderr, "xe tier: promotion cache: %lld promoted, %lld hits, %lld evicted, %lld skipped (cards busy), %lld bad; "
                     "%lld chunks, host push %.1f ms, poll %.1f ms\n", (long long) im.pc_done, (long long) im.pc_hit, (long long) im.pc_evicted,
                     (long long) im.pc_busy_skip, (long long) im.pc_bad, (long long) im.pc_chunks, im.pc_push_us / 1000.0, im.pc_poll_us / 1000.0);
    if (im.relq) { im.relq->wait(); im.relq.reset(); }
    if (im.os_calls || im.os_declined)
        std::fprintf(stderr, "xe tier: attention output split %lld calls (%zu layers), declined %lld\n", (long long) im.os_calls,
                     im.osplits.size(), (long long) im.os_declined);
    if (im.arm_used || im.arm_disarmed)
        std::fprintf(stderr, "xe tier: armed card shares used %lld, disarmed calls %lld\n", (long long) im.arm_used,
                     (long long) im.arm_disarmed);
    if (im.run_thr.joinable()) im.run_thr.join();
    prefetch_end();
    if (!im.pf_thr.empty()) {
        { std::lock_guard<std::mutex> lk(im.pf_mu); im.pf_stop = true; }
        im.pf_cv.notify_all();
        for (auto& t : im.pf_thr) t.join();
        im.pf_thr.clear();
    }
    for (auto& sl : im.pf) if (sl.stage) { sycl::free(sl.stage, im.ctx); sl.stage = nullptr; }
    for (auto& kv : im.heads)   // the head split's slices and scratch, on their cards (before the cards go)
        for (size_t i = 0; i < kv.second.c.size(); ++i) {
            sycl::queue& q = kv.second.c[i]->q;
            q.wait();
            sycl::free(kv.second.w[i], q); sycl::free(kv.second.xq[i], q); sycl::free(kv.second.xd[i], q); sycl::free(kv.second.y[i], q);
        }
    im.heads.clear();
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.free_experts();
        auto fr = [&](void* p) { if (p) sycl::free(p, c.q); };
        if (c.pbuf[0] == c.flex) c.pbuf[0] = c.pbuf[1] = nullptr;
        if (c.flex) { c.q.wait(); local_free(c.q, c.flex); }
        c.flex = nullptr;
        fr(c.stage_dev); fr(c.stage_host); fr(c.H); fr(c.OUT); fr(c.part_host); fr(c.arm_flag); fr(c.arm_seen); fr(c.arm_src);
        fr(c.cX); fr(c.cOUT); fr(c.cOUT_host); fr(c.cS); fr(c.cW); fr(c.cjobs); fr(c.cjobs_host); fr(c.dq_host); fr(c.cH); fr(c.cY); fr(c.stream);
        fr(c.gscratch); fr(c.cX2); fr(c.cX2h); fr(c.bounce); fr(c.dslots); fr(c.rstage); fr(c.xt); fr(c.ht);
        for (xe::XeQ8* a : {&c.xq, &c.hq, &c.cxq, &c.chq}) { fr(a->q); fr(a->d); fr(a->s); }
    }
    im.cards.clear();
    for (uint8_t* p : im.achunks) {
        if (sycl::get_pointer_type(p, im.ctx) == sycl::usm::alloc::host) sycl::free(p, im.ctx);
        else std::free(p);
    }
    im.achunks.clear();
    if (im.hsum) { im.hsum_ev.wait(); sycl::free(im.hsum, im.ctx); im.hsum = nullptr; }
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
    return im_->card_of[(size_t) im_->key(l, e)] >= 0;
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
