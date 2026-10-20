// tools/v41/engine/xe/ds41_xe_tier.cpp - see ds41_xe_tier.hpp.
#include "ds41_xe_tier.hpp"
#include "xe_kernels.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <thread>

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
constexpr int kMaxDecodeJobs = 64;               ///< a verify pass of 8 tokens routes at most 48 distinct experts
constexpr size_t kStageAlign = 256;

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
    /// decode / verify: the card's command sequence of a call (stage copy, quantise, gate/up, down, copy to the dense
    /// card's slot) recorded once per shape (jobs, tokens, kernel variant) and replayed - one submission, not five
    std::map<uint32_t, std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>>> dgraphs;
    // prompt-chunk scratch (grown on demand)
    float* cX = nullptr; size_t cX_cap = 0;          ///< floats
    float* cOUT = nullptr; size_t cOUT_cap = 0;      ///< floats
    float* cOUT_host = nullptr; size_t cOUT_host_cap = 0;
    int32_t* cS = nullptr; float* cW = nullptr; size_t cS_cap = 0;   ///< entries
    XeJob* cjobs = nullptr; float* cH = nullptr; float* cY = nullptr; int cjobs_cap = 0;
    uint8_t* stream = nullptr; size_t stream_bytes = 0;   ///< streamed (arena / file) experts of one chunk round
    uint8_t* gscratch = nullptr; size_t gscratch_cap = 0;  ///< the GEMM path's operands (bytes)
    float* cX2 = nullptr; size_t cX2_cap = 0;              ///< the dense card's copy of the other cards' sum
    uint8_t* bounce = nullptr; size_t bounce_cap = 0;      ///< pinned staging for streamed experts in pageable memory
    // the planar int8 path (xe_planar.cpp): every expert a card holds is stored repacked
    uint8_t* rstage = nullptr;      ///< [blob_max]: a file-layout blob on its way to its planar slot
    std::mutex rmu;                 ///< makes (copy into rstage, repack out of it) one unit for concurrent uploaders
    xe::XeQ8 xq, hq;                ///< decode / verify: X [8][n_embd], H [kMaxDecodeJobs][8][n_ff]
    xe::XeQ8 cxq; size_t cxq_rows = 0;   ///< prompt chunks: X [n][n_embd]
    xe::XeQ8 chq; int chq_jobs = 0;      ///< prompt chunks: H [jobs][8][n_ff]

    uint8_t* alloc_expert(size_t bytes) {
        if (slabs.empty() || slab_used + bytes > kSlab) {
            uint8_t* p = sycl::malloc_device<uint8_t>(kSlab, q);
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
        for (uint8_t* p : slabs) sycl::free(p, q);
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
    // STRATA_GPU_TL: the decode timeline's profiling tags - (id, event) on the dense queue; per card (start, end)
    std::vector<std::pair<int, sycl::event>> tl;
    std::vector<std::vector<std::pair<sycl::event, sycl::event>>> tl_card;
    double tl_sum[8][8] = {};   // [from id][to id] device us
    long tl_n[8][8] = {};
    std::vector<double> tl_csum;
    std::vector<long> tl_cn;
    long tl_tokens = 0;
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
    std::vector<std::pair<int32_t, int32_t>> dense_set;   ///< experts currently on the dense card
    int64_t resident_n = 0;
    // CPU pool for the arena experts of a decode / verify call
    std::unique_ptr<cpu::ExpertPool> pool;
    std::vector<cpu::ExpertJobMulti> cjobs;
    std::vector<cpu::ActQ> actq;           ///< per token
    std::vector<uint8_t> nact;             ///< [token][format slot][kNativeActBytes]
    std::vector<int> nact_fmt;             ///< format slots: distinct gu_act values
    std::vector<float> cpart;              ///< [job][8][n_embd] CPU expert outputs
    float* hsum = nullptr;                 ///< [8][n_embd] host-side sum (pinned: its copy to the dense card is async)
    std::vector<uint8_t> ftmp;             ///< file-tier reads (decode)
    Ds41MoeStats st;
    std::mutex mu;
    bool planar = true;                ///< STRATA_XE_PLANAR (default 1): experts repacked on the cards, int8 kernels
    bool tier_graphs = true;           ///< STRATA_XE_GRAPH (default 1): decode / verify card work as replayed SYCL graphs
    sycl::queue* dense_q = nullptr;    ///< set_dense_queue: decode sums go there, no host wait
    sycl::event last_sum;              ///< the previous decode call's sum on dense_q (its slots / stages are reused)
    sycl::event hsum_ev;               ///< the copy of hsum (the CPU part) into the dense card's slot

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
        if (c.dense) {
            c.dslots = sycl::malloc_device<float>((im.cards.size() + 1) * kMaxJobTok * H, c.q);
            if (!c.dslots) { err = "xe tier: slot allocation failed"; return false; }
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
    const int64_t pin_cap = (int64_t) (env_d("STRATA_XE_PINNED_GIB", 16.0) * kGiB);
    int64_t pinned = 0;
    for (size_t c = 0; c < nchunks; ++c) {
        uint8_t* p = nullptr;
        if (pinned + (int64_t) chunk_size[c] <= pin_cap) {
            p = sycl::malloc_host<uint8_t>(chunk_size[c], im.ctx);
            if (p) pinned += (int64_t) chunk_size[c];
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
    for (size_t i = 0; i < in_arena.size(); ++i)
        im.aptr[(size_t) im.key(in_arena[i].first, in_arena[i].second)] = im.achunks[where[i] >> 40] + (where[i] & (((size_t) 1 << 40) - 1));
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
    im.placed = true;
    std::fprintf(stderr, "xe tier: placed in %.1f s (cards %.1f s, arena %.1f s)\n%s", (now_ms() - t0) / 1000.0,
                 (t1 - t0) / 1000.0, (now_ms() - t1) / 1000.0, report().c_str());
    return true;
}

bool Ds41XeTier::seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    Ds41XeImpl& im = *im_;
    if (!im.placed && !build_arena_ranked(ranked, err)) return false;
    if (im.dense < 0 || !im.cache_enabled || !im.dense_set.empty()) return true;
    XeCard& c = *im.cards[(size_t) im.dense];
    // the dense card's share stays small: ggml-sycl's pools are virtual (VMM) and commit while decode runs, so "free
    // VRAM" right after a prompt overstates what is left, and the xe driver does not refuse an over-commit - it
    // spills buffers into host RAM (2026-10-19: card 0 at 98 % with 15 GiB of experts, 12 GiB spilled)
    const double cap = env_d("STRATA_XE_DENSE_GIB", 10.0);
    double gib = im.slot_gib;
    if (cap >= 0) gib = std::min(gib, cap);
    // whatever the engine measured, leave the dense half's graphs a reserve of what is free NOW
    const int64_t fr = c.free_vram();
    if (fr > 0) gib = std::min(gib, fr / kGiB - env_d("STRATA_XE_DENSE_RESERVE_GIB", 8.0));
    if (gib < 0.5) return true;
    c.budget = (int64_t) (gib * kGiB);
    const double t0 = now_ms();
    // the next experts in ranked order that live in the arena only.  Pageable arena experts go through a pinned
    // bounce: a DMA straight from pageable memory makes the driver pin the source pages, and 15 GiB queued at once
    // pushed larry's driver-held RAM past the watchdog (58 GiB, 2026-10-19; TTM's cap for all of it is ~60.7 GiB).
    const size_t half = std::max<size_t>((size_t) im.blob_max, (size_t) 64 << 20);
    if (c.bounce_cap < 2 * half) {
        if (c.bounce) sycl::free(c.bounce, c.q);
        c.bounce = sycl::malloc_host<uint8_t>(2 * half, c.q);
        c.bounce_cap = c.bounce ? 2 * half : 0;
        if (!c.bounce) { err = "xe tier: bounce allocation failed"; return false; }
    }
    sycl::event hev[2];
    int hcur = 0;
    size_t hused = 0;
    for (const auto& le : ranked) {
        if (le.first < 0 || le.first >= im.g.n_layers || le.second < 0 || le.second >= im.g.n_experts) continue;
        const size_t key = (size_t) im.key(le.first, le.second);
        if (im.card_of[key] >= 0 || !im.aptr[key]) continue;
        const size_t b = im.bbytes(le.first, le.second);
        if (c.used + (int64_t) b > c.budget) continue;
        uint8_t* p = c.alloc_expert(b);
        if (!p) break;
        if (sycl::get_pointer_type(im.aptr[key], im.ctx) == sycl::usm::alloc::host) {
            upload_planar(im, c, le.first, le.second, im.aptr[key], p);
        } else {
            if (hused + b > half) { hcur ^= 1; hused = 0; hev[hcur].wait(); }
            std::memcpy(c.bounce + (size_t) hcur * half + hused, im.aptr[key], b);
            hev[hcur] = upload_planar(im, c, le.first, le.second, c.bounce + (size_t) hcur * half + hused, p);
            hused += (b + 255) & ~(size_t) 255;
        }
        im.card_of[key] = (int8_t) c.idx;
        im.dptr[key] = p;
        im.dense_set.push_back(le);
    }
    c.q.wait();
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
    std::fprintf(stderr, "xe tier: dense card %d seeded with %zu experts (%.2f GiB) in %.2f s\n", c.idx,
                 im.dense_set.size(), c.used / kGiB, (now_ms() - t0) / 1000.0);
    return true;
}

void Ds41XeTier::set_slot_gib(double gib) { im_->slot_gib = gib; }
void Ds41XeTier::set_dense_queue(void* q) { im_->dense_q = (sycl::queue*) q; }

bool Ds41XeTier::tl_on() { static const bool on = env_i("STRATA_GPU_TL", 0) != 0; return on; }

void Ds41XeTier::tl_tag(int id) {
    Ds41XeImpl& im = *im_;
    if (!tl_on() || !im.dense_q) return;
    im.tl.emplace_back(id, sycl::ext::oneapi::experimental::submit_profiling_tag(*im.dense_q));
}

void Ds41XeTier::tl_flush() {
    Ds41XeImpl& im = *im_;
    if (!tl_on()) return;
    auto ns = [](const sycl::event& e) { return (double) e.get_profiling_info<sycl::info::event_profiling::command_end>(); };
    if (!im.tl.empty()) im.tl.back().second.wait();
    for (size_t i = 1; i < im.tl.size(); ++i) {
        const int a = im.tl[i - 1].first, b = im.tl[i].first;
        if (a < 0 || a > 7 || b < 0 || b > 7) continue;
        im.tl_sum[a][b] += 1e-3 * (ns(im.tl[i].second) - ns(im.tl[i - 1].second));
        ++im.tl_n[a][b];
    }
    im.tl.clear();
    im.tl_csum.resize(im.cards.size(), 0.0);
    im.tl_cn.resize(im.cards.size(), 0);
    im.tl_card.resize(im.cards.size());
    for (size_t c = 0; c < im.tl_card.size(); ++c) {
        for (auto& se : im.tl_card[c]) {
            se.second.wait();
            im.tl_csum[c] += 1e-3 * (ns(se.second) - ns(se.first));
            ++im.tl_cn[c];
        }
        im.tl_card[c].clear();
    }
    if (++im.tl_tokens % 64) return;
    std::fprintf(stderr, "gpu timeline (%ld tokens), dense card us per step:", im.tl_tokens);
    for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b)
            if (im.tl_n[a][b]) std::fprintf(stderr, " %d->%d %.1f (x%ld)", a, b, im.tl_sum[a][b] / im.tl_n[a][b], im.tl_n[a][b]);
    std::fprintf(stderr, "; cards (dispatch -> copy done):");
    for (size_t c = 0; c < im.tl_csum.size(); ++c)
        if (im.tl_cn[c]) std::fprintf(stderr, " %zu: %.1f (x%ld)", c, im.tl_csum[c] / im.tl_cn[c], im.tl_cn[c]);
    std::fprintf(stderr, "\n");
}
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

int64_t Ds41XeTier::shrink_cache_for(int64_t, std::string&) { return -1; }
int64_t Ds41XeTier::regrow_cache(std::string&) { return 0; }

// ================================ decode / verify ================================

namespace {

/// The CPU pool's share of a call: `ents` (all in the arena / file tier), outputs weighted into `hs` [8][n_embd].
void cpu_ents(Ds41XeImpl& im, int64_t l, const std::vector<const Ent*>& ents, const float* x, float* hs) {
    const int64_t H = im.g.n_embd;
    if (ents.empty()) return;
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

/// A card's share of a decode / verify call: its n jobs (staged in c.stage_dev, X at xoff) into dst [nt][n_embd].
sycl::event card_experts(Ds41XeImpl& im, XeCard& c, int n, int nt, int max_nt, size_t xoff, float* dst) {
    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
    const XeJob* jobs = (const XeJob*) c.stage_dev;
    const float* X = (const float*) (c.stage_dev + xoff);
    if (!im.planar) return xe::xe_experts(c.q, jobs, n, nt, X, c.H, dst, H, F, im.L, false);
    xe::xe_quant_rows(c.q, X, nt, H, c.xq);
    return xe::xe_experts_q8(c.q, jobs, n, nt, c.xq, c.hq, dst, H, F, im.L, false, max_nt);
}

/// One decode / verify call: nt <= 8 tokens.  out_host (nt x n_embd) or out_dev (the dense card) gets the sum.
bool run_small(Ds41XeImpl& im, int64_t l, int nt, const int32_t* ids, const float* w, const float* x, float* out_host,
               void* out_dev) {
    const int64_t H = im.g.n_embd;
    const int K = (int) im.g.top_k;
    const double t0 = now_ms();
    static thread_local std::vector<Ent> ents;
    static thread_local std::vector<int32_t> first;
    group(ids, w, nt, K, im.g.n_experts, ents, first);
    // per card: its jobs; the rest: the CPU pool
    std::vector<const Ent*> cpu_list;
    const size_t NC = im.cards.size();
    std::vector<int> nj(NC, 0), mnt(NC, 0);
    for (auto& cp : im.cards) {
        cp->busy = false;
        if (im.dense_q) cp->stage_ev.wait();   // a back-to-back call (no dense sync between): its stage copy is done
    }
    for (const Ent& en : ents) {
        const size_t key = (size_t) im.key(l, en.e);
        const int c = im.card_of[key];
        if (c < 0) { cpu_list.push_back(&en); continue; }
        XeCard& cd = *im.cards[(size_t) c];
        if (nj[(size_t) c] >= kMaxDecodeJobs) { cpu_list.push_back(&en); continue; }
        XeJob j = make_job(im, l, en.e, im.dptr[key]);
        j.nt = (uint8_t) en.nt;
        for (int s = 0; s < en.nt; ++s) { j.tok[s] = en.tok[s]; j.w[s] = en.w[s]; }
        std::memcpy(cd.stage_host + (size_t) nj[(size_t) c] * sizeof(XeJob), &j, sizeof(XeJob));
        ++nj[(size_t) c];
        mnt[(size_t) c] = std::max(mnt[(size_t) c], en.nt);
        im.st.hits += 1;
    }
    const size_t xoff = (kMaxDecodeJobs * sizeof(XeJob) + kStageAlign - 1) / kStageAlign * kStageAlign;
    const size_t n_out = (size_t) nt * H;
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    static const bool sprof = env_i("STRATA_XE_PROF", 0) != 0;
    static double sp[5] = {0, 0, 0, 0, 0};
    static int64_t sp_n = 0, sp_jobs = 0;
    double ts0 = sprof ? now_ms() : 0;
    if (out_dev && dc) {
        // DEVICE OUTPUT (every decode / verify step): each card's part goes straight into the dense card's slot
        // (card-to-card copy), the CPU pool's part into the last slot, and one sum kernel on the dense card - which
        // waits for the other cards' copies through their events, so the host waits once per layer, not per card.
        float* slots = dc->dslots;   // [NC + 1][8][H] on the dense card
        std::vector<sycl::event> deps;
        std::vector<int> used;
        for (auto& cp : im.cards) {
            XeCard& c = *cp;
            const int n = nj[(size_t) c.idx];
            if (n == 0) continue;
            c.busy = true;
            std::memcpy(c.stage_host + xoff, x, n_out * 4);
            float* dst = c.dense ? slots + (size_t) c.idx * kMaxJobTok * H : c.OUT;
            const int mv = mnt[(size_t) c.idx] <= 1 ? 1 : mnt[(size_t) c.idx] <= 2 ? 2 : mnt[(size_t) c.idx] <= 4 ? 4 : 8;
            // the whole sequence; its first command waits for `prev` (the previous call's sum reads the slots this
            // call's cards write) unless it is being recorded (a replay carries that dependency instead)
            auto enqueue = [&](const sycl::event* prev) {
                if (prev) {
                    c.q.submit([&](sycl::handler& h) {
                        h.depends_on(*prev);
                        h.memcpy(c.stage_dev, c.stage_host, xoff + n_out * 4);   // jobs and X in one copy
                    });
                } else {
                    c.q.memcpy(c.stage_dev, c.stage_host, xoff + n_out * 4);
                }
                sycl::event ev = card_experts(im, c, n, nt, mv, xoff, dst);
                if (!c.dense) ev = c.q.memcpy(slots + (size_t) c.idx * kMaxJobTok * H, c.OUT, n_out * 4);
                return ev;
            };
            sycl::event ev, tl_start;
            if (Ds41XeTier::tl_on()) tl_start = sycl::ext::oneapi::experimental::submit_profiling_tag(c.q);
            if (im.dense_q && im.planar && im.tier_graphs && nt == 1) {   // decode; verify shapes vary too much to amortise a recording
                namespace sx = sycl::ext::oneapi::experimental;
                const uint32_t key = (uint32_t) n | ((uint32_t) nt << 8) | ((uint32_t) mv << 12);
                auto it = c.dgraphs.find(key);
                if (it == c.dgraphs.end()) {
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
            if (Ds41XeTier::tl_on()) {
                if (im.tl_card.size() < NC) im.tl_card.resize(NC);
                im.tl_card[(size_t) c.idx].emplace_back(tl_start, sycl::ext::oneapi::experimental::submit_profiling_tag(c.q));
            }
            deps.push_back(ev);
            used.push_back(c.idx);
        }
        double ts1 = sprof ? now_ms() : 0;
        sycl::queue& qd = im.dense_q ? *im.dense_q : dc->q;
        if (!cpu_list.empty()) {
            float* hs = im.hsum;
            im.hsum_ev.wait();   // the previous call's copy of hsum has been made
            std::memset(hs, 0, n_out * 4);
            cpu_ents(im, l, cpu_list, x, hs);
            im.hsum_ev = qd.memcpy(slots + NC * kMaxJobTok * H, hs, n_out * 4);
            deps.push_back(im.hsum_ev);
            used.push_back((int) NC);
        }
        double ts2 = sprof ? now_ms() : 0;
        float* o = (float*) out_dev;
        if (used.empty()) {
            sycl::event ev = qd.memset(o, 0, n_out * 4);
            if (im.dense_q) im.last_sum = ev; else ev.wait();
        } else {
            int sl[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const int ns = (int) used.size();
            for (int k = 0; k < ns && k < 8; ++k) sl[k] = used[(size_t) k];
            const size_t stride = (size_t) kMaxJobTok * H;
            sycl::event ev = qd.submit([&](sycl::handler& h) {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(n_out), [=](sycl::id<1> i) {
                    float sum = 0.0f;
                    for (int k = 0; k < ns; ++k) sum += slots[(size_t) sl[k] * stride + i];
                    o[i] = sum;
                });
            });
            if (im.dense_q) im.last_sum = ev; else ev.wait();
        }
        if (Ds41XeTier::tl_on() && im.dense_q) im.tl.emplace_back(3, sycl::ext::oneapi::experimental::submit_profiling_tag(*im.dense_q));
        if (sprof) {
            const double ts3 = now_ms();
            sp[0] += ts0 - t0;      // grouping + job building (before the cards' submissions)
            sp[1] += ts1 - ts0;     // the cards' submissions
            sp[2] += ts2 - ts1;     // CPU-pool experts
            sp[3] += ts3 - ts2;     // sum submission + wait for everything
            for (int c = 0; c < (int) NC; ++c) sp_jobs += nj[(size_t) c];
            if (++sp_n % 2000 == 0)
                std::fprintf(stderr, "xe decode prof (%lld calls, %.2f card jobs/call): per call prep %.1f us, submit %.1f us, cpu %.1f us, "
                             "sum+wait %.1f us = %.1f us\n", (long long) sp_n, (double) sp_jobs / sp_n, 1e3 * sp[0] / sp_n,
                             1e3 * sp[1] / sp_n, 1e3 * sp[2] / sp_n, 1e3 * sp[3] / sp_n, 1e3 * (sp[0] + sp[1] + sp[2] + sp[3]) / sp_n);
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
std::mutex g_submit;

/// STRATA_XE_PROF=1: run_big's phases (wall ms of the whole call / per-card busy ms summed over the cards), printed
/// every 40 calls.
struct BigProf {
    double prep = 0, cards = 0, combine = 0, total = 0;
    std::atomic<int64_t> stream_us{0}, gemm_us{0}, small_us{0};
    std::atomic<int64_t> streamed{0}, gemm_e{0}, small_j{0};
    int64_t calls = 0;
};
BigProf g_bp;
bool xe_prof() { static const bool on = env_i("STRATA_XE_PROF", 0) != 0; return on; }

/// n tokens of one layer, expert-major.  out_host (n x n_embd) or out_dev (the dense card).
///
/// Per expert the call routes to: an expert with >= STRATA_XE_GEMM_MIN (8) tokens runs as two fp16 GEMMs on oneMKL
/// (gather its tokens, dequantise it once, gate|up GEMM, SwiGLU, down GEMM: XMX, each weight read once); a smaller one
/// as mat-vec jobs of <= 8 tokens.  Experts a card does not hold (arena / file tier) are streamed onto the card with
/// the fewest streamed bytes, in rounds that fit its stream buffer.  Every card runs its part on its own host thread;
/// each writes its experts' rows of its Y, a deterministic gather sums them into the card's part, and the parts are
/// copied card-to-card (P2P) onto the dense card and added there.
bool run_big(Ds41XeImpl& im, int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out_host,
             void* out_dev) {
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
    // which card computes each expert; streamed experts go to the card with the fewest streamed bytes
    std::vector<int64_t> sbytes(NC, 0);
    std::vector<std::vector<int32_t>> cl(NC);   // experts per card, resident first then streamed
    std::vector<std::vector<int32_t>> streamed(NC);
    for (int64_t e = 0; e < E; ++e) {
        if (by_e[(size_t) e].empty()) continue;
        const int c = im.card_of[(size_t) im.key(l, e)];
        if (c >= 0) cl[(size_t) c].push_back((int32_t) e);
    }
    for (int64_t e = 0; e < E; ++e) {
        if (by_e[(size_t) e].empty() || im.card_of[(size_t) im.key(l, e)] >= 0) continue;
        size_t best = 0;
        for (size_t k = 1; k < NC; ++k) if (sbytes[k] < sbytes[best]) best = k;
        sbytes[best] += (int64_t) ((im.bbytes(l, e) + 255) & ~(size_t) 255);
        streamed[best].push_back((int32_t) e);
        cl[best].push_back((int32_t) e);
    }
    const size_t stream_cap = (size_t) (env_d("STRATA_XE_STREAM_GIB", 1.0) * kGiB);
    const int BJ = env_i("STRATA_XE_CHUNK_JOBS", 512);
    // ---- per card: buffers, X / W upload
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.busy = !cl[(size_t) c.idx].empty();
        if (!c.busy) continue;
        if (!grow_dev(c.stream, c.stream_bytes, std::min<size_t>(stream_cap, std::max<size_t>((size_t) sbytes[(size_t) c.idx], 1)), c.q)) {
            std::fprintf(stderr, "xe tier: card %d: stream buffer allocation failed (%.2f GiB free)\n", c.idx, c.free_vram() / kGiB);
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
            c.cjobs = sycl::malloc_shared<XeJob>((size_t) BJ, c.q);
            c.cH = sycl::malloc_device<float>((size_t) BJ * kMaxJobTok * F, c.q);
            c.cY = sycl::malloc_device<float>((size_t) BJ * kMaxJobTok * H, c.q);
            if (!c.cjobs || !c.cH || !c.cY) {
                std::fprintf(stderr, "xe tier: card %d: chunk job buffers allocation failed (%.2f GiB free)\n", c.idx, c.free_vram() / kGiB);
                return false;
            }
            c.cjobs_cap = BJ;
        }
        c.q.memcpy(c.cX, x, (size_t) n * H * 4);
        c.q.memcpy(c.cW, w, (size_t) n * K * 4);
        c.q.memset(c.cOUT, 0, (size_t) n * H * 4);
    }
    const double t_prep = now_ms();
    // ---- every card's part on its own host thread
    std::vector<char> ok(NC, 1);
    std::atomic<int64_t> hits{0}, pcie{0};
    auto card_work = [&](XeCard& c) -> bool {
        std::vector<int32_t> S((size_t) n * K);
        std::vector<int64_t> soff((size_t) E, -1);
        std::vector<int32_t> round_of((size_t) E, 0);
        std::vector<int32_t> ptok;
        std::vector<uint8_t> ftmp;
        double tp = now_ms();
        auto lap = [&](std::atomic<int64_t>& acc) {
            if (!prof) return;
            c.q.wait();
            const double t = now_ms();
            acc += (int64_t) ((t - tp) * 1000.0);
            tp = t;
        };
        int nr = 1;
        size_t used = 0;
        for (int32_t e : streamed[(size_t) c.idx]) {
            const size_t b = (im.bbytes(l, e) + 255) & ~(size_t) 255;
            if (used + b > c.stream_bytes && used > 0) { ++nr; used = 0; }
            round_of[(size_t) e] = nr - 1;
            soff[(size_t) e] = (int64_t) used;
            used += b;
        }
        for (int r = 0; r < nr; ++r) {
            if (!streamed[(size_t) c.idx].empty()) {
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
                    std::lock_guard<std::mutex> lk(g_submit);
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
                        std::lock_guard<std::mutex> lk(g_submit);
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
                return im.card_of[key] >= 0 ? im.dptr[key] : c.stream + soff[(size_t) e];
            };
            auto in_round = [&](int32_t e) { return im.card_of[(size_t) im.key(l, e)] >= 0 ? r == 0 : round_of[(size_t) e] == r; };
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
            if (!ge.empty()) {
                const size_t need = (size_t) gm_max * H * 2 + (size_t) 2 * F * H * 2 + (size_t) H * F * 2 +
                                    (size_t) gm_max * 2 * F * 4 + (size_t) gm_max * F * 2 + (size_t) gp * H * 4 + (size_t) gp * 4 + 8192;
                if (c.gscratch_cap < need) c.q.wait();
                bool grown;
                {
                    std::lock_guard<std::mutex> lk(g_submit);
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
                    std::lock_guard<std::mutex> lk(g_submit);
                    c.q.memcpy(dtok, ptok.data(), ptok.size() * 4);
                    c.q.memcpy(c.cS, S.data(), S.size() * 4);
                }
                c.q.wait();   // S and ptok are reused below
                int64_t base = 0;
                for (int32_t e : ge) {
                    std::lock_guard<std::mutex> lk(g_submit);
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
                    std::lock_guard<std::mutex> lk(g_submit);
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
                    c.q.wait();   // cjobs (shared memory) and cS are read by the previous batch / the GEMM gather
                    std::fill(S.begin(), S.end(), -1);
                    int max_nt = 1;
                    for (size_t q = 0; q < nb; ++q) {
                        c.cjobs[q] = js[b0 + q];
                        max_nt = std::max(max_nt, (int) js[b0 + q].nt);
                        const auto& sl = jss[b0 + q];
                        for (size_t s = 0; s < sl.size(); ++s) S[(size_t) sl[s]] = (int32_t) (q * kMaxJobTok + s);
                    }
                    if (pl && !xq_done) {   // this chunk's X in int8, once per card per call
                        if (c.cxq_rows < (size_t) n) {
                            bool okq;
                            {
                                std::lock_guard<std::mutex> lk(g_submit);
                                okq = alloc_q8(c.cxq, (size_t) n, (size_t) H, c.q);
                            }
                            if (!okq) { std::fprintf(stderr, "xe tier: card %d: int8 X allocation failed (n %d)\n", c.idx, n); return false; }
                            c.cxq_rows = (size_t) n;
                        }
                        if (c.chq_jobs < BJ) {
                            bool okh;
                            {
                                std::lock_guard<std::mutex> lk(g_submit);
                                okh = alloc_q8(c.chq, (size_t) BJ * kMaxJobTok, (size_t) F, c.q);
                            }
                            if (!okh) { std::fprintf(stderr, "xe tier: card %d: int8 H allocation failed\n", c.idx); return false; }
                            c.chq_jobs = BJ;
                        }
                        std::lock_guard<std::mutex> lk(g_submit);
                        xe::xe_quant_rows(c.q, c.cX, n, (int) H, c.cxq);
                        xq_done = true;
                    }
                    {
                        std::lock_guard<std::mutex> lk(g_submit);
                        c.q.memcpy(c.cS, S.data(), S.size() * 4);
                    }
                    c.q.wait();
                    std::lock_guard<std::mutex> lk(g_submit);
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
        if (!out_dev) {
            std::lock_guard<std::mutex> lk(g_submit);
            c.q.memcpy(c.cOUT_host, c.cOUT, (size_t) n * H * 4);
        }
        c.q.wait();
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
    im.st.pcie += pcie;
    const double t_cards = now_ms();
    // ---- combine: the other cards' parts copied card-to-card onto the dense card and added there
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    const size_t NE = (size_t) n * H;
    if (out_dev) {
        if (!dc) { std::fprintf(stderr, "xe tier: device output without a dense card\n"); return false; }
        std::vector<const XeCard*> others;
        for (auto& cp : im.cards) if (cp->busy && !cp->dense) others.push_back(cp.get());
        if (!others.empty() && !grow_dev(dc->cX2, dc->cX2_cap, NE * others.size(), dc->q)) {
            std::fprintf(stderr, "xe tier: dense card: combine buffer allocation failed\n");
            return false;
        }
        for (size_t k = 0; k < others.size(); ++k) dc->q.memcpy(dc->cX2 + k * NE, others[k]->cOUT, NE * 4);
        const int np = (int) others.size();
        const float* parts = dc->cX2;
        const float* own = dc->busy ? dc->cOUT : nullptr;
        float* o = (float*) out_dev;
        dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
            float s = own ? own[i] : 0.0f;
            for (int k = 0; k < np; ++k) s += parts[(size_t) k * NE + i];
            o[i] = s;
        }).wait();
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
        g_bp.cards += t_cards - t_prep;
        g_bp.combine += now_ms() - t_cards;
        g_bp.total += now_ms() - t0;
        if (++g_bp.calls % 40 == 0)
            std::fprintf(stderr, "xe prof (%lld chunk calls, last n %d): wall %.0f ms = prep %.0f + cards %.0f + combine %.0f; "
                         "card-busy sums: stream %.0f (%lld experts), gemm %.0f (%lld experts), small %.0f (%lld jobs)\n",
                         (long long) g_bp.calls, n, g_bp.total, g_bp.prep, g_bp.cards, g_bp.combine, g_bp.stream_us / 1000.0,
                         (long long) g_bp.streamed.load(), g_bp.gemm_us / 1000.0, (long long) g_bp.gemm_e.load(),
                         g_bp.small_us / 1000.0, (long long) g_bp.small_j.load());
    }
    im.st.wall_ms += now_ms() - t0;
    return true;
}

}  // namespace

bool Ds41XeTier::run_chunk(int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out) {
    try {
        if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, out, nullptr);
        return run_big(*im_, l, n, ids, w, x, out, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xe tier: run_chunk (layer %lld, %d tokens): %s\n", (long long) l, n, e.what());
        return false;
    }
}
bool Ds41XeTier::run_chunk_dev(int64_t l, int n, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    try {
        if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, nullptr, out_dev);
        return run_big(*im_, l, n, ids, w, x, nullptr, out_dev);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "xe tier: run_chunk_dev (layer %lld, %d tokens): %s\n", (long long) l, n, e.what());
        return false;
    }
}

// ================================ the rest ================================

void Ds41XeTier::close() {
    if (!im_) return;
    Ds41XeImpl& im = *im_;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.free_experts();
        auto fr = [&](void* p) { if (p) sycl::free(p, c.q); };
        fr(c.stage_dev); fr(c.stage_host); fr(c.H); fr(c.OUT); fr(c.part_host);
        fr(c.cX); fr(c.cOUT); fr(c.cOUT_host); fr(c.cS); fr(c.cW); fr(c.cjobs); fr(c.cH); fr(c.cY); fr(c.stream);
        fr(c.gscratch); fr(c.cX2); fr(c.bounce); fr(c.dslots); fr(c.rstage);
        for (xe::XeQ8* a : {&c.xq, &c.hq, &c.cxq, &c.chq}) { fr(a->q); fr(a->d); fr(a->s); }
    }
    im.cards.clear();
    for (uint8_t* p : im.achunks) {
        if (sycl::get_pointer_type(p, im.ctx) == sycl::usm::alloc::host) sycl::free(p, im.ctx);
        else std::free(p);
    }
    im.achunks.clear();
    if (im.hsum) { im.hsum_ev.wait(); sycl::free(im.hsum, im.ctx); im.hsum = nullptr; }
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
