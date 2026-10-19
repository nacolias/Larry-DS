// tools/v41/engine/xe/ds41_xe_tier.cpp - see ds41_xe_tier.hpp.
#include "ds41_xe_tier.hpp"
#include "xe_kernels.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
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
    bool busy = false;              ///< this call enqueued work on it
    // prompt-chunk scratch (grown on demand)
    float* cX = nullptr; size_t cX_cap = 0;          ///< floats
    float* cOUT = nullptr; size_t cOUT_cap = 0;      ///< floats
    float* cOUT_host = nullptr; size_t cOUT_host_cap = 0;
    int32_t* cS = nullptr; float* cW = nullptr; size_t cS_cap = 0;   ///< entries
    XeJob* cjobs = nullptr; float* cH = nullptr; float* cY = nullptr; int cjobs_cap = 0;
    uint8_t* stream = nullptr; size_t stream_bytes = 0;   ///< streamed (arena / file) experts of one chunk round

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
    std::vector<float> hsum;               ///< [8][n_embd] host-side sum
    std::vector<uint8_t> ftmp;             ///< file-tier reads (decode)
    Ds41MoeStats st;
    std::mutex mu;

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
    im.hsum.assign((size_t) kMaxJobTok * H, 0.f);
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
    const double arena_cap_gib = aenv && *aenv ? std::atof(aenv) : (im.cfg.arena_gib > 0 ? im.cfg.arena_gib : 1e9);
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
    im.arena_usm = true;
    for (size_t c = 0; c < nchunks; ++c) {
        uint8_t* p = sycl::malloc_host<uint8_t>(chunk_size[c], im.ctx);
        if (!p) {
            im.arena_usm = false;
            void* q = nullptr;
            if (posix_memalign(&q, 4096, chunk_size[c]) != 0 || !q) { err = "xe tier: arena allocation failed"; return false; }
            p = (uint8_t*) q;
        }
        im.achunks.push_back(p);
    }
    for (size_t i = 0; i < in_arena.size(); ++i)
        im.aptr[(size_t) im.key(in_arena[i].first, in_arena[i].second)] = im.achunks[where[i] >> 40] + (where[i] & (((size_t) 1 << 40) - 1));
    std::fprintf(stderr, "xe tier: arena %.2f GiB in %zu chunks (%s), %lld experts; file tier %lld\n", abytes / kGiB, nchunks,
                 im.arena_usm ? "SYCL host USM, pinned" : "some pageable", (long long) im.arena_n, (long long) im.file_n);
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
        c.q.memcpy((void*) im.dptr[key], buf, im.bbytes(le.first, le.second)).wait();
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
    const double cap = env_d("STRATA_XE_DENSE_GIB", -1.0);
    double gib = im.slot_gib;
    if (cap >= 0) gib = std::min(gib, cap);
    // whatever the engine measured, leave the dense half's graphs a reserve of what is free NOW
    const int64_t fr = c.free_vram();
    if (fr > 0) gib = std::min(gib, fr / kGiB - env_d("STRATA_XE_DENSE_RESERVE_GIB", 4.0));
    if (gib < 0.5) return true;
    c.budget = (int64_t) (gib * kGiB);
    const double t0 = now_ms();
    // the next experts in ranked order that live in the arena only
    for (const auto& le : ranked) {
        if (le.first < 0 || le.first >= im.g.n_layers || le.second < 0 || le.second >= im.g.n_experts) continue;
        const size_t key = (size_t) im.key(le.first, le.second);
        if (im.card_of[key] >= 0 || !im.aptr[key]) continue;
        const size_t b = im.bbytes(le.first, le.second);
        if (c.used + (int64_t) b > c.budget) continue;
        uint8_t* p = c.alloc_expert(b);
        if (!p) break;
        c.q.memcpy(p, im.aptr[key], b);
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

/// One decode / verify call: nt <= 8 tokens.  out_host (nt x n_embd) or out_dev (the dense card) gets the sum.
bool run_small(Ds41XeImpl& im, int64_t l, int nt, const int32_t* ids, const float* w, const float* x, float* out_host,
               void* out_dev) {
    const int64_t H = im.g.n_embd, F = im.g.n_ff;
    const int K = (int) im.g.top_k;
    const double t0 = now_ms();
    static thread_local std::vector<Ent> ents;
    static thread_local std::vector<int32_t> first;
    group(ids, w, nt, K, im.g.n_experts, ents, first);
    // per card: its jobs; the rest: the CPU pool
    std::vector<const Ent*> cpu_list;
    const size_t NC = im.cards.size();
    std::vector<int> nj(NC, 0);
    for (auto& cp : im.cards) cp->busy = false;
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
        im.st.hits += 1;
    }
    const size_t xoff = (kMaxDecodeJobs * sizeof(XeJob) + kStageAlign - 1) / kStageAlign * kStageAlign;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        const int n = nj[(size_t) c.idx];
        if (n == 0) continue;
        c.busy = true;
        std::memcpy(c.stage_host + xoff, x, (size_t) nt * H * 4);
        // jobs [0, n) and X in two copies (the job block is small; X follows at xoff)
        c.q.memcpy(c.stage_dev, c.stage_host, (size_t) n * sizeof(XeJob));
        c.q.memcpy(c.stage_dev + xoff, c.stage_host + xoff, (size_t) nt * H * 4);
        xe::xe_experts(c.q, (const XeJob*) c.stage_dev, n, nt, (const float*) (c.stage_dev + xoff), c.H, c.OUT,
                       (int) H, (int) F, im.L, false);
        if (!(c.dense && out_dev)) c.q.memcpy(c.part_host, c.OUT, (size_t) nt * H * 4);
    }
    float* hs = im.hsum.data();
    std::memset(hs, 0, (size_t) nt * H * 4);
    cpu_ents(im, l, cpu_list, x, hs);
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.busy || (c.dense && out_dev)) continue;
        c.q.wait();
        for (int64_t i = 0; i < (int64_t) nt * H; ++i) hs[i] += c.part_host[i];
    }
    if (out_dev) {
        if (!dc) return false;
        const size_t n = (size_t) nt * H;
        if (dc->busy) {   // the dense card's own sum + the host's: OUT[0..n) + OUT[n..2n)
            dc->q.memcpy(dc->OUT + n, hs, n * 4);
            xe::xe_sum_parts(dc->q, dc->OUT, 2, (int) n, (float*) out_dev, false);
        } else {
            dc->q.memcpy(out_dev, hs, n * 4);
        }
        dc->q.wait();
    } else {
        std::memcpy(out_host, hs, (size_t) nt * H * 4);
    }
    im.st.wall_ms += now_ms() - t0;
    return true;
}

}  // namespace

bool Ds41XeTier::run(int64_t l, const int32_t* ids6, const float* w6, const float* x, float* out) {
    return run_small(*im_, l, 1, ids6, w6, x, out, nullptr);
}
bool Ds41XeTier::run_to_dev(int64_t l, const int32_t* ids6, const float* w6, const float* x, void* out_dev) {
    return run_small(*im_, l, 1, ids6, w6, x, nullptr, out_dev);
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

/// n tokens of one layer, expert-major.  out_host (n x n_embd) or out_dev (the dense card).
bool run_big(Ds41XeImpl& im, int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out_host,
             void* out_dev) {
    const int64_t H = im.g.n_embd, F = im.g.n_ff;
    const int K = (int) im.g.top_k;
    const double t0 = now_ms();
    std::vector<Ent> ents;
    std::vector<int32_t> first;
    group(ids, w, n, K, im.g.n_experts, ents, first);
    const size_t NC = im.cards.size();
    // streamed experts (arena / file tier) go to the card with the least streamed bytes so far
    struct Stream { int32_t e; int card; size_t off; };
    std::vector<int64_t> sbytes(NC, 0);
    std::vector<int32_t> scard((size_t) im.g.n_experts, -1);
    std::vector<int64_t> soff((size_t) im.g.n_experts, -1);
    std::vector<std::vector<int32_t>> card_ents(NC);   // indices into ents
    std::vector<std::vector<int32_t>> streamed(NC);     // experts each card streams
    const size_t stream_cap = (size_t) (env_d("STRATA_XE_STREAM_GIB", 1.0) * kGiB);
    for (size_t i = 0; i < ents.size(); ++i) {
        const int32_t e = ents[i].e;
        const size_t key = (size_t) im.key(l, e);
        int c = im.card_of[key];
        if (c < 0) {
            if (scard[(size_t) e] < 0) {
                int best = 0;
                for (size_t k = 1; k < NC; ++k) if (sbytes[k] < sbytes[(size_t) best]) best = (int) k;
                scard[(size_t) e] = best;
                soff[(size_t) e] = sbytes[(size_t) best];
                sbytes[(size_t) best] += (int64_t) ((im.bbytes(l, e) + 255) & ~(size_t) 255);
                streamed[(size_t) best].push_back(e);
            }
            c = scard[(size_t) e];
        }
        card_ents[(size_t) c].push_back((int32_t) i);
    }
    // one card at a time would serialise; every card runs its part concurrently (its own queue)
    std::vector<uint8_t> ftmp;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.busy = false;
        const auto& ce = card_ents[(size_t) c.idx];
        if (ce.empty()) continue;
        c.busy = true;
        // stream buffer: this card's streamed experts (one round when they fit the cap; else several, below)
        if (!grow_dev(c.stream, c.stream_bytes, std::min<size_t>(stream_cap, std::max<size_t>((size_t) sbytes[(size_t) c.idx], 1)), c.q))
            return false;
        if (!grow_dev(c.cX, c.cX_cap, (size_t) n * H, c.q) || !grow_dev(c.cOUT, c.cOUT_cap, (size_t) n * H, c.q)) return false;
        if (!grow_host(c.cOUT_host, c.cOUT_host_cap, (size_t) n * H, c.q)) return false;
        size_t scap = c.cS_cap;
        if (!grow_dev(c.cS, scap, (size_t) n * K, c.q)) return false;
        size_t wcap = c.cS_cap;
        if (!grow_dev(c.cW, wcap, (size_t) n * K, c.q)) return false;
        c.cS_cap = std::min(scap, wcap);
        c.q.memcpy(c.cX, x, (size_t) n * H * 4);
        c.q.memcpy(c.cW, w, (size_t) n * K * 4);
    }
    const int BJ = env_i("STRATA_XE_CHUNK_JOBS", 512);
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.busy) continue;
        size_t jcap = (size_t) c.cjobs_cap;
        if (c.cjobs_cap < BJ) {
            c.q.wait();
            if (c.cjobs) sycl::free(c.cjobs, c.q);
            if (c.cH) sycl::free(c.cH, c.q);
            if (c.cY) sycl::free(c.cY, c.q);
            c.cjobs = sycl::malloc_shared<XeJob>((size_t) BJ, c.q);
            c.cH = sycl::malloc_device<float>((size_t) BJ * kMaxJobTok * F, c.q);
            c.cY = sycl::malloc_device<float>((size_t) BJ * kMaxJobTok * H, c.q);
            if (!c.cjobs || !c.cH || !c.cY) return false;
            c.cjobs_cap = BJ;
        }
        (void) jcap;
    }
    // per card: rounds over its streamed experts (each round fills the stream buffer), each round's jobs in batches
    // of BJ; batch slot map S: -1 except the (token, k) entries the batch serves
    std::vector<int32_t> S((size_t) n * K);
    bool first_out[16] = {};
    for (auto& cp : im.cards) first_out[cp->idx] = true;
    std::vector<XeJob> jb;
    // the jobs that do not need streaming, then the streamed ones by round
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.busy) continue;
        const auto& ce = card_ents[(size_t) c.idx];
        // split ce into rounds: resident first (round 0 includes them), streamed by stream capacity
        std::vector<std::vector<int32_t>> rounds(1);
        size_t used = 0;
        std::vector<int32_t> round_of_e((size_t) im.g.n_experts, -1);
        int nr = 1;
        for (int32_t e : streamed[(size_t) c.idx]) {
            const size_t b = (im.bbytes(l, e) + 255) & ~(size_t) 255;
            if (used + b > c.stream_bytes && used > 0) { ++nr; used = 0; }
            round_of_e[(size_t) e] = nr - 1;
            soff[(size_t) e] = (int64_t) used;
            used += b;
        }
        rounds.resize((size_t) nr);
        for (int32_t i : ce) {
            const int32_t e = ents[(size_t) i].e;
            const int r = im.card_of[(size_t) im.key(l, e)] >= 0 ? 0 : round_of_e[(size_t) e];
            rounds[(size_t) r].push_back(i);
        }
        for (int r = 0; r < nr; ++r) {
            // stream this round's experts
            if (!streamed[(size_t) c.idx].empty()) {
                c.q.wait();
                for (int32_t e : streamed[(size_t) c.idx]) {
                    if (round_of_e[(size_t) e] != r) continue;
                    const uint8_t* hb = host_blob(im, l, e, ftmp);
                    c.q.memcpy(c.stream + soff[(size_t) e], hb, im.bbytes(l, e));
                    if (hb == ftmp.data()) c.q.wait();   // the file-tier buffer is reused
                    ++im.st.pcie;
                }
            }
            const auto& rl = rounds[(size_t) r];
            for (size_t b0 = 0; b0 < rl.size(); b0 += (size_t) BJ) {
                const size_t nb = std::min<size_t>((size_t) BJ, rl.size() - b0);
                c.q.wait();   // cjobs is shared memory the previous batch's kernels read
                std::fill(S.begin(), S.end(), -1);
                for (size_t q = 0; q < nb; ++q) {
                    const Ent& en = ents[(size_t) rl[b0 + q]];
                    const size_t key = (size_t) im.key(l, en.e);
                    const uint8_t* dp = im.card_of[key] >= 0 ? im.dptr[key] : c.stream + soff[(size_t) en.e];
                    XeJob j = make_job(im, l, en.e, dp);
                    j.nt = (uint8_t) en.nt;
                    for (int s = 0; s < en.nt; ++s) {
                        j.tok[s] = en.tok[s];
                        j.w[s] = en.w[s];
                        S[(size_t) en.k[s]] = (int32_t) (q * kMaxJobTok + (size_t) s);
                    }
                    c.cjobs[q] = j;
                    if (im.card_of[key] >= 0) ++im.st.hits;
                }
                c.q.memcpy(c.cS, S.data(), S.size() * 4).wait();   // S is reused by the next batch
                xe::xe_experts_y(c.q, c.cjobs, (int) nb, c.cX, c.cH, c.cY, (int) H, (int) F, im.L);
                xe::xe_gather(c.q, c.cY, c.cS, c.cW, n, K, (int) H, c.cOUT, !first_out[c.idx]);
                first_out[c.idx] = false;
            }
        }
        if (!(c.dense && out_dev)) c.q.memcpy(c.cOUT_host, c.cOUT, (size_t) n * H * 4);
    }
    // sum the cards' parts
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    std::vector<float> hs;
    bool have_host = false;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.busy || (c.dense && out_dev)) continue;
        c.q.wait();
        if (!have_host) { hs.assign(c.cOUT_host, c.cOUT_host + (size_t) n * H); have_host = true; }
        else for (size_t i = 0; i < (size_t) n * H; ++i) hs[i] += c.cOUT_host[i];
    }
    const size_t NE = (size_t) n * H;
    if (out_dev) {
        if (!dc) return false;
        if (dc->busy) {
            if (have_host) {
                float* tmp = nullptr;
                size_t cap = 0;
                if (!grow_dev(tmp, cap, NE, dc->q)) return false;
                dc->q.memcpy(tmp, hs.data(), NE * 4);
                // out = dense part + host part
                float* a = dc->cOUT;
                dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) { a[i] += tmp[i]; });
                dc->q.memcpy(out_dev, a, NE * 4).wait();
                sycl::free(tmp, dc->q);
            } else {
                dc->q.memcpy(out_dev, dc->cOUT, NE * 4).wait();
            }
        } else {
            if (!have_host) hs.assign(NE, 0.f);
            dc->q.memcpy(out_dev, hs.data(), NE * 4).wait();
        }
    } else {
        if (!have_host) hs.assign(NE, 0.f);
        std::memcpy(out_host, hs.data(), NE * 4);
    }
    im.st.wall_ms += now_ms() - t0;
    return true;
}

}  // namespace

bool Ds41XeTier::run_chunk(int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out) {
    if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, out, nullptr);
    return run_big(*im_, l, n, ids, w, x, out, nullptr);
}
bool Ds41XeTier::run_chunk_dev(int64_t l, int n, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, nullptr, out_dev);
    return run_big(*im_, l, n, ids, w, x, nullptr, out_dev);
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
    }
    im.cards.clear();
    for (uint8_t* p : im.achunks) {
        if (sycl::get_pointer_type(p, im.ctx) == sycl::usm::alloc::host) sycl::free(p, im.ctx);
        else std::free(p);
    }
    im.achunks.clear();
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
