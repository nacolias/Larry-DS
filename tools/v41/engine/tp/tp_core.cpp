// tools/v41/engine/tp/tp_core.cpp - larry: engine1's tensor-parallel decode core (see tp_core.hpp, engine1/DESIGN.md).
#include "tp_core.hpp"
#include "tp_kern.hpp"

#include "ds41_geometry.hpp"
#include "ggml.h"

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <list>
#include <map>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <tuple>
#include <unistd.h>

namespace sx = sycl::ext::oneapi::experimental;

namespace strata::v41 {

namespace {

double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool env_on(const char * k, bool dflt) {
    const char * v = std::getenv(k);
    return v ? std::atoi(v) != 0 : dflt;
}
int64_t next_pow2(int64_t v) { int64_t p = 1; while (p < v) p <<= 1; return p; }

constexpr int QK = 256, Q6B = 210;   // Q6_K: 256 values, 210 bytes (ql 128 | qh 64 | scales 16 | d 2)

// ---- [lo, hi) of `units` dealt contiguously over N ranks (sizes differ by at most one)
struct Range { int64_t lo = 0, hi = 0; int64_t n() const { return hi - lo; } };
Range split(int64_t units, int N, int k) { return { units * k / N, units * (k + 1) / N }; }

// ---- card-local device memory: Level Zero allocations resident on their own card only (SYCL's go through UR / UMF,
// which make them resident on every P2P peer: ~1 GiB of host RAM a GiB on larry - see the xe tier's local_alloc),
// bump-allocated from 256 MiB slabs
struct Arena {
    sycl::queue * q = nullptr;
    std::vector<void *> slabs;
    uint8_t * cur = nullptr;
    size_t left = 0;
    size_t total = 0;
    static constexpr size_t kSlab = (size_t) 256 << 20;
    void * get(size_t bytes, std::string & err) {
        bytes = (bytes + 255) & ~(size_t) 255;
        if (bytes > left) {
            const size_t sz = std::max(bytes, kSlab);
            ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q->get_context());
            ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q->get_device());
            ze_device_mem_alloc_desc_t d = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0 };
            void * p = nullptr;
            if (zeMemAllocDevice(zc, &d, sz, 4096, zd, &p) != ZE_RESULT_SUCCESS || !p) { err = "zeMemAllocDevice failed"; return nullptr; }
            if (zeContextMakeMemoryResident(zc, zd, p, sz) != ZE_RESULT_SUCCESS) { zeMemFree(zc, p); err = "zeContextMakeMemoryResident failed"; return nullptr; }
            slabs.push_back(p);
            cur = (uint8_t *) p;
            left = sz;
        }
        void * r = cur;
        cur += bytes;
        left -= bytes;
        total += bytes;
        return r;
    }
    void release() {
        if (!q) return;
        ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q->get_context());
        for (void * p : slabs) zeMemFree(zc, p);
        slabs.clear();
    }
};

// ---- a device buffer whose pages are committed on demand (Level Zero virtual memory): ranks >= 1 hold their KV
// replicas (compressed rows, indexer keys) only for the rows a pass can touch, not ctx_max rows from the start
// (STRATA_TP_LAZY_KV, default 1) - the rest stays with the tier (its flex region) until the context needs it.  One
// address for the kernels and the recorded graphs; bytes past `done` (but the tail, where the spare row lives) are
// never touched: passes and syncs map their rows first.
struct LazyBuf {
    void * va = nullptr;
    size_t cap = 0, page = 0, done = 0, tail = 0;   // reserved bytes, page size, bytes mapped from va, the tail's offset
    std::vector<ze_physical_mem_handle_t> phys;
    std::vector<std::pair<void *, size_t>> maps;
    bool map(ze_context_handle_t zc, ze_device_handle_t zd, size_t off, size_t bytes, std::string & err) {
        ze_physical_mem_desc_t d = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, bytes };
        ze_physical_mem_handle_t h = nullptr;
        if (zePhysicalMemCreate(zc, zd, &d, &h) != ZE_RESULT_SUCCESS || !h) { err = "zePhysicalMemCreate failed"; return false; }
        void * p = (uint8_t *) va + off;
        if (zeVirtualMemMap(zc, p, bytes, h, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE) != ZE_RESULT_SUCCESS) {
            zePhysicalMemDestroy(zc, h);
            err = "zeVirtualMemMap failed";
            return false;
        }
        phys.push_back(h);
        maps.emplace_back(p, bytes);
        return true;
    }
    // reserve `bytes`; map the pages from the one holding byte `spare` (the spare row) to the end, then `first` bytes
    bool make(ze_context_handle_t zc, ze_device_handle_t zd, size_t bytes, size_t spare, size_t first, std::string & err) {
        size_t pg = 0;
        if (zeVirtualMemQueryPageSize(zc, zd, bytes, &pg) != ZE_RESULT_SUCCESS || !pg) { err = "zeVirtualMemQueryPageSize failed"; return false; }
        page = std::max<size_t>(pg, (size_t) 2 << 20);
        cap = (bytes + page - 1) / page * page;
        if (zeVirtualMemReserve(zc, nullptr, cap, &va) != ZE_RESULT_SUCCESS || !va) { va = nullptr; err = "zeVirtualMemReserve failed"; return false; }
        tail = std::min(cap - page, spare / page * page);
        if (!map(zc, zd, tail, cap - tail, err)) return false;
        return grow(zc, zd, first, err);
    }
    size_t want_of(size_t bytes) const { return std::min(tail, (bytes + page - 1) / page * page); }
    size_t grow_bytes(size_t bytes) const { const size_t w = want_of(bytes); return w > done ? w - done : 0; }
    // map up to `bytes` (in pages; the tail is mapped already)
    bool grow(ze_context_handle_t zc, ze_device_handle_t zd, size_t bytes, std::string & err) {
        const size_t w = want_of(bytes);
        if (w <= done) return true;
        if (!map(zc, zd, done, w - done, err)) return false;
        done = w;
        return true;
    }
    size_t mapped() const { return va ? done + (cap - tail) : 0; }
    void release(ze_context_handle_t zc) {
        for (auto & m : maps) zeVirtualMemUnmap(zc, m.first, m.second);
        for (auto h : phys) zePhysicalMemDestroy(zc, h);
        if (va) zeVirtualMemFree(zc, va, cap);
        maps.clear();
        phys.clear();
        va = nullptr;
        done = 0;
    }
};

// ---- a weight shard on a card
struct Mat {
    const void * p = nullptr;
    int64_t K = 0, N = 0, G = 1;   // row length, rows (a batch), batches
    size_t gb = 0;                 // bytes a batch (Q6_K planar: its own plane set)
    bool f16 = false;
    int64_t nbt = 0;               // Q6_K: blocks of the plane set (0: N * K / 256) - a row prefix of a bigger tensor
    bool shared = false;           // the engine's memory (rank 0 reading its weights in place), not this core's
    int8_t form[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // Q6_K: the mat-vec form a pass width n takes (tpfmv::launch_form; 0 = fastmv's dispatch)
};

struct LW {   // one card's weights of one layer
    const sycl::half * hca_fn = nullptr, * hcf_fn = nullptr;   // [24][HC * D]
    const float * hca_sc = nullptr, * hca_bs = nullptr, * attn_norm = nullptr;
    const float * hcf_sc = nullptr, * hcf_bs = nullptr, * ffn_norm = nullptr;
    const float * sinks = nullptr;                             // this card's heads
    Mat qkv;                                                   // Q6_K [RQ + DH][D]: q_a rows, kv rows
    const float * qa_norm = nullptr, * kva_norm = nullptr;
    Mat qb;                                                    // Q6_K [heads * DH][RQ]
    Mat woa;                                                   // Q6_K groups x [OL][OGD]
    Mat wob;                                                   // Q6_K [D][groups * OL]
    Mat comp;                                                  // F16 [DH (ratio 1) | 2 DH (kv | gate)][D]
    const float * comp_norm = nullptr, * idx_knorm = nullptr;
    const sycl::half * idx_k = nullptr;                        // [ID][DH]
    Mat iqb, iproj;                                            // F16 [IH ID][RQ], [IH][D]
    Mat router;                                                // F16 [NE][D]
    const float * bias = nullptr, * rbias = nullptr;
    Mat sgu, sdn;                                              // Q6_K [2 F][D] (gate | up), [D][F]
    Mat ekv;                                                   // Q6_K [R][EC ED] (Engram layers)
    const float * wqk = nullptr;                               // [HC][D] engram q_norm * k_norm
};

struct LS {   // one layer's persistent state on a card
    float * raw = nullptr;                       // [RAW][DH] window ring
    uint16_t * comp = nullptr;                   // [cmax + 1][DH] bf16 (kv sources)
    uint16_t * idxk = nullptr;                   // [cmax + 1][ID] bf16
    float * st_kv = nullptr, * st_sc = nullptr;  // [2][DH] (ratio-2 kv sources)
    int32_t * sel = nullptr;                     // [NMAX][ITOP] (index sources)
    float * selm = nullptr;
    LazyBuf vcomp, vidxk;                        // ranks >= 1, STRATA_TP_LAZY_KV: comp / idxk's pages
};

// the per-pass input block (int32 / float words), the same layout in pinned memory and on every card
struct InLayout {
    size_t pid, pos, widx, wmask, wslot, nvis[3], cslot[3], cpos[3], sidx, sslot, lastblk, emb, words;
    void make(int NMAX, int SWA, int D) {
        size_t o = 0;
        auto take = [&](size_t n) { const size_t r = o; o += (n + 15) & ~(size_t) 15; return r; };
        pid = take(1); pos = take(NMAX); widx = take((size_t) NMAX * SWA); wmask = take((size_t) NMAX * SWA); wslot = take(NMAX);
        for (int r = 1; r <= 2; ++r) { nvis[r] = take(NMAX); cslot[r] = take(NMAX); cpos[r] = take(NMAX); }
        nvis[0] = cslot[0] = cpos[0] = 0;
        sidx = take(2 * (size_t) NMAX); sslot = take(2); lastblk = take(NMAX);
        emb = take((size_t) NMAX * D);
        words = o;
    }
    size_t bytes_for(int n, int D) const { return (emb + (size_t) n * D) * 4; }   // the used prefix of a pass of n
};

struct Card {
    int rank = 0;
    int gpu = 0;                 // its Level Zero GPU index
    sycl::device dev;
    sycl::queue q;
    Arena ar;
    void * peer_blk = nullptr;   // SYCL USM: what other cards' copy engines write
    std::vector<LW> lw;
    std::vector<LS> ls;
    // head
    Mat head;                    // Q6_K [V_k][D]
    const float * out_norm = nullptr;
    // geometry of this rank's shards
    Range grp, ffb, erow, vrow;  // o-groups, n_ff blocks (256), Engram rows (in 256-row units), vocab rows
    int64_t nhk = 0, fk = 0, rk = 0, vk = 0;   // heads, n_ff columns, Engram rows, vocab rows
    // buffers
    int32_t * in = nullptr;      // the input block
    float * x = nullptr, * pre = nullptr, * hap = nullptr, * fpost = nullptr, * fcomb = nullptr;
    float * hcs = nullptr;
    uint32_t * cnt = nullptr;
    float * xn = nullptr, * apre = nullptr, * apost = nullptr, * acomb = nullptr;
    int8_t * xq = nullptr; float * xd = nullptr;
    float * fn = nullptr; int8_t * fq = nullptr; float * fd = nullptr;
    float * qkv = nullptr, * qr = nullptr; int8_t * qrq = nullptr; float * qrd = nullptr;
    float * qh = nullptr, * kvn = nullptr, * rtab = nullptr, * apart = nullptr, * o = nullptr;
    int8_t * oq = nullptr; float * od = nullptr;
    float * oa = nullptr; int8_t * aq = nullptr; float * ad = nullptr;
    float * cy = nullptr, * iq = nullptr, * iw = nullptr;
    // the XMX mat-vecs' layout of the q8 activations ([K][8] fp16 integers, [K / 32][8] scales): written beside the
    // q8 rows by their producers in the passes whose consumer takes an XMX form
    sycl::half * xT = nullptr, * qrT = nullptr, * oT = nullptr, * aT = nullptr, * fT = nullptr, * shT = nullptr;
    float * xdT = nullptr, * qrdT = nullptr, * odT = nullptr, * adT = nullptr, * fdT = nullptr, * shdT = nullptr;
    tpdec::li_bf16 * qp = nullptr; float * ws = nullptr; uint32_t * bits = nullptr;
    float * score = nullptr, * bmax = nullptr;
    int32_t * tkid = nullptr; uint32_t * tkst = nullptr; int32_t * cand = nullptr;
    float * lg = nullptr; int32_t * ids = nullptr; float * wts = nullptr;
    float * gu = nullptr, * sh = nullptr; int8_t * shq = nullptr; float * shd = nullptr;
    float * ex = nullptr; int8_t * exq = nullptr; float * exd = nullptr;
    float * z = nullptr, * zn = nullptr; int8_t * zq = nullptr; float * zd = nullptr;
    float * lgt = nullptr;       // [NMAX][vk] this card's logits rows
    // routed experts: the residency table [NL][NE], this pass's jobs, the int8 activations / hidden rows
    tpk::TpExp * etab = nullptr;
    strata::v41::xe::XeJob * jobs = nullptr;
    int32_t * need_cpu = nullptr;
    strata::v41::xe::XeQ8 X, H;
    std::vector<uint8_t *> blobs;
    // peer-written
    float * ara = nullptr, * arb = nullptr, * eng = nullptr;   // [N][NMAX][D] x 2, [N][NMAX][Rmax]
    int32_t * flags = nullptr;                                 // [N][MAXS]
    int32_t * seq = nullptr, * err = nullptr;                  // [MAXS], [4] (card-local)
    uint64_t * stamps = nullptr;                               // STRATA_TP_STAMP_LAYER: the clock at each step of that layer
    // graphs: (n, cap1, cap2) -> the recorded pass
    std::map<std::tuple<int, int64_t, int64_t>, sx::command_graph<sx::graph_state::executable>> graphs;
};

}  // namespace

struct Ds41Tp::Impl {
    Ds41TpConfig cfg;
    Ds41TpHooks hooks;
    std::unique_ptr<GgufModel> model;
    Ds41Geometry G;
    std::string err;
    sycl::context ctx;
    std::vector<std::unique_ptr<Card>> cards;
    int N = 1, NL = 0, NMAX = 7;
    int64_t D = 0, NH = 0, DH = 0, ND = 0, RQ = 0, OL = 0, OG = 0, OGD = 0, SWA = 0, RAW = 0, NE = 0, NU = 0, NFF = 0;
    int64_t IH = 0, ID = 0, ITOP = 0, CB = 0, CBS = 0, EC = 0, ED = 0, V = 0, ER = 0, RMAX = 0;
    int64_t cmax[3] = { 0, 0, 0 };
    struct LInfo { int64_t ratio = 0; bool kv_src = false, idx_src = false, cand_src = false, uses_cand = false; int64_t kvl = -1, idxl = -1, eslot = -1; };
    std::vector<LInfo> li;
    InLayout IL;
    // pinned host
    int32_t * hin = nullptr;      // the input block
    float * hlog = nullptr;       // [NMAX][V] logits (the cards write their rows straight here)
    float * heng = nullptr;       // [2][NMAX][EC ED] Engram rows
    int32_t * herr = nullptr;     // [N][4] the cards' error counters after each pass
    // host gates and rank 0's mailboxes (pinned): flags [0] the Engram rows, [1 + l] layer l's CPU part; per layer the
    // routing + fn rank 0 copies out after the router (ids | wts | fn | seq) and the CPU part's output
    int32_t * hflag = nullptr;
    struct Mail { int32_t * ids; float * wts; float * fn; int32_t * seq; };
    std::vector<Mail> mail;
    uint8_t * hmail = nullptr;
    float * hcpu = nullptr;       // [NL][NMAX][D]
    float * hvh = nullptr;        // [NMAX][3][D] DSpark's taps
    uint64_t rtab_seen = ~(uint64_t) 0;
    static constexpr int S_MAIL = 128, S_ENG = 200;   // seq / flag indices (sync points stay below 128)
    const uint8_t * embd = nullptr;   // token_embd F16 rows (mmap)
    uint32_t pid = 0;
    int nsm = 32;
    double last_pass_us = 0;
    // STRATA_TP_PROF=1: per pass the host's split - inputs + table refresh, launches, the Engram read, the CPU pool's
    // experts (and the mailbox waits before them), the wait for the cards after the host's part
    double pf[6] = {0, 0, 0, 0, 0, 0};
    int64_t pf_n = 0, pf_cpu_layers = 0;
    double pf_eng = 0;   // STRATA_TP_ENG_ASYNC: the helper's Engram reads (beside the columns above)
    // STRATA_TP_STAMP_LAYER=l (diagnostic, off by default): a kernel reading the sub-group clock after every step of layer
    // l on every card, inside the pass graph (~0.6 us each: the steps' graph-mode times, which eager traces do not show);
    // per 128 passes each card's average per step.  l = -1: every layer (~0.65 ms a pass more), per layer the expert
    // step, card 0's wait for the CPU pool and the cards' waits at the two cross-card sums.  STRATA_TP_STAMP_TPU: clock
    // ticks a us (2790 with the clocks pinned).
    static constexpr int NST = 27;
    int stamp_l = -2;   // -2 off
    double st_tpu = 2790.0;
    std::vector<double> st_acc;   // [card][NST]
    int64_t st_n = 0;
    void ST(Card & c, int l, int i) {
        if (!c.stamps || (stamp_l >= 0 && l != stamp_l)) return;
        uint64_t * p = c.stamps + (stamp_l < 0 ? (size_t) l * NST : 0) + i;
        c.q.single_task([=] { *p = sx::clock<sx::clock_scope::sub_group>(); });
    }
    // STRATA_TP_ENG_ASYNC (default 1): a pass's Engram rows are read by a helper thread while the pass's own thread
    // takes the CPU pool's layers from layer 0 (inline, a CPU-pool layer among layers 0-3 - a quarter of the verify
    // passes - waited ~2 ms behind both Engram layers' rows).  The Engram reader is the dense half's, which nothing
    // else touches during a TP pass.
    std::thread eng_thr;
    std::mutex eng_mu;
    std::condition_variable eng_cv;
    std::function<void()> eng_job;
    std::atomic<int> eng_state{0};   // 0 idle, 1 posted, 2 done, 3 stop
    void eng_post(std::function<void()> job) {
        if (!eng_thr.joinable())
            eng_thr = std::thread([this] {
                for (;;) {
                    std::function<void()> j;
                    {
                        std::unique_lock<std::mutex> lk(eng_mu);
                        eng_cv.wait(lk, [this] { const int st = eng_state.load(); return st == 1 || st == 3; });
                        if (eng_state.load() == 3) return;
                        j = std::move(eng_job);
                    }
                    j();
                    eng_state.store(2, std::memory_order_release);
                }
            });
        {
            std::lock_guard<std::mutex> lk(eng_mu);
            eng_job = std::move(job);
            eng_state.store(1, std::memory_order_release);
        }
        eng_cv.notify_one();
    }
    void eng_join() {
        while (eng_state.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        eng_state.store(0, std::memory_order_release);
    }
    int pf_rec = 0;   // passes of the window that recorded their graphs
    std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> kb;   // keeps the cards' modules built
    int dm_layers[3] = { 36, 37, 38 };

    // STRATA_TP_LAZY_KV (default 1): ranks >= 1 map their KV replicas' rows as the context reaches them; rows_done[r]
    // = rows of every ratio-r source mapped (and mirroring card 0's) on every such card
    bool lazy = false;
    int64_t rows_done[3] = { 0, 0, 0 };
    std::function<bool(int, size_t)> make_room;   // the tier: room for that many more bytes on that GPU

    ~Impl() {
        if (eng_thr.joinable()) {
            { std::lock_guard<std::mutex> lk(eng_mu); eng_state.store(3); }
            eng_cv.notify_one();
            eng_thr.join();
        }
        for (auto & c : cards) {
            try { c->q.wait(); } catch (...) {}
            c->graphs.clear();
            if (c->peer_blk) sycl::free(c->peer_blk, c->q);
            ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
            for (auto & s : c->ls) { s.vcomp.release(zc); s.vidxk.release(zc); }
            c->ar.release();
        }
        if (hin) sycl::free(hin, ctx);
        if (hlog) sycl::free(hlog, ctx);
        if (heng) sycl::free(heng, ctx);
        if (herr) sycl::free(herr, ctx);
        if (hflag) sycl::free(hflag, ctx);
        if (hcpu) sycl::free(hcpu, ctx);
        if (hvh) sycl::free(hvh, ctx);
        if (hmail) sycl::free(hmail, ctx);
        if (hkeys) sycl::free(hkeys, ctx);
    }

    // ------------------------------------------------------------------ weights (host side)
    // a tensor's rows as Q6_K blocks (row-major), from the cache or requantised now (Q8_0 / F16 / F32 -> Q6_K per WHOLE
    // row, as ds41_dense's loader does with STRATA_DENSE_REQUANT=q6_K: the same bits)
    struct Q6 { const uint8_t * p = nullptr; int64_t rows = 0, K = 0; std::vector<uint8_t> own; void * map = nullptr; size_t len = 0; };
    std::map<std::string, std::shared_ptr<Q6>> q6cache;
    std::shared_ptr<Q6> q6(const std::string & name) {
        auto it = q6cache.find(name);
        if (it != q6cache.end()) return it->second;
        size_t s = 0;
        const TensorInfo * ti = model->find(name, &s);
        if (!ti || ti->shape.size() != 2) { err = "tp: no 2-D tensor " + name; return nullptr; }
        auto r = std::make_shared<Q6>();
        r->K = (int64_t) ti->shape[0];
        r->rows = (int64_t) ti->shape[1];
        if (r->K % QK) { err = "tp: " + name + " rows are not a multiple of 256"; return nullptr; }
        const size_t bytes = (size_t) r->rows * (size_t) (r->K / QK) * Q6B;
        std::string path;
        if (!cfg.cache_dir.empty()) {
            path = cfg.cache_dir + "/" + name + ".q6k";
            const int fd = ::open(path.c_str(), O_RDONLY);
            if (fd >= 0) {
                struct stat st {};
                if (fstat(fd, &st) == 0 && (size_t) st.st_size == bytes) {
                    void * m = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
                    if (m != MAP_FAILED) { r->map = m; r->len = bytes; r->p = (const uint8_t *) m; }
                }
                ::close(fd);
            }
        }
        if (!r->p) {
            const uint8_t * src = model->shard(s).tensor_data(*ti);
            const ggml_type st = (ggml_type) ti->type;
            const size_t src_row = ggml_row_size(st, r->K);
            r->own.resize(bytes);
            const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) r->K);
                    for (int64_t row = k; row < r->rows; row += nth) {
                        const uint8_t * sr = src + (size_t) row * src_row;
                        if (st == GGML_TYPE_F32) std::memcpy(f.data(), sr, (size_t) r->K * 4);
                        else ggml_get_type_traits(st)->to_float(sr, f.data(), r->K);
                        ggml_quantize_chunk(GGML_TYPE_Q6_K, f.data(), r->own.data() + (size_t) row * (r->K / QK) * Q6B, 0, 1, r->K, nullptr);
                    }
                });
            for (auto & t : th) t.join();
            r->p = r->own.data();
            if (!path.empty()) {
                const std::string tmp = path + ".tmp";
                FILE * f = std::fopen(tmp.c_str(), "wb");
                if (f && std::fwrite(r->own.data(), 1, bytes, f) == bytes && std::fclose(f) == 0) std::rename(tmp.c_str(), path.c_str());
                else if (f) std::fclose(f);
            }
        }
        q6cache[name] = r;
        return r;
    }
    void q6_drop() {
        for (auto & kv : q6cache) if (kv.second->map) munmap(kv.second->map, kv.second->len);
        q6cache.clear();
    }
    // ONE planar matrix (ql | qh | scales | d planes over all its blocks) from row ranges [r0, r1) x 256-blocks [b0, b1)
    // of Q6_K row-major tensors, stacked by rows in list order (the blocks of a row: b0..b1-1)
    struct Src { const Q6 * m; int64_t r0, r1, b0, b1; };
    static void planar(const std::vector<Src> & src, std::vector<uint8_t> & out) {
        int64_t NB = 0;
        for (const auto & s : src) NB += (s.r1 - s.r0) * (s.b1 - s.b0);
        const size_t base = out.size();
        out.resize(base + (size_t) NB * Q6B);
        uint8_t * ql = out.data() + base, * qh = ql + 128 * NB, * sc = qh + 64 * NB, * d = sc + 16 * NB;
        int64_t i = 0;
        for (const auto & s : src) {
            const int64_t bpr = s.m->K / QK;
            for (int64_t r = s.r0; r < s.r1; ++r)
                for (int64_t b = s.b0; b < s.b1; ++b, ++i) {
                    const uint8_t * blk = s.m->p + ((size_t) r * bpr + b) * Q6B;
                    std::memcpy(ql + i * 128, blk, 128);
                    std::memcpy(qh + i * 64, blk + 128, 64);
                    std::memcpy(sc + i * 16, blk + 192, 16);
                    std::memcpy(d + i * 2, blk + 208, 2);
                }
        }
    }
    const uint8_t * raw(const std::string & name, uint32_t want_type, size_t want_bytes) {
        size_t s = 0;
        const TensorInfo * ti = model->find(name, &s);
        if (!ti) { err = "tp: missing tensor " + name; return nullptr; }
        if (ti->type != want_type) { err = "tp: " + name + " has type " + ggml_type_name(ti->type); return nullptr; }
        if (want_bytes && ti->elements() * ggml_type_size((ggml_type) ti->type) / ggml_blck_size((ggml_type) ti->type) != want_bytes) {
            err = "tp: " + name + " size mismatch"; return nullptr;
        }
        return model->shard(s).tensor_data(*ti);
    }

    // device upload helpers (card c's arena; queue copies from host memory - init only)
    template <class T> T * up(Card & c, const void * src, size_t bytes) {
        void * p = c.ar.get(bytes, err);
        if (!p) return nullptr;
        c.q.memcpy(p, src, bytes).wait();
        return (T *) p;
    }
    template <class T> T * dalloc(Card & c, size_t count) {
        void * p = c.ar.get(count * sizeof(T), err);
        if (p) c.q.memset(p, 0, count * sizeof(T));
        return (T *) p;
    }
    bool up_f32(Card & c, const std::string & name, int64_t n, const float * & dst, int64_t off = 0) {
        const uint8_t * p = raw(name, ty::F32, 0);
        if (!p) return false;
        dst = up<float>(c, (const float *) p + off, (size_t) n * 4);
        return dst != nullptr;
    }
    bool up_f16(Card & c, const std::string & name, const sycl::half * & dst) {
        size_t s = 0;
        const TensorInfo * ti = model->find(name, &s);
        if (!ti || ti->type != ty::F16) { err = "tp: no F16 tensor " + name; return false; }
        dst = up<sycl::half>(c, model->shard(s).tensor_data(*ti), ti->elements() * 2);
        return dst != nullptr;
    }
    // F16 rows as a mat-vec weight (an F32 tensor converted, as STRATA_ROUTER_F16 does); several tensors stacked by rows
    bool mat_f16(Card & c, const std::vector<std::string> & names, Mat & m) {
        std::vector<uint16_t> buf;
        int64_t K = -1, rows = 0;
        for (const auto & n : names) {
            size_t s = 0;
            const TensorInfo * ti = model->find(n, &s);
            if (!ti || ti->shape.size() != 2) { err = "tp: no 2-D tensor " + n; return false; }
            if (K >= 0 && (int64_t) ti->shape[0] != K) { err = "tp: " + n + " row length differs"; return false; }
            K = (int64_t) ti->shape[0];
            const int64_t r = (int64_t) ti->shape[1];
            const size_t o = buf.size();
            buf.resize(o + (size_t) (r * K));
            const uint8_t * p = model->shard(s).tensor_data(*ti);
            if (ti->type == ty::F16) std::memcpy(buf.data() + o, p, (size_t) (r * K) * 2);
            else if (ti->type == ty::F32) ggml_fp32_to_fp16_row((const float *) p, (ggml_fp16_t *) (buf.data() + o), r * K);
            else { err = "tp: " + n + " is not F16 / F32"; return false; }
            rows += r;
        }
        m.p = up<uint8_t>(c, buf.data(), buf.size() * 2);
        m.K = K; m.N = rows; m.G = 1; m.gb = 0; m.f16 = true;
        return m.p != nullptr;
    }
    // Q6_K planar weight from row / block ranges of one or more tensors: G == 1 - the parts stacked by rows into one
    // matrix (one plane set); G > 1 - the parts ARE the batches (a plane set each, w_batch_bytes apart: wo_a's slices)
    struct Part { std::string name; int64_t r0, r1, b0, b1; };
    bool mat_q6(Card & c, const std::vector<Part> & parts, int64_t G, Mat & m) {
        std::vector<uint8_t> buf;
        std::vector<std::shared_ptr<Q6>> keep;
        std::vector<Src> src;
        int64_t K = -1, rows = 0;
        for (const auto & pt : parts) {
            auto t = q6(pt.name);
            if (!t) return false;
            keep.push_back(t);
            const int64_t k = (pt.b1 - pt.b0) * QK;
            if (K >= 0 && k != K) { err = "tp: mixed row lengths in a stacked weight"; return false; }
            K = k;
            src.push_back({ t.get(), pt.r0, pt.r1, pt.b0, pt.b1 });
            rows += pt.r1 - pt.r0;
        }
        if (G > 1) {
            if ((int64_t) parts.size() != G) { err = "tp: batch count"; return false; }
            for (const auto & sp : src) planar({ sp }, buf);
            rows /= G;
        } else {
            planar(src, buf);
        }
        m.p = up<uint8_t>(c, buf.data(), buf.size());
        m.K = K; m.N = rows; m.G = G; m.f16 = false;
        m.gb = (size_t) (rows * (K / QK)) * Q6B;
        return m.p != nullptr;
    }

    // rank 0: a weight the engine holds on card 0 (shared_weight), else null
    size_t shared_bytes = 0;
    const void * share(const Card & c, const std::string & name, size_t bytes) {
        if (c.rank != 0 || !hooks.shared_weight || !env_on("STRATA_TP_SHARE", true)) return nullptr;
        const void * p = hooks.shared_weight(name);
        if (p) shared_bytes += bytes;
        return p;
    }
    // a Q6_K row prefix [0, rows) of the engine's planar tensor `name` ([rows_all][K]), or false
    bool share_q6(Card & c, const std::string & name, int64_t K, int64_t rows, int64_t rows_all, Mat & m) {
        const void * p = share(c, name, (size_t) (rows * (K / QK)) * Q6B);
        if (!p) return false;
        m.p = p; m.K = K; m.N = rows; m.G = 1; m.gb = 0; m.f16 = false; m.nbt = rows_all * (K / QK); m.shared = true;
        return true;
    }
    bool share_f16(Card & c, const std::string & name, int64_t K, int64_t rows, Mat & m) {
        const void * p = share(c, name, (size_t) (K * rows) * 2);
        if (!p) return false;
        m.p = p; m.K = K; m.N = rows; m.G = 1; m.gb = 0; m.f16 = true; m.shared = true;
        return true;
    }

    bool load_weights(Card & c) {
        const int k = c.rank;
        c.grp = split(OG, N, k);
        c.ffb = split(NFF / QK, N, k);
        c.erow = split(ER / QK, N, k);
        c.vrow = split(V / 16, N, k);
        c.nhk = c.grp.n() * (NH / OG);
        c.fk = c.ffb.n() * QK;
        c.rk = c.erow.n() * QK;
        c.vk = c.vrow.n() * 16;
        if (c.grp.n() < 1 || c.ffb.n() < 1) { err = "tp: too many cards for the o-groups / n_ff blocks"; return false; }
        const int64_t h0 = c.grp.lo * (NH / OG);
        c.lw.assign((size_t) NL, {});
        for (int l = 0; l < NL; ++l) {
            LW & w = c.lw[(size_t) l];
            const std::string b = "blk." + std::to_string(l) + ".";
            const int64_t hcw = tpk::HC * D * tpk::NMIX;
            if (!(w.hca_fn = (const sycl::half *) share(c, b + "hc_attn_fn.weight", (size_t) hcw * 2)) && !up_f16(c, b + "hc_attn_fn.weight", w.hca_fn)) return false;
            if (!(w.hcf_fn = (const sycl::half *) share(c, b + "hc_ffn_fn.weight", (size_t) hcw * 2)) && !up_f16(c, b + "hc_ffn_fn.weight", w.hcf_fn)) return false;
            if (!up_f32(c, b + "hc_attn_scale.weight", 3, w.hca_sc) || !up_f32(c, b + "hc_attn_base.weight", tpk::NMIX, w.hca_bs) ||
                !up_f32(c, b + "attn_norm.weight", D, w.attn_norm) || !up_f32(c, b + "hc_ffn_scale.weight", 3, w.hcf_sc) ||
                !up_f32(c, b + "hc_ffn_base.weight", tpk::NMIX, w.hcf_bs) || !up_f32(c, b + "ffn_norm.weight", D, w.ffn_norm) ||
                !up_f32(c, b + "attn_sinks.weight", c.nhk, w.sinks, h0) || !up_f32(c, b + "attn_q_a_norm.weight", RQ, w.qa_norm) ||
                !up_f32(c, b + "attn_kv_a_norm.weight", DH, w.kva_norm) || !up_f32(c, b + "exp_probs_b.bias", NE, w.bias))
                return false;
            std::vector<float> zero((size_t) NE, 0.0f);
            w.rbias = up<float>(c, zero.data(), zero.size() * 4);
            if (!mat_q6(c, { { b + "attn_q_a.weight", 0, RQ, 0, D / QK }, { b + "attn_kv.weight", 0, DH, 0, D / QK } }, 1, w.qkv)) return false;
            if (!share_q6(c, b + "attn_q_b.weight", RQ, c.nhk * DH, NH * DH, w.qb) &&
                !mat_q6(c, { { b + "attn_q_b.weight", h0 * DH, (h0 + c.nhk) * DH, 0, RQ / QK } }, 1, w.qb)) return false;
            {   // wo_a: the engine's per-slice planes (prereorder_slices) - rank 0's slices are its first ones
                const size_t sb = (size_t) (OL * (OGD / QK)) * Q6B;
                if (const void * p = share(c, b + "attn_output_a.weight", sb * (size_t) c.grp.n())) {
                    w.woa.p = p; w.woa.K = OGD; w.woa.N = OL; w.woa.G = c.grp.n(); w.woa.gb = sb; w.woa.f16 = false; w.woa.shared = true;
                } else {
                    std::vector<Part> sl;
                    for (int64_t g = c.grp.lo; g < c.grp.hi; ++g) sl.push_back({ b + "attn_output_a.weight", g * OL, (g + 1) * OL, 0, OGD / QK });
                    if (!mat_q6(c, sl, c.grp.n(), w.woa)) return false;
                }
            }
            if (!mat_q6(c, { { b + "attn_output_b.weight", 0, D, c.grp.lo * OL / QK, c.grp.hi * OL / QK } }, 1, w.wob)) return false;
            const LInfo & L = li[(size_t) l];
            if (L.kv_src) {
                std::vector<std::string> nm = { b + "attn_compressor_kv.weight" };
                if (L.ratio == 2) nm.push_back(b + "attn_compressor_gate.weight");
                if (!mat_f16(c, nm, w.comp)) return false;
                if (!up_f32(c, b + "attn_compressor_norm.weight", DH, w.comp_norm) || !up_f32(c, b + "indexer.k_norm.weight", ID, w.idx_knorm) ||
                    !up_f16(c, b + "indexer.attn_k.weight", w.idx_k))
                    return false;
            }
            if (L.idx_src && ((!share_f16(c, b + "indexer.attn_q_b.weight", RQ, IH * ID, w.iqb) && !mat_f16(c, { b + "indexer.attn_q_b.weight" }, w.iqb)) ||
                              (!share_f16(c, b + "indexer.proj.weight", D, IH, w.iproj) && !mat_f16(c, { b + "indexer.proj.weight" }, w.iproj)))) return false;
            if (!share_f16(c, b + "ffn_gate_inp.weight", D, NE, w.router) && !mat_f16(c, { b + "ffn_gate_inp.weight" }, w.router)) return false;
            if (cfg.shexp && !mat_q6(c, { { b + "ffn_gate_shexp.weight", c.ffb.lo * QK, c.ffb.hi * QK, 0, D / QK },
                                          { b + "ffn_up_shexp.weight", c.ffb.lo * QK, c.ffb.hi * QK, 0, D / QK } }, 1, w.sgu)) return false;
            if (cfg.shexp && !mat_q6(c, { { b + "ffn_down_shexp.weight", 0, D, c.ffb.lo, c.ffb.hi } }, 1, w.sdn)) return false;
            if (L.eslot >= 0) {
                if (!share_q6(c, b + "engram_kv.weight", EC * ED, c.rk, ER, w.ekv) &&
                    !mat_q6(c, { { b + "engram_kv.weight", c.erow.lo * QK, c.erow.hi * QK, 0, EC * ED / QK } }, 1, w.ekv)) return false;
                const float * qn = (const float *) raw(b + "engram_q_norm.weight", ty::F32, 0);
                const float * kn = (const float *) raw(b + "engram_k_norm.weight", ty::F32, 0);
                if (!qn || !kn) return false;
                std::vector<float> wqk((size_t) (tpk::HC * D));
                for (size_t i = 0; i < wqk.size(); ++i) wqk[i] = qn[i] * kn[i];
                w.wqk = up<float>(c, wqk.data(), wqk.size() * 4);
            }
        }
        if (!share_q6(c, "output.weight", D, c.vk, V, c.head) &&
            !mat_q6(c, { { "output.weight", c.vrow.lo * 16, c.vrow.hi * 16, 0, D / QK } }, 1, c.head)) return false;
        if (!up_f32(c, "output_norm.weight", D, c.out_norm)) return false;
        return true;
    }

    // ------------------------------------------------------------------ phase A: routed experts from real blobs
    static int qk_of(int t) { return t == ty::Q3_K ? 256 : t == ty::Q2_0 ? 64 : 128; }
    static int bb_of(int t) { return t == ty::Q3_K ? 110 : t == ty::Q2_0 ? 18 : 28; }
    bool load_bench_experts() {
        const int nb = cfg.bench_experts;
        if (nb <= 0) return true;
        const int bench_rep = std::getenv("STRATA_TP_BENCH_REPLICAS") ? std::atoi(std::getenv("STRATA_TP_BENCH_REPLICAS")) : 0;
        struct BI { int gt, dt; size_t up_off, down_off, bytes; uint16_t gu_row, d_row; std::vector<uint8_t> host; };
        std::vector<BI> bi((size_t) nb);
        for (int b = 0; b < nb; ++b) {   // spread over the layers and the expert ids (a realistic type mix)
            const int l = b % NL, e = (int) ((int64_t) b * 97 % NE);
            const Ds41ExpertMap & X = G.experts[(size_t) l];
            BI & B = bi[(size_t) b];
            B.gt = X.gu_type[(size_t) e];
            B.dt = X.d_type[(size_t) e];
            B.gu_row = (uint16_t) (D / qk_of(B.gt) * bb_of(B.gt));
            B.d_row = (uint16_t) (NFF / qk_of(B.dt) * bb_of(B.dt));
            const size_t gb = (size_t) NFF * B.gu_row, db = (size_t) D * B.d_row;
            B.up_off = gb; B.down_off = 2 * gb; B.bytes = 2 * gb + db;
            B.host.resize(B.bytes);
            const std::string * nm[3] = { &X.gate_t.at(B.gt), &X.up_t.at(B.gt), &X.down_t.at(B.dt) };
            const size_t sz[3] = { gb, gb, db }, off[3] = { 0, gb, 2 * gb };
            const int slot[3] = { X.gu_slot[(size_t) e], X.gu_slot[(size_t) e], X.d_slot[(size_t) e] };
            for (int k = 0; k < 3; ++k) {
                size_t s = 0;
                const TensorInfo * ti = model->find(*nm[k], &s);
                if (!ti) { err = "tp: missing " + *nm[k]; return false; }
                std::memcpy(B.host.data() + off[k], model->shard(s).tensor_data(*ti) + (size_t) slot[k] * sz[k], sz[k]);
            }
        }
        for (auto & cp : cards) {
            Card & c = *cp;
            for (int b = 0; b < nb; ++b) {
                const BI & B = bi[(size_t) b];
                uint8_t * stage = (uint8_t *) c.ar.get(B.bytes, err);
                uint8_t * dst = (uint8_t *) c.ar.get(B.bytes, err);
                if (!stage || !dst) return false;
                c.q.memcpy(stage, B.host.data(), B.bytes).wait();
                strata::v41::xe::xe_repack_blob(c.q, stage, dst, B.gt, B.dt, B.up_off, B.down_off, (int) D, (int) NFF).wait();
                c.blobs.push_back(dst);
            }
            std::vector<tpk::TpExp> tab((size_t) (NL * NE));
            for (int l = 0; l < NL; ++l)
                for (int e = 0; e < NE; ++e) {
                    const int b = (int) (((int64_t) l * NE + e) % nb);
                    const BI & B = bi[(size_t) b];
                    tpk::TpExp & x = tab[(size_t) (l * NE + e)];
                    x.owner = (e + l) % N;
                    x.blob = x.owner == c.rank ? c.blobs[(size_t) b] : nullptr;
                    // STRATA_TP_BENCH_REPLICAS=k: every k-th expert off rank 0 has a copy on rank 0 too (its blob is there)
                    if (bench_rep > 0 && x.owner != 0 && e % bench_rep == 0) { x.alt = 0; x.alt_blob = cards[0]->blobs[(size_t) b]; }
                    x.up_off = (uint32_t) B.up_off; x.down_off = (uint32_t) B.down_off;
                    x.gu_type = (uint16_t) B.gt; x.d_type = (uint16_t) B.dt; x.gu_row = B.gu_row; x.d_row = B.d_row;
                }
            c.etab = up<tpk::TpExp>(c, tab.data(), tab.size() * sizeof(tpk::TpExp));
            if (!expert_buffers(c)) return false;
        }
        return true;
    }
    bool expert_buffers(Card & c) {
        const size_t njm = (size_t) NMAX * NU;
        if (!c.etab) c.etab = dalloc<tpk::TpExp>(c, (size_t) (NL * NE));
        c.jobs = dalloc<strata::v41::xe::XeJob>(c, njm);
        c.need_cpu = dalloc<int32_t>(c, 4);
        c.X.q = dalloc<int8_t>(c, (size_t) NMAX * D); c.X.d = dalloc<float>(c, (size_t) NMAX * D / 32); c.X.s = dalloc<int32_t>(c, (size_t) NMAX * D / 16);
        const size_t hr = njm * strata::v41::xe::kMaxJobTok;
        c.H.q = dalloc<int8_t>(c, hr * NFF); c.H.d = dalloc<float>(c, hr * NFF / 32); c.H.s = dalloc<int32_t>(c, hr * NFF / 16);
        c.q.wait();
        return err.empty();
    }
    // the tier's residency table -> every card's copy, when it changed (a kernel reads the pinned table: an H2D copy of
    // >= 128 KB would delay the next kernel ~300 us); rows of layers past NL are not copied
    std::vector<uint32_t> rt_keys;
    uint32_t * hkeys = nullptr;   // pinned: the changed keys a refresh copies
    size_t hkeys_cap = 0;
    void refresh_rtab() {
        if (!hooks.rtab || !hooks.rtab_version) return;
        const bool first = rtab_seen == ~(uint64_t) 0;
        // the dirty hook first, every pass: the tier publishes there the promotions whose copies finished since (they
        // run during the draft) - then the version says whether anything changed
        bool all = first;
        if (hooks.rtab_dirty && hooks.rtab_dirty(rt_keys)) all = true;
        const uint64_t v = *hooks.rtab_version;
        if (v == rtab_seen && !all) return;
        rtab_seen = v;
        const tpk::TpExp * src = (const tpk::TpExp *) hooks.rtab;
        const size_t nent = (size_t) (NL * NE);
        if (!hooks.rtab_dirty) all = true;   // no dirty keys: the whole table whenever it changed
        if (!all && rt_keys.size() > hkeys_cap) all = true;
        if (all) {
            for (auto & cp : cards) {
                tpk::TpExp * dst = cp->etab;
                cp->q.parallel_for(sycl::range<1>(nent), [=](sycl::id<1> i) { dst[i] = src[i]; });
            }
            return;
        }
        if (rt_keys.empty()) return;
        // only the changed entries (promotions: a few a pass): keys in pinned memory, a small gather kernel a card
        // (the previous refresh's kernels read hkeys: every queue's work before this pass is done - pass() waited)
        size_t nk = 0;
        for (uint32_t k : rt_keys) if (k < nent) hkeys[nk++] = k;
        const uint32_t * kp = hkeys;
        for (auto & cp : cards) {
            tpk::TpExp * dst = cp->etab;
            cp->q.parallel_for(sycl::range<1>(nk), [=](sycl::id<1> i) { dst[kp[i[0]]] = src[kp[i[0]]]; });
        }
    }

    // ------------------------------------------------------------------ buffers
    bool alloc_buffers(Card & c) {
        const size_t n = (size_t) NMAX;
        c.in = dalloc<int32_t>(c, IL.words);
        c.x = dalloc<float>(c, n * tpk::HC * D); c.pre = dalloc<float>(c, n * tpk::HC); c.hap = dalloc<float>(c, n * tpk::HC * D);
        c.fpost = dalloc<float>(c, n * tpk::HC); c.fcomb = dalloc<float>(c, n * tpk::HC * tpk::HC);
        c.hcs = dalloc<float>(c, tpk::hc_scratch_floats(D, (int64_t) n)); c.cnt = dalloc<uint32_t>(c, 64);
        c.xn = dalloc<float>(c, n * D); c.apre = dalloc<float>(c, n * tpk::HC); c.apost = dalloc<float>(c, n * tpk::HC);
        c.acomb = dalloc<float>(c, n * tpk::HC * tpk::HC); c.xq = dalloc<int8_t>(c, n * D); c.xd = dalloc<float>(c, n * D / 32);
        c.fn = dalloc<float>(c, n * D); c.fq = dalloc<int8_t>(c, n * D); c.fd = dalloc<float>(c, n * D / 32);
        c.qkv = dalloc<float>(c, n * (RQ + DH)); c.qr = dalloc<float>(c, n * RQ); c.qrq = dalloc<int8_t>(c, n * RQ); c.qrd = dalloc<float>(c, n * RQ / 32);
        c.qh = dalloc<float>(c, n * c.nhk * DH); c.kvn = dalloc<float>(c, n * DH); c.rtab = dalloc<float>(c, n * ND);
        c.apart = dalloc<float>(c, tpdec::scratch_floats((int) c.nhk, (int) (SWA + ITOP), (int) n));
        // attn_dec_q8's output: o [n][heads][DH] | q8 rows (groups x n) of OGD | scales
        c.o = dalloc<float>(c, n * c.nhk * DH); c.oq = dalloc<int8_t>(c, n * c.nhk * DH); c.od = dalloc<float>(c, n * c.nhk * DH / 32);
        c.oa = dalloc<float>(c, n * c.grp.n() * OL); c.aq = dalloc<int8_t>(c, n * c.grp.n() * OL); c.ad = dalloc<float>(c, n * c.grp.n() * OL / 32);
        c.xT = dalloc<sycl::half>(c, (size_t) D * 8); c.xdT = dalloc<float>(c, (size_t) D / 32 * 8);
        c.qrT = dalloc<sycl::half>(c, (size_t) RQ * 8); c.qrdT = dalloc<float>(c, (size_t) RQ / 32 * 8);
        c.oT = dalloc<sycl::half>(c, (size_t) (c.grp.n() * OGD) * 8); c.odT = dalloc<float>(c, (size_t) (c.grp.n() * OGD) / 32 * 8);
        c.aT = dalloc<sycl::half>(c, (size_t) (c.grp.n() * OL) * 8); c.adT = dalloc<float>(c, (size_t) (c.grp.n() * OL) / 32 * 8);
        c.fT = dalloc<sycl::half>(c, (size_t) D * 8); c.fdT = dalloc<float>(c, (size_t) D / 32 * 8);
        c.shT = dalloc<sycl::half>(c, (size_t) c.fk * 8); c.shdT = dalloc<float>(c, (size_t) c.fk / 32 * 8);
        c.cy = dalloc<float>(c, n * 2 * DH); c.iq = dalloc<float>(c, n * IH * ID); c.iw = dalloc<float>(c, n * IH);
        c.qp = dalloc<tpdec::li_bf16>(c, n * 4096); c.ws = dalloc<float>(c, n * tpdec::LIH);
        c.bits = dalloc<uint32_t>(c, n * (size_t) tpdec::li_words(cmax[1]));
        c.score = dalloc<float>(c, n * (size_t) cmax[1]); c.bmax = dalloc<float>(c, n * (size_t) (cmax[1] / 8));
        c.tkid = dalloc<int32_t>(c, n * (size_t) std::max(ITOP, CB)); c.tkst = dalloc<uint32_t>(c, n * tpk::TK_ROW_WORDS);
        c.cand = dalloc<int32_t>(c, n * (size_t) CB);
        c.lg = dalloc<float>(c, n * NE);
        c.ids = dalloc<int32_t>(c, 2 * n * NU);   // ids | wts back to back: one copy into rank 0's mailbox
        c.wts = (float *) (c.ids + n * NU);
        c.gu = dalloc<float>(c, n * 2 * c.fk); c.sh = dalloc<float>(c, n * c.fk); c.shq = dalloc<int8_t>(c, n * c.fk); c.shd = dalloc<float>(c, n * c.fk / 32);
        c.ex = dalloc<float>(c, n * EC * ED); c.exq = dalloc<int8_t>(c, n * EC * ED); c.exd = dalloc<float>(c, n * EC * ED / 32);
        c.z = dalloc<float>(c, n * D); c.zn = dalloc<float>(c, n * D); c.zq = dalloc<int8_t>(c, n * D); c.zd = dalloc<float>(c, n * D / 32);
        c.lgt = dalloc<float>(c, n * (size_t) c.vk);
        c.seq = dalloc<int32_t>(c, tpk::MAXS); c.err = dalloc<int32_t>(c, 4);
        if (stamp_l >= -1) c.stamps = dalloc<uint64_t>(c, stamp_l == -1 ? (size_t) NST * NL : NST);
        // persistent state
        c.ls.assign((size_t) NL, {});
        for (int l = 0; l < NL; ++l) {
            LS & s = c.ls[(size_t) l];
            const LInfo & L = li[(size_t) l];
            if (c.rank == 0 && !hooks.state.empty()) {   // the dense half's own tensors on card 0
                const Ds41TpLayerState & e = hooks.state[(size_t) l];
                s.raw = e.raw; s.comp = e.comp; s.idxk = e.idxk; s.st_kv = e.st_kv; s.st_sc = e.st_sc;
                if (!s.raw || (L.kv_src && (!s.comp || !s.idxk)) || (L.kv_src && L.ratio == 2 && (!s.st_kv || !s.st_sc))) {
                    err = "tp: layer " + std::to_string(l) + ": the engine's state is missing a tensor";
                    return false;
                }
            } else {
                s.raw = dalloc<float>(c, (size_t) (RAW * DH));
                if (L.kv_src) {
                    const int64_t rows = cmax[L.ratio] + 1;
                    if (lazy && c.rank > 0) {   // pages for the first rows_done rows and the spare row (row cmax)
                        ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
                        ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c.dev);
                        const int64_t r0 = rows_done[L.ratio];
                        std::string e;
                        if (!s.vcomp.make(zc, zd, (size_t) (rows * DH) * 2, (size_t) ((rows - 1) * DH) * 2, (size_t) (r0 * DH) * 2, e) ||
                            !s.vidxk.make(zc, zd, (size_t) (rows * ID) * 2, (size_t) ((rows - 1) * ID) * 2, (size_t) (r0 * ID) * 2, e)) {
                            err = "tp: KV replica of layer " + std::to_string(l) + ": " + e;
                            return false;
                        }
                        s.comp = (uint16_t *) s.vcomp.va;
                        s.idxk = (uint16_t *) s.vidxk.va;
                    } else {
                        s.comp = dalloc<uint16_t>(c, (size_t) (rows * DH));
                        s.idxk = dalloc<uint16_t>(c, (size_t) (rows * ID));
                    }
                    if (L.ratio == 2) { s.st_kv = dalloc<float>(c, (size_t) (2 * DH)); s.st_sc = dalloc<float>(c, (size_t) (2 * DH)); }
                }
            }
            if (L.idx_src) { s.sel = dalloc<int32_t>(c, n * (size_t) ITOP); s.selm = dalloc<float>(c, n * (size_t) ITOP); }
        }
        if (!err.empty()) return false;
        // peer-written buffers (SYCL USM: peers' copy engines write them)
        const size_t ar_f = (size_t) N * n * D, eng_f = (size_t) N * n * RMAX, fl = (size_t) N * tpk::MAXS;
        const size_t bytes = (ar_f * 2 + eng_f) * 4 + fl * 4;
        c.peer_blk = sycl::malloc_device(bytes, c.q);
        if (!c.peer_blk) { err = "tp: peer buffer allocation failed"; return false; }
        c.q.memset(c.peer_blk, 0, bytes);
        c.ara = (float *) c.peer_blk;
        c.arb = c.ara + ar_f;
        c.eng = c.arb + ar_f;
        c.flags = (int32_t *) (c.eng + eng_f);
        c.q.wait();
        return true;
    }

    // ------------------------------------------------------------------ the program (enqueued on card c's queue:
    // executed eagerly or recorded into the pass graph)
    struct PassK { int n; int64_t cap[3]; };
    int sync_idx = 0;   // the next cross-card sync point of the pass being emitted (the same on every card)
    bool push = false;  // STRATA_TP_PUSH: cross-card sums by P2P store kernels (peer access enabled at init)
    // route_jobs' balance between an expert's holders (STRATA_TP_BALANCE, default 1; the cost model's STRATA_TP_BAL_*):
    // set at init from the shapes - a rank with more shared-expert columns starts its layers that much later
    tpk::BalArgs bal;
    void bal_init() {
        auto envf = [](const char * k, float d) { const char * v = std::getenv(k); return v && *v ? (float) std::atof(v) : d; };
        bal.on = env_on("STRATA_TP_BALANCE", true);
        bal.job_us = envf("STRATA_TP_BAL_JOB_US", 19.0f);
        bal.us_per_byte = 1.0f / (envf("STRATA_TP_BAL_GBS", 837.0f) * 1e3f);
        bal.tok_us = envf("STRATA_TP_BAL_TOK_US", 3.0f);
        bal.cpu_us = envf("STRATA_TP_BAL_CPU_US", 85.0f);
        bal.cpu_us_per_byte = 1.0f / (envf("STRATA_TP_BAL_CPU_GBS", 30.0f) * 1e3f);
        const float blk_us = envf("STRATA_TP_BAL_SHEXP_US", 10.0f);   // a 256-column block of the shared expert
        int64_t fmin = INT64_MAX;
        for (auto & cp : cards) fmin = std::min(fmin, cp->fk);
        for (auto & cp : cards) if (cp->rank < 8) bal.base[cp->rank] = cfg.shexp ? (float) ((cp->fk - fmin) / QK) * blk_us : 0.0f;
        bal.dense = 0;
        bal.ncard = std::min(N, 8);
        bal.nff = NFF;
        bal.d = D;
    }
    bool tap_on = false;   // the pass records DSpark's taps (a graph per (n, caps, taps))

    tpdec::Rope rope_p(int64_t ratio) const {
        const bool yarn = ratio != 0;
        tpdec::Rope r;
        r.n_dims = (int) ND;
        const int n_ctx_orig = yarn ? (int) G.orig_ctx : 0;
        const float freq_base = (float) (yarn ? G.compress_rope_base : G.rope_base);
        r.freq_scale = yarn ? (float) (1.0 / G.rope_factor) : 1.0f;
        r.ext_factor = yarn ? 1.0f : 0.0f;
        r.attn_factor = yarn ? 1.0f / (1.0f + 0.1f * std::log(1.0f / r.freq_scale)) : 1.0f;
        const float beta_fast = yarn ? (float) G.beta_fast : 0.0f, beta_slow = yarn ? (float) G.beta_slow : 0.0f;
        r.theta_scale = powf(freq_base, -2.0f / r.n_dims);
        float corr[2];
        ggml_rope_yarn_corr_dims(r.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr);
        r.corr0 = corr[0];
        r.corr1 = corr[1];
        return r;
    }

    static tpfmv::Args q6_args(const Mat & w, const int8_t * xq, const float * xd, int n, float * y, int64_t ycs, int64_t ybs) {
        tpfmv::Args a;
        a.w = w.p; a.K = w.K; a.N = w.N; a.G = w.G; a.g_div = 1;
        a.n_blocks_total = w.nbt ? w.nbt : w.N * (w.K / QK);
        a.w_batch_bytes = w.gb;
        a.x = nullptr; a.x_col_stride = w.K * w.G; a.x_batch_stride = w.K;
        a.y = y; a.y_col_stride = ycs; a.y_batch_stride = ybs ? ybs : w.N;
        a.ncols = n; a.xq = xq; a.xd = xd;
        return a;
    }
    // xT / xdT: the activations in the XMX layout (when the width's form is an XMX one; its producer wrote them)
    void mv_q6(Card & c, const Mat & w, const int8_t * xq, const float * xd, int n, float * y, int64_t ycs, int64_t ybs = 0,
               const sycl::half * xT = nullptr, const float * xdT = nullptr) {
        const tpfmv::Args a = q6_args(w, xq, xd, n, y, ycs, ybs);
        const int f = w.form[std::min(n, 7)];
        if (f != 0 && tpfmv::launch_form(c.q, a, xT, xdT, f)) return;
        if (!tpfmv::launch(c.q, tpfmv::WType::Q6K_R8, a) && err.empty()) err = "tp: fastmv Q6_K shape not covered (K " + std::to_string(w.K) + ")";
    }
    static bool xmx(const Mat & w, int n) { return w.form[std::min(n, 7)] >= 5; }

    // STRATA_TP_MV_TUNE (default 1; 0: fastmv's dispatch everywhere): at init every Q6_K mat-vec shape of the cards gets,
    // per pass width, the fastest of fastmv's dispatch, the dp4a kernel's row blockings and the XMX kernel's K splits,
    // timed on the card's own layers (one matrix a layer: cold weights, as in a pass) in replayed graphs, a small kernel
    // between the launches (as in a pass: the XMX kernels start ~5 us slower after another kernel than after themselves,
    // bench/fm3).  An XMX form must win by its producer's extra cost (the q8 written a second time, in its layout: the hc
    // step ~3 us, attention ~2 us, the others ~0.5).  Cards with the same shapes share one timing.  STRATA_TP_MV_FORM=f:
    // form f for every width >= 2 instead (A/B).
    bool tune_mv() {
        const char * fe = std::getenv("STRATA_TP_MV_FORM");
        const int forced = fe && *fe ? std::atoi(fe) : -1;
        if (!env_on("STRATA_TP_MV_TUNE", true) && forced < 0) return true;
        const double t0 = now_us();
        struct Kind { const char * name; Mat LW::* m; double xmx_margin; };
        static const Kind kinds[] = { {"q_a+kv", &LW::qkv, 4.0}, {"q_b", &LW::qb, 0.5}, {"wo_a", &LW::woa, 2.5}, {"wo_b", &LW::wob, 0.5},
                                      {"shexp gate|up", &LW::sgu, 4.0}, {"shexp down", &LW::sdn, 0.5} };
        std::map<std::tuple<int64_t, int64_t, int64_t>, std::array<int8_t, 8>> done;
        for (auto & cp : cards) {
            Card & c = *cp;
            for (const Kind & k : kinds) {
                const Mat & m0 = c.lw[0].*(k.m);
                if (!m0.p || m0.f16) continue;
                const auto key = std::make_tuple(m0.K, m0.N, m0.G);
                std::array<int8_t, 8> f{};
                const auto it = done.find(key);
                if (it != done.end()) f = it->second;
                else if (forced >= 0) { for (int n = 2; n < 8; ++n) f[(size_t) n] = (int8_t) forced; }
                else if (!tune_one(c, k.m, k.name, k.xmx_margin, f)) return false;
                done[key] = f;
                for (int l = 0; l < NL; ++l) std::copy(f.begin(), f.end(), (c.lw[(size_t) l].*(k.m)).form);
            }
        }
        if (forced < 0) std::fprintf(stderr, "tp: mat-vec forms tuned in %.2f s\n", (now_us() - t0) / 1e6);
        return true;
    }
    bool tune_one(Card & c, Mat LW::* mp, const char * name, double xmx_margin, std::array<int8_t, 8> & f) {
        const Mat & m0 = c.lw[0].*mp;
        const int64_t K = m0.K, Nr = m0.N, Gb = m0.G;
        // scratch activations / outputs (zeros: only the time counts)
        const size_t nx = (size_t) (Gb * NMAX * K), ny = (size_t) (Gb * NMAX * Nr);
        int8_t * xq = sycl::malloc_device<int8_t>(nx, c.q);
        float * xd = sycl::malloc_device<float>(nx / 32, c.q);
        sycl::half * xT = sycl::malloc_device<sycl::half>((size_t) (Gb * K) * 8, c.q);
        float * xdT = sycl::malloc_device<float>((size_t) (Gb * K) / 32 * 8, c.q);
        float * y = sycl::malloc_device<float>(ny, c.q);
        if (!xq || !xd || !xT || !xdT || !y) { err = "tp: mat-vec tuning buffers"; return false; }
        c.q.memset(xq, 0, nx); c.q.memset(xd, 0, nx / 32 * 4); c.q.memset(xT, 0, (size_t) (Gb * K) * 8 * 2);
        c.q.memset(xdT, 0, (size_t) (Gb * K) / 32 * 8 * 4); c.q.memset(y, 0, ny * 4);
        c.q.wait();
        // the small kernel between the launches, and its own time
        float * sk = y;
        auto small = [&] { c.q.parallel_for(sycl::nd_range<1>(64 * 16, 64), [=](sycl::nd_item<1> it) { if (it.get_global_id(0) == 0) sk[0] += 1.0f; }); };
        auto timed = [&](const std::function<bool()> & body) -> double {   // a graph of NL launches (each + the small kernel)
            sx::command_graph<sx::graph_state::modifiable> g(ctx, c.dev);
            g.begin_recording(c.q);
            const bool ok = body();
            g.end_recording(c.q);
            if (!ok) return -1.0;
            auto ge = g.finalize();
            c.q.ext_oneapi_graph(ge).wait();
            const double ta = now_us();
            for (int r = 0; r < 3; ++r) c.q.ext_oneapi_graph(ge);
            c.q.wait();
            return (now_us() - ta) / (3.0 * NL);
        };
        const double t_small = timed([&] { for (int l = 0; l < NL; ++l) small(); return true; });
        char b[80];
        std::string line;
        for (int n = 1; n <= std::min(NMAX, 7); ++n) {
            double best = 1e30, bestx = 1e30, t_disp = 0;
            int bf = 0, bx = -1;
            for (int form = 0; form <= 8; ++form) {
                if (form >= 5 && (1 << (form - 5)) > K / QK) continue;
                double us = timed([&] {
                    for (int l = 0; l < NL; ++l) {
                        const tpfmv::Args a = q6_args(c.lw[(size_t) l].*mp, xq, xd, n, y + 16, Nr * Gb, Nr);
                        if (!tpfmv::launch_form(c.q, a, xT, xdT, form)) return false;
                        small();
                    }
                    return true;
                });
                if (us < 0) continue;
                us -= t_small;
                if (form == 0) t_disp = us;
                if (form < 5 && us < best) { best = us; bf = form; }
                if (form >= 5 && us < bestx) { bestx = us; bx = form; }
            }
            if (bx >= 0 && bestx + xmx_margin < best) { best = bestx; bf = bx; }
            f[(size_t) n] = (int8_t) bf;
            if (bx >= 0 && bf != bx) std::snprintf(b, sizeof b, " n%d f%d %.1f (%.1f, xmx %.1f)", n, bf, best, t_disp, bestx);
            else std::snprintf(b, sizeof b, " n%d f%d %.1f (%.1f)", n, bf, best, t_disp);
            line += b;
        }
        sycl::free(xq, c.q); sycl::free(xd, c.q); sycl::free(xT, c.q); sycl::free(xdT, c.q); sycl::free(y, c.q);
        std::fprintf(stderr, "tp: %s %lldx%lldx%lld on card %d, us a launch by width (fastmv's dispatch, the best XMX one):%s\n", name, (long long) K,
                     (long long) Nr, (long long) Gb, c.rank, line.c_str());
        return true;
    }
    void mv_f16(Card & c, const Mat & w, const float * x, int64_t xcs, int n, float * y, int64_t ycs) {
        tpfmv::Args a;
        a.w = w.p; a.K = w.K; a.N = w.N; a.G = 1; a.g_div = 1;
        a.w_row_bytes = (size_t) w.K * 2;
        a.x = x; a.x_col_stride = xcs; a.x_batch_stride = xcs * n;
        a.y = y; a.y_col_stride = ycs; a.y_batch_stride = ycs * n;
        a.ncols = n;
        if (!tpfmv::launch(c.q, tpfmv::WType::F16, a) && err.empty()) err = "tp: fastmv F16 shape not covered";
    }
    template <class T> T * inp(Card & c, size_t off) const { return (T *) (c.in + off); }

    // card c sends its partial (slot c.rank of `buf` on every peer) and waits for the peers' (sync point s)
    void exchange(Card & c, float * buf, size_t slot_floats, size_t bytes, int s, int sl = -1, int si = -1) {
        if (N == 1) return;
        const size_t off = (size_t) c.rank * slot_floats;
        if (push && bytes % 16 == 0) {   // STRATA_TP_PUSH: P2P stores by two kernels
            tpk::Peers P;
            for (int j = 0; j < N; ++j) {
                if (j == c.rank) continue;
                Card & p = *cards[(size_t) j];
                P.dst[P.np] = (buf == c.ara ? p.ara : buf == c.arb ? p.arb : p.eng) + off;
                P.flag[P.np] = p.flags + c.rank * tpk::MAXS + s;
                ++P.np;
            }
            tpk::push_data(c.q, buf + off, bytes / 16, P);
            tpk::push_flags(c.q, c.seq + s, P);
        } else {
            for (int j = 0; j < N; ++j) {
                if (j == c.rank) continue;
                Card & p = *cards[(size_t) j];
                float * peer_buf = buf == c.ara ? p.ara : buf == c.arb ? p.arb : p.eng;
                c.q.memcpy(peer_buf + off, buf + off, bytes);
            }
            for (int j = 0; j < N; ++j) {
                if (j == c.rank) continue;
                c.q.memcpy(cards[(size_t) j]->flags + c.rank * tpk::MAXS + s, c.seq + s, 4);
            }
        }
        if (si >= 0) ST(c, sl, si);
        const uint32_t mask = ((1u << N) - 1) & ~(1u << c.rank);
        tpk::wait_peers(c.q, c.flags, inp<int32_t>(c, IL.pid), s, mask, c.err);
    }

    void emit_prologue(Card & c, const PassK & P) {
        // the pass's inputs (the used prefix of the block), then the sync sequence numbers and the initial state
        c.q.memcpy(c.in, hin, IL.bytes_for(P.n, (int) D));
        c.q.memset(c.err, 0, 16);
        tpk::seq_init(c.q, inp<int32_t>(c, IL.pid), c.seq);
        tpk::init_state(c.q, inp<float>(c, IL.emb), P.n, (int) D, c.x, c.pre);
    }

    void emit_layer(Card & c, int l, const PassK & P) {
        const LInfo & L = li[(size_t) l];
        const LW & w = c.lw[(size_t) l];
        LS & S = c.ls[(size_t) l];
        const int n = P.n;
        const int64_t r = L.ratio, cap = r ? P.cap[r] : 0;
        const size_t ar_slot = (size_t) NMAX * D;
        ST(c, l, 0);
        // ---- the previous layer's finish, Engram, then the attention hc
        tpk::HcOut ha;
        ha.sc = w.hca_sc; ha.bs = w.hca_bs; ha.pw = w.attn_norm; ha.eps = (float) G.rms_eps; ha.hc_eps = (float) G.hc_eps;
        ha.n_iter = (int) G.hc_iters; ha.xn = c.xn; ha.pre = c.apre; ha.post = c.apost; ha.comb = c.acomb; ha.xq = c.xq; ha.xd = c.xd;
        if (xmx(w.qkv, n)) { ha.xT = c.xT; ha.xdT = c.xdT; }
        tpk::HcIn hi;
        hi.fn = w.hca_fn; hi.fs1 = tpk::HC * D; hi.w16 = true;
        hi.pre = c.pre; hi.ps1 = tpk::HC;
        hi.out = c.hcs; hi.cnt = c.cnt; hi.D = D; hi.n = n;
        tpk::PSum yb { c.arb, (int64_t) ar_slot, D, N };
        const bool fin_in = l > 0 && L.eslot < 0;   // the previous layer's finish folded into this hc (STRATA_FIN_DEFER)
        if (l > 0 && !fin_in) {   // Engram layer: the finish on its own first
            finish(c, P);
        }
        if (L.eslot >= 0) engram(c, l, P);
        if (fin_in) {
            hi.x = c.hap; hi.xs1 = D; hi.xs2 = tpk::HC * D;
            hi.y = yb; hi.post = c.fpost; hi.pos1 = tpk::HC; hi.comb = c.fcomb; hi.cs1 = tpk::HC; hi.cs2 = tpk::HC * tpk::HC;
            hi.hdst = c.x;
        } else {
            hi.x = c.x; hi.xs1 = D; hi.xs2 = tpk::HC * D;
        }
        tpk::hc_fab(c.q, hi, ha, fin_in);
        // DSpark's main_hidden taps (rank 0): the state leaving layers 36 / 37 / 38 = entering 37 / 38 / 39
        for (int j = 0; j < 3; ++j)
            if (c.rank == 0 && tap_on && l == dm_layers[j] + 1) tpk::hc_mean(c.q, c.x, n, (int) D, hvh + (size_t) j * D, 3 * D);
        ST(c, l, 1);
        // ---- q_a + kv, q_a's norm (+ q8), q_b (this card's heads), the window row
        mv_q6(c, w.qkv, c.xq, c.xd, n, c.qkv, RQ + DH, 0, c.xT, c.xdT);
        ST(c, l, 2);
        {
            tpdec::RmsQ8Args a;
            a.x = c.qkv; a.x_cs = RQ + DH; a.w = w.qa_norm; a.eps = (float) G.rms_eps;
            a.y = c.qr; a.q = c.qrq; a.d = c.qrd; a.K = (int) RQ; a.n = n;
            if (xmx(w.qb, n)) { a.xT = c.qrT; a.xdT = c.qrdT; }
            tpdec::rms_q8(c.q, a);
        }
        ST(c, l, 3);
        mv_q6(c, w.qb, c.qrq, c.qrd, n, c.qh, c.nhk * DH, 0, c.qrT, c.qrdT);
        ST(c, l, 4);
        {
            tpdec::KvFinArgs a;
            a.x = c.qkv + RQ; a.x_cs = RQ + DH; a.w = w.kva_norm; a.pos = inp<int32_t>(c, IL.pos); a.eps = (float) G.rms_eps;
            a.rope = rope_p(r); a.fq = cfg.act_quant; a.y = c.kvn; a.y_cs = DH;
            a.ring = S.raw; a.ring_cs = DH; a.slot = inp<int32_t>(c, IL.wslot);
            a.slot_from = n - (int) std::min<int64_t>(n, RAW); a.tab = c.rtab; a.D = (int) DH; a.n = n;
            tpdec::kv_fin(c.q, a);
        }
        ST(c, l, 5);
        // ---- the compressor (kv sources) and the indexer (index sources)
        const uint16_t * comp = nullptr;
        const int32_t * sel = nullptr;
        const float * selm = nullptr;
        int K = 0;
        if (r) {
            if (L.kv_src) {
                mv_f16(c, w.comp, c.xn, D, n, c.cy, w.comp.N);
                tpk::CompArgs a;
                a.y = c.cy; a.ratio = (int) r; a.st_kv = S.st_kv; a.st_sc = S.st_sc; a.sidx = inp<int32_t>(c, IL.sidx);
                a.cnorm = w.comp_norm; a.attn_k = w.idx_k; a.knorm = w.idx_knorm;
                a.cpos = inp<int32_t>(c, IL.cpos[r]); a.cslot = inp<int32_t>(c, IL.cslot[r]);
                a.rope = rope_p(r); a.eps = (float) G.rms_eps; a.fq = cfg.act_quant;
                a.comp = S.comp; a.idxk = S.idxk; a.n = n;
                tpk::comp_fin(c.q, a);
                if (r == 2) tpk::comp_state(c.q, c.cy, n, inp<int32_t>(c, IL.sslot), S.st_kv, S.st_sc);
            }
            ST(c, l, 6);
            if (cap > 0) {
                const LS & KS = c.ls[(size_t) L.kvl];
                comp = KS.comp;
                K = (int) std::min<int64_t>(cap, ITOP);
                if (L.idx_src) {
                    mv_f16(c, w.iqb, c.qr, RQ, n, c.iq, IH * ID);
                    mv_f16(c, w.iproj, c.xn, D, n, c.iw, IH);
                    tpdec::LiArgs a;
                    a.n = n; a.q = c.iq; a.q_h = ID; a.q_b = IH * ID; a.tab = c.rtab; a.n_dims = (int) ND; a.fq = cfg.act_quant;
                    a.w = c.iw; a.w_b = IH; a.wscale = 1.0f / std::sqrt((float) (ID * IH));
                    a.k = (const tpdec::li_bf16 *) KS.idxk; a.ld_k = ID; a.NKV = cap;
                    a.nvis = inp<float>(c, IL.nvis[r]);
                    const int64_t nblk = cap / CBS, ncb = std::min<int64_t>(CB, nblk);
                    if (L.uses_cand) { a.cand = c.cand; a.cand_b = (size_t) CB; a.ncb = (int) ncb; }
                    a.o = c.score; a.ld_o = cap;
                    if (L.cand_src) { a.lastblk = inp<float>(c, IL.lastblk); a.bmax = c.bmax; a.ld_bm = nblk; }
                    a.qp = c.qp; a.ws = c.ws; a.bits = c.bits;
                    tpdec::li_prep(c.q, a);
                    tpdec::li_run(c.q, a);
                    if (L.cand_src) topk(c, c.bmax, nblk, (int) nblk, n, (int) ncb, c.cand, CB);
                    topk(c, c.score, cap, (int) cap, n, K, c.tkid, K);
                    tpdec::SelArgs s;
                    s.ids = c.tkid; s.ids_b = (size_t) K; s.nvis = inp<float>(c, IL.nvis[r]);
                    s.sel = S.sel; s.sel_b = (size_t) ITOP; s.selm = S.selm; s.selm_b = (size_t) ITOP; s.K = K; s.n = n;
                    tpdec::sel_fin(c.q, s);
                    sel = S.sel; selm = S.selm;
                } else {
                    const LS & IS = c.ls[(size_t) L.idxl];
                    sel = IS.sel; selm = IS.selm;
                }
            }
            ST(c, l, 7);
        } else {
            ST(c, l, 6);
            ST(c, l, 7);
        }
        // ---- attention over this card's heads (+ its q8 for wo_a), the output projection -> partial A
        {
            tpdec::AttnArgs a;
            a.q = c.qh; a.q_h = DH; a.q_b = (size_t) (c.nhk * DH); a.tab = c.rtab; a.n_dims = (int) ND;
            a.ring = S.raw; a.ring_cs = DH; a.RAW = (int) RAW; a.rows = c.kvn; a.rows_cs = DH;
            a.widx = inp<int32_t>(c, IL.widx); a.wi_b = SWA; a.wmask = inp<float>(c, IL.wmask); a.wm_b = SWA;
            if (comp && sel) {
                a.comp = comp; a.comp_t = 1; a.comp_rb = (size_t) DH * 2;
                a.sel = sel; a.sel_b = ITOP; a.selm = selm; a.sm_b = ITOP; a.K = K;
            }
            a.swa = (int) SWA; a.sinks = w.sinks; a.o = c.o; a.o_h = DH; a.o_b = (size_t) (c.nhk * DH);
            a.NH = (int) c.nhk; a.n = n; a.scale = 1.0f / std::sqrt((float) DH); a.part = c.apart;
            a.hpg = (int) (OGD / DH); a.oq = c.oq; a.od = c.od;
            if (xmx(w.woa, n)) { a.oT = c.oT; a.odT = c.odT; }
            tpdec::attn_dec(c.q, a);
        }
        ST(c, l, 8);
        mv_q6(c, w.woa, c.oq, c.od, n, c.oa, c.grp.n() * OL, OL, c.oT, c.odT);
        ST(c, l, 9);
        if (xmx(w.wob, n)) tpfmv::quant_xT(c.q, c.oa, c.grp.n() * OL, c.grp.n() * OL, n, 1, c.grp.n() * OL, c.aT, c.adT);
        else tpfmv::quant_x(c.q, c.oa, c.grp.n() * OL, c.grp.n() * OL, n, 1, c.grp.n() * OL, c.aq, c.ad);
        ST(c, l, 10);
        float * pa = c.ara + (size_t) c.rank * ar_slot;
        mv_q6(c, w.wob, c.aq, c.ad, n, pa, D, 0, c.aT, c.adT);
        ST(c, l, 11);
        exchange(c, c.ara, ar_slot, (size_t) n * D * 4, sync_idx++, l, 12);
        ST(c, l, 13);
        // ---- the FFN hc (the attention output = the cards' partials summed), router, shared expert -> partial B
        tpk::HcOut hf;
        hf.sc = w.hcf_sc; hf.bs = w.hcf_bs; hf.pw = w.ffn_norm; hf.eps = (float) G.rms_eps; hf.hc_eps = (float) G.hc_eps;
        hf.n_iter = (int) G.hc_iters; hf.xn = c.fn; hf.pre = c.pre; hf.post = c.fpost; hf.comb = c.fcomb; hf.xq = c.fq; hf.xd = c.fd;
        if (cfg.shexp && xmx(w.sgu, n)) { hf.xT = c.fT; hf.xdT = c.fdT; }
        tpk::HcIn fi;
        fi.x = c.x; fi.xs1 = D; fi.xs2 = tpk::HC * D;
        fi.fn = w.hcf_fn; fi.fs1 = tpk::HC * D; fi.w16 = true;
        fi.pre = c.apre; fi.ps1 = tpk::HC;
        fi.y = tpk::PSum { c.ara, (int64_t) ar_slot, D, N };
        fi.post = c.apost; fi.pos1 = tpk::HC; fi.comb = c.acomb; fi.cs1 = tpk::HC; fi.cs2 = tpk::HC * tpk::HC;
        fi.out = c.hcs; fi.hdst = c.hap; fi.cnt = c.cnt; fi.D = D; fi.n = n;
        tpk::hc_fab(c.q, fi, hf, true);
        ST(c, l, 14);
        mv_f16(c, w.router, c.fn, D, n, c.lg, NE);
        ST(c, l, 15);
        {
            tpk::RouterArgs a;
            a.lg = c.lg; a.bias = w.bias; a.rbias = w.rbias; a.ne = (int) NE; a.k = (int) NU; a.n = n;
            a.norm = G.norm_topk; a.scale = (float) G.weights_scale; a.ids = c.ids; a.wts = c.wts;
            tpk::router(c.q, a);
        }
        ST(c, l, 16);
        if (c.rank == 0 && hooks.cpu_part) {   // the routing + the experts' input out to the host (the CPU pool's share)
            const Mail & m = mail[(size_t) l];
            c.q.memcpy(m.ids, c.ids, (size_t) 2 * NMAX * NU * 4);
            c.q.memcpy(m.fn, c.fn, (size_t) n * D * 4);
            c.q.memcpy(m.seq, c.seq + S_MAIL + l, 4);
        }
        ST(c, l, 17);
        float * pb = c.arb + (size_t) c.rank * ar_slot;
        if (cfg.shexp) {
            mv_q6(c, w.sgu, c.fq, c.fd, n, c.gu, 2 * c.fk, 0, c.fT, c.fdT);
            ST(c, l, 18);
            tpk::swiglu_q8(c.q, c.gu, c.fk, n, (float) G.swiglu_limit, c.sh, c.shq, c.shd, xmx(w.sdn, n) ? c.shT : nullptr, c.shdT);
            ST(c, l, 19);
            mv_q6(c, w.sdn, c.shq, c.shd, n, pb, D, 0, c.shT, c.shdT);
            ST(c, l, 20);
        } else {
            c.q.memset(pb, 0, (size_t) n * D * 4);
            ST(c, l, 18); ST(c, l, 19); ST(c, l, 20);
        }
        if (c.etab) {   // this card's routed experts, added to the shared expert's partial
            const int njm = n * (int) NU;
            static const bool route_par = env_on("STRATA_TP_ROUTE_PAR", true);   // a lane a pick (0: the one-work-item form)
            if (route_par) tpk::route_jobs(c.q, c.ids, c.wts, n, (int) NU, c.etab + (size_t) l * NE, c.rank, c.jobs, njm, c.need_cpu, bal);
            else tpk::route_jobs_serial(c.q, c.ids, c.wts, n, (int) NU, c.etab + (size_t) l * NE, c.rank, c.jobs, njm, c.need_cpu);
            ST(c, l, 21);
            strata::v41::xe::xe_quant_rows(c.q, c.fn, n, (int) D, c.X);
            ST(c, l, 22);
            // engine1 options of the expert kernels: STRATA_TP_NJ_DEV (default 1) the kernels stop at route_jobs' real job count,
            // STRATA_TP_JOB_GRID=j (default 8) gate/up launched for j job slots (looping over the jobs; 0: one slot a pick) -
            // bit-identical, a 5-token pass -1 %; STRATA_XE_DOWN_SPLIT=1 the down kernel's jobs over its sub-groups (measured
            // slower: +1.9 % a 5-token pass)
            static const bool nj_dev = env_on("STRATA_TP_NJ_DEV", true), dsplit = env_on("STRATA_XE_DOWN_SPLIT", false);
            static const int jgrid = std::getenv("STRATA_TP_JOB_GRID") ? std::atoi(std::getenv("STRATA_TP_JOB_GRID")) : 8;
            strata::v41::xe::xe_experts_q8(c.q, c.jobs, njm, n, c.X, c.H, pb, (int) D, (int) NFF, (float) G.swiglu_limit, true, n,
                                           nj_dev ? c.need_cpu + 1 : nullptr, nj_dev && jgrid > 0 ? jgrid : 0, dsplit);
            ST(c, l, 23);
            if (c.rank == 0 && hooks.cpu_part) {   // the CPU pool's experts: wait for the host only if the routing has any
                tpk::wait_host(c.q, hflag + 16 * (1 + l), inp<int32_t>(c, IL.pid), S_MAIL + l, c.need_cpu, c.err);
                const float * co = hcpu + (size_t) l * NMAX * D;
                const int32_t * need = c.need_cpu;
                const size_t cnt = (size_t) n * D;
                c.q.parallel_for(sycl::range<1>(cnt), [=](sycl::id<1> i) { if (*need) pb[i] += co[i]; });
            }
            ST(c, l, 24);
        } else {
            ST(c, l, 21); ST(c, l, 22); ST(c, l, 23); ST(c, l, 24);
        }
        exchange(c, c.arb, ar_slot, (size_t) n * D * 4, sync_idx++, l, 25);
        ST(c, l, 26);
    }

    // x = hc_post(the FFN output summed over the cards, hap, post, comb): the finish on its own (before an Engram layer
    // and after the last layer)
    void finish(Card & c, const PassK & P) {
        const size_t ar_slot = (size_t) NMAX * D;
        const tpk::PSum y { c.arb, (int64_t) ar_slot, D, N };
        const float * res = c.hap, * post = c.fpost, * comb = c.fcomb;
        float * xo = c.x;
        const int n = P.n;
        const int64_t Dd = D, nq = D / 4;
        c.q.parallel_for(sycl::range<1>((size_t) (n * nq)), [=](sycl::id<1> id) {
            const int64_t i = ((int64_t) id[0] % nq) * 4, t = (int64_t) id[0] / nq;
            const float * yp = y.p + t * y.ts + i;
            sycl::float4 xv = *(const sycl::float4 *) yp;
            for (int k = 1; k < y.np; ++k) xv += *(const sycl::float4 *) (yp + k * y.stride);
            sycl::float4 rr[tpk::HC];
#pragma unroll
            for (int s = 0; s < tpk::HC; ++s) rr[s] = *(const sycl::float4 *) (res + (t * tpk::HC + s) * Dd + i);
            const float * po = post + t * tpk::HC;
            const float * co = comb + t * tpk::HC * tpk::HC;
#pragma unroll
            for (int d = 0; d < tpk::HC; ++d) {
                sycl::float4 o = xv * po[d];
#pragma unroll
                for (int s = 0; s < tpk::HC; ++s) o += rr[s] * co[d + s * tpk::HC];
                *(sycl::float4 *) (xo + (t * tpk::HC + d) * Dd + i) = o;
            }
        });
    }

    void engram(Card & c, int l, const PassK & P) {
        const LInfo & L = li[(size_t) l];
        const LW & w = c.lw[(size_t) l];
        const int n = P.n;
        const int64_t E = EC * ED;
        // the rows (pinned: the host reads them while layer 0 runs) -> q8 -> this card's projection rows
        if (hooks.engram) tpk::wait_host(c.q, hflag + 16 * (1 + NL) * (L.eslot > 0 ? 1 : 0), inp<int32_t>(c, IL.pid), S_ENG + (int) L.eslot, nullptr, c.err);
        c.q.memcpy(c.ex, heng + (size_t) L.eslot * NMAX * E, (size_t) n * E * 4);
        tpfmv::quant_x(c.q, c.ex, E, E, n, 1, E, c.exq, c.exd);
        const size_t eslot = (size_t) NMAX * RMAX;
        float * mine = c.eng + (size_t) c.rank * eslot;
        mv_q6(c, w.ekv, c.exq, c.exd, n, mine, c.rk);
        exchange(c, c.eng, eslot, (size_t) n * c.rk * 4, sync_idx++);
        // the full [n][(HC + 1) D] from the cards' row blocks, then the gated add (into kv_full: c.ex is free again)
        float * full = c.ex;   // (n * E floats >= n * ER? E = 6144 < ER = 25600: use the hc scratch instead)
        full = c.hcs;          // hc_scratch_floats(D, NMAX) >= NMAX * 5 * D
        const float * eng = c.eng;
        int64_t r0[9];
        for (int k = 0; k <= N; ++k) r0[k] = split(ER / QK, N, std::min(k, N - 1)).lo * QK;
        r0[N] = ER;
        int64_t rl[8];
        for (int k = 0; k < N; ++k) rl[k] = r0[k + 1] - r0[k];
        const int NN = N;
        const int64_t ER_ = ER;
        struct B { int64_t r0[9]; int64_t rl[8]; } bb;
        std::memcpy(bb.r0, r0, sizeof r0);
        std::memcpy(bb.rl, rl, sizeof rl);
        c.q.parallel_for(sycl::range<1>((size_t) (n * ER)), [=](sycl::id<1> id) {
            const int64_t t = (int64_t) id[0] / ER_, rr = (int64_t) id[0] % ER_;
            int k = 0;
            while (k + 1 < NN && rr >= bb.r0[k + 1]) ++k;
            full[t * ER_ + rr] = eng[(size_t) k * eslot + t * bb.rl[k] + (rr - bb.r0[k])];
        });
        tpk::engram_add(c.q, c.x, full, w.wqk, n, (int) D, (float) G.rms_eps);
    }

    void topk(Card & c, const float * src, int64_t ld, int ncols, int nrows, int k, int32_t * dst, int64_t ld_dst) {
        const int wg = 1024;
        if (ncols < 65536 || nrows >= 2 * nsm) {
            tpk::topk_rows(c.q, src, ld, ncols, nrows, k, dst, ld_dst, wg);
            return;
        }
        const int64_t target = 2 * nsm;
        const int parts = (int) std::max<int64_t>(1, std::min<int64_t>(target / nrows, ncols / 4096));
        if (parts <= 1) { tpk::topk_rows(c.q, src, ld, ncols, nrows, k, dst, ld_dst, wg); return; }
        tpk::topk_split(c.q, src, ld, ncols, nrows, k, dst, ld_dst, parts, wg, c.tkst);
    }

    void emit_epilogue(Card & c, const PassK & P) {
        finish(c, P);
        tpk::hc_pre(c.q, c.x, c.pre, P.n, (int) D, c.z);
        tpdec::RmsQ8Args a;
        a.x = c.z; a.x_cs = D; a.w = c.out_norm; a.eps = (float) G.rms_eps;
        a.y = c.zn; a.q = c.zq; a.d = c.zd; a.K = (int) D; a.n = P.n;
        tpdec::rms_q8(c.q, a);
        // this card's vocab rows -> device, then out to the pinned logits (one copy a token row: rows of V apart)
        mv_q6(c, c.head, c.zq, c.zd, P.n, c.lgt, c.vk);
        for (int t = 0; t < P.n; ++t) c.q.memcpy(hlog + (size_t) t * V + c.vrow.lo * 16, c.lgt + (size_t) t * c.vk, (size_t) c.vk * 4);
        c.q.memcpy(herr + c.rank * 4, c.err, 16);
    }

    // ------------------------------------------------------------------ the pass's host inputs (ds41_dense's
    // upload_inputs: positions, window maps, slot maps; the embeddings from the F16 rows)
    void make_inputs(const int * tids, int n, int pos0) {
        std::memset(hin, 0, IL.emb * 4);   // the maps (the embedding rows are all written below)
        if (++pid >= (1u << 19)) pid = 1;
        hin[IL.pid] = (int32_t) pid;
        int32_t * pos = hin + IL.pos;
        for (int t = 0; t < n; ++t) pos[t] = pos0 + t;
        int32_t * widx = hin + IL.widx;
        float * wm = (float *) (hin + IL.wmask);
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t;
            for (int64_t j = 0; j < SWA; ++j) {
                const int64_t q = p - SWA + 1 + j;
                int32_t row = 0;
                float m = 0.0f;
                if (q < 0) m = -INFINITY;
                else if (q >= pos0) row = (int32_t) (RAW + (q - pos0));
                else row = (int32_t) (q % RAW);
                widx[t * SWA + j] = row;
                wm[t * SWA + j] = m;
            }
        }
        {
            const int64_t k = std::min<int64_t>(n, RAW);
            for (int64_t j = 0; j < k; ++j) hin[IL.wslot + j] = (int32_t) ((pos0 + n - k + j) % RAW);
        }
        for (int r = 1; r <= 2; ++r) {
            float * nv = (float *) (hin + IL.nvis[r]);
            for (int t = 0; t < n; ++t) {
                const int64_t p = pos0 + t;
                nv[t] = (float) ((p + 1) / r);
                const bool done = (p + 1) % r == 0;
                hin[IL.cslot[r] + t] = (int32_t) (done ? p / r : cmax[r]);
                hin[IL.cpos[r] + t] = (int32_t) ((p / r) * r);
            }
            if (r == 1) {
                float * lb = (float *) (hin + IL.lastblk);
                for (int t = 0; t < n; ++t) lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / CBS : -1);
            }
        }
        {
            auto row = [&](int64_t q) { return (int32_t) (q >= pos0 ? 2 + (q - pos0) : q % 2); };
            for (int t = 0; t < n; ++t) {
                const int64_t p = pos0 + t, g0 = (p / 2) * 2;
                hin[IL.sidx + 2 * t] = row(g0);
                hin[IL.sidx + 2 * t + 1] = row(p >= g0 + 1 ? g0 + 1 : g0);
            }
            const int64_t k = std::min<int64_t>(n, 2);
            for (int64_t j = 0; j < k; ++j) hin[IL.sslot + j] = (int32_t) ((pos0 + n - k + j) % 2);
        }
        float * emb = (float *) (hin + IL.emb);
        for (int t = 0; t < n; ++t)   // (the row converter: per-value calls cost ~100 us a token)
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(embd) + (size_t) tids[t] * (size_t) D, emb + (size_t) t * D, D);
    }
    // the compressed rows a pass's indexer scans (a graph per bucket): next_pow2 of the visible rows as the dense half,
    // but at least STRATA_TP_CAP_MIN (1024: one bucket for every context below it - rows past the visible ones score
    // -inf, top-k K = 512 then from the start, the attention's key count fixed)
    // every card's pass graph for (n, caps, taps), recorded once
    bool record(const PassK & P, const std::tuple<int, int64_t, int64_t> & key) {
        tap_on = (std::get<0>(key) & 1) != 0;
        for (auto & cp : cards) {
            Card & c = *cp;
            if (c.graphs.count(key)) continue;
            if (&c == cards[0].get()) ++pf_rec;
            sx::command_graph<sx::graph_state::modifiable> g(ctx, c.dev);
            g.begin_recording(c.q);
            sync_idx = 0;
            emit_prologue(c, P);
            for (int l = 0; l < NL; ++l) emit_layer(c, l, P);
            emit_epilogue(c, P);
            g.end_recording(c.q);
            if (!err.empty()) return false;
            c.graphs.emplace(key, g.finalize());
        }
        return true;
    }
    // STRATA_TP_LAZY_KV: ranks >= 1 map the pages of rows [0, want[r]) of every ratio-r source (up to cmax[r]) before
    // anything touches them, the tier making room first (its flex region gives VRAM back); the newly mapped rows are
    // copied from card 0, so the mapped rows of a replica always mirror card 0's.  Between passes only.
    bool ensure_rows(const int64_t want_in[3]) {
        if (!lazy || N < 2) return true;
        int64_t want[3] = { 0, 0, 0 };
        bool any = false;
        for (int r = 1; r <= 2; ++r) {
            want[r] = std::min<int64_t>(cmax[r], std::max<int64_t>(want_in[r], rows_done[r]));
            any = any || want[r] > rows_done[r];
        }
        if (!any) return true;
        ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
        const Card & c0 = *cards[0];
        size_t total = 0;
        for (int k = 1; k < N; ++k) {
            Card & c = *cards[(size_t) k];
            ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c.dev);
            size_t extra = 0;
            for (int l = 0; l < NL; ++l) {
                const LInfo & L = li[(size_t) l];
                if (!L.kv_src) continue;
                const LS & s = c.ls[(size_t) l];
                extra += s.vcomp.grow_bytes((size_t) (want[L.ratio] * DH) * 2) + s.vidxk.grow_bytes((size_t) (want[L.ratio] * ID) * 2);
            }
            if (extra && make_room && !make_room(c.gpu, extra)) { err = "tp: no VRAM on GPU " + std::to_string(c.gpu) + " for the KV rows"; return false; }
            total += extra;
            for (int l = 0; l < NL; ++l) {
                const LInfo & L = li[(size_t) l];
                if (!L.kv_src) continue;
                LS & s = c.ls[(size_t) l];
                const int64_t r = L.ratio, r0 = rows_done[r], r1 = want[r];
                if (r1 <= r0) continue;
                std::string e;
                if (!s.vcomp.grow(zc, zd, (size_t) (r1 * DH) * 2, e) || !s.vidxk.grow(zc, zd, (size_t) (r1 * ID) * 2, e)) {
                    err = "tp: KV rows of layer " + std::to_string(l) + ": " + e;
                    return false;
                }
                const LS & a = c0.ls[(size_t) l];
                c.q.memcpy(s.comp + r0 * DH, a.comp + r0 * DH, (size_t) ((r1 - r0) * DH) * 2);
                c.q.memcpy(s.idxk + r0 * ID, a.idxk + r0 * ID, (size_t) ((r1 - r0) * ID) * 2);
            }
        }
        try {
            for (int k = 1; k < N; ++k) cards[(size_t) k]->q.wait_and_throw();
        } catch (const std::exception & e) {
            err = std::string("tp: KV rows copy: ") + e.what();
            return false;
        }
        std::fprintf(stderr, "tp: KV replicas now %lld / %lld rows (ratio 1 / 2), +%.0f MiB on the other cards\n",
                     (long long) want[1], (long long) want[2], total / 1048576.0);
        for (int r = 1; r <= 2; ++r) rows_done[r] = want[r];
        return true;
    }
    // the VRAM a rank's lazy replicas have not mapped yet (what the tier's flex region can hold meanwhile)
    size_t deferred(const Card & c) const {
        size_t b = 0;
        for (const LS & s : c.ls) {
            if (s.vcomp.va) b += s.vcomp.cap - s.vcomp.mapped();
            if (s.vidxk.va) b += s.vidxk.cap - s.vidxk.mapped();
        }
        return b;
    }
    // the cards' free VRAM (GiB; what the driver reports: the xe driver spills an over-commit into host RAM)
    std::string free_str() const {
        std::string r;
        char b[48];
        for (const auto & c : cards) {
            if (!c->dev.has(sycl::aspect::ext_intel_free_memory)) return "";
            std::snprintf(b, sizeof b, " r%d %.2f", c->rank, c->dev.get_info<sycl::ext::intel::info::device::free_memory>() / 1073741824.0);
            r += b;
        }
        return " | free GiB" + r;
    }
    size_t lazy_mapped(const Card & c) const {
        size_t b = 0;
        for (const LS & s : c.ls) b += s.vcomp.mapped() + s.vidxk.mapped();
        return b;
    }
    void caps(int pos0, int n, int64_t out[3]) const {
        static const int64_t cap_min = std::getenv("STRATA_TP_CAP_MIN") ? std::atoll(std::getenv("STRATA_TP_CAP_MIN")) : 1024;
        out[0] = 0;
        for (int r = 1; r <= 2; ++r) {
            const int64_t nvis_last = (pos0 + n) / r;
            out[r] = nvis_last == 0 && cap_min <= 0 ? 0 : std::min<int64_t>(next_pow2(std::max<int64_t>({ nvis_last, CBS, cap_min })), cmax[r]);
        }
    }
};

Ds41Tp::Ds41Tp() : p_(std::make_unique<Impl>()) {}
Ds41Tp::~Ds41Tp() = default;
int64_t Ds41Tp::vocab() const { return p_->V; }
int64_t Ds41Tp::n_embd() const { return p_->D; }
int Ds41Tp::n_layer() const { return p_->NL; }
int Ds41Tp::n_cards() const { return p_->N; }
const std::string & Ds41Tp::last_error() const { return p_->err; }
bool Ds41Tp::shard_desc(const std::string & name, int rank, Ds41TpShard & d) const {
    const Impl & im = *p_;
    d = Ds41TpShard();
    if (rank < 0 || rank >= im.N) return false;
    const Card & c = *im.cards[(size_t) rank];
    d.gpu = c.gpu;
    const Mat * m = nullptr;
    if (name == "output.weight") {
        m = &c.head; d.kind = 0; d.a0 = c.vrow.lo * 16; d.a1 = c.vrow.hi * 16;
    } else {
        int l = -1, n = 0;
        if (std::sscanf(name.c_str(), "blk.%d.%n", &l, &n) != 1 || l < 0 || l >= im.NL || n <= 0) return false;
        const std::string t = name.substr((size_t) n);
        const LW & w = c.lw[(size_t) l];
        const int64_t h0 = c.grp.lo * (im.NH / im.OG);
        if (t == "attn_q_b.weight") { m = &w.qb; d.kind = 0; d.a0 = h0 * im.DH; d.a1 = (h0 + c.nhk) * im.DH; }
        else if (t == "attn_output_a.weight") {
            m = &w.woa; d.kind = 3; d.a0 = c.grp.lo; d.a1 = c.grp.hi; d.slice = w.woa.gb;
            if (w.woa.G != c.grp.n() || !w.woa.gb) return false;
        }
        else if (t == "attn_output_b.weight") { m = &w.wob; d.kind = 2; d.a0 = c.grp.lo * im.OL / QK; d.a1 = c.grp.hi * im.OL / QK; }
        else if (t == "ffn_gate_up_shexp.weight") { m = &w.sgu; d.kind = 1; d.a0 = c.ffb.lo * QK; d.a1 = c.ffb.hi * QK; d.half = im.NFF; }
        else if (t == "ffn_down_shexp.weight") { m = &w.sdn; d.kind = 2; d.a0 = c.ffb.lo; d.a1 = c.ffb.hi; }
        else if (t == "engram_kv.weight" && im.li[(size_t) l].eslot >= 0) { m = &w.ekv; d.kind = 0; d.a0 = c.erow.lo * QK; d.a1 = c.erow.hi * QK; }
        else return false;
    }
    if (!m || !m->p || m->shared || m->f16 || m->nbt) return false;   // only this core's own Q6_K copies
    d.p = m->p;
    return true;
}

const void * Ds41Tp::shard(const std::string & name, int rank, int64_t & r0, int64_t & r1, int & gpu) const {
    const Impl & im = *p_;
    if (rank < 0 || rank >= im.N) return nullptr;
    const Card & c = *im.cards[(size_t) rank];
    gpu = c.gpu;
    if (name == "output.weight") {
        r0 = c.vrow.lo * 16; r1 = c.vrow.hi * 16;
        return c.head.G == 1 && !c.head.f16 && !c.head.nbt ? c.head.p : nullptr;
    }
    int l = -1;
    if (std::sscanf(name.c_str(), "blk.%d.engram_kv.weight", &l) == 1 && l >= 0 && l < im.NL && im.li[(size_t) l].eslot >= 0) {
        const Mat & m = c.lw[(size_t) l].ekv;
        r0 = c.erow.lo * QK; r1 = c.erow.hi * QK;
        return m.G == 1 && !m.f16 && !m.nbt ? m.p : nullptr;
    }
    return nullptr;
}

std::vector<std::pair<int, size_t>> Ds41Tp::deferred_vram() const {
    std::vector<std::pair<int, size_t>> v;
    for (const auto & c : p_->cards) if (const size_t b = p_->deferred(*c)) v.emplace_back(c->gpu, b);
    return v;
}

void Ds41Tp::set_vram_hook(std::function<bool(int, size_t)> make_room) { p_->make_room = std::move(make_room); }

std::vector<void *> Ds41Tp::queues() const {
    std::vector<void *> v;
    for (auto & c : p_->cards) v.push_back(&c->q);
    return v;
}

bool Ds41Tp::init(const Ds41TpConfig & cfg, const Ds41TpHooks & hooks, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    im.hooks = hooks;
    try {
        im.model = std::make_unique<GgufModel>(GgufModel::open(cfg.gguf));
    } catch (const std::exception & e) {
        err = std::string("tp: ") + e.what();
        return false;
    }
    if (!(err = ds41_read_geometry(*im.model, im.G)).empty()) return false;
    if (!(err = ds41_check_tensors(*im.model, im.G, ds41_file_layers(*im.model, im.G))).empty()) return false;
    const Ds41Geometry & G = im.G;
    im.N = cfg.n_cards;
    im.NMAX = cfg.max_n;
    im.NL = cfg.n_layer > 0 ? std::min<int>(cfg.n_layer, (int) G.n_layer) : (int) G.n_layer;
    im.D = G.n_embd; im.NH = G.n_head; im.DH = G.d_head; im.ND = G.d_rope; im.RQ = G.r_q; im.OL = G.o_lora; im.OG = G.o_groups;
    im.OGD = G.n_head * G.d_head / G.o_groups; im.SWA = G.n_swa; im.RAW = im.SWA + 16; im.NE = G.n_expert; im.NU = G.n_used;
    im.NFF = G.n_ff; im.IH = G.idx_heads; im.ID = G.idx_dim; im.ITOP = G.idx_topk; im.CB = G.cand_blocks; im.CBS = G.cand_bsize;
    im.EC = (G.engram_ngram - 1) * G.engram_heads; im.ED = G.engram_dim; im.V = G.vocab; im.ER = (tpk::HC + 1) * G.n_embd;
    im.cmax[1] = cfg.ctx_max; im.cmax[2] = cfg.ctx_max / 2;
    im.lazy = env_on("STRATA_TP_LAZY_KV", true) && cfg.n_cards > 1;
    for (int r = 1; r <= 2; ++r) im.rows_done[r] = im.lazy ? std::min<int64_t>(im.cmax[r], 1024) : im.cmax[r];
    if (im.D != 5120 || G.hc != tpk::HC || im.DH != tpdec::DD || im.ND != 64 || im.IH != tpdec::LIH || im.ID != tpdec::LID ||
        im.ITOP > 512 || im.CBS != 8 || im.NFF % QK || im.ER % QK || im.V % 16 || (im.EC * im.ED) % QK || cfg.ctx_max % 256 ||
        im.N < 1 || im.N > 8 || im.NMAX < 1 || im.NMAX > 7) {
        err = "tp: unsupported geometry / config";
        return false;
    }
    im.li.assign((size_t) im.NL, {});
    for (int l = 0; l < im.NL; ++l) {
        Impl::LInfo & L = im.li[(size_t) l];
        L.ratio = G.ratio[(size_t) l];
        L.kv_src = G.is_kv_source(l);
        L.idx_src = G.is_index_source(l);
        L.cand_src = l == G.cand_layer;
        L.uses_cand = G.cand_layer >= 0 && l > G.cand_layer && L.idx_src;
        L.kvl = G.kv_src(l);
        L.idxl = G.idx_src(l);
        L.eslot = G.engram_slot(l);
        if (L.ratio && (L.kvl >= im.NL || L.idxl >= im.NL)) { err = "tp: layer prefix cuts a source"; return false; }
    }
    im.RMAX = 0;
    for (int k = 0; k < im.N; ++k) im.RMAX = std::max(im.RMAX, split(im.ER / QK, im.N, k).n() * QK);
    {
        size_t s = 0;
        const TensorInfo * te = im.model->find("token_embd.weight", &s);
        if (!te || te->type != ty::F16) { err = "tp: token_embd.weight (F16) missing"; return false; }
        im.embd = im.model->shard(s).tensor_data(*te);
    }
    im.IL.make(im.NMAX, (int) im.SWA, (int) im.D);
    // ---- the cards
    std::vector<sycl::device> gpus;
    sycl::platform plat;
    for (const auto & p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        auto d = p.get_devices(sycl::info::device_type::gpu);
        if (d.empty()) continue;
        gpus = d;
        plat = p;
        break;
    }
    if (gpus.empty()) { err = "tp: no Level Zero GPU"; return false; }
    im.ctx = plat.khr_get_default_context();
    std::vector<int> devs = cfg.devices;
    if (devs.empty()) for (int k = 0; k < im.N; ++k) devs.push_back(k);
    if ((int) devs.size() != im.N) { err = "tp: devices list does not match n_cards"; return false; }
    for (int k = 0; k < im.N; ++k) {
        if (devs[(size_t) k] < 0 || devs[(size_t) k] >= (int) gpus.size()) { err = "tp: no GPU " + std::to_string(devs[(size_t) k]); return false; }
        auto c = std::make_unique<Card>();
        c->rank = k;
        c->gpu = devs[(size_t) k];
        c->dev = gpus[(size_t) devs[(size_t) k]];
        c->q = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        c->ar.q = &c->q;
        im.cards.push_back(std::move(c));
    }
    im.nsm = (int) (im.cards[0]->dev.get_info<sycl::info::device::max_compute_units>() / 16);
    im.push = env_on("STRATA_TP_PUSH", false) && im.N > 1;
    if (const char * e = std::getenv("STRATA_TP_STAMP_LAYER")) im.stamp_l = std::atoi(e);
    if (const char * e = std::getenv("STRATA_TP_STAMP_TPU")) im.st_tpu = std::atof(e);
    if (im.push) {   // the push kernels store into the peers' (SYCL USM) slot buffers
        for (auto & a : im.cards)
            for (auto & b : im.cards)
                if (a != b) {
                    if (!a->dev.ext_oneapi_can_access_peer(b->dev)) { im.push = false; break; }
                    try { a->dev.ext_oneapi_enable_peer_access(b->dev); } catch (const std::exception &) {}
                }
        std::fprintf(stderr, "tp: cross-card sums by %s\n", im.push ? "P2P store kernels" : "copy-engine copies (no peer access)");
    }
    {   // every kernel of this library built for every card now: a kernel's first launch on a card otherwise loads its
        // module then (~1 s), and a card waiting for a peer that is still loading gives up
        const double tb = now_us();
        std::vector<sycl::kernel_id> mine;
        for (const auto & id : sycl::get_kernel_ids()) {
            const std::string nm = id.get_name();
            if (nm.find("tpk") != std::string::npos || nm.find("tpdec") != std::string::npos || nm.find("tpfmv") != std::string::npos ||
                nm.find("Ds41Tp") != std::string::npos)
                mine.push_back(id);
        }
        std::vector<sycl::device> dv;
        for (auto & c : im.cards) dv.push_back(c->dev);
        try {
            im.kb = std::make_unique<sycl::kernel_bundle<sycl::bundle_state::executable>>(
                sycl::get_kernel_bundle<sycl::bundle_state::executable>(im.ctx, dv, mine));
        } catch (const std::exception & e) {
            std::fprintf(stderr, "tp: kernel bundle build failed (%s) - first launches load lazily\n", e.what());
        }
        std::fprintf(stderr, "tp: %zu kernels built for %d cards in %.2f s\n", mine.size(), im.N, (now_us() - tb) / 1e6);
    }
    im.hin = sycl::malloc_host<int32_t>(im.IL.words, im.ctx);
    im.hlog = sycl::malloc_host<float>((size_t) im.NMAX * im.V, im.ctx);
    im.heng = sycl::malloc_host<float>((size_t) 2 * im.NMAX * im.EC * im.ED, im.ctx);
    im.herr = sycl::malloc_host<int32_t>((size_t) im.N * 4, im.ctx);
    if (!im.hin || !im.hlog || !im.heng || !im.herr) { err = "tp: pinned allocation failed"; return false; }
    if (!im.hooks.state.empty() && (int) im.hooks.state.size() < im.NL) { err = "tp: hooks.state has too few layers"; return false; }
    im.hflag = sycl::malloc_host<int32_t>((size_t) (2 + im.NL) * 16, im.ctx);   // (a cache line a flag)
    im.hcpu = sycl::malloc_host<float>((size_t) im.NL * im.NMAX * im.D, im.ctx);
    im.hvh = sycl::malloc_host<float>((size_t) im.NMAX * 3 * im.D, im.ctx);
    const size_t mb = ((size_t) 2 * im.NMAX * im.NU + (size_t) im.NMAX * im.D + 16) * 4;   // ids | wts | fn | seq
    im.hmail = (uint8_t *) sycl::malloc_host((size_t) im.NL * mb, im.ctx);
    if (!im.hflag || !im.hcpu || !im.hvh || !im.hmail) { err = "tp: pinned allocation failed"; return false; }
    std::memset(im.hflag, 0, (size_t) (2 + im.NL) * 16 * 4);
    for (int l = 0; l < im.NL; ++l) {
        uint8_t * b = im.hmail + (size_t) l * mb;
        Impl::Mail m;
        m.ids = (int32_t *) b;
        m.wts = (float *) (m.ids + im.NMAX * im.NU);
        m.fn = m.wts + im.NMAX * im.NU;
        m.seq = (int32_t *) (m.fn + im.NMAX * im.D);
        *m.seq = 0;
        im.mail.push_back(m);
    }
    // (the flags are spaced a cache line apart: hflag + 16 k)
    std::memset(im.heng, 0, (size_t) 2 * im.NMAX * im.EC * im.ED * 4);
    const double t0 = now_us();
    for (auto & c : im.cards) {
        if (!im.load_weights(*c) || !im.alloc_buffers(*c)) { err = im.err; return false; }
        std::fprintf(stderr, "tp: rank %d: %lld heads, %lld o-groups, n_ff %lld, Engram rows %lld, vocab rows %lld - %.2f GiB%s%s\n",
                     c->rank, (long long) c->nhk, (long long) c->grp.n(), (long long) c->fk, (long long) c->rk, (long long) c->vk,
                     (c->ar.total + im.lazy_mapped(*c)) / 1073741824.0, c->rank == 0 && im.shared_bytes ?
                     (" (+ " + std::to_string((int) (im.shared_bytes >> 20)) + " MiB of the engine's weights shared)").c_str() : "",
                     im.deferred(*c) ? (" (+ " + std::to_string((int) (im.deferred(*c) >> 20)) + " MiB of KV rows mapped as the context grows)").c_str() : "");
    }
    im.q6_drop();
    im.bal_init();
    if (!im.load_bench_experts()) { err = im.err; return false; }
    if (im.hooks.rtab) {   // the tier's experts: every card computes the selected ones it holds
        static_assert(sizeof(tpk::TpExp) == 40, "TpExp mirrors the tier's RtEnt");
        for (auto & c : im.cards) if (!im.expert_buffers(*c)) { err = im.err; return false; }
        im.refresh_rtab();
        for (auto & c : im.cards) c->q.wait();
    }
    std::fprintf(stderr, "tp: %d cards loaded in %.1f s%s\n", im.N, (now_us() - t0) / 1e6, im.free_str().c_str());
    // the first card-to-card copy of every pair now (timed: the first one sets up the peer path)
    for (int k = 0; k < im.N; ++k)
        for (int j = 0; j < im.N; ++j) {
            if (j == k) continue;
            const double tc = now_us();
            im.cards[(size_t) k]->q.memcpy(im.cards[(size_t) j]->flags + tpk::MAXS - 1, im.cards[(size_t) k]->seq, 4).wait();
            const double dt = now_us() - tc;
            if (dt > 2000) std::fprintf(stderr, "tp: first copy card %d -> %d took %.1f ms\n", k, j, dt / 1e3);
        }
    if (!im.tune_mv()) { err = im.err; return false; }
    // a throw-away pass: every kernel's first launch on every card (module loads, command lists) before any pass whose
    // cards wait on each other (a card still loading made its peers' bounded waits give up); no state exists yet
    {
        const double tw = now_us();
        std::vector<int> t(1, 0);
        if (!pass(t.data(), 1, 0, nullptr, false)) std::fprintf(stderr, "tp: warm-up pass: %s (ignored)\n", im.err.c_str());
        im.err.clear();
        std::fprintf(stderr, "tp: warm-up pass %.2f s%s\n", (now_us() - tw) / 1e6, im.free_str().c_str());
    }
    return true;
}

void Ds41Tp::fill_state(int64_t pos, uint32_t seed) {
    Impl & im = *p_;
    { const int64_t all[3] = { 0, im.cmax[1], im.cmax[2] }; im.ensure_rows(all); }   // lazy replicas: every row mapped
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    // window rows: FP8-ish values; compressed rows / indexer keys: bf16 of small values; compressor state: floats
    std::vector<float> raw((size_t) (im.RAW * im.DH));
    for (int l = 0; l < im.NL; ++l) {
        const Impl::LInfo & L = im.li[(size_t) l];
        for (auto & v : raw) v = U(rng);
        std::vector<uint16_t> comp, idxk;
        std::vector<float> st;
        if (L.kv_src) {
            const int64_t rows = std::min<int64_t>(im.cmax[L.ratio] + 1, pos / L.ratio + 1);
            comp.resize((size_t) ((im.cmax[L.ratio] + 1) * im.DH), 0);
            idxk.resize((size_t) ((im.cmax[L.ratio] + 1) * im.ID), 0);
            for (int64_t i = 0; i < rows * im.DH; ++i) { const float f = U(rng); uint32_t u; std::memcpy(&u, &f, 4); comp[(size_t) i] = (uint16_t) (u >> 16); }
            for (int64_t i = 0; i < rows * im.ID; ++i) { const float f = U(rng); uint32_t u; std::memcpy(&u, &f, 4); idxk[(size_t) i] = (uint16_t) (u >> 16); }
            st.resize((size_t) (2 * im.DH));
            for (auto & v : st) v = U(rng);
        }
        for (auto & c : im.cards) {
            const LS & S = c->ls[(size_t) l];
            c->q.memcpy(S.raw, raw.data(), raw.size() * 4);
            if (S.comp) c->q.memcpy(S.comp, comp.data(), comp.size() * 2);
            if (S.idxk) c->q.memcpy(S.idxk, idxk.data(), idxk.size() * 2);
            if (S.st_kv) { c->q.memcpy(S.st_kv, st.data(), st.size() * 4); c->q.memcpy(S.st_sc, st.data(), st.size() * 4); }
            c->q.wait();
        }
    }
}

bool Ds41Tp::pass(const int * tids, int n, int pos0, float * logits, float * main_hidden, bool graph) {
    Impl & im = *p_;
    im.err.clear();
    if (n < 1 || n > im.NMAX) { im.err = "tp: pass width"; return false; }
    if (pos0 < 0 || (int64_t) pos0 + n > im.cfg.ctx_max) { im.err = "tp: pass past the context"; return false; }
    for (int t = 0; t < n; ++t) if (tids[t] < 0 || tids[t] >= im.V) { im.err = "tp: token id out of range"; return false; }
    Impl::PassK P;
    P.n = n;
    im.caps(pos0, n, P.cap);
    if (!im.ensure_rows(P.cap)) return false;   // lazy replicas: the rows the indexer scans (a cap bucket)
    const double t0 = now_us();
    static const bool prof = env_on("STRATA_TP_PROF", false);
    im.make_inputs(tids, n, pos0);
    im.tap_on = main_hidden != nullptr;
    im.refresh_rtab();
    const double t1 = now_us();
    std::vector<sycl::event> ev;
    if (graph) {
        const auto key = std::make_tuple(n * 2 + (im.tap_on ? 1 : 0), P.cap[1], P.cap[2]);
        if (!im.record(P, key)) return false;
        for (auto & cp : im.cards) ev.push_back(cp->q.ext_oneapi_graph(cp->graphs.at(key)));
    } else {
        // eager: every card's commands layer by layer (the cards meet twice a layer; the host stays ahead)
        im.sync_idx = 0;
        for (auto & cp : im.cards) im.emit_prologue(*cp, P);
        for (int l = 0; l < im.NL; ++l) {
            const int s0 = im.sync_idx;
            int s1 = s0;
            for (auto & cp : im.cards) { im.sync_idx = s0; im.emit_layer(*cp, l, P); s1 = im.sync_idx; }
            im.sync_idx = s1;
        }
        for (auto & cp : im.cards) im.emit_epilogue(*cp, P);
        if (!im.err.empty()) return false;
    }
    // ---- the host's part while the cards run: the Engram rows (layers 1 / 14 wait for them), then layer by layer the
    // CPU pool's experts (rank 0 waits for those only on layers that route to one)
    const int32_t want0 = (int32_t) im.pid * 4096;
    bool host_ok = true;
    const double t2 = now_us();
    double t_cpu = 0, t_mail = 0;
    bool eng_ok = true;
    double t_eng = 0;
    // the first Engram layer's rows, its gate; then the second's (its layer is ~13 layers later).  On failure the rows
    // are stale but the gates open anyway: the pass must not hang; the error is returned.
    auto engram_rows = [&im, tids, n, pos0, want0, &eng_ok, &t_eng] {
        const double te = now_us();
        float * outp[2] = { im.heng, im.heng + (size_t) im.NMAX * im.EC * im.ED };
        const bool split = im.hooks.engram_slot != nullptr;
        if (split) {
            if (!im.hooks.engram_slot(tids, n, pos0, 0, outp[0])) eng_ok = false;
        } else if (!im.hooks.engram(tids, n, pos0, outp)) eng_ok = false;
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<volatile int32_t *>(im.hflag)[0] = want0 + Impl::S_ENG;
        if (split && !im.hooks.engram_slot(tids, n, pos0, 1, outp[1])) eng_ok = false;
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<volatile int32_t *>(im.hflag + 16 * (1 + im.NL))[0] = want0 + Impl::S_ENG + 1;
        t_eng = now_us() - te;
    };
    static const bool eng_async = env_on("STRATA_TP_ENG_ASYNC", true);
    const bool eng_posted = im.hooks.engram && eng_async && im.hooks.cpu_part;
    if (eng_posted) im.eng_post(engram_rows);
    else if (im.hooks.engram) engram_rows();
    const double t3 = now_us();
    if (im.hooks.cpu_part) {
        for (int l = 0; l < im.NL && host_ok; ++l) {
            const Impl::Mail & m = im.mail[(size_t) l];
            const int32_t want = want0 + Impl::S_MAIL + l;
            const double tw = now_us();
            for (int it = 0; *reinterpret_cast<volatile int32_t *>(m.seq) != want; ++it) {
#if defined(__x86_64__)
                __builtin_ia32_pause();   // (the SMT sibling runs the CPU pool / the promotion helper)
#endif
                if ((it & 1023) == 0 && now_us() - tw > 5e6) { host_ok = false; im.err = "tp: rank 0's mailbox of layer " + std::to_string(l) + " never came"; break; }
            }
            if (!host_ok) break;
            std::atomic_thread_fence(std::memory_order_acquire);
            const double tc = now_us();
            t_mail += tc - tw;
            float * out = im.hcpu + (size_t) l * im.NMAX * im.D;
            if (im.hooks.cpu_part(l, n, m.ids, m.wts, m.fn, out) > 0) {
                std::atomic_thread_fence(std::memory_order_release);
                reinterpret_cast<volatile int32_t *>(im.hflag + 16 * (1 + l))[0] = want;
                ++im.pf_cpu_layers;
            }
            t_cpu += now_us() - tc;
        }
    }
    if (eng_posted) im.eng_join();
    if (!eng_ok) { host_ok = false; if (im.err.empty()) im.err = "tp: Engram rows failed"; }
    const double t4 = now_us();
    for (auto & e : ev) e.wait();
    for (auto & cp : im.cards) cp->q.wait();
    const double t5 = now_us();
    im.last_pass_us = t5 - t0;
    if (im.stamp_l == -1 || (im.stamp_l >= 0 && im.stamp_l < im.NL)) {
        static const char * nm[Impl::NST] = {"", "hc_a", "qkv", "rms_q8", "q_b", "kv_fin", "comp", "indexer", "attn", "wo_a", "quant",
                                             "wo_b", "xA-copy", "xA-wait", "hc_f", "rt-mv", "router", "mailbox", "sh-gu", "swiglu", "sh-dn",
                                             "route", "q-rows", "experts", "cpu-wait", "xB-copy", "xB-wait"};
        const int nls = im.stamp_l == -1 ? im.NL : 1;
        const size_t per = (size_t) nls * Impl::NST;
        im.st_acc.resize((size_t) im.N * per, 0.0);
        std::vector<uint64_t> h(per);
        for (int k = 0; k < im.N; ++k) {
            Card & c = *im.cards[(size_t) k];
            c.q.memcpy(h.data(), c.stamps, per * 8).wait();
            for (int ll = 0; ll < nls; ++ll)
                for (int i = 1; i < Impl::NST; ++i)
                    im.st_acc[(size_t) k * per + (size_t) ll * Impl::NST + i] +=
                        (double) (int64_t) (h[(size_t) ll * Impl::NST + i] - h[(size_t) ll * Impl::NST + i - 1]) / im.st_tpu;
        }
        if (nls > 1 && ++im.st_n % 128 == 0) {   // every layer: per layer the expert step, the CPU wait, the sums' waits
            auto at = [&](int k, int ll, int i) { return im.st_acc[(size_t) k * per + (size_t) ll * Impl::NST + i] / 128.0; };
            double se[8] = {0}, sw[8] = {0}, sa[8] = {0}, sl[8] = {0}, scpu = 0;
            for (int ll = 0; ll < im.NL; ++ll) {
                std::string line;
                char b[64];
                for (int k = 0; k < im.N; ++k) {
                    double tl = 0;
                    for (int i = 1; i < Impl::NST; ++i) tl += at(k, ll, i);
                    se[k] += at(k, ll, 23); sw[k] += at(k, ll, 26); sa[k] += at(k, ll, 13); sl[k] += tl;
                    std::snprintf(b, sizeof b, " | c%d %.0f: ex %.0f xA %.0f xB %.0f", k, tl, at(k, ll, 23), at(k, ll, 13), at(k, ll, 26));
                    line += b;
                }
                scpu += at(0, ll, 24);
                std::fprintf(stderr, "tp stamps L%02d (passes %lld-%lld): cpu-wait %.0f%s\n", ll, (long long) im.st_n - 127, (long long) im.st_n,
                             at(0, ll, 24), line.c_str());
            }
            std::string sum;
            char b[96];
            for (int k = 0; k < im.N; ++k) {
                std::snprintf(b, sizeof b, " | c%d layers %.0f: experts %.0f, xA-wait %.0f, xB-wait %.0f", k, sl[k], se[k], sa[k], sw[k]);
                sum += b;
            }
            std::fprintf(stderr, "tp stamps all layers (us a pass): cpu-wait %.0f%s\n", scpu, sum.c_str());
            std::fill(im.st_acc.begin(), im.st_acc.end(), 0.0);
        } else if (nls == 1 && ++im.st_n % 128 == 0) {
            for (int k = 0; k < im.N; ++k) {
                std::string line;
                double tot = 0;
                char b[48];
                for (int i = 1; i < Impl::NST; ++i) {
                    const double v = im.st_acc[(size_t) k * Impl::NST + i] / 128.0;
                    tot += v;
                    std::snprintf(b, sizeof b, " %s %.1f", nm[i], v);
                    line += b;
                }
                std::fprintf(stderr, "tp stamps layer %d card %d (passes %lld-%lld, us):%s = %.1f\n", im.stamp_l, k, (long long) im.st_n - 127,
                             (long long) im.st_n, line.c_str(), tot);
            }
            std::fill(im.st_acc.begin(), im.st_acc.end(), 0.0);
        }
    }
    if (prof) {
        im.pf[0] += t1 - t0; im.pf[1] += t2 - t1; im.pf[2] += t3 - t2; im.pf[3] += t_mail; im.pf[4] += t_cpu; im.pf[5] += t5 - t4;
        if (eng_posted) im.pf_eng += t_eng;
        if (++im.pf_n % 128 == 0) {
            const double k = 1.0 / 128.0;
            std::fprintf(stderr, "tp prof (passes %lld-%lld, %d recorded, %.2f CPU-pool layers a pass): inputs %.0f us, launch %.0f, engram %.0f, "
                         "mailbox waits %.0f, cpu experts %.0f, card wait %.0f = %.0f us a pass (Engram helper %.0f)%s\n", (long long) im.pf_n - 127,
                         (long long) im.pf_n, im.pf_rec, im.pf_cpu_layers * k, im.pf[0] * k, im.pf[1] * k, im.pf[2] * k, im.pf[3] * k, im.pf[4] * k,
                         im.pf[5] * k, (im.pf[0] + im.pf[1] + im.pf[2] + im.pf[3] + im.pf[4] + im.pf[5]) * k, im.pf_eng * k, im.free_str().c_str());
            for (double & v : im.pf) v = 0;
            im.pf_eng = 0;
            im.pf_cpu_layers = 0;
            im.pf_rec = 0;
        }
    }
    if (!host_ok) return false;
    if (main_hidden) std::memcpy(main_hidden, im.hvh, (size_t) n * 3 * im.D * 4);
    if (env_on("STRATA_TP_SPINSTAT", false)) {
        std::string st;
        for (int k = 0; k < im.N; ++k) st += " r" + std::to_string(k) + ": max " + std::to_string(im.herr[k * 4 + 2]) + " polls, " + std::to_string(im.herr[k * 4 + 3]) + " waits";
        std::fprintf(stderr, "tp spins:%s\n", st.c_str());
    }
    for (int k = 0; k < im.N; ++k)
        if (im.herr[k * 4] || im.herr[k * 4 + 1]) {
            im.err = "tp: rank " + std::to_string(k) + " gave up waiting (" + std::to_string(im.herr[k * 4]) + " peer, " +
                     std::to_string(im.herr[k * 4 + 1]) + " host)";
            return false;
        }
    if (logits) std::memcpy(logits, im.hlog, (size_t) n * im.V * 4);
    return true;
}

bool Ds41Tp::attach_experts(const void * rtab, const volatile uint64_t * rtab_version,
                            std::function<int(int, int, const int32_t *, const float *, const float *, float *)> cpu_part,
                            std::function<bool(std::vector<uint32_t> &)> rtab_dirty) {
    Impl & im = *p_;
    im.err.clear();
    if (!rtab || !rtab_version) { im.err = "tp: no residency table"; return false; }
    im.hooks.rtab = rtab;
    im.hooks.rtab_version = rtab_version;
    im.hooks.cpu_part = std::move(cpu_part);
    im.hooks.rtab_dirty = std::move(rtab_dirty);
    if (!im.hkeys) {
        im.hkeys_cap = 4096;
        im.hkeys = sycl::malloc_host<uint32_t>(im.hkeys_cap, im.ctx);
        if (!im.hkeys) im.hkeys_cap = 0;
    }
    for (auto & c : im.cards) if (!c->jobs && !im.expert_buffers(*c)) return false;
    const tpk::TpExp * t = (const tpk::TpExp *) rtab;
    for (int64_t i = 0; i < (int64_t) im.NL * im.NE; ++i)
        if (t[i].owner >= im.N) { im.err = "tp: an expert lives on a card past the TP ranks (the tier's cards must be the TP cards)"; return false; }
    for (auto & c : im.cards) c->graphs.clear();   // recorded without the expert part
    im.rtab_seen = ~(uint64_t) 0;
    im.refresh_rtab();
    for (auto & c : im.cards) c->q.wait();
    // the graphs of the first cap bucket (every context below ~1K positions), every width, with and without DSpark's
    // taps: recorded now, not inside the first requests (~40 ms a width; STRATA_TP_PRERECORD=0: lazily)
    if (env_on("STRATA_TP_PRERECORD", true)) {
        const double t0 = now_us();
        int nk = 0;
        for (int n = 1; n <= im.NMAX; ++n)
            for (int taps = 0; taps < 2; ++taps) {
                Impl::PassK P;
                P.n = n;
                im.caps(0, n, P.cap);
                if (!im.record(P, std::make_tuple(n * 2 + taps, P.cap[1], P.cap[2]))) return false;
                ++nk;
            }
        im.pf_rec = 0;
        std::fprintf(stderr, "tp: %d pass graphs recorded a card in %.2f s%s\n", nk, (now_us() - t0) / 1e6, im.free_str().c_str());
    }
    return true;
}

bool Ds41Tp::sync_state(int64_t lo, int64_t hi, bool all) {
    Impl & im = *p_;
    if (im.N < 2) return true;
    const bool every = all || hi > im.cmax[1] || lo < 0;
    if (im.lazy && !every) {   // the replicas' pages for the rows up to hi first (as a pass there would: a cap bucket)
        int64_t want[3] = { 0, 0, 0 };
        for (int r = 1; r <= 2; ++r) want[r] = std::min<int64_t>(im.cmax[r], next_pow2(std::max<int64_t>(1024, (hi + r - 1) / r + 1)));
        if (!im.ensure_rows(want)) return false;
    }
    const Card & c0 = *im.cards[0];
    for (int k = 1; k < im.N; ++k) {
        Card & c = *im.cards[(size_t) k];   // card k pulls (a P2P read by its copy engine from card 0's buffers)
        for (int l = 0; l < im.NL; ++l) {
            const Impl::LInfo & L = im.li[(size_t) l];
            const LS & a = c0.ls[(size_t) l];
            LS & b = c.ls[(size_t) l];
            c.q.memcpy(b.raw, a.raw, (size_t) (im.RAW * im.DH) * 4);
            if (!L.kv_src) continue;
            const int64_t r = L.ratio, rows = im.cmax[r] + 1;
            const int64_t top = im.lazy ? im.rows_done[r] : rows - 1;   // rows past the mapped ones: copied when mapped
            int64_t r0 = every ? 0 : std::max<int64_t>(0, lo / r), r1 = every ? top : std::min<int64_t>(top, (hi + r - 1) / r + 1);
            if (r1 > r0) {
                c.q.memcpy(b.comp + r0 * im.DH, a.comp + r0 * im.DH, (size_t) ((r1 - r0) * im.DH) * 2);
                c.q.memcpy(b.idxk + r0 * im.ID, a.idxk + r0 * im.ID, (size_t) ((r1 - r0) * im.ID) * 2);
            }
            // the spare row (an open group's)
            c.q.memcpy(b.comp + (rows - 1) * im.DH, a.comp + (rows - 1) * im.DH, (size_t) im.DH * 2);
            c.q.memcpy(b.idxk + (rows - 1) * im.ID, a.idxk + (rows - 1) * im.ID, (size_t) im.ID * 2);
            if (L.ratio == 2) {
                c.q.memcpy(b.st_kv, a.st_kv, (size_t) (2 * im.DH) * 4);
                c.q.memcpy(b.st_sc, a.st_sc, (size_t) (2 * im.DH) * 4);
            }
        }
    }
    try {
        for (int k = 1; k < im.N; ++k) im.cards[(size_t) k]->q.wait_and_throw();
    } catch (const std::exception & e) {
        im.err = std::string("tp: state sync: ") + e.what();
        return false;
    }
    return true;
}

void Ds41Tp::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= im.NL) return;
    for (auto & c : im.cards) c->q.memcpy(const_cast<float *>(c->lw[(size_t) il].rbias), bias, (size_t) im.NE * 4);
    for (auto & c : im.cards) c->q.wait();
}

bool Ds41Tp::dump(const std::string & dir, int n) {
    Impl & im = *p_;
    for (auto & cp : im.cards) {
        Card & c = *cp;
        const int64_t D = im.D, HC = tpk::HC;
        struct B { const char * name; const void * p; size_t bytes; };
        const size_t sl = (size_t) im.NMAX * D;
        std::vector<B> bs = {
            { "x", c.x, (size_t) n * HC * D * 4 }, { "pre", c.pre, (size_t) n * HC * 4 }, { "hap", c.hap, (size_t) n * HC * D * 4 },
            { "xn", c.xn, (size_t) n * D * 4 }, { "apre", c.apre, (size_t) n * HC * 4 }, { "apost", c.apost, (size_t) n * HC * 4 },
            { "acomb", c.acomb, (size_t) n * HC * HC * 4 }, { "qkv", c.qkv, (size_t) n * (im.RQ + im.DH) * 4 },
            { "qr", c.qr, (size_t) n * im.RQ * 4 }, { "qh", c.qh, (size_t) n * c.nhk * im.DH * 4 }, { "kvn", c.kvn, (size_t) n * im.DH * 4 },
            { "o", c.o, (size_t) n * c.nhk * im.DH * 4 }, { "oa", c.oa, (size_t) n * c.grp.n() * im.OL * 4 },
            { "ara", c.ara + (size_t) c.rank * sl, (size_t) n * D * 4 }, { "ara_all", c.ara, (size_t) im.N * sl * 4 },
            { "fn", c.fn, (size_t) n * D * 4 }, { "lg", c.lg, (size_t) n * im.NE * 4 }, { "ids", c.ids, (size_t) n * im.NU * 4 },
            { "wts", c.wts, (size_t) n * im.NU * 4 }, { "gu", c.gu, (size_t) n * 2 * c.fk * 4 }, { "sh", c.sh, (size_t) n * c.fk * 4 },
            { "arb", c.arb + (size_t) c.rank * sl, (size_t) n * D * 4 }, { "arb_all", c.arb, (size_t) im.N * sl * 4 },
            { "z", c.z, (size_t) n * D * 4 }, { "zn", c.zn, (size_t) n * D * 4 },
        };
        for (int l = 0; l < im.NL; ++l) {   // the index sources' picks
            const LS & S = c.ls[(size_t) l];
            if (!S.sel) continue;
            static std::list<std::string> names;   // (stable addresses: bs keeps the c_str()s)
            names.push_back("sel" + std::to_string(l));
            bs.push_back({ names.back().c_str(), S.sel, (size_t) n * im.ITOP * 4 });
            names.push_back("selm" + std::to_string(l));
            bs.push_back({ names.back().c_str(), S.selm, (size_t) n * im.ITOP * 4 });
        }
        for (const auto & b : bs) {
            std::vector<uint8_t> h(b.bytes);
            c.q.memcpy(h.data(), b.p, b.bytes).wait();
            const std::string f = dir + "/r" + std::to_string(c.rank) + "_" + b.name + ".bin";
            FILE * fp = std::fopen(f.c_str(), "wb");
            if (!fp) return false;
            std::fwrite(h.data(), 1, h.size(), fp);
            std::fclose(fp);
        }
    }
    return true;
}

std::string Ds41Tp::report() const {
    char b[128];
    std::snprintf(b, sizeof b, "last pass %.1f us (%.1f us/layer)", p_->last_pass_us, p_->last_pass_us / std::max(1, p_->NL));
    return b;
}

}  // namespace strata::v41
