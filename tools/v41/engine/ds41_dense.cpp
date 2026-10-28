// tools/v41/engine/ds41_dense.cpp - see ds41_dense.hpp.  Line references are to DeepSeek's inference/model.py.
#include "ds41_dense.hpp"

#include "ds41_engram.hpp"
#include "ds41_ops.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#endif   // ggml_cgraph::uid (graph reuse)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <set>
#include <thread>
#include <fcntl.h>
#include <spawn.h>
#include <dirent.h>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <sys/syscall.h>
#include <condition_variable>

#include <strings.h>

using namespace strata;
using namespace strata::v41;

namespace {

constexpr float NEG_INF = -INFINITY;

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

bool env_on(const char * k, bool dflt) {
    const char * e = std::getenv(k);
    return e ? std::atoi(e) != 0 : dflt;
}

// The decode graphs are built once and reused; stamping a uid lets ggml-cuda reuse its captured CUDA graph.  One
// allocator serves every cached graph (they run one at a time); when it has to grow, every graph allocated before
// is re-allocated on its next use (the epoch), as tools/ds4/ds4_dense.cpp alloc_graph does.
struct Allo {
    ggml_gallocr_t a = nullptr;
    uint64_t epoch = 0;
    std::map<const ggml_cgraph *, uint64_t> graph_epoch;
    std::set<const void *> compute_bufs;
    ~Allo() { if (a) ggml_gallocr_free(a); }
    // alloc() returns at once (the graph keeps its addresses): nothing is (re)allocated under in-flight kernels
    bool kept(const ggml_cgraph * g, bool reuse) const {
        auto it = graph_epoch.find(g);
        return reuse && g->uid != 0 && it != graph_epoch.end() && it->second == epoch;
    }
    bool alloc(ggml_cgraph * g, bool reuse) {
        auto it = graph_epoch.find(g);
        if (reuse && g->uid != 0 && it != graph_epoch.end() && it->second == epoch) return true;
        if (it != graph_epoch.end() && it->second != epoch) {
            auto computed = [&](const ggml_tensor * t) { return t && t->buffer && compute_bufs.count(t->buffer); };
            auto unbind = [&](ggml_tensor * t) {
                if (computed(t) || (t->view_src && computed(t->view_src))) { t->data = nullptr; t->buffer = nullptr; }
            };
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                ggml_tensor * t = ggml_graph_node(g, i);
                for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) unbind(t->src[j]);
                unbind(t);
            }
            for (int i = 0; i < g->n_leafs; ++i) unbind(g->leafs[i]);
        }
        const size_t before = ggml_gallocr_get_buffer_size(a, 0);
        if (!ggml_gallocr_alloc_graph(a, g)) return false;
        if (ggml_gallocr_get_buffer_size(a, 0) != before && before != 0) ++epoch;
        auto note = [&](const ggml_tensor * t) {
            if (!t || !t->buffer) return;
            if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) compute_bufs.insert(t->buffer);
            else compute_bufs.erase(t->buffer);
        };
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            const ggml_tensor * t = ggml_graph_node(g, i);
            note(t);
            for (int j = 0; j < GGML_MAX_SRC; ++j) note(t->src[j]);
        }
        for (int i = 0; i < g->n_leafs; ++i) note(g->leafs[i]);
        graph_epoch[g] = epoch;
        if (reuse) g->uid = ggml_graph_next_uid();
        return true;
    }
};

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;
    ~WStore() {
        for (auto b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "[ds41_dense] missing weight %s\n", n.c_str()); std::abort(); }
        return it->second;
    }
    bool has(const std::string & n) const { return t.count(n) != 0; }
};

bool is_routed(const std::string & n) {
    return n.find("_exps.") != std::string::npos || n.find("ffn_exps_") != std::string::npos;
}

}  // namespace

// ================================================================== Impl

// larry: a host mirror the dense card's rows are read back into - pinned (the device's host buffer type: SYCL
// malloc_host) when a backend is given, so the async readbacks really are async (a device -> host copy into pageable
// memory is staged and blocks the host until the card reaches it); a plain vector otherwise
template <typename T> struct HostRows {
    std::vector<T> v;
    ggml_backend_buffer_t buf = nullptr;
    T * p = nullptr;
    size_t n = 0;
    HostRows() = default;
    HostRows(const HostRows &) = delete;
    HostRows & operator=(const HostRows &) = delete;
    ~HostRows() { if (buf) ggml_backend_buffer_free(buf); }
    void resize(size_t count, ggml_backend_t be = nullptr) {
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        n = count;
        ggml_backend_buffer_type_t hbt = be && count ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(be)) : nullptr;
        if (hbt && (buf = ggml_backend_buft_alloc_buffer(hbt, count * sizeof(T))) != nullptr) {
            p = (T *) ggml_backend_buffer_get_base(buf);
            std::memset(p, 0, count * sizeof(T));
            v.clear();
            return;
        }
        v.assign(count, T());
        p = v.data();
    }
    T * data() { return p; }
    const T * data() const { return p; }
    size_t size() const { return n; }
    bool empty() const { return n == 0; }
};

struct Ds41Dense::Impl {
    std::function<bool()> oom_hook;       // set_oom_hook: frees device memory when a graph buffer does not fit
    Ds41DenseConfig cfg;
    Ds41Geometry g;
    std::unique_ptr<GgufModel> model;
    std::vector<int64_t> layers;          // blocks in the file (0..nl-1)
    int64_t nl = 0;
    ggml_backend_t backend = nullptr;
    bool own_backend = false, cpu = true;
    bool sycl = false;   // larry: the backend is ggml-sycl (its DSV4 prompt flash attention needs no 256-padding)
    WStore w;
    Ds41Engram engram;
    std::string err;

    int64_t D = 0, HC = 0, DH = 0, DHR = 0, NH = 0, OG = 0, OL = 0, OGD = 0, RQ = 0, SWA = 0, RAW = 0;
    int64_t IH = 0, ID = 0, ITOP = 0, NE = 0, NU = 0, NFF = 0, NC = 0, CB = 0, CBS = 0, EC = 0, ED = 0;
    int64_t cmax[3] = { 0, 0, 0 };        // compressed rows per ratio (index 1 / 2)

    // ---- persistent state (sctx / sbuf)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;   ///< larry (cfg.kv_alloc): over the caller's KV units
    std::vector<std::pair<ggml_tensor *, int>> kv_units;   ///< larry: (comp / idxk tensor, its ratio) in kv_alloc's order
    ggml_tensor * x_state = nullptr;      // [D, HC, NC]
    ggml_tensor * pre_state = nullptr;    // [HC, NC]   the next attention's pre-mix
    ggml_tensor * hap = nullptr;          // [D, HC, NC] the FFN residual (hc_post of the attention)
    ggml_tensor * post_f = nullptr;       // [HC, NC]
    ggml_tensor * comb_f = nullptr;       // [HC, HC, NC]
    ggml_tensor * shexp = nullptr;        // [D, NC]
    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
    // larry: o_fn's own buffer, NCmax columns for the engine's life (the expert cards read it card-to-card; a fresh
    // buffer per batch paid the peer mapping at the tier's first copy)
    ggml_context * fn_ctx = nullptr;
    ggml_backend_buffer_t fn_buf = nullptr;
    // larry, STRATA_TP_OPROJ=2: between a decode layer's two attention halves (<= 8 columns) - the de-RoPEd attention
    // output, the attention hc mixes (hc_fb's pre | post | comb planes) and the output projection
    ggml_tensor * o_att = nullptr;        // [OGD * OG, <= 8]
    ggml_tensor * o_mix = nullptr;        // [2 HC + HC * HC, <= 8]
    ggml_tensor * o_proj = nullptr;       // [D, <= 8]
    // larry, STRATA_Q8_FUSE: o_fn's int8 form (q8: ggml_dsv41_q8_floats(D, n) floats) for the shared expert's graph
    ggml_tensor * o_fn8 = nullptr;        // F32 [q8 of D x <= 8]
    ggml_tensor * o_ids = nullptr;        // I32 [NU, NC]
    ggml_tensor * o_wts = nullptr;        // [NU, NC]
    ggml_tensor * routed = nullptr;       // [D, NC]
    ggml_tensor * cand = nullptr;         // I32 [CB, NC]  candidate blocks of cand_layer
    struct Layer {
        int64_t ratio = 0;
        bool kv_src = false, idx_src = false, cand_src = false, uses_cand = false;
        int64_t kvl = -1, idxl = -1;      // the layers whose compressed KV / top-k this one reads
        int64_t eslot = -1;               // Engram slot
        ggml_tensor * raw = nullptr;      // [DH, RAW]       window K (= V), post-RoPE
        ggml_tensor * comp = nullptr;     // [DH, cmax + 1]  compressed KV (kv sources); row cmax = spare
        ggml_tensor * idxk = nullptr;     // [ID, cmax + 1]  indexer keys (kv sources)
        ggml_tensor * st_kv = nullptr;    // [DH, 2]         ratio-2 compressor state (values)
        ggml_tensor * st_sc = nullptr;    // [DH, 2]         ... (scores)
        ggml_tensor * sel = nullptr;      // I32 [ITOP, NC]  top-k rows (index sources)
        ggml_tensor * selm = nullptr;     // F32 [ITOP, NC]  0 / -inf (row visible to the query)
        ggml_tensor * raw_snap = nullptr, * st_kv_snap = nullptr, * st_sc_snap = nullptr;   // snapshot()
        ggml_tensor * rbias = nullptr;    // F32 [NE]        selection-only routing bias (set_route_bias), 0 = model routing
    };
    int snap_pos = -1;
    // larry, engine1: what was written since take_written() (an external decode core re-syncs from it; at first: all)
    int64_t wr_lo = 0, wr_hi = INT64_MAX;
    bool wr_changed = true;
    void wrote(int64_t lo, int64_t hi) { wr_lo = std::min(wr_lo, lo); wr_hi = std::max(wr_hi, hi); wr_changed = true; }
    // (larry, batch slots: by layer l and kind k - 0 raw_snap, 1 st_kv_snap, 2 st_sc_snap, 3 comp, 4 idxk - so a parked
    // conversation comes back into whichever conversation is bound)
    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; int l = -1, k = -1; };
    // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites; larry: a pool of
    // slots, grown on demand (patch-parkpool.py)
    struct ParkSlot { std::vector<ParkedRows> rows; int pos = -1; int64_t n = 0; size_t bytes = 0; };
    std::vector<ParkSlot> pk;
    std::vector<Layer> ly;
    // larry (batch slots, patch-seqs-dense.py): every conversation's state tensors (per layer: raw, raw_snap, comp, idxk,
    // st_kv, st_sc, st_kv_snap, st_sc_snap - the Layer members are the bound one's), snapshot position, written range
    struct SeqT { std::vector<std::array<ggml_tensor *, 8>> t; int snap_pos = -1; int64_t wr_lo = 0, wr_hi = INT64_MAX; bool wr_changed = true; };
    std::vector<SeqT> seqs;
    int seq = 0;
    std::vector<int> kv_unit_seq;   // the conversation of each kv_units entry

    // ---- per-pass inputs (ictx / ibuf), uploaded by begin()
    ggml_context * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    // ---- the NC-column tensors (state columns + their per-pass inputs) in their own context / buffer, so
    // set_columns() can resize them: --serve keeps S columns while it decodes and grows to the prompt batch only for a
    // layer-major batch (~0.31 MiB a column: 8192 columns = 2.6 GiB that decode gives to expert slots instead)
    ggml_context * nctx = nullptr;
    ggml_backend_buffer_t nbuf = nullptr;
    int64_t NCmax = 0;
    ggml_tensor * i_emb = nullptr;        // [D, NC]
    ggml_tensor * i_pos = nullptr;        // I32 [NC]
    ggml_tensor * i_widx = nullptr;       // I32 [SWA * NC]   per query: rows of [ring | pass rows]
    ggml_tensor * i_wmask = nullptr;      // F32 [SWA, NC]
    ggml_tensor * i_wslot = nullptr;      // I32 [min(NC, RAW)]  ring slots of the pass's last rows
    ggml_tensor * i_nvis[3] = {};         // F32 [NC]   visible compressed rows per query (ratio 1 / 2)
    ggml_tensor * i_cslot[3] = {};        // I32 [NC]   compressed row each token writes (spare if its group is open)
    ggml_tensor * i_cpos[3] = {};         // I32 [NC]   RoPE position of that row (group start)
    ggml_tensor * i_sidx = nullptr;       // I32 [2 * NC] ratio-2 state rows per token, of [state ring | pass states]
    ggml_tensor * i_sslot = nullptr;      // I32 [min(NC, 2)]
    // upload_span's last (pos0, col0, n): its four index tensors depend only on these, not on the layer, so the 39
    // layers after the first skip identical re-uploads.  Those tiny H2D copies queued behind the chunk prestage's
    // ~1.5 GiB DMA of the next layer's experts and cost ~65 ms a layer (~2.5 s a turn).  Any other writer of the four
    // tensors (upload_inputs) or their reallocation (alloc_state) resets it.
    int64_t span_memo_pos0 = -1;
    int span_memo_col0 = -1, span_memo_n = -1;
    ggml_tensor * i_lastblk = nullptr;    // F32 [NC]   (nvis - 1) / block size: the pinned candidate block (ratio 1)
    ggml_tensor * i_img = nullptr;        // larry (vision): F32 [NC]  1 at an image position (VL routing bias, no Engram)
    std::vector<Ds41Dense::ImageSpan> img;   // larry (vision): the prompt's image spans (set_images)
    const float * img_row(int64_t p) const {
        for (const auto & s : img) if (p >= s.p0 && p < s.p0 + s.n) return s.rows + (size_t) (p - s.p0) * (size_t) D;
        return nullptr;
    }
    bool img_in(int64_t p0, int64_t n) const {
        for (const auto & s : img) if (s.p0 < p0 + n && s.p0 + s.n > p0) return true;
        return false;
    }
    ggml_tensor * i_eng[2] = {};          // F32 [EC * ED, NC]  Engram rows
    ggml_tensor * neg_ramp = nullptr;     // F32 [cmax1]  -i     (constant)
    ggml_tensor * blk_ramp = nullptr;     // F32 [cmax1 / CBS + 1]  i (constant)

    // ---- the pass
    int pos0 = 0, n = 0;
    bool fin_pending = false;   // larry, STRATA_FIN_DEFER: the last finish runs inside the next attention graph
    int64_t pass_cap[3] = { 0, 0, 0 };
    // ---- the layer-major prompt batch (begin_batch): S = the dense sub-batch size, batch = B, span_cap per sub-batch
    int64_t S = 1, batch = 0, col0 = 0;
    int batch_pos0 = 0;                             // the batch's first position
    std::vector<std::array<int64_t, 3>> span_cap;   // [sub-batch][ratio] pass_cap
    HostRows<float> host_fn, host_wts;
    std::vector<float> host_logits, host_emb;
    // larry, STRATA_EMB_PIN: the token table pinned (f16 rows), a batch's rows staged pinned, i_emb f16
    bool emb_f16 = false;
    HostRows<uint16_t> emb_pin, emb_stage;
    ggml_backend_event_t emb_ev = nullptr;
    HostRows<int32_t> host_ids;
    const uint8_t * embd = nullptr;       // token_embd F16 rows on the mmap
    std::vector<std::vector<float>> eng_rows;
    // The pass's Engram reads, started (async) by upload_inputs and waited at the first Engram layer's span /
    // attn_router: the whole pass's rows are read from disk while the embeddings, the init graph and the layers below
    // the first Engram layer (1) run.  Null once uploaded (or when Engram is off) - see engram_ready().
    std::shared_ptr<Ds41Engram::Batch> eng_batch;
    std::shared_ptr<Ds41Engram::Batch> eng_pf;   // larry: the next batch's rows being read into the row cache
    // larry, STRATA_ENG_START_ASYNC: a prompt batch's start_into on a thread (eng_batch is set when it is resolved)
    std::future<std::shared_ptr<Ds41Engram::Batch>> eng_fut;
    std::string eng_fut_err;
    // larry, STRATA_ENG_PIN: a prompt batch's rows in pinned memory, uploaded async (eng_pin_n floats a slot; 0 = the
    // batch used eng_rows), the next batch's writes waiting on eng_pin_ev
    ggml_backend_buffer_t eng_pin_buf = nullptr;
    float * eng_pin[2] = {};
    size_t eng_pin_n = 0;
    ggml_backend_event_t eng_pin_ev = nullptr;
    std::vector<int> eng_pf_tids;                // larry: a prefetch request waiting for this batch's own reads
    int eng_pf_pos0 = -1;

    // ---- graphs
    ggml_context * gctx = nullptr;        // node structs of the cached (n = 1) graphs
    std::vector<uint8_t> pf_meta;         // node structs of a pass graph with n > 1 (built per pass)
    Allo allo_dec, allo_pass;
    struct GKey { int il, kind; int64_t cap, n; int seq; bool operator<(const GKey & o) const {   // (larry: + the conversation)
        return std::tie(il, kind, cap, n, seq) < std::tie(o.il, o.kind, o.cap, o.n, o.seq); } };
    std::map<GKey, ggml_cgraph *> graphs;
    std::map<int, ggml_tensor *> logits_out;   // decode head graph output, by nrows
    std::map<int, ggml_tensor *> predict_out;  // predict graph output, by layer
    std::map<std::pair<int, int>, ggml_tensor *> predict_out_n;   // predict_multi output, by (layer, pass width)
    std::map<int, ggml_tensor *> fpredict_out; // finish_predict graph output, by the predicted layer
    std::map<std::pair<int, int>, std::vector<ggml_tensor *>> fpredict_ahead; // finish_predict_ahead extra outputs

    ~Impl() {
        if (backend) ggml_backend_synchronize(backend);   // async decode graphs may still be queued
        if (rb_ev) ggml_backend_event_free(rb_ev);
        if (up_ev) ggml_backend_event_free(up_ev);
        if (eng_pin_ev) ggml_backend_event_free(eng_pin_ev);
        if (eng_pin_buf) ggml_backend_buffer_free(eng_pin_buf);
        if (nbuf) ggml_backend_buffer_free(nbuf);
        if (nctx) ggml_free(nctx);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (kv_buf) ggml_backend_buffer_free(kv_buf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (sctx) ggml_free(sctx);
        if (ictx) ggml_free(ictx);
        if (gctx) ggml_free(gctx);
        if (own_backend && backend) ggml_backend_free(backend);
    }
    ggml_tensor * W(const std::string & n) const { return w.get(n); }
    ggml_tensor * BL(int64_t il, const std::string & s) const { return w.get("blk." + std::to_string(il) + "." + s); }
    // the router as the prediction graphs read it: the STRATA_PRED_ROUTER copy when the loader made one, else the F32
    ggml_tensor * BLpred(int64_t il) const {
        const auto it = w.t.find("blk." + std::to_string(il) + ".ffn_gate_inp_pred.weight");
        return it != w.t.end() ? it->second : BL(il, "ffn_gate_inp.weight");
    }
    bool compute(ggml_cgraph * gf, bool cached, bool async = false, bool nosync = false);
    double last_alloc_ms = 0;   // larry, patch-logitsdiag: the last compute()'s graph allocation
    std::mutex * sub_mu = nullptr;            // larry: set_submit_lock
    bool shexp_skip = false;                  // larry, STRATA_SHEXP_TIER: set_shexp_skip
    bool fn_host = true;                      // larry: set_fn_host
    std::function<void(int, int)> pre_wait;   // larry: set_pre_wait_hook
    std::function<void(int)> page_need, page_touch;   // larry, engine1: set_page_hooks
    std::vector<std::vector<std::pair<ggml_tensor *, size_t>>> paged_lay;   // larry, engine1: cfg.paged_alloc's units
    std::function<void(int, int, const void *, const void *, const void *, const float *)> route_hook;   // larry: set_route_hook
    std::function<bool(int, int, const ggml_tensor *, const ggml_tensor *, const float *, float *)> oproj_hook;   // larry
    std::atomic<int> * sub_prio = nullptr;    // larry: raised while waiting for sub_mu (the tier's threads hold off)
    ggml_backend_event_t rb_ev = nullptr;   // larry: the router readbacks' event (async decode / verify layers)
    // larry: a decode / verify pass's inputs go up through a pinned staging buffer with async copies (each
    // ggml_backend_tensor_set is a device-wide wait plus a staged pageable copy: ~70 us, a dozen per pass)
    HostRows<uint8_t> up_stage;
    HostRows<uint8_t> span_stage;             // larry: upload_span's index maps (pinned; async copies)
    bool eng_waited[2] = {};                  // larry: engram_wait_host ran for eng_batch's slot
    bool eng_up[2] = {};                      // larry: eng_batch's slot uploaded (the batch goes once both are)
    Allo allo_pass2;                          // larry: set_pass_ring - the span graphs' second allocator
    bool pass_ring = false;
    int pass_turn = 0, pass_k = 0;
    ggml_backend_event_t pass_ev[2] = { nullptr, nullptr };
    bool pass_ev_rec[2] = { false, false };
    size_t up_off = 0;
    bool up_async = false;                    // this pass stages (set by upload_inputs)
    ggml_backend_event_t up_ev = nullptr;     // after the last staged copy: the buffer's reuse waits for it
    void up_set(ggml_tensor * t, const void * data, size_t size);
    bool upload_inputs(const int * tids, int n, int pos0);
};

// ================================================================== weights

static bool load_weights(Ds41Dense::Impl & im, std::string & err) {
    WStore & w = im.w;
    const GgufModel & model = *im.model;
    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, true };
    w.ctx = ggml_init(ip);
    auto make = [&](const TensorInfo & ti, int type = -1) {   // type: the in-memory type (default the file's)
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
        ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) (type >= 0 ? type : (int) ti.type), (int) ti.shape.size(), ne);
        ggml_set_name(t, ti.name.c_str());
        return t;
    };
    // routed experts never reach ggml (PTQ1_0 = 143 is past GGML_TYPE_COUNT); the I32 maps are read by the geometry
    auto skip = [&](const std::string & n) { return is_routed(n) || n == "token_embd.weight"; };
    if (im.cpu) {
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & ts = sh.tensors();
            if (ts.empty()) continue;
            uint8_t * base = const_cast<uint8_t *>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            w.bufs.push_back(buf);
            for (const auto & ti : ts) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = make(ti);
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[ti.name] = t;
            }
        }
    } else {
        // STRATA_DENSE_REQUANT=<ggml type name> (q6_K, q5_K, q4_K, ...; quality experiment, OFF by default): every
        // Q8_0 dense matrix (attention, shared expert; output.weight unless STRATA_DENSE_REQUANT_HEAD=0) is
        // re-quantized in memory at load - rows dequantized, quantized to that type.  The file is not touched.
        ggml_type rq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_DENSE_REQUANT"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) rq = (ggml_type) t;
        const bool rq_head = !(std::getenv("STRATA_DENSE_REQUANT_HEAD") && std::getenv("STRATA_DENSE_REQUANT_HEAD")[0] == '0');
        auto requant = [&](const TensorInfo & ti) {
            return rq != GGML_TYPE_COUNT && (ggml_type) ti.type == GGML_TYPE_Q8_0 && ti.shape.size() == 2 &&
                   (int64_t) ti.shape[0] % ggml_blck_size(rq) == 0 && (rq_head || ti.name != "output.weight");
        };
        // STRATA_PRED_ROUTER=<ggml type> (q8_0, f16; OFF by default): the PREDICTION graphs (the prefetch top-16 and the
        // speculative reads' rows) read a copy of each layer's F32 router in that type (q8_0: 7.9 -> 2.1 MB a read); the
        // routing itself (build_attn) keeps the file's F32.  Only which experts are prefetched can change.
        ggml_type pq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_PRED_ROUTER"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) pq = (ggml_type) t;
        auto pred_name = [&](const TensorInfo & ti) -> std::string {   // "" = no prediction copy for this tensor
            static const std::string suf = "ffn_gate_inp.weight";
            const std::string & n = ti.name;
            if (pq == GGML_TYPE_COUNT || (ggml_type) ti.type != GGML_TYPE_F32 || ti.shape.size() != 2 ||
                (int64_t) ti.shape[0] % ggml_blck_size(pq) != 0 || n.size() <= suf.size() ||
                n.compare(n.size() - suf.size(), suf.size(), suf) != 0) return std::string();
            return n.substr(0, n.size() - suf.size()) + "ffn_gate_inp_pred.weight";
        };
        // larry, STRATA_ROUTER_F16 (default 1): each layer's F32 router (ffn_gate_inp, 7.9 MB read whole by every decode
        // step) stored F16 - its values are bf16, exact in F16 above 6.1e-5 (~0.1% of them are smaller: absolute error
        // <= 3e-8).  The decode mat-vec reads half the bytes; the routing GEMM asks GGML_PREC_F32 and stays on f32 math.
        // STRATA_ENGRAM_REQUANT=<ggml type> (q6_K, q8_0; off unless set): the Engram KV projections (F16 [6144 x 25600],
        // 315 MB, read whole every token on the two Engram layers) re-quantized at load like the dense matrices.
        const bool router_f16 = env_on("STRATA_ROUTER_F16", true);
        ggml_type eq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_ENGRAM_REQUANT"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) eq = (ggml_type) t;
        auto ends_with = [](const std::string & n, const char * suf) {
            const size_t k = std::strlen(suf);
            return n.size() > k && n.compare(n.size() - k, k, suf) == 0;
        };
        size_t cv_router = 0, cv_engram = 0;
        auto mem_type = [&](const TensorInfo & ti) -> int {   // the in-memory type, -1 = the file's
            if (requant(ti)) return (int) rq;
            if (router_f16 && (ggml_type) ti.type == GGML_TYPE_F32 && ti.shape.size() == 2 && ends_with(ti.name, ".ffn_gate_inp.weight")) {
                ++cv_router;
                return GGML_TYPE_F16;
            }
            if (eq != GGML_TYPE_COUNT && eq != (ggml_type) ti.type && (ggml_type) ti.type == GGML_TYPE_F16 && ti.shape.size() == 2 &&
                ends_with(ti.name, ".engram_kv.weight") && (int64_t) ti.shape[0] % ggml_blck_size(eq) == 0) {
                ++cv_engram;
                return (int) eq;
            }
            return -1;
        };
        size_t rq_n = 0, rq_from = 0, rq_to = 0, pq_n = 0;
        // larry, STRATA_QKV_FUSE (default 0): each layer's attn_kv created right after its attn_q_a - the allocator then
        // puts them back to back (one [K, Nq + Nkv] weight over both, below)
        const bool qkv_fuse = im.sycl && env_on("STRATA_QKV_FUSE", false);
        std::map<std::string, const TensorInfo *> kv_info;
        if (qkv_fuse)
            for (size_t s = 0; s < model.size(); ++s)
                for (const auto & ti : model.shard(s).tensors())
                    if (ends_with(ti.name, ".attn_kv.weight")) kv_info[ti.name] = &ti;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                if (qkv_fuse && w.t.count(ti.name)) continue;   // an attn_kv created early
                w.t[ti.name] = make(ti, mem_type(ti));
                if (qkv_fuse && ends_with(ti.name, ".attn_q_a.weight")) {
                    const std::string kn = ti.name.substr(0, ti.name.size() - std::strlen("attn_q_a.weight")) + "attn_kv.weight";
                    auto ik = kv_info.find(kn);
                    if (ik != kv_info.end() && !skip(kn)) w.t[kn] = make(*ik->second, mem_type(*ik->second));
                }
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    ggml_tensor * tp = make(ti, (int) pq);
                    ggml_set_name(tp, pn.c_str());
                    w.t[pn] = tp;
                }
            }
        // larry, engine1 (cfg.paged_alloc): each layer's weights the TP cards hold between them - q_b, wo_a, wo_b, the
        // shared expert's gate | up (back to back: one weight below) and down, the Engram projection - in a unit of the
        // caller's memory, bound before the buffer below takes the rest.  Q6_K 2-D tensors only.
        if (im.sycl && im.cfg.paged_alloc) {
            const ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(im.backend);
            const size_t al = std::max<size_t>(256, ggml_backend_buft_get_alignment(bt));
            auto up_to = [](size_t v, size_t a) { return (v + a - 1) / a * a; };
            const bool gu_fuse = env_on("STRATA_SHEXP_GU", true);
            int nl = 0;
            while (w.t.count("blk." + std::to_string(nl) + ".attn_q_b.weight")) ++nl;
            std::vector<std::vector<std::pair<ggml_tensor *, size_t>>> lay((size_t) nl);
            std::vector<size_t> ub((size_t) nl, 0), uo;
            for (int il = 0; il < nl; ++il) {
                const std::string b = "blk." + std::to_string(il) + ".";
                auto q6 = [&](const char * n) -> ggml_tensor * {
                    auto it = w.t.find(b + n);
                    return it != w.t.end() && it->second->type == GGML_TYPE_Q6_K && ggml_n_dims(it->second) == 2 ? it->second : nullptr;
                };
                size_t off = 0;
                auto put = [&](ggml_tensor * t, bool packed) {   // packed: right after the previous one (gate | up)
                    if (!packed) off = up_to(off, al);
                    lay[(size_t) il].push_back({ t, off });
                    off += ggml_backend_buft_get_alloc_size(bt, t);
                };
                for (const char * n : { "attn_q_b.weight", "attn_output_a.weight", "attn_output_b.weight" })
                    if (ggml_tensor * t = q6(n)) put(t, false);
                ggml_tensor * g = q6("ffn_gate_shexp.weight"), * u = q6("ffn_up_shexp.weight");
                if (gu_fuse && g && u && g->ne[0] == u->ne[0] && g->ne[1] == u->ne[1] && ggml_backend_buft_get_alloc_size(bt, g) == ggml_nbytes(g)) {
                    put(g, false);
                    put(u, true);
                }
                if (ggml_tensor * t = q6("ffn_down_shexp.weight")) put(t, false);
                if (ggml_tensor * t = q6("engram_kv.weight")) put(t, false);
                ub[(size_t) il] = up_to(off, al);
            }
            if (ggml_backend_buffer_t pb = im.cfg.paged_alloc(ub, uo)) {
                uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(pb);
                for (int il = 0; il < nl; ++il)
                    for (auto & e : lay[(size_t) il])
                        if (ggml_backend_tensor_alloc(pb, e.first, base + uo[(size_t) il] + e.second) != GGML_STATUS_SUCCESS) {
                            err = std::string("cannot bind the paged weight ") + ggml_get_name(e.first);
                            ggml_backend_buffer_free(pb);
                            return false;
                        }
                w.bufs.push_back(pb);
                im.paged_lay = std::move(lay);
            }
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!buf) { err = "cannot allocate the dense weights on the backend"; return false; }
        w.bufs.push_back(buf);
        // rows of the file's tensor `ti` (bytes `src`) converted to t->type and uploaded; returns the bytes uploaded
        auto convert_upload = [&](ggml_tensor * t, const TensorInfo & ti, const uint8_t * src) -> size_t {
            const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
            const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
            std::vector<uint8_t> dst((size_t) nr * dst_row);
            const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) n0);
                    for (int64_t r = k; r < nr; r += nth) {
                        if ((ggml_type) ti.type == GGML_TYPE_F32) std::memcpy(f.data(), src + (size_t) r * src_row, (size_t) n0 * 4);
                        else ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                        ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
            return dst.size();
        };
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = w.t[ti.name];
                const uint8_t * src = model.shard(s).tensor_data(ti);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    convert_upload(w.t[pn], ti, src);
                    ++pq_n;
                }
                if ((int) t->type == (int) ti.type) {
                    ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
                    continue;
                }
                rq_to += convert_upload(t, ti, src);
                ++rq_n;
                rq_from += (size_t) ggml_nrows(t) * ggml_row_size((ggml_type) ti.type, t->ne[0]);
            }
        if (rq_n) std::fprintf(stderr, "dense requant: %zu matrices converted (%s for Q8_0; %zu routers F16, %zu Engram KV %s), %.2f -> %.2f GB\n",
                               rq_n, rq != GGML_TYPE_COUNT ? ggml_type_name(rq) : "-", cv_router, cv_engram,
                               eq != GGML_TYPE_COUNT ? ggml_type_name(eq) : "-", rq_from / 1e9, rq_to / 1e9);
        if (pq_n) std::fprintf(stderr, "prediction routers: %zu F32 copies as %s\n", pq_n, ggml_type_name(pq));
        // larry (SYCL): the shared expert's gate and up as ONE [K, 2N] weight over their bytes - the file stores them back
        // to back and the buffer keeps them so - one mat-vec instead of two (STRATA_SHEXP_GU, default 1).  shexp_ffn is
        // their only user: the halves alone must never reach ggml-sycl's reorder (it would reorder the bytes twice).
        if (im.sycl && env_on("STRATA_SHEXP_GU", true)) {
            static const std::string suf = "ffn_gate_shexp.weight";
            std::vector<std::pair<std::string, ggml_tensor *>> add;
            for (const auto & kv : w.t) {
                const std::string & n = kv.first;
                if (n.size() <= suf.size() || n.compare(n.size() - suf.size(), suf.size(), suf) != 0) continue;
                const std::string pre = n.substr(0, n.size() - suf.size());
                auto iu = w.t.find(pre + "ffn_up_shexp.weight");
                if (iu == w.t.end()) continue;
                ggml_tensor * g = kv.second, * u = iu->second;
                if (g->type != u->type || g->ne[0] != u->ne[0] || g->ne[1] != u->ne[1] || g->buffer != u->buffer ||
                    (const char *) u->data != (const char *) g->data + ggml_nbytes(g)) continue;
                ggml_tensor * gu = ggml_new_tensor_2d(w.ctx, g->type, g->ne[0], 2 * g->ne[1]);
                ggml_set_name(gu, (pre + "ffn_gate_up_shexp.weight").c_str());
                if (ggml_backend_tensor_alloc(g->buffer, gu, g->data) != GGML_STATUS_SUCCESS) continue;
                add.emplace_back(gu->name, gu);
            }
            for (auto & a : add) w.t[a.first] = a.second;
            std::fprintf(stderr, "shared expert: %zu gate+up pairs as one weight\n", add.size());
            for (size_t il = 0; il < im.paged_lay.size(); ++il)   // larry, engine1: a paged pair is paged as its fused weight
                for (auto & e : im.paged_lay[il])
                    if (std::strstr(ggml_get_name(e.first), ".ffn_gate_shexp.weight") && !w.t.count("blk." + std::to_string(il) + ".ffn_gate_up_shexp.weight")) {
                        err = "paged shared expert of layer " + std::to_string(il) + " has no fused weight";
                        return false;
                    }
        }
        if (qkv_fuse) {   // STRATA_QKV_FUSE: q_a | kv as one weight over their bytes (same type, rows, buffer, adjacent)
            size_t nf = 0;
            for (int64_t il = 0;; ++il) {
                const std::string pre = "blk." + std::to_string(il) + ".";
                auto iq = w.t.find(pre + "attn_q_a.weight"), ik = w.t.find(pre + "attn_kv.weight");
                if (iq == w.t.end() || ik == w.t.end()) break;
                ggml_tensor * q = iq->second, * k = ik->second;
                if (q->type != k->type || q->ne[0] != k->ne[0] || q->ne[2] != 1 || k->ne[2] != 1 || q->buffer != k->buffer ||
                    (const char *) k->data != (const char *) q->data + ggml_nbytes(q)) continue;
                ggml_tensor * f = ggml_new_tensor_2d(w.ctx, q->type, q->ne[0], q->ne[1] + k->ne[1]);
                ggml_set_name(f, (pre + "attn_qkv_a.weight").c_str());
                if (ggml_backend_tensor_alloc(q->buffer, f, q->data) != GGML_STATUS_SUCCESS) continue;
                w.t[f->name] = f;
                ++nf;
            }
            std::fprintf(stderr, "attention: %zu q_a+kv pairs as one weight (STRATA_QKV_FUSE)\n", nf);
        }
    }
    size_t si = 0;
    const TensorInfo * te = model.find("token_embd.weight", &si);
    im.embd = model.shard(si).tensor_data(*te);
    return true;
}

// ================================================================== graph helpers

namespace {

struct B {
    Ds41Dense::Impl * im;
    ggml_context * c;
    const Ds41Geometry * g;

    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * cols2(ggml_tensor * t, int64_t n0, int64_t n1, int64_t c0 = 0) const {   // columns c0.. of a [n0, *] tensor
        return ggml_view_2d(c, t, n0, n1, t->nb[1], (size_t) c0 * t->nb[1]);
    }
    ggml_tensor * cols3(ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2, int64_t c0 = 0) const {
        return ggml_view_3d(c, t, n0, n1, n2, t->nb[1], t->nb[2], (size_t) c0 * t->nb[2]);
    }
    ggml_tensor * flat2(ggml_tensor * t, int64_t ne0, int64_t ne1) const {
        return ggml_is_contiguous(t) ? ggml_reshape_2d(c, t, ne0, ne1) : ggml_cont_2d(c, t, ne0, ne1);
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wt);
    }
    // hc_mixes (model.py Block.hc_mixes + kernel hc_split_sinkhorn): x [D, HC, n] -> pre [HC, n], post [HC, n],
    // comb [HC(dst), HC(src), n]
    void hc_mixes(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                  ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t HC = g->hc, nt = x->ne[2], D = g->n_embd;
        // larry, STRATA_DEC_FUSE: decode / verify widths take the norm folded into the mat-vec (one kernel, not two)
        static const bool dec_fuse = env_on("STRATA_DEC_FUSE", true);
        // larry, STRATA_HC_NORM_MV_WIDE (patch-normmvw.py, default 0): prompt widths too - ggml-sycl's wide norm_mv reads
        // the state once (the rms_norm copy + GEMM read it twice and wrote it once: 323 -> ~100 us at 512 tokens)
        static const bool nmv_wide = env_on("STRATA_HC_NORM_MV_WIDE", false);
        ggml_tensor * mixes = nullptr;
        if (dec_fuse && im->sycl && (nt <= 8 || nmv_wide) && ggml_is_contiguous(x)) {
            mixes = ggml_dsv41_norm_mv(c, fn, ggml_reshape_2d(c, x, HC * D, nt), (float) g->rms_eps);   // [24, n]
        } else {
            ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, nt), (float) g->rms_eps);
            mixes = ggml_mul_mat(c, fn, flat);                                                 // [24, n]
            ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
        }
#if defined(STRATA_DSV41_HC_MIX)
        // -DSTRATA_DSV41_HC_MIX=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fuse.patch: the two mixes as
        // ONE op (GGML_OP_DSV41_HC_MIXES), the same mul/add/sigmoid/scale steps bit for bit (test_ds41_hcmix).  It
        // replaces 8 of the layer graph's ~82 tiny kernels per call.  STRATA_HC_MIX_COMPOSITE=1: the composite.
        static const bool composite = std::getenv("STRATA_HC_MIX_COMPOSITE") != nullptr;
        if (!composite) {
            ggml_tensor * mix = ggml_dsv41_hc_mixes(c, mixes, v1(sc, 2, 0), v1(bs, 2 * HC, 0), (float) g->hc_eps);
            *pre  = ggml_view_2d(c, mix, HC, nt, mix->nb[1], 0);
            *post = ggml_view_2d(c, mix, HC, nt, mix->nb[1], (size_t) nt * mix->nb[1]);
            *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
            return;
        }
#endif
        ggml_tensor * m_pre = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
        ggml_tensor * m_post = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
        *pre = ggml_add(c, ggml_mul(c, m_pre, v1(sc, 1, 0)), v1(bs, HC, 0));
        *pre = ggml_scale_bias(c, ggml_sigmoid(c, *pre), 1.0f, (float) g->hc_eps);
        *post = ggml_add(c, ggml_mul(c, m_post, v1(sc, 1, sizeof(float))), v1(bs, HC, (size_t) HC * sizeof(float)));
        *post = ggml_scale(c, ggml_sigmoid(c, *post), 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
    }
    // larry, STRATA_HC_FUSE (default 1): decode / verify widths run each sub-layer's hc step as two fused ops
    // (ggml_dsv41_hc_fa + _hc_fb: the composite's hc_post, norm_mv, hc_mixes, hc_comb, hc_pre and rms_norm * w) and
    // the finish as one (ggml_dsv41_hc_post2: add, hc_post and the state write) - 14 tiny kernels a layer -> 5;
    // STRATA_HC_MERGE (default 1): each hc_fa + hc_fb pair as ONE kernel (ggml_dsv41_hc_merge) - 3
    ggml_tensor * hc_merge(ggml_tensor * fb) const {
        static const bool on = env_on("STRATA_HC_MERGE", true);
        // (larry, patch-hcwide.py: the merged kernel takes <= 64 tokens - hc_fb's op_params[2] is its n)
        return on && g->n_embd % 512 == 0 && fb->op_params[2] <= 64 ? ggml_dsv41_hc_merge(fb) : fb;
    }
    bool hc_fused(int64_t nt) const {
        static const bool on = env_on("STRATA_HC_FUSE", true);
        // STRATA_HC_FUSE_MAXN (default 8 = decode / verify): wider is slower - at 512-token spans the per-token mixes
        // dots lose to the composite GEMM (31.6K prompt 26.2 -> 27.7 s)
        static const int64_t maxn = std::getenv("STRATA_HC_FUSE_MAXN") ? std::atoll(std::getenv("STRATA_HC_FUSE_MAXN")) : 8;
        return on && im->sycl && nt <= maxn;
    }
    // the views of a ggml_dsv41_hc_fb result: xn [D, n] | pre [HC, n] | post [HC, n] | comb [HC, HC, n]
    void hc_fb_views(ggml_tensor * fb, int64_t nt, ggml_tensor ** xn, ggml_tensor ** pre, ggml_tensor ** post,
                     ggml_tensor ** comb) const {
        const int64_t D = g->n_embd, HC = g->hc;
        size_t o = 0;
        *xn = ggml_view_2d(c, fb, D, nt, (size_t) D * 4, o);
        o += (size_t) (D * nt) * 4;
        *pre = ggml_view_2d(c, fb, HC, nt, (size_t) HC * 4, o);
        o += (size_t) (HC * nt) * 4;
        *post = ggml_view_2d(c, fb, HC, nt, (size_t) HC * 4, o);
        o += (size_t) (HC * nt) * 4;
        *comb = ggml_view_3d(c, fb, HC, HC, nt, (size_t) HC * 4, (size_t) (HC * HC) * 4, o);
    }
    // RoPE over the last `nd` dims of each `ne0` row (model.py apply_rotary_emb: adjacent pairs = NORMAL mode).
    // ratio > 0: compress_rope_theta with YaRN; ratio 0: rope_theta, no YaRN (model.py Attention.__init__)
    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t ratio, bool back = false, bool inplace = false) const {
        const bool yarn = ratio != 0;
        const float base = (float) (yarn ? g->compress_rope_base : g->rope_base);
        const float fscale = yarn ? (float) (1.0 / g->rope_factor) : 1.0f;
        const float ext = yarn ? 1.0f : 0.0f;
        // ggml multiplies by mscale = attn_factor * (1 + 0.1 ln(1/fscale)) under YaRN; model.py has no such scale
        const float attn = yarn ? 1.0f / (1.0f + 0.1f * std::log(1.0f / fscale)) : 1.0f;
        const float bf = yarn ? (float) g->beta_fast : 0.0f, bsl = yarn ? (float) g->beta_slow : 0.0f;
        const int n_ctx = yarn ? (int) g->orig_ctx : 0;
        const int nd = (int) g->d_rope;
        ggml_tensor * r = nullptr;
        if (inplace) {   // larry, STRATA_ROPE_INPLACE: a view of a (ggml_rope_ext_back is ggml_rope_ext with the op swapped)
            r = ggml_rope_ext_inplace(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);
            if (back) r->op = GGML_OP_ROPE_BACK;
        } else {
            r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl)
                     : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);
        }
        return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
    }
    // the same RoPE as the fused decode ops take it (ggml_dsv41_kv_fin / _attn_dec)
    ggml_dsv41_rope rope_p(int64_t ratio) const {
        const bool yarn = ratio != 0;
        ggml_dsv41_rope p;
        p.n_dims = (int) g->d_rope;
        p.n_ctx_orig = yarn ? (int) g->orig_ctx : 0;
        p.freq_base = (float) (yarn ? g->compress_rope_base : g->rope_base);
        p.freq_scale = yarn ? (float) (1.0 / g->rope_factor) : 1.0f;
        p.ext_factor = yarn ? 1.0f : 0.0f;
        p.attn_factor = yarn ? 1.0f / (1.0f + 0.1f * std::log(1.0f / p.freq_scale)) : 1.0f;
        p.beta_fast = yarn ? (float) g->beta_fast : 0.0f;
        p.beta_slow = yarn ? (float) g->beta_slow : 0.0f;
        return p;
    }
};

}  // namespace

// STRATA_DENSE_PROF=1 (diagnostic, off by default): where a layer-major batch's dense time goes - graph build, gallocr
// allocation, GPU compute, Engram wait, readback - summed per prompt and printed by prof_report()
static const bool g_dprof_on = std::getenv("STRATA_DENSE_PROF") != nullptr;
static double g_dprof[8] = {};   // build, alloc, compute, engram, readback, finish build, upload, calls
static double g_dlock = 0, g_dfin = 0;   // larry: submit-lock waits (pipelined batch), finish_span wall
static double dprof_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int g_dprof_slot = -1;    // which of build/alloc/compute buckets compute() charges: set by the span callers

// STRATA_DEC_PROF=1 (larry, diagnostic): host time per decode layer in attn_router / finish - the attention graph's
// submission, the readbacks' wait (the card finishing everything queued), the shared expert's submission, finish's
static const bool g_decprof_on = std::getenv("STRATA_DEC_PROF") != nullptr;
static double g_decprof[6] = {};   // attn submit, get enqueue, sync wait, shexp submit, finish submit, calls
static void decprof_report() {
    const double n = g_decprof[5];
    std::fprintf(stderr, "dec prof (%.0f layer calls): attn submit %.1f us, get enqueue %.1f us, sync wait %.1f us, "
                 "shexp submit %.1f us, finish submit %.1f us\n", n, 1e3 * g_decprof[0] / n, 1e3 * g_decprof[1] / n,
                 1e3 * g_decprof[2] / n, 1e3 * g_decprof[3] / n, 1e3 * g_decprof[4] / n);
}

// larry: async = the kept decode / verify graphs (STRATA_DENSE_ASYNC, default on) are queued without ggml's
// end-of-graph synchronize - the card runs attention, shared expert, routed sum and finish back to back while the host
// queues ahead; it waits only for the router readbacks.  A graph whose allocation is not the kept fast path syncs
// first (gallocr may free a buffer that queued kernels still use).
void (*ds41_dense_tl_hook)(int id) = nullptr;
void (*ds41_dense_ptl_hook)(int tag, int il, int col0, double t0) = nullptr;   // larry, STRATA_GPU_PTL

static bool dense_async_on() { static const bool on = env_on("STRATA_DENSE_ASYNC", true); return on; }

bool Ds41Dense::Impl::compute(ggml_cgraph * gf, bool cached, bool async, bool nosync) {
    Allo & a = cached ? allo_dec : pass_ring && pass_k == 1 ? allo_pass2 : allo_pass;
    const double tq0 = g_dprof_on ? dprof_now() : 0;
    const bool reuse = cached && env_on("DS41_GRAPH_REUSE", true);
    async = async && cached && dense_async_on();
    if (async && !a.kept(gf, reuse)) ggml_backend_synchronize(backend);
    const double tqa = dprof_now();   // larry, patch-logitsdiag
    bool ok_alloc = a.alloc(gf, reuse);
    last_alloc_ms = dprof_now() - tqa;
    const double tq1 = g_dprof_on ? dprof_now() : 0;
    if (!ok_alloc && oom_hook && oom_hook()) {   // something gave VRAM back: once more
        std::fprintf(stderr, "ds41_dense: graph buffer did not fit - freed the tier's prestage region, retrying\n");
        ok_alloc = a.alloc(gf, false);
    }
    if (!ok_alloc) { err = "graph allocation failed"; return false; }
    const ggml_status st = async || nosync ? ggml_backend_graph_compute_async(backend, gf) : ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
    if (g_dprof_on && g_dprof_slot >= 0) { g_dprof[1] += tq1 - tq0; g_dprof[2] += dprof_now() - tq1; }
    return true;
}

// ================================================================== the attention graph (+ router + shared expert)

// the model's activation rounding: ds41_ops.hpp's stock-op composite (~45 ops per call), or - built with
// -DSTRATA_DSV41_FQ_OP=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fake-quant.patch - the fused
// GGML_OP_DSV41_FAKE_QUANT, one kernel per call (bit-identical on CUDA and CPU: test_ds41_fq; ~3,000 fewer tiny
// kernels per decode token).  STRATA_FQ_COMPOSITE=1 in such a build: the composite.
// larry, STRATA_CONT_SKIP (default 1): a ggml_cont only where the tensor is not contiguous already (ggml_cont is
// always an op - a copy - even of a contiguous tensor)
static ggml_tensor * cont_if(ggml_context * gc, ggml_tensor * x) {
    static const bool skip = env_on("STRATA_CONT_SKIP", true);
    return skip && ggml_is_contiguous(x) ? x : ggml_cont(gc, x);
}

static ggml_tensor * fake_quant(ggml_context * gc, ggml_tensor * x, Ds41Quant mode) {
#if defined(STRATA_DSV41_FQ_OP)
    static const bool composite = std::getenv("STRATA_FQ_COMPOSITE") != nullptr;
    return composite ? ds41_fake_quant(gc, x, mode) : ggml_dsv41_fake_quant(gc, x, (int) mode);
#else
    return ds41_fake_quant(gc, x, mode);
#endif
}

// larry, STRATA_ATTN_FUSE (default 1): at the ggml-sycl DSV4 decode attention's widths (< 8 queries) the window row
// is ggml_dsv41_kv_fin (rms_norm * w, RoPE, the FP8 rounding, its ring write) and the attention ggml_dsv41_attn_dec
// (q RoPE, the window / compressed key gather, the masks, sinks, o de-RoPE) - 11 kernels a layer -> 3
static bool attn_fused(const Ds41Dense::Impl & im, int il, int64_t n) {
    static const bool on = env_on("STRATA_ATTN_FUSE", true) && env_on("STRATA_KGATHER", true) && env_on("STRATA_DEC_FUSE", true);
    static const bool dec_off = [] { const char * v = std::getenv("GGML_SYCL_DSV4_DEC"); return v && v[0] == '0'; }();
    return on && !dec_off && im.sycl && n < 8 && im.DH == 512 && im.NH % 32 == 0 && im.ly[(size_t) il].raw->type == GGML_TYPE_F32;
}
// larry, STRATA_Q8_FUSE (default 1; with the fused attention and hc ops): the ops in front of the decode layer's Q6_K
// mat-vecs write their result's int8 form too (hc_fb: q_a|kv's and the shared expert's input; rms_q8: q_b's; the
// attention: wo_a's; swiglu_q8: the shared expert's down) and the products read it (ggml_dsv41_mvq / _oproj_q8) - the
// separate activation quantisation (quant_x, 2-4 us + a launch each) goes, and wo_a writes wo_b's layout (no cont)
static bool q8_fused(const Ds41Dense::Impl & im, int il, int64_t n) {
    static const bool on = env_on("STRATA_Q8_FUSE", true) && env_on("STRATA_HC_FUSE", true);
    static const int64_t hc_maxn = std::getenv("STRATA_HC_FUSE_MAXN") ? std::atoll(std::getenv("STRATA_HC_FUSE_MAXN")) : 8;
    return on && n <= hc_maxn && attn_fused(im, il, n);
}

// the shared expert (model.py Expert, clamped SwiGLU) on the routed experts' input.  larry: fn8 = fn's q8 (STRATA_Q8_FUSE)
// out (with fn8): the down product writes straight into it (the returned node is a view of out, no copy needed)
static ggml_tensor * shexp_ffn(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor * fn, ggml_tensor * fn8 = nullptr,
                               ggml_tensor * out = nullptr) {
    const float lim = (float) im.g.swiglu_limit;
    ggml_tensor * up = nullptr, * gt = nullptr;
    auto igu = im.w.t.find("blk." + std::to_string(il) + ".ffn_gate_up_shexp.weight");
    if (igu != im.w.t.end() && fn8) {   // larry: q8 in, q8 between (swiglu_q8), q8 into down
        ggml_tensor * y = ggml_dsv41_mvq(gc, igu->second, fn, fn8);                            // [2N, n]
        const int64_t N = igu->second->ne[1] / 2, n = fn->ne[1];
        ggml_tensor * sw = ggml_dsv41_swiglu_q8(gc, y, lim);
        ggml_tensor * h = ggml_view_2d(gc, sw, N, n, (size_t) N * 4, 0);
        ggml_tensor * h8 = ggml_view_1d(gc, sw, ggml_dsv41_q8_floats(N, n), (size_t) (N * n) * 4);
        return out ? ggml_dsv41_mvq_into(gc, im.BL(il, "ffn_down_shexp.weight"), h, h8, out)
                   : ggml_dsv41_mvq(gc, im.BL(il, "ffn_down_shexp.weight"), h, h8);
    }
    if (igu != im.w.t.end()) {   // larry: one mat-vec for both (rows: gate, then up)
        ggml_tensor * y = ggml_mul_mat(gc, igu->second, fn);                                   // [2N, n]
        const int64_t N = igu->second->ne[1] / 2, n = fn->ne[1];
        static const bool dec_fuse = env_on("STRATA_DEC_FUSE", true);
        if (dec_fuse && im.sycl && n <= 8)   // larry: clamp / clamp / swiglu (and the conts) as one kernel
            return ggml_mul_mat(gc, im.BL(il, "ffn_down_shexp.weight"), ggml_dsv41_swiglu_lim(gc, y, lim));
        gt = ggml_view_2d(gc, y, N, n, y->nb[1], 0);
        up = ggml_view_2d(gc, y, N, n, y->nb[1], (size_t) N * ggml_element_size(y));
        if (n > 1) { gt = ggml_cont(gc, gt); up = ggml_cont(gc, up); }
    } else {
        up = ggml_mul_mat(gc, im.BL(il, "ffn_up_shexp.weight"), fn);
        gt = ggml_mul_mat(gc, im.BL(il, "ffn_gate_shexp.weight"), fn);
    }
    if (lim > 0) {
        up = ggml_clamp(gc, up, -lim, lim);
        gt = ggml_clamp(gc, gt, NEG_INF, lim);
    }
    return ggml_mul_mat(gc, im.BL(il, "ffn_down_shexp.weight"), ggml_swiglu_split(gc, gt, up));
}

// with_shexp = false (larry, decode): the shared expert is left out; attn_router runs it as its own graph (build_shexp)
// after the router readback, so the dense card computes it while the expert cards compute the routed experts
// post_in (larry, STRATA_FIN_DEFER): the previous layer's finish (x_state = hc_post(shexp + routed, hap, post_f, comb_f))
// is this graph's first step - inside the attention hc kernel (ggml_dsv41_hc_fa2), which also writes x_state
static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0,
                                bool with_shexp = true, int64_t max_nodes = 8192, bool dec = false, int part = 0,
                                bool post_in = false) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, max_nodes, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH, NH = im.NH, SWA = im.SWA, RAW = im.RAW;
    const int64_t r = L.ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    std::vector<ggml_tensor *> outs, late;    // late: ring / state write-backs, after every read
    // larry, STRATA_TP_OPROJ=2 (decode widths, fused hc): part 1 ends with the de-RoPEd attention output and the attention
    // hc mixes in o_att / o_mix (every ring / state write-back with it: no read of them follows); the projection runs
    // between the two graphs; part 2 starts from o_proj.  Part 0: the whole layer.
    ggml_tensor * x = nullptr, * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr, * attn = nullptr, * fb_a = nullptr;
    const bool hcf = b.hc_fused(n);
    const bool q8f = hcf && q8_fused(im, il, n);
    GGML_ASSERT(part == 0 || (hcf && col0 == 0 && n <= im.o_att->ne[1]));
    // larry (vision): image positions among this graph's columns - their VL routing bias and no Engram for them
    ggml_tensor * mimg = !im.img.empty() && im.img_in((int64_t) im.pos0 + col0, n) ? b.v1(im.i_img, n, (size_t) col0 * 4) : nullptr;
    if (part == 2) {
        x = b.cols3(im.x_state, D, HC, n, col0);
        ggml_tensor * mx = ggml_view_1d(gc, im.o_mix, (2 * HC + HC * HC) * n, 0);
        a_pre = ggml_view_2d(gc, mx, HC, n, (size_t) HC * 4, 0);
        a_post = ggml_view_2d(gc, mx, HC, n, (size_t) HC * 4, (size_t) (HC * n) * 4);
        a_comb = ggml_view_3d(gc, mx, HC, HC, n, (size_t) HC * 4, (size_t) (HC * HC) * 4, (size_t) (2 * HC * n) * 4);
        attn = b.cols2(im.o_proj, D, n, 0);
    } else {
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    // ---- Engram (model.py Engram.forward), before the block, on the hc state
    x = b.cols3(im.x_state, D, HC, n, col0);
    GGML_ASSERT(!post_in || (hcf && L.eslot < 0));
    if (L.eslot >= 0 && im.engram.enabled()) {
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);          // [(HC+1) D, n]
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = ggml_mul(gc, im.BL(il, "engram_q_norm.weight"), im.BL(il, "engram_k_norm.weight"));   // [D, HC]
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn));    // [1, HC, n]
        dot = ggml_scale(gc, dot, 1.0f / std::sqrt((float) D));
        // gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))  (signed sqrt, as the training kernel)
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (mimg) gate = ggml_mul(gc, gate, ggml_reshape_3d(gc, ggml_scale_bias(gc, mimg, -1.0f, 1.0f), 1, 1, n));   // (larry, vision)
        ggml_tensor * gv = ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate);
        x = ggml_add(gc, x, gv);
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));   // the block input (residual) = state + Engram
    }

    // ---- attention hc: mixes of the block input; its pre-mix is the previous FFN's (pre_state)
    ggml_tensor * xn = nullptr, * xn8 = nullptr;
    if (hcf) {
        ggml_tensor * fa = nullptr;
        if (post_in) {   // h = hc_post(shexp + routed, hap, post_f, comb_f) -> x_state, and the block input
            fa = ggml_dsv41_hc_fa2(gc, b.cols3(im.hap, D, HC, n, col0), im.BL(il, "hc_attn_fn.weight"), b.cols2(im.pre_state, HC, n, col0),
                                   b.cols2(im.shexp, D, n, col0), b.cols2(im.routed, D, n, col0), b.cols2(im.post_f, HC, n, col0),
                                   b.cols3(im.comb_f, HC, HC, n, col0));
            ggml_dsv41_write_to(fa, x, nullptr, nullptr, nullptr, nullptr);
            x = ggml_view_3d(gc, fa, D, HC, n, (size_t) D * 4, (size_t) (HC * D) * 4, 0);
        } else {
            fa = ggml_dsv41_hc_fa(gc, x, im.BL(il, "hc_attn_fn.weight"), b.cols2(im.pre_state, HC, n, col0), nullptr, nullptr, nullptr);
        }
        ggml_tensor * fb = b.hc_merge((q8f ? ggml_dsv41_hc_fb_q8 : ggml_dsv41_hc_fb)(gc, fa, im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
                                            im.BL(il, "attn_norm.weight"), (float) G.rms_eps, (float) G.hc_eps, (int) G.hc_iters));
        fb_a = fb;
        b.hc_fb_views(fb, n, &xn, &a_pre, &a_post, &a_comb);
        if (q8f) xn8 = ggml_view_1d(gc, fb, ggml_dsv41_q8_floats(D, n), (size_t) (n * (D + 2 * HC + HC * HC)) * 4);
    } else {
        b.hc_mixes(x, im.BL(il, "hc_attn_fn.weight"), im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
                   &a_pre, &a_post, &a_comb);
        ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));         // [D, n]
        xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));
    }

    const bool afuse = attn_fused(im, il, n);

    // ---- q (no per-head norm in V4.1).  larry, STRATA_QKV_FUSE: q_a and kv as one product (rows: q_a, then kv)
    ggml_tensor * qa_y = nullptr, * kv_y = nullptr;
    if (im.w.has("blk." + std::to_string(il) + ".attn_qkv_a.weight")) {
        ggml_tensor * wf = im.BL(il, "attn_qkv_a.weight");
        const int64_t nq = im.BL(il, "attn_q_a.weight")->ne[1], nk = wf->ne[1] - nq;
        ggml_tensor * y = xn8 ? ggml_dsv41_mvq(gc, wf, xn, xn8) : ggml_mul_mat(gc, wf, xn);     // [nq + nk, n]
        qa_y = ggml_view_2d(gc, y, nq, n, y->nb[1], 0);
        kv_y = ggml_view_2d(gc, y, nk, n, y->nb[1], (size_t) nq * ggml_element_size(y));
    } else if (xn8) {
        qa_y = ggml_dsv41_mvq(gc, im.BL(il, "attn_q_a.weight"), xn, xn8);
        kv_y = ggml_dsv41_mvq(gc, im.BL(il, "attn_kv.weight"), xn, xn8);
    } else {
        qa_y = ggml_mul_mat(gc, im.BL(il, "attn_q_a.weight"), xn);
        kv_y = ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn);
    }
    ggml_tensor * qr = nullptr, * q = nullptr;
    if (q8f) {   // larry: q_a's norm with its q8 for q_b (rms_norm * w + quant_x -> one kernel)
        const int64_t RQ = qa_y->ne[0];
        ggml_tensor * rq = ggml_dsv41_rms_q8(gc, qa_y, im.BL(il, "attn_q_a_norm.weight"), (float) G.rms_eps);
        qr = ggml_view_2d(gc, rq, RQ, n, (size_t) RQ * 4, 0);
        ggml_tensor * qr8 = ggml_view_1d(gc, rq, ggml_dsv41_q8_floats(RQ, n), (size_t) (RQ * n) * 4);
        q = ggml_reshape_3d(gc, ggml_dsv41_mvq(gc, im.BL(il, "attn_q_b.weight"), qr, qr8), DH, NH, n);
    } else {
        qr = b.rms_w(qa_y, im.BL(il, "attn_q_a_norm.weight"));
        q = ggml_reshape_3d(gc, ggml_mul_mat(gc, im.BL(il, "attn_q_b.weight"), qr), DH, NH, n);
    }
    // larry, STRATA_ROPE_INPLACE (default 0): in place - ggml-sycl then rotates only the 64 rope dims (q is read rotated only)
    static const bool rope_ip = env_on("STRATA_ROPE_INPLACE", false);
    if (!afuse) q = b.rope(q, pos, r, false, rope_ip && im.sycl);

    // ---- window K (= V): [ring | this pass's rows] -> each query's SWA rows; the ring keeps the last RAW
    ggml_tensor * widx_v = b.v1(im.i_widx, SWA * n, (size_t) col0 * SWA * 4);
    ggml_tensor * Mw = b.cols2(im.i_wmask, SWA, n, col0);
    ggml_tensor * wsrc = nullptr, * Kw = nullptr, * kvn = nullptr, * rtab = nullptr;
    const int ND = (int) G.d_rope;
    if (afuse) {
        // the pass's rows also go straight into the ring: its slots hold positions RAW - SWA (16) older than any window;
        // and the pass's RoPE table (the attention's q / o and the indexer's q rotate by it)
        ggml_tensor * kf = ggml_dsv41_kv_fin(gc, kv_y, im.BL(il, "attn_kv_a_norm.weight"), pos, L.raw, b.v1(im.i_wslot, std::min<int64_t>(n, RAW), 0),
                                             (float) G.rms_eps, b.rope_p(r), im.cfg.act_quant);
        kvn = ggml_view_2d(gc, kf, DH, n, (size_t) DH * 4, 0);
        rtab = ggml_view_1d(gc, kf, (int64_t) ND * n, (size_t) (DH * n) * 4);
    } else {
        ggml_tensor * kv = b.rms_w(kv_y, im.BL(il, "attn_kv_a_norm.weight"));
        kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
        ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
        if (im.cfg.act_quant) kv2 = fake_quant(gc, cont_if(gc, kv2), Ds41Quant::FP8_POW2_32);   // model.py _window_kv
        wsrc = ggml_concat(gc, L.raw->type == GGML_TYPE_F32 ? L.raw : ggml_cast(gc, L.raw, GGML_TYPE_F32), kv2, 1);     // [DH, RAW + n]
        Kw = ggml_reshape_3d(gc, ggml_get_rows(gc, wsrc, widx_v), DH, SWA, n);
        const int64_t k = std::min<int64_t>(n, RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV (kv sources write it; every ratio > 0 layer reads its source's)
    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    ggml_tensor * kg_comp = nullptr, * kg_sel = nullptr, * kg_selm = nullptr;   // Kc's source rows, their ids and mask
    ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;   // this pass's writes (later reads depend on them)
    if (r) {
        if (L.kv_src) {
            ggml_tensor * latent = nullptr;                                                  // [DH, n]
            if (r == 1) {   // model.py Compressor.forward: ratio 1 = a plain projection, no gate
                latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            } else {        // ratio 2: softmax-gated pooling over the group's 2 states, per dim (fp32)
                ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
                ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
                ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
                ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);                     // [DH, 2 + n]
                ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
                ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
                ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
                ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
                vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));                       // [2, DH, n]
                vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
                latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
                const int64_t k = std::min<int64_t>(n, 2);
                late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
                late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
            }
            latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
            ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
            ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
            // indexer keys from the UNROTATED latent (model.py Indexer.forward, owns_k)
            ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
            ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
            if (im.cfg.act_quant) ik = fake_quant(gc, cont_if(gc, ik), Ds41Quant::FP4_POW2_32);
            idxk_w = ggml_set_rows(gc, L.idxk, ik, cslot);
            outs.push_back(idxk_w);
            ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
            if (im.cfg.act_quant) lr = fake_quant(gc, cont_if(gc, lr), Ds41Quant::FP4_E4M3_16);
            comp_w = ggml_set_rows(gc, L.comp, lr, cslot);
            outs.push_back(comp_w);
        }
        if (cap > 0) {
            const L_t & KS = im.ly[(size_t) L.kvl];
            const int64_t K = std::min<int64_t>(cap, im.ITOP);
            ggml_tensor * comp = comp_w ? comp_w : KS.comp;
            ggml_tensor * idxk = idxk_w ? idxk_w : KS.idxk;
            ggml_tensor * sel = nullptr, * selm = nullptr;
            // larry, STRATA_LI_FUSE (default 1; with the fused attention): the indexer from its two raw products in one op
            // (q RoPE + FP4 rounding, the head-weight scale, the visibility / candidate masks and the candidate source's
            // block maxima in-kernel: 2 kernels, was ~14) and its top-k canonicalised straight into L.sel / L.selm
            static const bool li_fuse = env_on("STRATA_LI_FUSE", true);
            const bool lfuse = li_fuse && afuse && L.idx_src && im.CBS == 8 && idxk->type == GGML_TYPE_BF16 && im.ID == 128 &&
                               im.IH == 32 && cap % 8 == 0 && K <= 512;
            if (lfuse) {
                ggml_tensor * iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                ggml_tensor * iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                ggml_tensor * keys = ggml_view_2d(gc, idxk, im.ID, cap, idxk->nb[1], 0);
                ggml_tensor * nvs = b.v1(im.i_nvis[r], n, (size_t) col0 * 4);
                const int64_t nblk = cap / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                ggml_tensor * cand_v = ggml_view_2d(gc, im.cand, ncb, n, im.cand->nb[1], (size_t) col0 * im.cand->nb[1]);
                ggml_tensor * li = ggml_dsv41_li_dec(gc, ggml_reshape_3d(gc, iq, im.ID, im.IH, n), keys, iw, nvs, rtab,
                                                     L.uses_cand ? cand_v : nullptr, L.cand_src ? b.v1(im.i_lastblk, n, (size_t) col0 * 4) : nullptr,
                                                     1.0f / std::sqrt((float) (im.ID * im.IH)), ND, im.cfg.act_quant);
                ggml_tensor * sc = L.cand_src ? ggml_view_2d(gc, li, cap, n, (size_t) cap * 4, 0) : li;
                if (L.cand_src) {   // model.py select_candidate_blocks on the op's block maxima (newest block pinned)
                    ggml_tensor * bm = ggml_view_2d(gc, li, nblk, n, (size_t) nblk * 4, (size_t) (cap * n) * 4);
                    outs.push_back(ggml_cpy(gc, ggml_top_k(gc, bm, (int) ncb), cand_v));
                }
                selm = ggml_view_2d(gc, L.selm, K, n, L.selm->nb[1], (size_t) col0 * L.selm->nb[1]);
                sel = ggml_dsv41_sel_fin(gc, ggml_top_k(gc, sc, (int) K), nvs,
                                         ggml_view_2d(gc, L.sel, K, n, L.sel->nb[1], (size_t) col0 * L.sel->nb[1]), selm);
            } else if (L.idx_src) {
                // ---- indexer (model.py Indexer.forward): scores over the visible rows of the shared index keys
                ggml_tensor * iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                iq = b.rope(ggml_reshape_3d(gc, iq, im.ID, im.IH, n), pos, r);
                if (im.cfg.act_quant) iq = fake_quant(gc, cont_if(gc, iq), Ds41Quant::FP4_POW2_32);
                ggml_tensor * iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                iw = ggml_scale(gc, iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));         // softmax_scale * heads^-0.5
                ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, cap, idxk->nb[1], idxk->nb[1], 0);
                ggml_tensor * ramp = ggml_view_2d(gc, im.neg_ramp, cap, 1, (size_t) cap * 4, 0);
                // larry: the [cap, n] tensors below in column slices of at most STRATA_LI_CHUNK_MIB (512) MiB an F32 one -
                // at a 1M-token context a 512-column span took ~9 GiB on card 0 and a 984K prompt's last batch failed.
                // Each query column ranks on its own: the same ids.  A slice's outputs are pushed before the next slice
                // is built (STRATA_CPY_LAST expands their sources in that order, so its temporaries are freed first).
                static const int64_t li_chunk = (std::getenv("STRATA_LI_CHUNK_MIB") ? std::atoll(std::getenv("STRATA_LI_CHUNK_MIB")) : 512) << 20;
                int64_t ncol = n;
                if (li_chunk > 0 && cap * n * 4 > li_chunk) ncol = std::min<int64_t>(n, std::max<int64_t>(16, (li_chunk / (cap * 4)) & ~(int64_t) 15));
                // larry, STRATA_LI_TOPK (patch-litopk.py, default 0; with STRATA_LI_NVIS): a source that ranks every visible row
                // (not the candidate layer, not the candidates' users) takes its ids from ggml_lightning_indexer_topk - the
                // scores and their top-k in one op that never holds [cap, n] scores in the graph (and can hand a share of its
                // queries to the expert cards: ggml_backend_sycl_li_remote) - one call a span
                static const bool li_topk_on = env_on("STRATA_LI_TOPK", false) && env_on("STRATA_LI_NVIS", false);
                const bool topk_op = li_topk_on && im.sycl && !L.uses_cand && !L.cand_src && idxk->type == GGML_TYPE_BF16 &&
                                     idxk->nb[1] == 256 && cap % 32 == 0 && im.ID == 128 && im.IH == 32;
                if (topk_op) ncol = n;
                std::vector<ggml_tensor *> sel_parts, selm_parts;
                for (int64_t q0 = 0; q0 < n; q0 += ncol) {
                    const int64_t nq = std::min<int64_t>(ncol, n - q0);
                    const int64_t qc0 = col0 + q0;   // the slice's first column of the state
                    ggml_tensor * iqs = nq == n ? iq : ggml_view_3d(gc, iq, im.ID, im.IH, nq, iq->nb[1], iq->nb[2], (size_t) q0 * iq->nb[2]);
                    ggml_tensor * iws = nq == n ? iw : ggml_view_2d(gc, iw, iw->ne[0], nq, iw->nb[1], (size_t) q0 * iw->nb[1]);
                    // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                    ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], nq, (size_t) qc0 * 4), 1, nq);
                    // larry, STRATA_LI_NVIS (patch-linvis.py, default 0): ggml-sycl's indexer takes nvis itself - no [cap, nq]
                    // mask (repeat + add + step + log + F16 copy a call), the selected rows' mask from their ids below
                    static const bool li_nvis = env_on("STRATA_LI_NVIS", false);
                    ggml_tensor * vis = nullptr, * sc = nullptr;
                    if (topk_op) {   // (patch-litopk.py) the ids straight from the scores
                        // (the rows its last query sees are written by now: (position + 1) / ratio of it)
                        sel = ggml_lightning_indexer_topk(gc, iqs, ksrc, iws, nvs, (int) K, (int) (((int64_t) im.pos0 + qc0 + nq) / r));
                    } else {
                    if (li_nvis && im.sycl) {
                        sc = ggml_lightning_indexer_nvis(gc, iqs, ksrc, iws, nvs);   // [cap, nq], -inf past nvis
                    } else {
                        vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, nq)), nvs)));
                        sc = ggml_lightning_indexer(gc, iqs, ksrc, iws, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, nq]
                    }
                    sc = ggml_reshape_2d(gc, sc, cap, nq);
                    if (L.uses_cand) {   // rank only inside cand_layer's candidate blocks
                        const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                        ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, nq), NEG_INF);
                        ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, nq), 0.0f);
                        m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], (size_t) qc0 * im.cand->nb[1])));
                        m = ggml_view_2d(gc, m, cap, nq, m->nb[2], 0);
                        sc = ggml_add(gc, sc, m);
                    }
                    if (L.cand_src) {   // model.py select_candidate_blocks: block score = max, newest block pinned
                        const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                        GGML_ASSERT(cap % im.CBS == 0);
                        ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);   // [nblk, nq]
                        // pin: + 1e30 at block (nvis - 1) / CBS of each query
                        ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                        ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, nq, (size_t) qc0 * 4), 1, nq);
                        ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                        bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                        ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));                     // I32 [ncb, nq]
                        outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], (size_t) qc0 * im.cand->nb[1])));
                    }
                    sel = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                                     // I32 [K, nq]
                    }   // (patch-litopk.py)
                // Canonicalise the top-k order.  ggml_cuda_op_top_k asks CUB's DeviceTopK for
                // output_ordering::unsorted, and CUB hands the k ids back in an order its atomically assigned
                // output slots produce - run-to-run unstable, while the SET is stable (measured: order-only on
                // every differing row).  The ids drive the gather of the compressed rows, so that order is the
                // flash-attn's accumulation order; left alone the whole run stops being reproducible from the
                // first query whose compressed rows are attended (the SWA window's 128).  Sort ascending.
                    sel = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sel, 1, K, nq),
                                                            ggml_argsort(gc, ggml_cast(gc, sel, GGML_TYPE_F32),
                                                                         GGML_SORT_ORDER_ASC)), K, nq);
                    if (vis) {
                        selm = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, nq), sel), K, nq);   // vis at sel
                    } else {   // 0 where sel < nvis, else -inf: log(step(nvis - sel)) on [K, nq]
                        selm = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_scale(gc, ggml_cast(gc, sel, GGML_TYPE_F32), -1.0f), nvs)));
                    }
                    outs.push_back(ggml_cpy(gc, sel, ggml_view_2d(gc, L.sel, K, nq, L.sel->nb[1], (size_t) qc0 * L.sel->nb[1])));
                    outs.push_back(ggml_cpy(gc, selm, ggml_view_2d(gc, L.selm, K, nq, L.selm->nb[1], (size_t) qc0 * L.selm->nb[1])));
                    sel_parts.push_back(sel);
                    selm_parts.push_back(selm);
                }
                sel = sel_parts[0];
                selm = selm_parts[0];
                for (size_t i = 1; i < sel_parts.size(); ++i) {   // the span's ids / masks for its attention
                    sel = ggml_concat(gc, sel, sel_parts[i], 1);
                    selm = ggml_concat(gc, selm, selm_parts[i], 1);
                }
            } else {
                const L_t & IS = im.ly[(size_t) L.idxl];
                sel = ggml_view_2d(gc, IS.sel, K, n, IS.sel->nb[1], (size_t) col0 * IS.sel->nb[1]);
                selm = ggml_view_2d(gc, IS.selm, K, n, IS.selm->nb[1], (size_t) col0 * IS.selm->nb[1]);
            }
            Kc = ggml_reshape_3d(gc, ggml_get_rows(gc, comp, ggml_cont(gc, ggml_reshape_1d(gc, ggml_cont(gc, sel), K * n))), DH, K, n);
            kg_comp = comp;
            kg_sel = sel;
            kg_selm = selm;
            Mc = cont_if(gc, selm);
            // the source's rows must be read AFTER this pass wrote its own (a kv source attends to its new row)
        }
    }

    // ---- attention: per query its window rows + its top-k compressed rows, one flash-attn with the query as batch
    ggml_tensor * o = nullptr, * o8 = nullptr;
    if (afuse && q8f && part == 0) {   // + o's q8 for wo_a (its 8 head groups as batches)
        ggml_tensor * ao = ggml_dsv41_attn_dec_q8(gc, q, L.raw, kvn, widx_v, Mw, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr,
                                                  Kc ? kg_selm : nullptr, im.BL(il, "attn_sinks.weight"), rtab, (int) SWA,
                                                  1.0f / std::sqrt((float) DH), ND, (int) (im.OGD / DH));
        o = ggml_view_3d(gc, ao, DH, NH, n, (size_t) DH * 4, (size_t) (DH * NH) * 4, 0);
        o8 = ggml_view_1d(gc, ao, ggml_dsv41_q8_floats(im.OGD, n * im.OG), (size_t) (DH * NH * n) * 4);
    } else if (afuse) {
        o = ggml_dsv41_attn_dec(gc, q, L.raw, kvn, widx_v, Mw, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, Kc ? kg_selm : nullptr,
                                im.BL(il, "attn_sinks.weight"), rtab, (int) SWA, 1.0f / std::sqrt((float) DH), ND);   // [DH, NH, n]
    } else {
    ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;                                 // [DH, M, n]
    ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;                                 // [M, n]
    int64_t M = Kall->ne[1];
    // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256; ggml-sycl's DSV4 prompt path (>= 8
    // queries, fattn-dsv4.cpp in larry's ggml-sycl patch) takes any multiple of 16 - padding there only moves bytes
    const char * mqa_env = std::getenv("GGML_SYCL_DSV4_MQA");
    const bool sycl_mqa = im.sycl && n >= 8 && M % 16 == 0 && !(mqa_env && mqa_env[0] == '0');
    // larry: ggml-sycl's DSV4 decode / verify attention (< 8 queries, fattn-dsv4-dec.hpp) reads f32 keys of any count -
    // no padding, no key cast (GGML_SYCL_DSV4_DEC=0: the generic path)
    const char * dec_env = std::getenv("GGML_SYCL_DSV4_DEC");
    const bool sycl_dec = im.sycl && !sycl_mqa && DH == 512 && NH % 32 == 0 && !(dec_env && dec_env[0] == '0');
    // larry, STRATA_ATTN_SDPA (patch-sdpa-dense.py): a fixed key count's span as one oneDNN SDPA (keys 576 wide + the sink)
    static const bool attn_sdpa = env_on("STRATA_ATTN_SDPA", false);
    static const int sdpa_dp = std::getenv("STRATA_ATTN_SDPA_DP") ? std::atoi(std::getenv("STRATA_ATTN_SDPA_DP")) : 576;
    if (attn_sdpa && sycl_mqa && DH == 512 && (M == SWA || (Kc && M == SWA + 512))) {
        ggml_tensor * kp = ggml_dsv41_kgather_pad(gc, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, (int) SWA, sdpa_dp);
        // larry, STRATA_SDPA_O16 (patch-sdpao16.py, default 0): the output in fp16 when wo_a takes it as one op (STRATA_OPROJ_A)
        static const bool sdpa_o16 = env_on("STRATA_SDPA_O16", false) && env_on("STRATA_OPROJ_A", false);
        const bool o16 = sdpa_o16 && im.sycl && n > 8 && im.BL(il, "attn_output_a.weight")->type == GGML_TYPE_Q6_K;
        o = (o16 ? ggml_dsv41_attn_sdpa_o16 : ggml_dsv41_attn_sdpa)(gc, q, kp, Mall, im.BL(il, "attn_sinks.weight"),
                                                                   1.0f / std::sqrt((float) DH));   // [DH, NH, n]
        o = b.rope(o, pos, r, true, rope_ip && im.sycl);                                       // de-RoPE the output
    } else {
    if (!im.cpu && !sycl_mqa && !sycl_dec && M % 256) {
        const int64_t pad = 256 - M % 256;
        Kall = ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
        Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, n), NEG_INF), 0);
        M += pad;
    }
    // larry: STRATA_KGATHER (default 1): the prompt path's keys gathered straight into f16 by one op - the get_rows /
    // concat / cast chain behind Kall wrote and re-read ~2 GB per 512-query pass (~40% of a long prompt's card-0 time);
    // Kw / Kc / Kall are then unused and drop out of the graph
    static const bool kgather = env_on("STRATA_KGATHER", true);
    ggml_tensor * k4 = nullptr;
    static const bool dec_fuse = env_on("STRATA_DEC_FUSE", true);
    if (sycl_mqa && kgather) {
        k4 = ggml_dsv41_kgather(gc, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, (int) SWA);
        GGML_ASSERT(k4->ne[1] == M);
    } else if (sycl_dec && kgather && dec_fuse && (!Kc || kg_sel->nb[0] == 4)) {
        // larry: decode / verify: the window rows, the compressed rows and their concat as one f32 gather
        k4 = ggml_dsv41_kgather2(gc, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, (int) SWA, GGML_TYPE_F32);
        GGML_ASSERT(k4->ne[1] == M);
    } else {
        k4 = ggml_reshape_4d(gc, Kall, DH, M, 1, n);
        if (!sycl_dec) k4 = ggml_cast(gc, k4, GGML_TYPE_F16);
    }
    ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);    // [DH, 1, NH, n]
    // the mask stays f16 (ggml_flash_attn_ext asserts it; it is M halves)
    ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, n), GGML_TYPE_F16);
    o = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(o, im.BL(il, "attn_sinks.weight"));
    o = ggml_reshape_3d(gc, o, DH, NH, n);
    o = b.rope(o, pos, r, true, rope_ip && im.sycl);                                           // de-RoPE the output
    }   // larry, STRATA_ATTN_SDPA's else
    }
    if (part == 1) {
        outs.push_back(ggml_cpy(gc, o, b.cols2(im.o_att, im.OGD * im.OG, n, 0)));
        outs.push_back(ggml_cpy(gc, ggml_view_1d(gc, fb_a, (2 * HC + HC * HC) * n, (size_t) (D * n) * 4),
                                ggml_view_1d(gc, im.o_mix, (2 * HC + HC * HC) * n, 0)));
        for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
        for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
        return gf;
    }
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    // larry, STRATA_TP_OPROJ=1 (decode / verify graphs): wo_a and wo_b as one op that ggml-sycl hands to the xe tier, which
    // splits wo_a's head groups (and wo_b's matching columns) over the four cards and adds the partial sums on this one
    static const int tp_oproj = std::getenv("STRATA_TP_OPROJ") ? std::atoi(std::getenv("STRATA_TP_OPROJ")) : 0;
    if (o8) {   // larry, STRATA_Q8_FUSE: wo_a on the attention's q8, writing wo_b's input layout (the tier's split, if set, reads o)
        attn = ggml_dsv41_oproj_q8(gc, o, o8, wo_a, im.BL(il, "attn_output_b.weight"));           // [D, n]
    } else if (dec && tp_oproj == 1 && im.sycl && n <= 8) {
        attn = ggml_dsv41_oproj(gc, o, wo_a, im.BL(il, "attn_output_b.weight"));                  // [D, n]
    } else {
        // larry, STRATA_OPROJ_A (patch-oproja.py, default 0): prompt widths take wo_a as one op that writes wo_b's input
        // layout (the mul_mat's [OL, n, OG] was permuted and copied back: 0.16 ms a 512-column span)
        static const bool oproj_a = env_on("STRATA_OPROJ_A", false);
        if (oproj_a && im.sycl && n > 8 && ggml_is_contiguous(o) && wo_a->type == GGML_TYPE_Q6_K) {
            attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), ggml_dsv41_oproj_a(gc, o, wo_a));   // [D, n]
        } else {
        // grouped output LoRA (model.py: einsum over groups) then wo_b
        o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);           // [OGD, n, OG]
        ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);          // [OL, OG, n]
        attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]
        }
    }
    }   // part != 2

    // ---- hc_post(attention) -> FFN hc: its mixes give the next layer's pre-mix; its input uses the attention's
    ggml_tensor * h = nullptr, * fn = nullptr, * fn8 = nullptr;                             // [D, HC, n], [D, n]
    ggml_tensor * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    ggml_tensor * fa_f = nullptr, * fb_f = nullptr;   // the FFN hc ops (STRATA_DIRECT_OUT writes their outputs in place)
    if (hcf) {
        ggml_tensor * fa = ggml_dsv41_hc_fa(gc, x, im.BL(il, "hc_ffn_fn.weight"), a_pre, attn, a_post, a_comb);
        fa_f = fa;
        h = ggml_view_3d(gc, fa, D, HC, n, (size_t) D * 4, (size_t) (HC * D) * 4, 0);
        ggml_tensor * fb = b.hc_merge((q8f ? ggml_dsv41_hc_fb_q8 : ggml_dsv41_hc_fb)(gc, fa, im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"),
                                            im.BL(il, "ffn_norm.weight"), (float) G.rms_eps, (float) G.hc_eps, (int) G.hc_iters));
        b.hc_fb_views(fb, n, &fn, &f_pre, &f_post, &f_comb);
        if (q8f) fn8 = ggml_view_1d(gc, fb, ggml_dsv41_q8_floats(D, n), (size_t) (n * (D + 2 * HC + HC * HC)) * 4);
        fb_f = fb;
    } else {
        h = ggml_dsv4_hc_post(gc, attn, x, a_post, a_comb);
        b.hc_mixes(h, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"),
                   &f_pre, &f_post, &f_comb);
        fn = b.rms_w(ggml_dsv4_hc_pre(gc, h, a_pre), im.BL(il, "ffn_norm.weight"));
    }

    // ---- router (model.py Gate): sqrt(softplus), top-k on score + bias, weights from the raw scores
    ggml_tensor * rl = ggml_mul_mat(gc, im.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_sqrt(gc, ggml_softplus(gc, rl));                                  // [NE, n]
    // selection = score + exp_probs_b + the selection-only rbias (soft prune / cache bonus); the weights below come
    // from the unbiased scores, as model.py's Gate - all-zero rbias (the default) = the model's routing bit for bit
    ggml_tensor * s0 = ggml_add(gc, sc, im.BL(il, "exp_probs_b.bias"));                        // the model's own selection score
    if (mimg) {   // larry (vision): an image position selects with the VL bias (model.py Gate: bias_vl where image_mask)
        ggml_tensor * dvl = ggml_sub(gc, im.BL(il, "exp_probs_b_vl.bias"), im.BL(il, "exp_probs_b.bias"));   // [NE]
        s0 = ggml_add(gc, s0, ggml_mul(gc, ggml_repeat(gc, ggml_reshape_2d(gc, dvl, im.NE, 1), s0),
                                         ggml_repeat(gc, ggml_reshape_2d(gc, mimg, 1, n), s0)));
    }
    ggml_tensor * rb = ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1);
    ggml_tensor * sel = nullptr;
    // SMART route bias (opt-in; the flat bias changes ~every choice it can: -14 pt API top-1, repetition loops):
    //   STRATA_RB_PROTECT=P  the exact top-P experts are always kept (+1e4); the bias only competes for the other slots
    //   STRATA_RB_MARGIN=d   the bias only applies to experts whose exact score is within d of the cutoff (midpoint of the
    //                        exact NU-th / NU+1-th scores); experts confidently above it (> cut + d) are always kept -
    //                        only near-ties can flip
    // Both may be set.  Weights still come from the unbiased scores.  Unset = the flat bias, bit for bit as before.
    static const int rb_protect = std::getenv("STRATA_RB_PROTECT") ? std::atoi(std::getenv("STRATA_RB_PROTECT")) : 0;
    static const float rb_margin = std::getenv("STRATA_RB_MARGIN") ? (float) std::atof(std::getenv("STRATA_RB_MARGIN")) : -1.0f;
    if (rb_protect > 0 || rb_margin >= 0.0f) {
        const int kt = (int) im.NU + 1;
        ggml_tensor * ti = ggml_argsort_top_k(gc, s0, kt);                                         // [kt, n]
        ggml_tensor * tv = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s0, 1, im.NE, n), ti), kt, n);   // [kt, n] desc
        auto row = [&](int r) { return ggml_view_2d(gc, tv, 1, n, tv->nb[1], (size_t) r * tv->nb[0]); };   // [1, n]
        ggml_tensor * like = ggml_new_tensor_2d(gc, GGML_TYPE_F32, im.NE, n);
        ggml_tensor * protect = nullptr;
        if (rb_protect > 0) {   // s0 >= the P-th largest exact score
            ggml_tensor * vp = ggml_repeat(gc, ggml_cont(gc, row(std::min(rb_protect, (int) im.NU) - 1)), like);
            protect = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, vp), 1.0f, 1e-6f));
        }
        ggml_tensor * gate = nullptr;
        if (rb_margin >= 0.0f) {
            ggml_tensor * cut = ggml_repeat(gc, ggml_scale(gc, ggml_add(gc, ggml_cont(gc, row((int) im.NU - 1)), ggml_cont(gc, row((int) im.NU))), 0.5f), like);
            ggml_tensor * dv = ggml_sub(gc, s0, cut);                                              // [NE, n]
            ggml_tensor * above = ggml_step(gc, ggml_scale_bias(gc, dv, 1.0f, -rb_margin));          // dv > d
            ggml_tensor * below = ggml_step(gc, ggml_scale_bias(gc, dv, -1.0f, -rb_margin));         // dv < -d
            protect = protect ? ggml_clamp(gc, ggml_add(gc, protect, above), 0.0f, 1.0f) : above;
            gate = ggml_scale_bias(gc, ggml_clamp(gc, ggml_add(gc, above, below), 0.0f, 1.0f), -1.0f, 1.0f);   // in the band
        }
        ggml_tensor * unprot = ggml_scale_bias(gc, protect, -1.0f, 1.0f);
        if (gate) unprot = ggml_mul(gc, unprot, gate);
        sel = ggml_add(gc, ggml_add(gc, s0, ggml_mul(gc, ggml_repeat(gc, rb, like), unprot)), ggml_scale(gc, protect, 1e4f));
    } else {
        sel = ggml_add(gc, s0, rb);
    }
    ggml_tensor * ids = nullptr, * wts = nullptr, * ro = nullptr;
    // larry: ggml-sycl's fused Gate (GGML_OP_DSV41_ROUTER: one kernel for the ~11 below) with the flat bias (the default;
    // STRATA_ROUTER_FUSED=0: the composite).  Its ids are floats; the cpy into o_ids converts them.
    static const bool fused_router = [] { const char * v = std::getenv("STRATA_ROUTER_FUSED"); return !(v && v[0] == '0'); }();
    if (im.sycl && fused_router && rb_protect <= 0 && rb_margin < 0.0f && !mimg) {   // (larry, vision: one bias only)
        ro = ggml_dsv41_router(gc, rl, im.BL(il, "exp_probs_b.bias"), ggml_reshape_1d(gc, rb, im.NE), (int) im.NU,
                               G.norm_topk, (float) G.weights_scale);   // [2 NU, n]
        ids = ggml_view_2d(gc, ro, im.NU, n, ro->nb[1], 0);
        wts = ggml_view_2d(gc, ro, im.NU, n, ro->nb[1], (size_t) im.NU * 4);
    } else {
        ids = ggml_argsort_top_k(gc, sel, (int) im.NU);   // [NU, n]
        wts = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, im.NE, n), ids), im.NU, n);
        if (G.norm_topk) wts = ggml_div(gc, wts, ggml_scale_bias(gc, ggml_sum_rows(gc, wts), 1.0f, 1e-20f));
        wts = ggml_scale(gc, wts, (float) G.weights_scale);
    }

    // ---- shared expert
    if (with_shexp) {
        ggml_tensor * sx = b.cols2(im.shexp, D, n, col0);
        if (im.shexp_skip && !dec)   // larry, STRATA_SHEXP_TIER: the expert cards add it into the routed sum - zeros here
            outs.push_back(ggml_cpy(gc, ggml_scale(gc, fn, 0.0f), sx));
        else
            outs.push_back(fn8 ? shexp_ffn(im, gc, il, fn, fn8, sx) : ggml_cpy(gc, shexp_ffn(im, gc, il, fn), sx));
    }
    ggml_tensor * o_fn8 = fn8 && !with_shexp ? ggml_view_1d(gc, im.o_fn8, ggml_dsv41_q8_floats(D, n), 0) : nullptr;   // the shexp graph's

    // larry, STRATA_DIRECT_OUT (default 1; fused hc + fused router): the FFN hc ops and the router write the layer's
    // outputs straight into their state tensors (hap; o_fn, pre_state, post_f, comb_f, o_fn8; o_ids as ints, o_wts) -
    // the copy kernels after them go (a batched copy and the ids' float -> int copy).  pre_state is read above (hc_fa).
    static const bool direct_out = env_on("STRATA_DIRECT_OUT", true);
    if (direct_out && fa_f && fb_f && ro) {
        ggml_dsv41_write_to(fa_f, b.cols3(im.hap, D, HC, n, col0), nullptr, nullptr, nullptr, nullptr);
        ggml_dsv41_write_to(fb_f, b.cols2(im.o_fn, D, n, col0), b.cols2(im.pre_state, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                            b.cols3(im.comb_f, HC, HC, n, col0), o_fn8);
        ggml_dsv41_write_to(ro, b.cols2(im.o_ids, im.NU, n, col0), b.cols2(im.o_wts, im.NU, n, col0), nullptr, nullptr, nullptr);
        outs.push_back(ro);
    } else {
    if (o_fn8) outs.push_back(ggml_cpy(gc, fn8, o_fn8));
    outs.push_back(ggml_cpy(gc, h, b.cols3(im.hap, D, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, f_post, b.cols2(im.post_f, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, ggml_reshape_3d(gc, f_comb, HC, HC, n), b.cols3(im.comb_f, HC, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, fn, b.cols2(im.o_fn, D, n, col0)));
    outs.push_back(ggml_cpy(gc, ids, b.cols2(im.o_ids, im.NU, n, col0)));
    outs.push_back(ggml_cpy(gc, wts, b.cols2(im.o_wts, im.NU, n, col0)));
    outs.push_back(ggml_cpy(gc, f_pre, b.cols2(im.pre_state, HC, n, col0)));   // reads of pre_state happen above
    }
    // larry, STRATA_CPY_LAST (default 1): every output copy's source first, then the copies - consecutive plain copies
    // run as one batched kernel in ggml-sycl (each used to sit right after its source: its own command)
    static const bool cpy_last = env_on("STRATA_CPY_LAST", true);
    if (cpy_last) for (ggml_tensor * t : outs) if (t->op == GGML_OP_CPY && t->src[0]) ggml_build_forward_expand(gf, t->src[0]);
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// enc-skip: the persistent-KV half of build_attn for one span.  Runs hc_pre / attn_norm, writes the raw window K/V
// ring, and on a kv source layer writes the compressed-KV row + indexer-key row (+ the ratio-2 compressor state).
// Everything else build_attn does (q, the attention + output LoRA, hc_post, the router, the shared expert, the indexer
// selection / candidate blocks, and the pre_state / hap / shexp / o_* writes) only ever flows into THIS layer's own
// output columns, so a prompt token whose later layers are never read does not need them.  The state it does write is
// exactly what build_attn writes for those positions, so a later query's window / compressed reads see the same
// values as the full path (verified by the enc-skip gate).
static ggml_cgraph * build_attn_kv_only(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH;
    const int64_t r = L.ratio;
    std::vector<ggml_tensor *> outs, late;   // late: the ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);
    ggml_tensor * mimg = !im.img.empty() && im.img_in((int64_t) im.pos0 + col0, n) ? b.v1(im.i_img, n, (size_t) col0 * 4) : nullptr;   // (larry, vision)

    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {   // the Engram add still feeds the K/V + compressor projections
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = ggml_mul(gc, im.BL(il, "engram_q_norm.weight"), im.BL(il, "engram_k_norm.weight"));
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_scale(gc, ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn)), 1.0f / std::sqrt((float) D));
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (mimg) gate = ggml_mul(gc, gate, ggml_reshape_3d(gc, ggml_scale_bias(gc, mimg, -1.0f, 1.0f), 1, 1, n));   // (larry, vision)
        x = ggml_add(gc, x, ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate));
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    }
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- window K (= V): the same post-RoPE raw row build_attn writes into the ring.  STRATA_QKV_FUSE: through the fused
    // weight (attn_kv alone is not usable once the pair is reordered) - its q_a rows are computed and dropped
    ggml_tensor * kv_y = nullptr;
    if (im.w.has("blk." + std::to_string(il) + ".attn_qkv_a.weight")) {
        ggml_tensor * wf = im.BL(il, "attn_qkv_a.weight");
        const int64_t nq = im.BL(il, "attn_q_a.weight")->ne[1], nk = wf->ne[1] - nq;
        ggml_tensor * y = ggml_mul_mat(gc, wf, xn);
        kv_y = ggml_view_2d(gc, y, nk, n, y->nb[1], (size_t) nq * ggml_element_size(y));
    } else {
        kv_y = ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn);
    }
    ggml_tensor * kv = b.rms_w(kv_y, im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, cont_if(gc, kv2), Ds41Quant::FP8_POW2_32);
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV + indexer keys (kv source layers; the only other persistent state a later query reads)
    if (r && L.kv_src) {
        ggml_tensor * latent = nullptr;
        if (r == 1) {
            latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
        } else {
            ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
            ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
            ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
            ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);
            ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
            ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
            ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
            ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
            vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));
            vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
            latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
            const int64_t k = std::min<int64_t>(n, 2);
            late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
            late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
        }
        latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
        ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
        ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
        ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
        ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
        if (im.cfg.act_quant) ik = fake_quant(gc, cont_if(gc, ik), Ds41Quant::FP4_POW2_32);
        outs.push_back(ggml_set_rows(gc, L.idxk, ik, cslot));
        ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
        if (im.cfg.act_quant) lr = fake_quant(gc, cont_if(gc, lr), Ds41Quant::FP4_E4M3_16);
        outs.push_back(ggml_set_rows(gc, L.comp, lr, cslot));
    }
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// predict: router(rms_norm(hc_pre(state, pre)) * ffn_norm) + bias, top 16 of token 0
static ggml_cgraph * build_predict(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor ** out, int64_t n = 1) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_dsv4_hc_pre(gc, b.cols3(im.x_state, D, HC, n), b.cols2(im.pre_state, HC, n));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(x, im.BL(il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
    *out = ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*out);
    ggml_build_forward_expand(gf, *out);
    return gf;
}

// the shared expert of layer il on o_fn (what build_attn(with_shexp = false) wrote) into shexp
static ggml_cgraph * build_shexp(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D;
    // larry, STRATA_Q8_FUSE: the attention graph (same layer, same width, built with the same test) left fn's q8 in o_fn8
    ggml_tensor * fn8 = col0 == 0 && b.hc_fused(n) && q8_fused(im, il, n) ? ggml_view_1d(gc, im.o_fn8, ggml_dsv41_q8_floats(D, n), 0) : nullptr;
    ggml_tensor * sx = b.cols2(im.shexp, D, n, col0);
    ggml_tensor * fv = b.cols2(im.o_fn, D, n, col0);
    // STRATA_DEBUG_NO_SHEXP=1 (timing bound only: WRONG output) - the shared expert's work left out (its output zero)
    static const bool no_shexp = env_on("STRATA_DEBUG_NO_SHEXP", false);
    if (no_shexp) { ggml_build_forward_expand(gf, ggml_scale_inplace(gc, sx, 0.0f)); return gf; }
    ggml_build_forward_expand(gf, fn8 ? shexp_ffn(im, gc, il, fv, fn8, sx) : ggml_cpy(gc, shexp_ffn(im, gc, il, fv), sx));
    return gf;
}

// finish: x_state = hc_post(shexp + routed, hap, post_f, comb_f).  col0: the batch column (0 on the single-pass paths).
static ggml_cgraph * build_finish(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    if (b.hc_fused(n)) {   // larry: the add, the hc_post and the state write in one kernel
        ggml_build_forward_expand(gf, ggml_dsv41_hc_post2(gc, b.cols2(im.shexp, D, n, col0), b.cols2(im.routed, D, n, col0),
                                                          b.cols3(im.hap, D, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                                                          b.cols3(im.comb_f, HC, HC, n, col0), b.cols3(im.x_state, D, HC, n, col0)));
        return gf;
    }
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, n, col0), b.cols2(im.routed, D, n, col0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                                        b.cols3(im.comb_f, HC, HC, n, col0));
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    return gf;
}

// finish (n = 1) + the next layer's routing prediction on the state it writes (build_predict's ops on the cpy result)
static ggml_cgraph * build_finish_predict(Ds41Dense::Impl & im, ggml_context * gc, int next_il, ggml_tensor ** pout,
                                          int ahead = 0, std::vector<ggml_tensor *> * aout = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 128, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, 1, 0), b.cols2(im.routed, D, 1, 0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, 1, 0), b.cols2(im.post_f, HC, 1, 0),
                                        b.cols3(im.comb_f, HC, HC, 1, 0));
    ggml_tensor * st = ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, 1, 0));
    ggml_build_forward_expand(gf, st);
    ggml_tensor * xp = ggml_dsv4_hc_pre(gc, st, b.cols2(im.pre_state, HC, 1));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(next_il), b.rms_w(xp, im.BL(next_il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(next_il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) next_il].rbias, im.NE, 1));
    // the layers after next_il on the same state (speculative reads): their scores are stacked under next_il's so ONE
    // argsort gives all the rows and ONE readback fetches them ([16, 1 + ahead] I32)
    for (int a = 1; a <= ahead && aout; ++a) {
        const int il = next_il + a;
        if (il >= (int) im.ly.size()) break;
        ggml_tensor * ra = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(xp, im.BL(il, "ffn_norm.weight")));
        ggml_mul_mat_set_prec(ra, GGML_PREC_F32);
        ggml_tensor * sa = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, ra)), im.BL(il, "exp_probs_b.bias")),
                                    ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
        sc = ggml_concat(gc, sc, sa, 1);
        aout->push_back(nullptr);   // a row of *pout
    }
    *pout = ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*pout);
    ggml_build_forward_expand(gf, *pout);
    return gf;
}

// head: hc_pre(state, pre_state) -> output_norm -> output, rows row0..row0+nr-1
static ggml_cgraph * build_head(Ds41Dense::Impl & im, ggml_context * gc, int64_t row0, int64_t nr, ggml_tensor ** lg) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = b.cols3(im.x_state, D, HC, nr, row0);
    ggml_tensor * p = b.cols2(im.pre_state, HC, nr, row0);
    ggml_tensor * h = b.rms_w(ggml_dsv4_hc_pre(gc, x, p), im.W("output_norm.weight"));
    *lg = ggml_mul_mat(gc, im.W("output.weight"), h);
    ggml_set_output(*lg);
    ggml_build_forward_expand(gf, *lg);
    return gf;
}

// begin: x_state = repeat(embd), pre_state = one-hot (model.py make_identity_pre_mix).  col0: the batch column the
// pass's n tokens live at (0 on the single-pass paths).
static ggml_cgraph * build_init(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * e = b.cols2(im.i_emb, D, n, col0);
    if (e->type != GGML_TYPE_F32) e = ggml_cast(gc, e, GGML_TYPE_F32);   // larry, STRATA_EMB_PIN: f16 rows (exact)
    e = ggml_reshape_3d(gc, e, D, 1, n);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_repeat_4d(gc, e, D, HC, n, 1), b.cols3(im.x_state, D, HC, n, col0)));
    ggml_tensor * one = ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, n), 1.0f);
    ggml_tensor * oh = ggml_pad(gc, one, (int) (HC - 1), 0, 0, 0);                            // [1, 0, 0, 0] per token
    ggml_build_forward_expand(gf, ggml_cpy(gc, oh, b.cols2(im.pre_state, HC, n, col0)));
    return gf;
}

// ================================================================== public API

static bool fin_flush(Ds41Dense::Impl & im);   // (STRATA_FIN_DEFER, below)

std::vector<ggml_tensor *> Ds41Dense::slice_reorder_weights() const {
    std::vector<ggml_tensor *> v;
    const std::string suf = "attn_output_a.weight";
    for (const auto & kv : p_->w.t) {
        ggml_tensor * t = kv.second;
        if (t->type != GGML_TYPE_Q6_K || t->ne[2] != 1 || t->ne[3] != 1 || !t->buffer) continue;
        if (kv.first.size() > suf.size() && kv.first.compare(kv.first.size() - suf.size(), suf.size(), suf) == 0) v.push_back(t);
    }
    return v;
}

Ds41Dense::Ds41Dense() : p_(new Impl()) {}
Ds41Dense::~Ds41Dense() = default;
const Ds41Geometry & Ds41Dense::geom() const { return p_->g; }
std::vector<ggml_tensor *> Ds41Dense::reorder_weights() const {
    std::vector<ggml_tensor *> v;
    auto ends = [](const std::string & n, const char * suf) { const size_t k = std::strlen(suf); return n.size() > k && n.compare(n.size() - k, k, suf) == 0; };
    for (const auto & kv : p_->w.t) {
        ggml_tensor * t = kv.second;
        if (t->type != GGML_TYPE_Q6_K || t->ne[2] != 1 || t->ne[3] != 1 || !t->buffer) continue;
        // wo_a is read through 3-D views only: decode reorders it per slice, the prompt reads its standard layout - never whole
        if (ends(kv.first, "attn_output_a.weight")) continue;
        if (ends(kv.first, "ffn_gate_shexp.weight") || ends(kv.first, "ffn_up_shexp.weight")) {
            const std::string pre = kv.first.substr(0, kv.first.size() - (ends(kv.first, "ffn_gate_shexp.weight") ? 21 : 19));
            if (p_->w.t.count(pre + "ffn_gate_up_shexp.weight")) continue;
        }
        if (ends(kv.first, "attn_q_a.weight") || ends(kv.first, "attn_kv.weight")) {   // STRATA_QKV_FUSE: the pair as one
            const std::string pre = kv.first.substr(0, kv.first.size() - (ends(kv.first, "attn_q_a.weight") ? 15 : 14));
            if (p_->w.t.count(pre + "attn_qkv_a.weight")) continue;
        }
        v.push_back(t);
    }
    return v;
}

ggml_tensor * Ds41Dense::weight(const std::string & name) const { return p_->w.get(name); }
int64_t Ds41Dense::n_layer() const { return p_->nl; }
ggml_backend_t Ds41Dense::backend() const { return p_->backend; }
const std::string & Ds41Dense::last_error() const { return p_->err; }

// The NC-column tensors (see Impl::nctx): (re)created for `NC` columns, zeroed.  Every one is per-pass scratch (a pass
// writes the columns it reads), so a resize between passes loses nothing.
static bool alloc_cols(Ds41Dense::Impl & im, int64_t NC, std::string & err) {
    if (im.nbuf) { ggml_backend_buffer_free(im.nbuf); im.nbuf = nullptr; }
    if (im.nctx) { ggml_free(im.nctx); im.nctx = nullptr; }
    im.NC = NC;
    ggml_init_params cp = { ggml_tensor_overhead() * 256, nullptr, true };
    im.nctx = ggml_init(cp);
    ggml_context * c = im.nctx;
    const int64_t D = im.D, HC = im.HC;
    im.x_state = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.pre_state = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.hap = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.post_f = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.comb_f = ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, NC);
    im.shexp = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    if (!im.fn_buf) {   // larry: allocated once, NCmax columns (views take the first NC)
        ggml_init_params fp = { ggml_tensor_overhead() * 2, nullptr, true };
        im.fn_ctx = ggml_init(fp);
        ggml_tensor * fn = ggml_new_tensor_2d(im.fn_ctx, GGML_TYPE_F32, D, std::max(NC, im.NCmax));
        im.fn_buf = ggml_backend_alloc_ctx_tensors(im.fn_ctx, im.backend);
        if (!im.fn_buf) { err = "cannot allocate the FFN input columns"; return false; }
        ggml_backend_buffer_clear(im.fn_buf, 0);
        im.o_fn = fn;
    }
    {
        const int64_t NO = std::min<int64_t>(NC, 8);
        im.o_att = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.OGD * im.OG, NO);
        im.o_mix = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * HC + HC * HC, NO);
        im.o_proj = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NO);
        im.o_fn8 = ggml_new_tensor_1d(c, GGML_TYPE_F32, ggml_dsv41_q8_floats(D, NO));
    }
    im.o_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.NU, NC);
    im.o_wts = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.NU, NC);
    im.routed = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.cand = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.CB, NC);
    for (auto & L : im.ly)
        if (L.idx_src) { L.sel = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.ITOP, NC); L.selm = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.ITOP, NC); }
    im.i_emb = ggml_new_tensor_2d(c, im.emb_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, D, NC);   // larry, STRATA_EMB_PIN: f16
    im.i_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    im.i_widx = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.SWA * NC);
    im.span_memo_pos0 = -1;   // new tensors: nothing uploaded yet
    im.i_wmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.SWA, NC);
    im.i_wslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW));
    for (int r = 1; r <= 2; ++r) {
        im.i_nvis[r] = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
        im.i_cslot[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
        im.i_cpos[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    }
    im.i_sidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2 * NC);
    im.i_sslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, 2));
    im.i_lastblk = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
    im.i_img = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);   // larry (vision)
    for (int e = 0; e < 2; ++e) im.i_eng[e] = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.EC * im.ED, NC);
    // STRATA_VRAM_TRACE: WHICH state tensor owns the columns' VRAM (s13c: a batch's state alloc was 2.45-2.76 GiB and
    // a 2193.81 MiB allocation later failed with 6.83 GiB "free" - name the owner before sizing anything).
    if (std::getenv("STRATA_VRAM_TRACE")) {
        double tot = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) tot += (double) ggml_nbytes(t);
        std::fprintf(stderr, "vram trace: state NC=%d wants %.2f GiB (i_eng %lld x %d = %.2f GiB each, i_wmask %.1f MiB, "
                             "i_widx %.1f MiB)\n", NC, tot / 1073741824.0, (long long) im.EC, (int) im.ED,
                     (double) ggml_nbytes(im.i_eng[0]) / 1073741824.0, (double) ggml_nbytes(im.i_wmask) / 1048576.0,
                     (double) ggml_nbytes(im.i_widx) / 1048576.0);
    }
    im.nbuf = ggml_backend_alloc_ctx_tensors(c, im.backend);
    if (!im.nbuf) { err = "cannot allocate " + std::to_string(NC) + " state columns"; return false; }
    ggml_backend_buffer_clear(im.nbuf, 0);
    return true;
}

bool Ds41Dense::set_columns(int nc) {
    Impl & im = *p_;
    const int64_t want = std::max<int64_t>(im.S, std::min<int64_t>(nc, im.NCmax));
    if (want == im.NC) return true;
    const double tc0 = g_dprof_on ? dprof_now() : 0;
    ggml_backend_synchronize(im.backend);
    // every cached graph holds views of the old columns (their addresses are fixed at the first allocation), and the
    // allocators remember graphs by address - all of it goes, and is rebuilt lazily for the new tensors
    im.graphs.clear();
    im.logits_out.clear();
    im.predict_out.clear();
    im.predict_out_n.clear();
    im.fpredict_out.clear();
    im.fpredict_ahead.clear();
    for (Allo * al : { &im.allo_dec, &im.allo_pass, &im.allo_pass2 }) {
        if (al->a) ggml_gallocr_free(al->a);
        al->a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
        al->graph_epoch.clear();
        al->compute_bufs.clear();
        ++al->epoch;
    }
    if (im.gctx) ggml_free(im.gctx);
    // larry: 4x upstream's - DSpark's verify widths (2..8) x the pass caps x 40 layers filled the old one every few
    // minutes in --serve (each reset rebuilds every kept graph); metadata only, its pages are touched as graphs land
    ggml_init_params gp = { ggml_tensor_overhead() * 800000 + ggml_graph_overhead_custom(8192, false) * 1600, nullptr, true };
    im.gctx = ggml_init(gp);
    const bool ok = alloc_cols(im, want, im.err);
    if (g_dprof_on) std::fprintf(stderr, "dense prof: %lld state columns in %.0f ms\n", (long long) want, dprof_now() - tc0);
    return ok;
}

int Ds41Dense::columns() const { return (int) p_->NC; }

bool Ds41Dense::init(const std::string & path, const Ds41DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    try {
        im.model = std::make_unique<GgufModel>(GgufModel::open(path));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (!(err = ds41_read_geometry(*im.model, im.g)).empty()) return false;
    im.layers = ds41_file_layers(*im.model, im.g);
    if (!(err = ds41_check_tensors(*im.model, im.g, im.layers)).empty()) return false;
    for (size_t i = 0; i < im.layers.size(); ++i)
        if (im.layers[i] != (int64_t) i) { err = "the file's blocks must be 0..k"; return false; }
    im.nl = (int64_t) im.layers.size();
    const Ds41Geometry & G = im.g;
    im.D = G.n_embd; im.HC = G.hc; im.DH = G.d_head; im.DHR = G.d_rope; im.NH = G.n_head; im.OG = G.o_groups;
    im.OL = G.o_lora; im.OGD = G.n_head * G.d_head / G.o_groups; im.RQ = G.r_q; im.SWA = G.n_swa;
    im.IH = G.idx_heads; im.ID = G.idx_dim; im.ITOP = G.idx_topk; im.NE = G.n_expert; im.NU = G.n_used; im.NFF = G.n_ff;
    im.CB = G.cand_blocks; im.CBS = G.cand_bsize; im.EC = (G.engram_ngram - 1) * G.engram_heads; im.ED = G.engram_dim;
    // gate-only overrides (the golden takes the same): small top-k / candidate counts make the selection logic bite
    // on a short prompt
    if (const char * e = std::getenv("DS41_TEST_TOPK")) im.ITOP = std::atoi(e);
    if (const char * e = std::getenv("DS41_TEST_CAND_BLOCKS")) im.CB = std::atoi(e);
    im.S = std::max(1, cfg.max_pass);                       // the dense sub-batch size of the layer-major walk
    im.NCmax = std::max<int64_t>(im.S, std::max(1, cfg.prompt_batch));   // state columns: the whole prompt batch, or the pass
    im.NC = cfg.elastic_cols ? im.S : im.NCmax;                           // elastic: S until a batch asks for more
    // the window ring: every row is gathered BY POSITION (slot = pos % RAW), so its size never changes results.  DSpark
    // verify passes write up to k rejected draft positions; a rejected q evicts q - RAW, which a later query q' (>= the
    // last accepted position, re-fed) needs iff q - RAW >= q' - SWA + 1  ->  safe for k <= RAW - SWA = 16 drafts.
    im.RAW = im.SWA + 16;
    im.cmax[1] = cfg.ctx_max; im.cmax[2] = cfg.ctx_max / 2;
    if (im.ITOP > im.cmax[2] || im.CBS <= 0 || cfg.ctx_max % 256) { err = "ctx_max must be a multiple of 256 and >= 1024"; return false; }

    if (cfg.backend) { im.backend = cfg.backend; }
    else {
        im.backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(im.backend, cfg.n_threads);
        im.own_backend = true;
    }
    im.cpu = ggml_backend_is_cpu(im.backend);
    im.sycl = !im.cpu && std::strstr(ggml_backend_name(im.backend), "SYCL") != nullptr;
    if (!load_weights(im, err)) return false;
    im.emb_f16 = im.sycl && env_on("STRATA_EMB_PIN", false);   // larry (before the columns are allocated)
    // larry, STRATA_ENGRAM_THREADS: the row reader's pool = its queue depth over both NVMe drives (a cold 16K prompt
    // batch is ~1.4M random reads: at 32 a cold 31.6K prompt waited ~4 s for its first batch's rows)
    const int eng_threads = std::getenv("STRATA_ENGRAM_THREADS") ? std::atoi(std::getenv("STRATA_ENGRAM_THREADS")) : cfg.engram_threads;
    if (!cfg.engram_dirs.empty() && !im.engram.open(G, cfg.engram_dirs, eng_threads, err)) return false;
    if (cfg.n_seqs > 1) im.engram.set_seqs(cfg.n_seqs, cfg.ctx_max);   // larry (batch slots): a history a conversation

    // layers
    im.ly.resize((size_t) im.nl);
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        L.ratio = G.ratio[(size_t) l];
        L.kv_src = G.is_kv_source(l);
        L.idx_src = G.is_index_source(l);
        L.cand_src = l == G.cand_layer;
        L.uses_cand = G.cand_layer >= 0 && l > G.cand_layer && L.idx_src;
        L.kvl = G.kv_src(l);
        L.idxl = G.idx_src(l);
        L.eslot = G.engram_slot(l);
    }
    // persistent state
    if (cfg.kv_fp4 && (!cfg.act_quant || im.cpu || im.DH % 64 || im.ID % 32)) {
        err = "--kv-fp4 needs act_quant (the FP4 grids), the CUDA backend, d_head % 64 == 0 and idx_dim % 32 == 0";
        return false;
    }
    const ggml_type kvt = cfg.kv_type >= 0 ? (ggml_type) cfg.kv_type : cfg.act_quant ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (512 + 384 * std::max(0, cfg.n_seqs - 1)), nullptr, true };
    im.sctx = ggml_init(sp);
    auto T = [&](ggml_type t, std::initializer_list<int64_t> ne) {
        const int64_t * d = ne.begin();
        switch (ne.size()) {
            case 1: return ggml_new_tensor_1d(im.sctx, t, d[0]);
            case 2: return ggml_new_tensor_2d(im.sctx, t, d[0], d[1]);
            default: return ggml_new_tensor_3d(im.sctx, t, d[0], d[1], d[2]);
        }
    };
    const int64_t D = im.D, NCM = im.NCmax;
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        // larry: the window ring in F32 on SYCL (STRATA_RAW_F32, default 1): its rows are FP8-rounded, exact in either
        // type, and attention reads it as F32 every layer of every token (a bf16 ring costs a cast kernel each time)
        static const bool raw_f32 = env_on("STRATA_RAW_F32", true);
        const ggml_type rawt = im.sycl && raw_f32 && cfg.kv_type < 0 ? GGML_TYPE_F32 : kvt;
        L.raw = T(rawt, { im.DH, im.RAW });
        L.raw_snap = T(rawt, { im.DH, im.RAW });
        L.rbias = T(GGML_TYPE_F32, { im.NE });   // zeroed with the buffer
        if (L.kv_src) {
            // STRATA_KV_FP4_ONLY=comp|idx (bisect tap): pack only one of the two caches
            static const char * only = std::getenv("STRATA_KV_FP4_ONLY");
            const bool fc = cfg.kv_fp4 && !(only && std::string(only) == "idx");
            const bool fi = cfg.kv_fp4 && !(only && std::string(only) == "comp");
            L.comp = T(fc ? GGML_TYPE_NVFP4 : kvt, { im.DH, im.cmax[L.ratio] + 1 });
            L.idxk = T(fi ? GGML_TYPE_MXFP4 : kvt, { im.ID, im.cmax[L.ratio] + 1 });
            if (L.ratio == 2) {
                L.st_kv = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_sc = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_kv_snap = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_sc_snap = T(GGML_TYPE_F32, { im.DH, 2 });
            }
        }
    }
    // larry (batch slots, patch-seqs-dense.py): conversation 0's tensors are the Layer members; the others' the same
    // shapes, from the same context (allocated below with them)
    im.seqs.assign((size_t) std::max(1, cfg.n_seqs), {});
    for (size_t q = 0; q < im.seqs.size(); ++q) {
        auto & S = im.seqs[q];
        S.t.assign((size_t) im.nl, {});
        for (int64_t l = 0; l < im.nl; ++l) {
            auto & L = im.ly[(size_t) l];
            auto & a = S.t[(size_t) l];
            if (q == 0) { a = { L.raw, L.raw_snap, L.comp, L.idxk, L.st_kv, L.st_sc, L.st_kv_snap, L.st_sc_snap }; continue; }
            auto dup = [&](ggml_tensor * t) { return t ? ggml_dup_tensor(im.sctx, t) : nullptr; };
            a = { dup(L.raw), dup(L.raw_snap), dup(L.comp), dup(L.idxk), dup(L.st_kv), dup(L.st_sc), dup(L.st_kv_snap), dup(L.st_sc_snap) };
        }
    }
    // larry (cfg.kv_alloc, STRATA_DENSE_LAZY_KV): the compressed KV and indexer keys in the caller's virtual memory,
    // mapped as the context grows - a 1M context's rows stay out of card 0's VRAM until a conversation reaches them
    // (batch slots: every conversation's units, each mapped as its own conversation grows)
    if (im.sycl && im.cfg.kv_alloc) {
        const ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(im.backend);
        std::vector<size_t> ub, spare, uo;
        for (size_t q = 0; q < im.seqs.size(); ++q)
            for (int64_t l = 0; l < im.nl; ++l)
                for (ggml_tensor * t : { im.seqs[q].t[(size_t) l][2], im.seqs[q].t[(size_t) l][3] })
                    if (t) {
                        im.kv_units.push_back({ t, (int) im.ly[(size_t) l].ratio });
                        im.kv_unit_seq.push_back((int) q);
                        ub.push_back(ggml_backend_buft_get_alloc_size(bt, t));
                        spare.push_back((size_t) (t->ne[1] - 1) * t->nb[1]);   // the spare row (open compressor groups)
                    }
        if (!im.kv_units.empty())
            if (ggml_backend_buffer_t kb = im.cfg.kv_alloc(ub, spare, uo)) {
                uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(kb);
                for (size_t i = 0; i < im.kv_units.size(); ++i)
                    if (ggml_backend_tensor_alloc(kb, im.kv_units[i].first, base + uo[i]) != GGML_STATUS_SUCCESS) {
                        err = "cannot bind the KV state in the lazy map";
                        ggml_backend_buffer_free(kb);
                        return false;
                    }
                im.kv_buf = kb;
            }
        if (!im.kv_buf) { im.kv_units.clear(); im.kv_unit_seq.clear(); }
    }
    im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
    if (!im.sbuf) { err = "cannot allocate the decode state"; return false; }
    ggml_backend_buffer_clear(im.sbuf, 0);
    // the ratio-2 state scores start at -1e30 (an unwritten state gets softmax weight 0); sel = 0 (a valid row)
    for (auto & S : im.seqs)   // (larry, batch slots: every conversation's)
        for (auto & a : S.t)
            if (ggml_tensor * sc = a[5]) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); ggml_backend_tensor_set(sc, v.data(), 0, v.size() * 4); }
    // inputs
    ggml_init_params ipp = { ggml_tensor_overhead() * 64, nullptr, true };
    im.ictx = ggml_init(ipp);
    auto I = [&](ggml_type t, int64_t a, int64_t b2 = 1) { return b2 == 1 ? ggml_new_tensor_1d(im.ictx, t, a) : ggml_new_tensor_2d(im.ictx, t, a, b2); };
    im.neg_ramp = I(GGML_TYPE_F32, im.cmax[1] + 1);
    im.blk_ramp = I(GGML_TYPE_F32, im.cmax[1] / im.CBS + 2);
    im.ibuf = ggml_backend_alloc_ctx_tensors(im.ictx, im.backend);
    if (!im.ibuf) { err = "cannot allocate the inputs"; return false; }
    ggml_backend_buffer_clear(im.ibuf, 0);
    {
        std::vector<float> v((size_t) im.neg_ramp->ne[0]);
        for (size_t i = 0; i < v.size(); ++i) v[i] = -(float) i;
        ggml_backend_tensor_set(im.neg_ramp, v.data(), 0, v.size() * 4);
        v.assign((size_t) im.blk_ramp->ne[0], 0);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (float) i;
        ggml_backend_tensor_set(im.blk_ramp, v.data(), 0, v.size() * 4);
    }
    // larry: 4x upstream's - DSpark's verify widths (2..8) x the pass caps x 40 layers filled the old one every few
    // minutes in --serve (each reset rebuilds every kept graph); metadata only, its pages are touched as graphs land
    ggml_init_params gp = { ggml_tensor_overhead() * 800000 + ggml_graph_overhead_custom(8192, false) * 1600, nullptr, true };
    im.gctx = ggml_init(gp);
    im.allo_dec.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass2.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    if (!alloc_cols(im, im.NC, err)) return false;
    // the host mirrors are sized for the largest batch once (pinned registrations must not move)
    ggml_backend_t pin_be = im.sycl ? im.backend : nullptr;   // SYCL: pinned through the device's host buffer type
    im.host_fn.resize((size_t) (D * NCM), pin_be);
#ifdef DS41_HAVE_CUDA
    // pinned: the fn rows go device -> host -> the tier's device input every layer (pageable copies ran at ~2 GB/s)
    if (!im.cpu && !im.host_fn.empty()) ggml_backend_cuda_register_host_buffer(im.host_fn.data(), im.host_fn.size() * 4);
#endif
    im.host_ids.resize((size_t) (im.NU * NCM), pin_be);
    im.host_wts.resize((size_t) (im.NU * NCM), pin_be);
    if (im.emb_f16) {   // larry, STRATA_EMB_PIN: the table into pinned memory once (8 threads), the batch stage
        const double te0 = dprof_now();
        const size_t nrow = (size_t) im.g.vocab, rowb = (size_t) D * 2;
        im.emb_pin.resize(nrow * (size_t) D, pin_be);
        im.emb_stage.resize((size_t) D * (size_t) NCM, pin_be);
        if (im.emb_pin.buf && im.emb_stage.buf) {
            std::vector<std::thread> th;
            for (int k = 0; k < 8; ++k)
                th.emplace_back([&, k] {
                    const size_t r0 = nrow * (size_t) k / 8, r1 = nrow * (size_t) (k + 1) / 8;
                    std::memcpy(im.emb_pin.data() + r0 * (size_t) D, im.embd + r0 * rowb, (r1 - r0) * rowb);
                });
            for (auto & t : th) t.join();
            std::fprintf(stderr, "ds41_dense: token table pinned (%.2f GB, %.0f ms), batch embeddings staged f16\n",
                         nrow * rowb / 1e9, dprof_now() - te0);
        } else {
            std::fprintf(stderr, "ds41_dense: STRATA_EMB_PIN: pinned allocation failed - embeddings from the mmap (f16)\n");
            im.emb_pin.resize(0);
        }
    }
#ifdef DS41_HAVE_CUDA
    // pinned too: attn_router reads fn / ids / wts back with three async copies and one sync (truly async only from
    // pinned memory)
    if (!im.cpu) {
        ggml_backend_cuda_register_host_buffer(im.host_ids.data(), im.host_ids.size() * 4);
        ggml_backend_cuda_register_host_buffer(im.host_wts.data(), im.host_wts.size() * 4);
    }
#endif
    return true;
}

void Ds41Dense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ly[(size_t) il].rbias) return;
    ggml_backend_tensor_set(im.ly[(size_t) il].rbias, bias, 0, (size_t) im.NE * 4);
}

void Ds41Dense::set_oom_hook(std::function<bool()> hook) { p_->oom_hook = std::move(hook); }

void Ds41Dense::release_pass() {
    Impl & im = *p_;
    if (!im.allo_pass.a) return;
    ggml_gallocr_free(im.allo_pass.a);
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.graph_epoch.clear();
    im.allo_pass.compute_bufs.clear();
    ++im.allo_pass.epoch;
    if (im.allo_pass2.a) {   // larry: the pass ring's second allocator
        ggml_gallocr_free(im.allo_pass2.a);
        im.allo_pass2.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
        im.allo_pass2.graph_epoch.clear();
        im.allo_pass2.compute_bufs.clear();
        ++im.allo_pass2.epoch;
    }
}

// pass_cap (per ratio) of a pass of n tokens ending at pos0+n: the visible compressed rows it ranks over.
static void span_pass_caps(const Ds41Dense::Impl & im, int pos0, int n, int64_t out[3]) {
    out[0] = 0;
    for (int r = 1; r <= 2; ++r) {
        const int64_t nvis_last = (pos0 + n) / r;
        out[r] = nvis_last == 0 ? 0 : std::min<int64_t>(next_pow2(std::max<int64_t>(nvis_last, im.CBS)), im.cmax[r]);
    }
}

// STRATA_ENG_PROF: a prompt batch's Engram phases (start on the caller, the host wait, the rows' upload), per batch
static const bool g_eng_prof = std::getenv("STRATA_ENG_PROF") != nullptr;
static double eng_t_start = 0, eng_t_wait = 0, eng_t_wait1 = 0;   // larry: eng_t_wait1 = slot 1's wait
static int eng_t_n = 0;
static double eng_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// larry: the deferred next-batch prefetch (Ds41Dense::engram_prefetch), launched once this batch's own reads are done
static void eng_pf_launch(Ds41Dense::Impl & im) {
    if (im.eng_pf_pos0 < 0 || im.eng_pf) return;
    const int n = (int) im.eng_pf_tids.size();
    for (int t = 0; t < n; ++t)   // (larry, vision: an image position as DEAD)
        im.engram.set_token(im.eng_pf_pos0 + t, im.img_row((int64_t) im.eng_pf_pos0 + t) ? -1 : im.eng_pf_tids[(size_t) t]);
    std::string e2;
    im.eng_pf = im.engram.prefetch(im.eng_pf_pos0, n, e2);
    im.eng_pf_pos0 = -1;
    im.eng_pf_tids.clear();
}

// larry: the pinned rows of a prompt batch of `per` floats a slot (sized once for the widest batch); false = the
// pageable eng_rows path (off, not SYCL, or the allocation failed)
static bool eng_pin_ok(Ds41Dense::Impl & im, size_t per) {
    static const bool on = env_on("STRATA_ENG_PIN", true);
    const int ns = im.engram.n_slots();
    if (!on || !im.sycl || ns < 1 || ns > 2) return false;
    if (im.eng_pin_ev) ggml_backend_event_synchronize(im.eng_pin_ev);   // the previous batch's rows are up
    const size_t per_max = std::max(per, (size_t) im.cfg.prompt_batch * (size_t) (im.EC * im.ED));
    if (!im.eng_pin_buf || ggml_backend_buffer_get_size(im.eng_pin_buf) < (size_t) ns * per * sizeof(float)) {
        if (im.eng_pin_buf) { ggml_backend_buffer_free(im.eng_pin_buf); im.eng_pin_buf = nullptr; }
        ggml_backend_buffer_type_t ht = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(im.backend));
        if (!ht) return false;
        im.eng_pin_buf = ggml_backend_buft_alloc_buffer(ht, (size_t) ns * per_max * sizeof(float));
        if (!im.eng_pin_buf) { std::fprintf(stderr, "dense: Engram pinned rows (%.0f MB) not allocated - pageable\n", ns * per_max * 4 / 1e6); return false; }
        if (!im.eng_pin_ev) im.eng_pin_ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
        if (!im.eng_pin_ev) { ggml_backend_buffer_free(im.eng_pin_buf); im.eng_pin_buf = nullptr; return false; }
    }
    float * base = (float *) ggml_backend_buffer_get_base(im.eng_pin_buf);
    const size_t stride = ggml_backend_buffer_get_size(im.eng_pin_buf) / sizeof(float) / (size_t) ns;
    for (int e = 0; e < ns; ++e) im.eng_pin[e] = base + (size_t) e * stride;
    return true;
}

// Wait for the pass's Engram reads (started by upload_inputs) and push the rows into the state tensor's columns.  A
// no-op once done (eng_batch is null), so it is safe to call at every layer - but the caller gates it on L.eslot >= 0
// so the layers BELOW the first Engram layer run while the disk is still reading.  The whole pass's rows (both Engram
// slots, all n columns) go up in one shot, so the second Engram layer (14) is a plain read of the same tensor.
// larry: the Engram batch's host-side wait alone for one slot (no device work: a pipelined span runs it before the
// submit lock)
// larry, STRATA_ENG_START_ASYNC: the started batch from its thread (a no-op without one)
static bool eng_resolve(Ds41Dense::Impl & im) {
    if (!im.eng_fut.valid()) return true;
    im.eng_batch = im.eng_fut.get();
    if (!im.eng_batch) { im.err = im.eng_fut_err.empty() ? "engram: batch start failed" : im.eng_fut_err; im.eng_pin_n = 0; return false; }
    return true;
}

static bool engram_wait_host(Ds41Dense::Impl & im, int slot) {
    if (!eng_resolve(im)) return false;
    if (!im.eng_batch || slot < 0 || slot > 1 || im.eng_waited[slot] || im.eng_up[slot]) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.engram.wait_slot(im.eng_batch, slot, im.err)) { im.eng_batch.reset(); return false; }
    if (g_eng_prof) (slot ? eng_t_wait1 : eng_t_wait) += eng_now() - t0;
    im.eng_waited[slot] = true;
    return true;
}

// Slot `slot`'s rows up (after its reads): pinned rows async in queue order (before this layer's graph); the batch is
// done - and the next batch's prefetch launched - once every slot is up
static bool engram_ready(Ds41Dense::Impl & im, int slot) {
    if (!eng_resolve(im)) return false;
    if (!im.eng_batch) return true;
    const int ns = std::min(im.engram.n_slots(), 2);
    if (slot < 0 || slot >= ns || im.eng_up[slot]) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.eng_waited[slot] && !im.engram.wait_slot(im.eng_batch, slot, im.err)) { im.eng_batch.reset(); return false; }
    const double t1 = g_eng_prof ? eng_now() : 0;
    if (g_eng_prof) (slot ? eng_t_wait1 : eng_t_wait) += t1 - t0;
    im.eng_up[slot] = true;
    if (im.eng_pin_n) {
        ggml_backend_tensor_set_async(im.backend, im.i_eng[slot], im.eng_pin[slot], 0, im.eng_pin_n * sizeof(float));
        ggml_backend_event_record(im.eng_pin_ev, im.backend);
    } else if ((size_t) slot < im.eng_rows.size()) {
        im.up_set(im.i_eng[slot], im.eng_rows[(size_t) slot].data(), im.eng_rows[(size_t) slot].size() * 4);
    }
    bool all = true;
    for (int e = 0; e < ns; ++e) all = all && im.eng_up[e];
    if (all) {
        im.engram.wait(im.eng_batch, im.err);   // done already: the batch's read stats
        im.eng_batch.reset();
        for (int e = 0; e < 2; ++e) im.eng_waited[e] = im.eng_up[e] = false;
        im.eng_pin_n = 0;
        eng_pf_launch(im);
        if (g_eng_prof && eng_t_n > 64) {
            std::fprintf(stderr, "engram prompt batch: %d tokens, start %.1f ms, host wait %.1f ms (layer 1) + %.1f ms (layer 14), "
                         "upload %.1f ms\n", eng_t_n, eng_t_start, eng_t_wait, eng_t_wait1, eng_now() - t1);
            eng_t_start = eng_t_wait = eng_t_wait1 = 0;
            eng_t_n = 0;
        }
    }
    if (im.up_async) ggml_backend_event_record(im.up_ev, im.backend);   // the staging buffer's reuse waits for these too
    return true;
}

// every slot of a still-pending batch up (a new pass / an engine1 read flushes it first)
static bool engram_ready_all(Ds41Dense::Impl & im) {
    for (int e = 0; e < 2; ++e) if (!engram_ready(im, e)) return false;
    return true;
}

void Ds41Dense::Impl::up_set(ggml_tensor * t, const void * data, size_t size) {
    const size_t a = (up_off + 63) & ~(size_t) 63;
    if (!up_async || a + size > up_stage.size()) { ggml_backend_tensor_set(t, data, 0, size); return; }
    std::memcpy(up_stage.data() + a, data, size);
    ggml_backend_tensor_set_async(backend, t, up_stage.data() + a, 0, size);
    up_off = a + size;
}

// Per-token inputs of n tokens at positions pos0.. (embeddings, positions, the window / compressed / indexer index
// maps, the Engram rows and the per-pass ring slots).  Shared by begin() and begin_batch().
bool Ds41Dense::Impl::upload_inputs(const int * tids, int n, int pos0) {
    Impl & im = *this;
    im.up_async = im.sycl && n <= 8 && dense_async_on();
    if (im.up_async) {
        if (im.up_stage.empty()) im.up_stage.resize((size_t) 4 << 20, im.backend);
        if (!im.up_ev) im.up_ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
        else ggml_backend_event_synchronize(im.up_ev);   // the previous pass's staged copies have run
        im.up_off = 0;
        im.up_async = im.up_ev && !im.up_stage.empty();
    }
    im.span_memo_pos0 = -1;   // this writes i_widx / i_wslot / i_sidx / i_sslot: upload_span's memo is stale
    const int64_t D = im.D, SWA = im.SWA, RAW = im.RAW;
    // embeddings (F16 rows on the mmap)
    const double tu0 = g_dprof_on && n > 64 ? dprof_now() : 0;   // larry, STRATA_DENSE_PROF: the batch inputs' parts
    for (int t = 0; t < n; ++t)
        if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
    double tu1 = 0;
    if (im.emb_f16) {   // larry, STRATA_EMB_PIN: the rows as they are (f16), from the pinned table
        const uint16_t * tab = !im.emb_pin.empty() ? im.emb_pin.data() : reinterpret_cast<const uint16_t *>(im.embd);
        const size_t rowb = (size_t) D * 2;
        if (n > 8 && (size_t) n * (size_t) D <= im.emb_stage.size()) {
            if (im.emb_ev) ggml_backend_event_synchronize(im.emb_ev);   // the previous batch's upload has read the stage
            else im.emb_ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
            uint16_t * st = im.emb_stage.data();
            const int nt = n >= 1024 ? 8 : 1;
            std::vector<std::thread> th;
            auto part = [&](int k) {
                for (int t = n * k / nt; t < n * (k + 1) / nt; ++t)
                    std::memcpy(st + (size_t) t * (size_t) D, tab + (size_t) tids[t] * (size_t) D, rowb);
            };
            for (int k = 1; k < nt; ++k) th.emplace_back(part, k);
            part(0);
            for (auto & t : th) t.join();
            if (!im.img.empty())   // larry (vision): an image position's row is the encoder's, not the token table's
                for (int t = 0; t < n; ++t)
                    if (const float * r = im.img_row((int64_t) pos0 + t))
                        for (int64_t i = 0; i < D; ++i) st[(size_t) t * (size_t) D + (size_t) i] = ggml_fp32_to_fp16(r[i]);
            tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
            ggml_backend_tensor_set_async(im.backend, im.i_emb, st, 0, (size_t) n * rowb);
            if (im.emb_ev) ggml_backend_event_record(im.emb_ev, im.backend);
            else ggml_backend_synchronize(im.backend);
        } else {
            std::vector<uint16_t> rows((size_t) n * (size_t) D);
            for (int t = 0; t < n; ++t) std::memcpy(rows.data() + (size_t) t * (size_t) D, tab + (size_t) tids[t] * (size_t) D, rowb);
            for (int t = 0; t < n && !im.img.empty(); ++t)   // larry (vision)
                if (const float * r = im.img_row((int64_t) pos0 + t))
                    for (int64_t i = 0; i < D; ++i) rows[(size_t) t * (size_t) D + (size_t) i] = ggml_fp32_to_fp16(r[i]);
            im.up_set(im.i_emb, rows.data(), rows.size() * 2);
        }
    } else {
    im.host_emb.resize((size_t) (D * n));
    for (int t = 0; t < n; ++t) {
        const uint16_t * row = reinterpret_cast<const uint16_t *>(im.embd) + (size_t) tids[t] * (size_t) D;
        for (int64_t i = 0; i < D; ++i) im.host_emb[(size_t) (t * D + i)] = ggml_fp16_to_fp32(row[i]);
    }
    for (int t = 0; t < n && !im.img.empty(); ++t)   // larry (vision)
        if (const float * r = im.img_row((int64_t) pos0 + t)) std::memcpy(&im.host_emb[(size_t) (t * D)], r, (size_t) D * 4);
    tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
    im.up_set(im.i_emb, im.host_emb.data(), im.host_emb.size() * 4);
    }
    const double tu2 = g_dprof_on && n > 64 ? dprof_now() : 0;
    std::vector<int32_t> pos(n);
    for (int t = 0; t < n; ++t) pos[t] = pos0 + t;
    im.up_set(im.i_pos, pos.data(), (size_t) n * 4);
    if (im.img_in(pos0, n)) {   // larry (vision): these columns' image mask
        std::vector<float> m((size_t) n);
        for (int t = 0; t < n; ++t) m[(size_t) t] = im.img_row((int64_t) pos0 + t) ? 1.0f : 0.0f;
        im.up_set(im.i_img, m.data(), (size_t) n * 4);
    }
    // window: per query the SWA positions [p - SWA + 1, p] as rows of [ring (RAW) | pass rows]
    std::vector<int32_t> widx((size_t) (SWA * n));
    std::vector<float> wm((size_t) (SWA * n));
    for (int t = 0; t < n; ++t) {
        const int64_t p = pos0 + t;
        for (int64_t j = 0; j < SWA; ++j) {
            const int64_t q = p - SWA + 1 + j;
            int32_t row = 0;
            float m = 0.0f;
            if (q < 0) m = NEG_INF;
            else if (q >= pos0) row = (int32_t) (RAW + (q - pos0));
            else row = (int32_t) (q % RAW);
            widx[(size_t) (t * SWA + j)] = row;
            wm[(size_t) (t * SWA + j)] = m;
        }
    }
    im.up_set(im.i_widx, widx.data(), widx.size() * 4);
    im.up_set(im.i_wmask, wm.data(), wm.size() * 4);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pos0 + n - k + j) % RAW);
        im.up_set(im.i_wslot, ws.data(), ws.size() * 4);
    }
    // compressed rows per ratio
    for (int r = 1; r <= 2; ++r) {
        std::vector<float> nv(n);
        std::vector<int32_t> cs(n), cp(n);
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t;
            nv[t] = (float) ((p + 1) / r);
            const bool done = (p + 1) % r == 0;
            cs[t] = (int32_t) (done ? p / r : im.cmax[r]);   // an open group writes the spare row
            cp[t] = (int32_t) ((p / r) * r);
        }
        im.up_set(im.i_nvis[r], nv.data(), (size_t) n * 4);
        im.up_set(im.i_cslot[r], cs.data(), (size_t) n * 4);
        im.up_set(im.i_cpos[r], cp.data(), (size_t) n * 4);
        if (r == 1) {
            std::vector<float> lb(n);
            for (int t = 0; t < n; ++t) lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / im.CBS : -1);
            im.up_set(im.i_lastblk, lb.data(), (size_t) n * 4);
        }
    }
    {   // ratio-2 state: rows of [ring (2) | pass states] for each token's group (2g, 2g+1); open groups repeat 2g
        std::vector<int32_t> si((size_t) (2 * n)), ss;
        auto row = [&](int64_t q) { return (int32_t) (q >= pos0 ? 2 + (q - pos0) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        const int64_t k = std::min<int64_t>(n, 2);
        for (int64_t j = 0; j < k; ++j) ss.push_back((int32_t) ((pos0 + n - k + j) % 2));
        im.up_set(im.i_sidx, si.data(), si.size() * 4);
        im.up_set(im.i_sslot, ss.data(), ss.size() * 4);
    }
    const double tu3 = g_dprof_on && n > 64 ? dprof_now() : 0;
    // Engram rows: hash the whole pass now and START the (dedup/sort/coalesce) reads; they are waited - and the rows
    // moved to the state tensor - by the first Engram layer's span (engram_ready), so the disk read overlaps the
    // embeddings upload, the init graph and the layers below it.  A tiny batch (decode) still runs on the caller.
    if (im.engram.enabled()) {
        if (!engram_ready_all(im)) return false;    // flush a still-pending batch from a previous pass first
        im.eng_pf_pos0 = -1;   // larry: a prefetch request never launched is dropped (this pass reads its own rows)
        im.eng_pf_tids.clear();
        if (im.eng_pf) {   // larry: the prefetched rows are in the cache once it is done (start() reads the cache)
            std::string e2;
            im.engram.wait(im.eng_pf, e2);   // best effort: a failed read is read again below
            im.eng_pf.reset();
        }
        for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, im.img_row((int64_t) pos0 + t) ? -1 : tids[t]);   // (larry, vision)
        static const bool eng_async = env_on("STRATA_ENG_ASYNC", true);
        if (n <= 8 && im.up_async && eng_async) {   // larry: decode / verify - start the reads (worker threads) now and
            // wait at the first Engram layer (engram_ready); layer 0 runs meanwhile.  Synchronous, they cost ~0.66 ms a token
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            for (int e = 0; e < 2; ++e) im.eng_waited[e] = im.eng_up[e] = false;
            if (!im.eng_batch) return false;
        } else if (n <= 64) {   // decode / verify: the rows are needed at layer 1 - read them now (synchronously)
            static const bool eprof = std::getenv("STRATA_ENG_PROF") != nullptr;
            static double eng_ms = 0; static long eng_n = 0;
            const auto te = std::chrono::steady_clock::now();
            if (!im.engram.rows(pos0, n, im.eng_rows, im.err)) return false;
            if (eprof) {
                eng_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
                if (++eng_n % 64 == 0) std::fprintf(stderr, "engram decode reads: %.3f ms/call over %ld calls\n", eng_ms / eng_n, eng_n);
            }
            for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
                im.up_set(im.i_eng[e], im.eng_rows[e].data(), im.eng_rows[e].size() * 4);
        } else {
            const double t0 = g_eng_prof ? eng_now() : 0;
            const size_t per = (size_t) n * (size_t) (im.EC * im.ED);
            static const bool start_async = env_on("STRATA_ENG_START_ASYNC", false);
            if (start_async && eng_pin_ok(im, per)) {   // larry: the start on a thread, resolved at the first wait
                im.eng_pin_n = per;
                im.eng_fut_err.clear();
                float * const * outp = im.eng_pin;
                im.eng_fut = std::async(std::launch::async, [&im, pos0, n, outp]() {
                    return im.engram.start_into(pos0, n, outp, im.eng_fut_err);
                });
            } else if (eng_pin_ok(im, per)) {
                im.eng_batch = im.engram.start_into(pos0, n, im.eng_pin, im.err);
                im.eng_pin_n = im.eng_batch ? per : 0;
            } else {
                im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            }
            if (g_eng_prof) { eng_t_start = eng_now() - t0; eng_t_wait = eng_t_wait1 = 0; eng_t_n = n; }
            for (int e = 0; e < 2; ++e) im.eng_waited[e] = im.eng_up[e] = false;
            if (!im.eng_batch && !im.eng_fut.valid()) return false;
        }
    }
    if (im.up_async) ggml_backend_event_record(im.up_ev, im.backend);   // the staging buffer's reuse waits for it
    if (g_dprof_on && n > 64)
        std::fprintf(stderr, "dense prof: begin_batch inputs (%d tokens): embeddings %.1f ms, their upload %.1f ms, maps %.1f ms, "
                     "Engram start %.1f ms\n", n, tu1 - tu0, tu2 - tu1, tu3 - tu2, dprof_now() - tu3);
    return true;
}

// larry: the kept graphs (per layer, cap and pass width) only ever accumulate in gctx; when it is more than 60 % used
// (a long serve session: verify passes of several widths across many cap buckets) they all go and are rebuilt lazily -
// at a pass's start, when no graph of the previous one is held
static void reset_kept_graphs(Ds41Dense::Impl & im) {
    ggml_backend_synchronize(im.backend);
    im.graphs.clear();
    im.logits_out.clear();
    im.predict_out.clear();
    im.predict_out_n.clear();
    im.fpredict_out.clear();
    im.fpredict_ahead.clear();
    for (Allo * al : { &im.allo_dec, &im.allo_pass, &im.allo_pass2 }) {
        al->graph_epoch.clear();
        al->compute_bufs.clear();
        ++al->epoch;
    }
    const size_t sz = ggml_get_mem_size(im.gctx);
    ggml_free(im.gctx);
    ggml_init_params gp = { sz, nullptr, true };
    im.gctx = ggml_init(gp);
    std::fprintf(stderr, "ds41_dense: kept graphs reset (their context was full)\n");
}

bool Ds41Dense::begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    im.fin_pending = false;
    if (im.gctx && ggml_used_mem(im.gctx) > ggml_get_mem_size(im.gctx) / 10 * 6) reset_kept_graphs(im);
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    im.pos0 = pos0;
    im.n = n;
    if (!im.upload_inputs(tids, n, pos0)) return false;
    im.wrote(pos0, pos0 + n);
    span_pass_caps(im, pos0, n, im.pass_cap);
    // embeddings -> state
    ggml_context * gc = nullptr;
    ggml_cgraph * gf = nullptr;
    static const bool pass_cache = [] { const char * v = std::getenv("STRATA_PASS_CACHE"); return !(v && v[0] == '0'); }();
    const bool cached = n == 1 || (pass_cache && n <= 8);
    std::vector<uint8_t> meta;
    if (cached) {
        const Impl::GKey key { -1, 0, 0, (int64_t) n, im.seq };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_init(im, im.gctx, n)).first;
        gf = it->second;
    } else {
        meta.resize(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_init(im, gc, n);
    }
    const bool ok = im.compute(gf, cached, im.sycl);
    if (gc) ggml_free(gc);
    return ok;
}

// ================================================================== the layer-major prompt batch

// larry (vision, patch-vision-engine.py)
void Ds41Dense::set_images(const std::vector<ImageSpan> & spans) { p_->img = spans; }
void Ds41Dense::clear_images() { p_->img.clear(); }
void Ds41Dense::engram_tokens(int64_t pos0, const int * tids, int n) {
    Impl & im = *p_;
    if (!im.engram.enabled()) return;
    for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, im.img_row(pos0 + t) ? -1 : tids[t]);
}

void Ds41Dense::engram_prefetch(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    static const bool on = env_on("STRATA_ENG_PREFETCH", true);
    if (!on || !im.engram.enabled() || n <= 64 || im.eng_pf || pos0 < 0 || pos0 + n > im.cfg.ctx_max) return;
    // launched once this batch's own reads are done (eng_pf_launch); the tokens are set then (that batch's own
    // inputs rewrite them)
    im.eng_pf_tids.assign(tids, tids + n);
    im.eng_pf_pos0 = pos0;
    if (!im.eng_batch && !im.eng_fut.valid()) eng_pf_launch(im);   // larry: not under a batch still starting
}

bool Ds41Dense::begin_batch(const int * tids, int B, int pos0) {
    Impl & im = *p_;
    if (B < 1 || B > im.cfg.prompt_batch || B > im.NC || pos0 < 0 || pos0 + B > im.cfg.ctx_max) { im.err = "bad batch"; return false; }
    if (!im.upload_inputs(tids, B, pos0)) return false;
    im.wrote(pos0, pos0 + B);
    im.pos0 = pos0; im.batch_pos0 = pos0; im.batch = B; im.n = B; im.col0 = 0;
    // per-sub-batch pass_cap (the compressed rows each sub-batch ranks over) - depends only on (pos0, len)
    const int64_t nb = (B + im.S - 1) / im.S;
    im.span_cap.assign((size_t) nb, { 0, 0, 0 });
    for (int64_t i = 0; i < nb; ++i)
        span_pass_caps(im, pos0 + (int) (i * im.S), (int) std::min<int64_t>(im.S, (int64_t) B - i * im.S),
                       im.span_cap[(size_t) i].data());
    // the batch's initial state, one sub-batch at a time (x_state = embd, pre_state = one-hot)
    const double ti0 = g_dprof_on ? dprof_now() : 0;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
    for (int64_t i = 0; i < nb; ++i) {
        const int64_t c0 = i * im.S, sn = std::min<int64_t>(im.S, (int64_t) B - c0);
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        ggml_cgraph * gf = build_init(im, gc, sn, c0);
        const bool ok = im.compute(gf, false);
        ggml_free(gc);
        if (!ok) { im.err = "batch init failed"; return false; }
    }
    if (g_dprof_on) std::fprintf(stderr, "dense prof: begin_batch init graphs (%lld spans) %.1f ms\n", (long long) nb, dprof_now() - ti0);
    return true;
}

// The per-span inputs of one layer-major sub-batch: the sub-batch's own pass_cap (the graphs read it) and the
// PASS-LOCAL row maps (the pass's own rows / states sit at the end of the [ring | pass rows] they index, so they are
// recomputed from the sub-batch's own position - upload_inputs laid them out for the whole batch, correct only at
// col0 = 0).  Shared by attn_router_span and attn_router_span_kv.
static bool upload_span(Ds41Dense::Impl & im, int col0, int n) {
    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const size_t si = (size_t) (col0 / im.S);
    if (si >= im.span_cap.size()) { im.err = "bad span"; return false; }
    im.pass_cap[0] = im.span_cap[si][0];
    im.pass_cap[1] = im.span_cap[si][1];
    im.pass_cap[2] = im.span_cap[si][2];
    const int64_t pbase = (int64_t) im.batch_pos0 + col0;
    static const bool memo_on = !(std::getenv("STRATA_SPAN_MEMO") && std::getenv("STRATA_SPAN_MEMO")[0] == '0');
    if (memo_on && im.span_memo_pos0 == (int64_t) im.batch_pos0 && im.span_memo_col0 == col0 && im.span_memo_n == n) return true;
    im.span_memo_pos0 = (int64_t) im.batch_pos0;
    im.span_memo_col0 = col0;
    im.span_memo_n = n;
    // larry: with a submit lock (pipelined batch) the maps go up through a pinned buffer, async - a pageable
    // ggml_backend_tensor_set is a blocking copy, and it held the lock ~2 ms a span under the tier's PCIe traffic.
    // The span ends with a full synchronize, so the buffer is free again by the next span.
    // pass ring: half pass_k (each half sized for S columns; resized only once, after a full wait)
    const size_t half = ((size_t) (im.SWA * im.S + 2 * im.S + im.RAW + 2) * 4 + 4 * 64 + 63) & ~(size_t) 63;
    const bool staged = im.sub_mu && !im.cpu;
    if (staged && im.span_stage.size() < 2 * half) { ggml_backend_synchronize(im.backend); im.span_stage.resize(2 * half, im.backend); }
    size_t soff = im.pass_ring ? (size_t) im.pass_k * half : 0;
    auto put = [&](ggml_tensor * t, const void * data, size_t off, size_t size) {
        if (!staged || (im.pass_ring ? (size_t) (im.pass_k + 1) * half : im.span_stage.size()) < soff + size) {
            ggml_backend_tensor_set(t, data, off, size);
            return;
        }
        std::memcpy(im.span_stage.data() + soff, data, size);
        const double tq = g_dprof_on ? dprof_now() : 0;
        ggml_backend_tensor_set_async(im.backend, t, im.span_stage.data() + soff, off, size);
        if (g_dprof_on) {   // larry (diagnostic): a slow async upload
            static int nslow = 0;
            const double dt = dprof_now() - tq;
            if (dt > 1.0 && nslow++ < 12) std::fprintf(stderr, "dense prof: tensor_set_async %s %zu bytes took %.2f ms (ring %d k %d)\n", t->name, size, dt, (int) im.pass_ring, im.pass_k);
        }
        soff = (soff + size + 63) & ~(size_t) 63;
    };
    {
        std::vector<int32_t> widx((size_t) (im.SWA * n));
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t;
            for (int64_t j = 0; j < im.SWA; ++j) {
                const int64_t q = p - im.SWA + 1 + j;
                widx[(size_t) (t * im.SWA + j)] = q < 0 ? 0 : (int32_t) (q >= pbase ? im.RAW + (q - pbase) : q % im.RAW);
            }
        }
        put(im.i_widx, widx.data(), (size_t) col0 * im.SWA * 4, widx.size() * 4);
    }
    {
        std::vector<int32_t> si((size_t) (2 * n));
        auto row = [&](int64_t q) { return (int32_t) (q >= pbase ? 2 + (q - pbase) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        put(im.i_sidx, si.data(), (size_t) col0 * 2 * 4, si.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pbase + n - k + j) % im.RAW);
        put(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, 2);
        std::vector<int32_t> ss((size_t) k);
        for (int64_t j = 0; j < k; ++j) ss[(size_t) j] = (int32_t) ((pbase + n - k + j) % 2);
        put(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    return true;
}

// larry, pass ring: the next allocator / staging half - after the graph that used it last (2 spans back) is done -
// and, under the submit lock after the span's last enqueue, the event that marks this span's end
static void ring_take(Ds41Dense::Impl & im) {
    if (!im.pass_ring) return;
    const int k = im.pass_turn;
    im.pass_turn ^= 1;
    im.pass_k = k;
    if (!im.pass_ev[k]) im.pass_ev[k] = ggml_backend_event_new(ggml_backend_get_device(im.backend));
    else if (im.pass_ev_rec[k]) ggml_backend_event_synchronize(im.pass_ev[k]);
    im.pass_ev_rec[k] = false;
}
static void ring_mark(Ds41Dense::Impl & im) {
    if (!im.pass_ring || !im.pass_ev[im.pass_k]) return;
    ggml_backend_event_record(im.pass_ev[im.pass_k], im.backend);
    im.pass_ev_rec[im.pass_k] = true;
}

void Ds41Dense::set_pass_ring(bool on) {
    Impl & im = *p_;
    ggml_backend_synchronize(im.backend);
    im.pass_ring = on && !im.cpu && im.sub_mu;
    im.pass_turn = 0;
    im.pass_ev_rec[0] = im.pass_ev_rec[1] = false;
}
void Ds41Dense::pass_sync() { ggml_backend_synchronize(p_->backend); }

namespace {
// larry, engine1 (set_page_hooks): layer il's paged weights mapped while a call queues work on them - need() before
// the submit lock (it may wait for a slot's earlier work), touch() once the call's work is queued
struct PageScope {
    Ds41Dense::Impl & im;
    int il;
    PageScope(Ds41Dense::Impl & i, int l) : im(i), il(l) { if (im.page_need) im.page_need(il); }
    ~PageScope() { if (im.page_touch) im.page_touch(il); }
};
}  // namespace

std::vector<Ds41DensePaged> Ds41Dense::paged_weights() const {
    const Impl & im = *p_;
    std::vector<Ds41DensePaged> v;
    for (size_t il = 0; il < im.paged_lay.size(); ++il)
        for (auto & e : im.paged_lay[il]) {
            ggml_tensor * t = e.first;
            const std::string n = ggml_get_name(t);
            if (n.find(".ffn_up_shexp.weight") != std::string::npos) continue;   // with its gate, as the fused weight
            Ds41DensePaged d;
            d.layer = (int) il;
            d.data = t->data;
            d.K = t->ne[0];
            d.N = t->ne[1];
            d.bytes = ggml_nbytes(t);
            d.name = n;
            if (n.find(".ffn_gate_shexp.weight") != std::string::npos) {
                d.name = "blk." + std::to_string(il) + ".ffn_gate_up_shexp.weight";
                d.N *= 2;
                d.bytes *= 2;
            }
            v.push_back(d);
        }
    return v;
}

void Ds41Dense::set_page_hooks(std::function<void(int)> need, std::function<void(int)> touch) {
    p_->page_need = std::move(need);
    p_->page_touch = std::move(touch);
}

bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;
    PageScope ps(im, il);
    if (im.sub_mu && im.ly[(size_t) il].eslot >= 0 && !engram_wait_host(im, (int) im.ly[(size_t) il].eslot)) return false;
    ring_take(im);
    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) { if (im.sub_prio) im.sub_prio->fetch_add(1); sl = std::unique_lock<std::mutex>(*im.sub_mu); if (im.sub_prio) im.sub_prio->fetch_sub(1); }
    if (!upload_span(im, col0, n)) return false;
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;   // the Engram add feeds this layer's K/V
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params pk = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gck = ggml_init(pk);
    ggml_cgraph * gk = build_attn_kv_only(im, gck, il, n, col0);
    const bool okk = im.compute(gk, false, false, sl.owns_lock());
    ggml_free(gck);
    ring_mark(im);
    if (sl.owns_lock()) { sl.unlock(); if (!im.pass_ring) ggml_backend_synchronize(im.backend); }
    return okk;
}

bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    const double tptl = ds41_dense_ptl_hook ? dprof_now() : 0;   // larry, STRATA_GPU_PTL
    PageScope ps(im, il);
    if (im.sub_mu && im.ly[(size_t) il].eslot >= 0 && !engram_wait_host(im, (int) im.ly[(size_t) il].eslot)) return false;
    ring_take(im);
    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) {
        const double tl = g_dprof_on ? dprof_now() : 0;
        { if (im.sub_prio) im.sub_prio->fetch_add(1); sl = std::unique_lock<std::mutex>(*im.sub_mu); if (im.sub_prio) im.sub_prio->fetch_sub(1); }
        if (g_dprof_on) g_dlock += dprof_now() - tl;
    }
    double tp = g_dprof_on ? dprof_now() : 0;
    auto lap = [&](int i) { if (g_dprof_on) { const double t = dprof_now(); g_dprof[i] += t - tp; tp = t; } };
    if (!upload_span(im, col0, n)) return false;
    if (g_dprof_on && dprof_now() - tp > 50) std::fprintf(stderr, "dense prof: slow upload_span layer %d col0 %d n %d: %.0f ms\n", il, col0, n, dprof_now() - tp);
    lap(6);
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;   // wait the slot's Engram reads here
    lap(3);
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_attn(im, gc, il, n, col0);
    lap(0);
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false, false, sl.owns_lock());
    g_dprof_slot = -1;
    if (g_dprof_on) { tp = dprof_now(); g_dprof[7] += 1; }
    ggml_free(gc);
    if (!ok) return false;
    // the sub-batch's ids / wts / fn land at its own batch columns (token-major: token col0+t at t * NU / t * D)
    if (im.fn_host) ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data() + (size_t) col0 * im.D, (size_t) col0 * im.D * 4, (size_t) n * im.D * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ring_mark(im);
    if (ds41_dense_ptl_hook) ds41_dense_ptl_hook(1, il, col0, tptl);   // larry, STRATA_GPU_PTL
    if (sl.owns_lock()) sl.unlock();
    if (!im.pass_ring) ggml_backend_synchronize(im.backend);
    lap(4);
    return true;
}

void Ds41Dense::set_pre_wait_hook(std::function<void(int, int)> h) { p_->pre_wait = std::move(h); }
void Ds41Dense::set_route_hook(std::function<void(int, int, const void *, const void *, const void *, const float *)> h) { p_->route_hook = std::move(h); }
void Ds41Dense::set_oproj_hook(std::function<bool(int, int, const ggml_tensor *, const ggml_tensor *, const float *, float *)> h) {
    p_->oproj_hook = std::move(h);
}
void Ds41Dense::set_submit_lock(std::mutex * m, std::atomic<int> * prio) { p_->sub_mu = m; p_->sub_prio = m ? prio : nullptr; }
void Ds41Dense::set_shexp_skip(bool on) { p_->shexp_skip = on; }   // larry, STRATA_SHEXP_TIER

bool Ds41Dense::upload_routed(const float * routed_host, int B) {
    Impl & im = *p_;
    if (!routed_host || B < 1 || B > im.batch) { im.err = "bad routed batch"; return false; }
    ggml_backend_tensor_set(im.routed, routed_host, 0, (size_t) B * im.D * 4);
    return true;
}

float * Ds41Dense::routed_dev() { return p_->cpu || !p_->routed ? nullptr : (float *) p_->routed->data; }

bool Ds41Dense::finish_span(int il, int col0, int n) {
    Impl & im = *p_;
    (void) il;
    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const double tptl = ds41_dense_ptl_hook ? dprof_now() : 0;   // larry, STRATA_GPU_PTL
    ring_take(im);
    const double tf0 = g_dprof_on ? dprof_now() : 0;
    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) {
        { if (im.sub_prio) im.sub_prio->fetch_add(1); sl = std::unique_lock<std::mutex>(*im.sub_mu); if (im.sub_prio) im.sub_prio->fetch_sub(1); }
        if (g_dprof_on) g_dlock += dprof_now() - tf0;
    }
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_finish(im, gc, n, col0);
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false, false, sl.owns_lock());
    g_dprof_slot = -1;
    ggml_free(gc);
    ring_mark(im);
    if (ds41_dense_ptl_hook) ds41_dense_ptl_hook(2, il, col0, tptl);   // larry, STRATA_GPU_PTL
    if (sl.owns_lock()) { sl.unlock(); if (!im.pass_ring) ggml_backend_synchronize(im.backend); }
    if (g_dprof_on) g_dfin += dprof_now() - tf0;
    return ok;
}

void Ds41Dense::prof_report(const char * tag) {
    if (!g_dprof_on) return;
    std::fprintf(stderr, "dense prof %s: %d attn spans: build %.0f ms, alloc %.0f, gpu compute %.0f, engram wait %.0f, "
                 "readback %.0f, upload %.0f, finish build %.0f\n", tag, (int) g_dprof[7], g_dprof[0], g_dprof[1], g_dprof[2],
                 g_dprof[3], g_dprof[4], g_dprof[6], g_dprof[5]);
    std::fprintf(stderr, "dense prof %s: finish spans %.0f ms, submit-lock waits %.0f ms\n", tag, g_dfin, g_dlock);
    for (double & v : g_dprof) v = 0;
    g_dfin = g_dlock = 0;
}

int Ds41Dense::batch_size() const { return (int) p_->batch; }
const int32_t * Ds41Dense::batch_ids() const { return p_->host_ids.data(); }
const float * Ds41Dense::batch_wts() const { return p_->host_wts.data(); }
const float * Ds41Dense::batch_fn() const { return p_->host_fn.data(); }
const float * Ds41Dense::batch_fn_dev() const { return p_->cpu || !p_->o_fn ? nullptr : (const float *) p_->o_fn->data; }
void Ds41Dense::set_fn_host(bool on) { p_->fn_host = on; }

bool Ds41Dense::attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev) {
    Impl & im = *p_;
    PageScope ps(im, il);
    const int64_t n = im.n;
    // Engram layer: wait the pass's reads (started in upload_inputs) before the graph that reads i_eng runs.  For the
    // chunk path this is the first Engram layer too, so the layers below it ran while the disk was reading.
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;
    const int64_t r = im.ly[(size_t) il].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    // larry: verify passes (n <= 8) keep their graphs too, one per (layer, cap, width) - built and allocated once, not
    // per layer per pass (STRATA_PASS_CACHE=0: rebuilt every call, as before)
    static const bool pass_cache = [] { const char * v = std::getenv("STRATA_PASS_CACHE"); return !(v && v[0] == '0'); }();
    const bool cached = n == 1 || (pass_cache && n <= 8);
    // larry: decode runs the shared expert as its own graph after the readback (STRATA_SHEXP_INLINE=1: in the attention graph)
    static const bool split_shexp = std::getenv("STRATA_SHEXP_INLINE") == nullptr;
    // larry, STRATA_TP_OPROJ=2: the kept graph in two halves, the output projection queued between them (oproj_hook)
    static const int tp_mode = std::getenv("STRATA_TP_OPROJ") ? std::atoi(std::getenv("STRATA_TP_OPROJ")) : 0;
    static const bool hc_fuse = env_on("STRATA_HC_FUSE", true);
    static const int64_t hc_maxn = std::getenv("STRATA_HC_FUSE_MAXN") ? std::atoll(std::getenv("STRATA_HC_FUSE_MAXN")) : 8;
    const bool halves = cached && tp_mode == 2 && im.sycl && im.oproj_hook && hc_fuse && n <= hc_maxn && n <= im.o_att->ne[1];
    ggml_cgraph * gf2 = nullptr;
    if (halves) {
        const Impl::GKey ka { il, 11, cap, (int64_t) n, im.seq }, kb { il, 12, cap, (int64_t) n, im.seq };
        auto ia = im.graphs.find(ka);
        if (ia == im.graphs.end()) ia = im.graphs.emplace(ka, build_attn(im, im.gctx, il, n, 0, !split_shexp, 2048, true, 1)).first;
        auto ib = im.graphs.find(kb);
        if (ib == im.graphs.end()) ib = im.graphs.emplace(kb, build_attn(im, im.gctx, il, n, 0, !split_shexp, 2048, true, 2)).first;
        gf = ia->second;
        gf2 = ib->second;
    } else if (cached) {
        const Impl::GKey key { il, im.fin_pending ? 13 : 1, cap, (int64_t) n, im.seq };
        auto it = im.graphs.find(key);
        // a kept graph is a few hundred nodes: room for 2048 (the gctx holds hundreds of them: layers x caps x widths)
        if (it == im.graphs.end())
            it = im.graphs.emplace(key, build_attn(im, im.gctx, il, n, 0, !split_shexp, 2048, true, 0, im.fin_pending)).first;
        gf = it->second;
    } else {
        im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
        ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
        gc = ggml_init(p);
        gf = build_attn(im, gc, il, n, 0, true, 8192, true);
    }
    // async (kept graphs): the readbacks get their own event and the shared expert is queued BEFORE the host waits,
    // so the host's wait ends when the router's rows land, not when this card is idle
    const bool async = cached && dense_async_on() && im.sycl;
    if (async && !im.rb_ev) im.rb_ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
    const bool tl = ds41_dense_tl_hook && async;
    const int tlo = n == 1 ? 0 : 8;   // timeline ids: decode 0.., verify passes 8..
    if (tl) ds41_dense_tl_hook(tlo + 0);
    const double tp0 = g_decprof_on ? dprof_now() : 0;
    GGML_ASSERT(!im.fin_pending || (cached && !halves));
    bool ok = im.compute(gf, cached, async);
    if (ok) im.fin_pending = false;
    if (ok && gf2) {
        ok = im.oproj_hook(il, (int) n, im.BL(il, "attn_output_a.weight"), im.BL(il, "attn_output_b.weight"),
                           (const float *) im.o_att->data, (float *) im.o_proj->data);
        if (!ok) im.err = "attention output projection failed";
        ok = ok && im.compute(gf2, cached, async);
    }
    if (gc) ggml_free(gc);
    if (!ok) return false;
    const double tp1 = g_decprof_on ? dprof_now() : 0;
    // one sync for the three readbacks (was one per tensor_get: 3 x 40 stream syncs per decode token)
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    // larry, STRATA_XE_GROUTE: the tier's route kernel behind the fn readback (the expert cards read that copy)
    if (async && im.rb_ev && im.route_hook) im.route_hook(il, (int) n, im.o_ids->data, im.o_wts->data, im.o_fn->data, im.host_fn.data());
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * n) * 4);
    if (async && im.rb_ev) ggml_backend_event_record(im.rb_ev, im.backend);
    if (tl) ds41_dense_tl_hook(tlo + 1);
    double ts = 0;
    auto shexp = [&]() -> bool {   // queued now: it runs on this card while the tier's cards run the routed experts
        if (!(cached && split_shexp)) return true;
        const Impl::GKey key { il, 7, 0, (int64_t) n, im.seq };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_shexp(im, im.gctx, il, n)).first;
        const double ts0 = g_decprof_on ? dprof_now() : 0;
        if (!im.compute(it->second, true, async)) return false;
        // STRATA_DEBUG_SHEXP_TWICE=1 (timing only; the output is right): the shared expert twice - its marginal cost
        static const bool twice = env_on("STRATA_DEBUG_SHEXP_TWICE", false);
        if (twice && !im.compute(it->second, true, async)) return false;
        if (g_decprof_on) ts += dprof_now() - ts0;
        return true;
    };
    if (async && im.rb_ev && !shexp()) return false;
    if (tl) ds41_dense_tl_hook(tlo + 2);
    const double tp2 = g_decprof_on ? dprof_now() : 0;
    if (async && im.rb_ev && im.pre_wait) im.pre_wait(il, (int) n);   // larry: the tier arms its cards meanwhile
    if (async && im.rb_ev) ggml_backend_event_synchronize(im.rb_ev);
    else ggml_backend_synchronize(im.backend);
    const double tp3 = g_decprof_on ? dprof_now() : 0;
    std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * n) * 4);
    std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * n) * 4);
    *fn_host = im.host_fn.data();
    if (fn_dev) *fn_dev = im.o_fn;
    if (!(async && im.rb_ev) && !shexp()) return false;
    if (g_decprof_on && n == 1) {
        g_decprof[0] += tp1 - tp0; g_decprof[1] += tp2 - tp1 - ts; g_decprof[2] += tp3 - tp2; g_decprof[3] += ts;
        if ((long) ++g_decprof[5] % 2560 == 0) decprof_report();
    }
    return true;
}

bool Ds41Dense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (!fin_flush(im)) return false;
    if (im.n != 1 || n_top < 1 || n_top > 16) return false;
    const Impl::GKey key { il, 3, 0, 1, im.seq };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out[il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    ggml_backend_tensor_get(im.predict_out[il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::predict_multi(int il, int * top_ids) {
    Impl & im = *p_;
    if (!fin_flush(im)) return false;
    const int64_t n = im.n;
    if (n < 1) return false;
    if (n == 1) return predict(il, top_ids, 16);
    const Impl::GKey key { il, 3, 0, (int) n };   // one graph per (layer, pass width): verify passes reuse a few widths
    auto it = im.graphs.find(key);
    ggml_tensor * o = nullptr;
    if (it == im.graphs.end()) {
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o, n);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out_n[{ il, (int) n }] = o;
    }
    o = im.predict_out_n[{ il, (int) n }];
    if (!im.compute(it->second, true)) return false;
    ggml_backend_tensor_get(o, top_ids, 0, sizeof(int32_t) * 16 * (size_t) n);
    return true;
}

// larry, STRATA_FIN_DEFER (default 1): a decode / verify pass's finish of layer il (x_state = hc_post(shexp + routed,
// ..)) runs as the first step of layer il + 1's attention hc kernel (build_attn post_in) - one kernel and one graph a
// layer less.  Not before an Engram layer (its Engram add reads x_state first), after the last layer, on the TP_OPROJ=2
// halves or uncached graphs; every reader of x_state through this API flushes a pending finish first (fin_flush).
static bool fin_defer_ok(const Ds41Dense::Impl & im, int il, int64_t n) {
    static const bool on = env_on("STRATA_FIN_DEFER", true);
    static const bool pass_cache = [] { const char * v = std::getenv("STRATA_PASS_CACHE"); return !(v && v[0] == '0'); }();
    static const int tp_mode = std::getenv("STRATA_TP_OPROJ") ? std::atoi(std::getenv("STRATA_TP_OPROJ")) : 0;
    static const bool hc_fuse = env_on("STRATA_HC_FUSE", true);
    static const int64_t hc_maxn = std::getenv("STRATA_HC_FUSE_MAXN") ? std::atoll(std::getenv("STRATA_HC_FUSE_MAXN")) : 8;
    if (!on || !im.sycl || !dense_async_on() || tp_mode == 2 || !hc_fuse || n > hc_maxn) return false;
    if (!(n == 1 || (pass_cache && n <= 8))) return false;
    if (il + 1 >= (int) im.ly.size() || im.ly[(size_t) il + 1].eslot >= 0) return false;
    return q8_fused(im, il + 1, n);
}

static bool finish_now(Ds41Dense::Impl & im);
static bool fin_flush(Ds41Dense::Impl & im) {
    if (!im.fin_pending) return true;
    im.fin_pending = false;
    return finish_now(im);
}

bool Ds41Dense::finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) (im.D * n) * 4);
    if (fin_defer_ok(im, il, n)) { im.fin_pending = true; return true; }
    return finish_now(im);
}

static bool finish_now(Ds41Dense::Impl & im) {
    const int64_t n = im.n;
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    static const bool pass_cache = [] { const char * v = std::getenv("STRATA_PASS_CACHE"); return !(v && v[0] == '0'); }();
    const bool cached = n == 1 || (pass_cache && n <= 8);
    std::vector<uint8_t> meta;
    if (cached) {
        const Ds41Dense::Impl::GKey key { -1, 2, 0, (int64_t) n };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_finish(im, im.gctx, n)).first;
        gf = it->second;
    } else {
        meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_finish(im, gc, n);
    }
    const double tf0 = g_decprof_on ? dprof_now() : 0;
    const bool ok = im.compute(gf, cached, im.sycl);
    if (g_decprof_on && n == 1) g_decprof[4] += dprof_now() - tf0;
    if (ds41_dense_tl_hook && cached && im.sycl && dense_async_on()) ds41_dense_tl_hook(n == 1 ? 4 : 12);
    if (gc) ggml_free(gc);
    return ok;
}

bool Ds41Dense::finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top) {
    Impl & im = *p_;
    (void) il;
    if (im.n != 1 || n_top < 1 || n_top > 16 || next_il < 0 || next_il >= (int) im.ly.size()) return false;
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 5, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.fpredict_out[next_il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    ggml_backend_tensor_get(im.fpredict_out[next_il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                                     int * ahead_ids) {
    Impl & im = *p_;
    (void) il;
    if (ahead <= 0 || !ahead_ids) return finish_predict(il, routed_sum, next_il, top_ids, n_top);
    if (im.n != 1 || n_top < 1 || n_top > 16 || next_il < 0 || next_il >= (int) im.ly.size()) return false;
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 6, ahead, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        std::vector<ggml_tensor *> ao;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o, ahead, &ao);
        it = im.graphs.emplace(key, gf).first;
        ao.insert(ao.begin(), o);   // [0] = next_il's prediction, [1 + a] = next_il + 1 + a's
        im.fpredict_ahead[{ next_il, ahead }] = ao;
    }
    if (!im.compute(it->second, true)) return false;
    const auto & ao = im.fpredict_ahead[{ next_il, ahead }];   // [0] = the output, then one entry per extra row
    // argsort_top_k is a VIEW of the full argsort: row r starts at r * nb[1] (NE ids apart), not r * 16 - one copy of
    // the span covering every row's first 16
    const int rows = (int) ao.size();
    const size_t rs = ao[0]->nb[1] / 4;
    std::vector<int32_t> buf(rs * (size_t) (rows - 1) + 16);
    ggml_backend_tensor_get(ao[0], buf.data(), 0, buf.size() * 4);
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[(size_t) i];
    for (int a = 0; a < ahead; ++a) {
        if (a + 1 < rows) std::copy(buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1)), buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1) + 16),
                                    ahead_ids + (size_t) a * 16);
        else std::fill(ahead_ids + (size_t) a * 16, ahead_ids + (size_t) (a + 1) * 16, -1);
    }
    return true;
}

// larry, STRATA_STALL_WATCH (patch-stallwatch.py, diagnostic, default on): while a watched call runs past 300 ms, the
// calling thread's kernel stack is sampled every 100 ms (/proc/self/task/TID/stack: root only) and printed at the end
struct StallWatch {
    const char * what;
    pid_t tid = 0;
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    std::map<std::string, int> stacks;
    explicit StallWatch(const char * w) : what(w) {
        static const bool on = !(std::getenv("STRATA_STALL_WATCH") && std::getenv("STRATA_STALL_WATCH")[0] == '0');
        if (!on) return;
        tid = (pid_t) syscall(SYS_gettid);
        th = std::thread([this] {
            std::unique_lock<std::mutex> lk(mu);
            if (cv.wait_for(lk, std::chrono::milliseconds(300), [this] { return done; })) return;
            // larry, STRATA_STALL_PERF (patch-stallwatch3): a perf sample of this thread's user + kernel stacks
            static const bool perf_on = std::getenv("STRATA_STALL_PERF") && std::getenv("STRATA_STALL_PERF")[0] == '1';
            static int perf_n = 0;
            if (perf_on && perf_n < 4) {   // posix_spawn (vfork-like): no fork() of a process holding GiBs of DMA-pinned memory
                char ts[16], out[160];
                std::snprintf(ts, sizeof ts, "%d", (int) tid);
                std::snprintf(out, sizeof out, "/opt/ai/dsv41/logs/stall.%d.%d.perf", (int) getpid(), perf_n++);
                const char * av[] = { "perf", "record", "-q", "-F", "499", "--call-graph", "dwarf,32768", "-t", ts, "-o", out, "--", "sleep", "0.8", nullptr };
                pid_t pp = 0;
                if (posix_spawnp(&pp, "perf", nullptr, nullptr, const_cast<char * const *>(av), ::environ) == 0)
                    std::fprintf(stderr, "stall watch (%s): perf sampling thread %d -> %s\n", what, (int) tid, out);
            }
            // larry, STRATA_STALL_GDB (patch-stallwatch4): every thread's backtrace, once the stall is 400 ms old
            static const bool gdb_on = std::getenv("STRATA_STALL_GDB") && std::getenv("STRATA_STALL_GDB")[0] == '1';
            static int gdb_n = 0;
            if (gdb_on && gdb_n < 3 && !cv.wait_for(lk, std::chrono::milliseconds(100), [this] { return done; })) {
                char pidb[16], out[160];
                std::snprintf(pidb, sizeof pidb, "%d", (int) getpid());
                std::snprintf(out, sizeof out, "/opt/ai/dsv41/logs/stall.%d.%d.gdb", (int) getpid(), gdb_n++);
                posix_spawn_file_actions_t fa;
                posix_spawn_file_actions_init(&fa);
                posix_spawn_file_actions_addopen(&fa, 1, out, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                posix_spawn_file_actions_adddup2(&fa, 1, 2);
                const char * av[] = { "gdb", "-p", pidb, "-batch", "-nx", "-ex", "set pagination off", "-ex", "thread apply all bt 40", nullptr };
                pid_t pp = 0;
                if (posix_spawnp(&pp, "gdb", &fa, nullptr, const_cast<char * const *>(av), ::environ) == 0)
                    std::fprintf(stderr, "stall watch (%s): gdb backtraces -> %s\n", what, out);
                posix_spawn_file_actions_destroy(&fa);
            }
            while (!done) {
                char path[64];
                std::snprintf(path, sizeof path, "/proc/self/task/%d/stack", (int) tid);
                std::ifstream f(path);
                std::stringstream ss;
                ss << f.rdbuf();
                std::snprintf(path, sizeof path, "/proc/self/task/%d/wchan", (int) tid);
                std::ifstream g(path);
                std::string wc;
                std::getline(g, wc);
                ++stacks["wchan " + wc + "\n" + ss.str()];
                // larry, patch-stallwatch2: every other thread inside the GPU driver (the lock holder)
                if (DIR * d = opendir("/proc/self/task")) {
                    while (dirent * de = readdir(d)) {
                        if (de->d_name[0] == '.' || std::atoi(de->d_name) == (int) tid) continue;
                        const std::string base = std::string("/proc/self/task/") + de->d_name;
                        std::ifstream gw(base + "/wchan");
                        std::string w2;
                        std::getline(gw, w2);
                        std::ifstream gs(base + "/stack");
                        std::stringstream s2;
                        s2 << gs.rdbuf();
                        const std::string st = s2.str();
                        auto has = [&](const char * k) { return st.find(k) != std::string::npos || w2.find(k) != std::string::npos; };
                        if (!(has("xe_") || has("drm_") || has("ttm_") || has("dma_fence") || has("rwsem") || has("i915"))) continue;
                        std::ifstream gc(base + "/comm");
                        std::string cm;
                        std::getline(gc, cm);
                        ++stacks["  other thread " + std::string(de->d_name) + " (" + cm + ") wchan " + w2 + "\n" + st];
                    }
                    closedir(d);
                }
                cv.wait_for(lk, std::chrono::milliseconds(100), [this] { return done; });
            }
        });
    }
    ~StallWatch() {
        if (!th.joinable()) return;
        { std::lock_guard<std::mutex> lk(mu); done = true; }
        cv.notify_all();
        th.join();
        for (const auto & kv : stacks)
            std::fprintf(stderr, "stall watch (%s): %d samples of 100 ms:\n%s", what, kv.second, kv.first.c_str());
    }
};

bool Ds41Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    StallWatch sw("batch head");   // larry, STRATA_STALL_WATCH
    const double tl0 = dprof_now();   // larry: a head taking > 300 ms is reported (one 16K prompt's took 7.7 s, abwT60)
    if (!fin_flush(im)) return false;
    const double tl1 = dprof_now();
    if (row0 < 0 || nrows < 1 || row0 + nrows > im.n) { im.err = "bad logits rows"; return false; }
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_tensor * lg = nullptr;
    ggml_cgraph * gf = build_head(im, gc, row0, nrows, &lg);
    // larry, patch-logitsdiag: queued (allocation inside), stamped, then waited for - the same work as compute(gf, false)
    const double tlb = dprof_now();
    auto stamp = [&](int tag, double t) {
        if (!ds41_dense_ptl_hook) return;
        std::unique_lock<std::mutex> sl;
        if (im.sub_mu) sl = std::unique_lock<std::mutex>(*im.sub_mu);
        ds41_dense_ptl_hook(tag, -1, nrows, t);
    };
    stamp(50, tlb);
    const bool ok = im.compute(gf, false, false, true);
    const double tlq = dprof_now();
    stamp(51, tlq);
    if (ok) ggml_backend_synchronize(im.backend);
    const double tl2 = dprof_now();
    if (ok) ggml_backend_tensor_get(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    const double tl3 = dprof_now();
    if (tl3 - tl0 > 300) {
        size_t fr = 0, tot = 0;
        if (im.backend) ggml_backend_dev_memory(ggml_backend_get_device(im.backend), &fr, &tot);
        std::fprintf(stderr, "ds41_dense: slow logits (%d rows): flush %.0f ms, build %.0f, alloc %.0f, queue %.0f, wait %.0f, read %.0f ms; "
                     "%.2f GiB free\n", nrows, tl1 - tl0, tlb - tl1, im.last_alloc_ms, tlq - tlb - im.last_alloc_ms, tl2 - tlq, tl3 - tl2,
                     fr / 1073741824.0);
    }
    return ok;
}

bool Ds41Dense::read_state(std::vector<float> & x, std::vector<float> & pre) {
    Impl & im = *p_;
    if (!fin_flush(im)) return false;
    x.resize((size_t) (im.D * im.HC * im.n));
    pre.resize((size_t) (im.HC * im.n));
    ggml_backend_tensor_get(im.x_state, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_get(im.pre_state, pre.data(), 0, pre.size() * 4);
    return true;
}

bool Ds41Dense::read_state_cols(int c0, int n, std::vector<float> & x) {
    Impl & im = *p_;
    if (!fin_flush(im)) return false;
    if (c0 < 0 || n < 1 || c0 + n > im.NC) { im.err = "read_state_cols: bad range"; return false; }
    const size_t col = (size_t) (im.D * im.HC) * sizeof(float);   // x_state [D, HC, NC]: a column is contiguous
    x.resize((size_t) (im.D * im.HC * n));
    ggml_backend_tensor_get(im.x_state, x.data(), (size_t) c0 * col, (size_t) n * col);
    return true;
}

bool Ds41Dense::snapshot(int pos) {
    Impl & im = *p_;
    for (auto & L : im.ly) {
        ggml_backend_tensor_copy(L.raw, L.raw_snap);
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv, L.st_kv_snap); ggml_backend_tensor_copy(L.st_sc, L.st_sc_snap); }
    }
    ggml_backend_synchronize(im.backend);
    im.snap_pos = pos;
    return true;
}

bool Ds41Dense::restore() {
    Impl & im = *p_;
    if (im.snap_pos < 0) return false;
    for (auto & L : im.ly) {
        ggml_backend_tensor_copy(L.raw_snap, L.raw);
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv_snap, L.st_kv); ggml_backend_tensor_copy(L.st_sc_snap, L.st_sc); }
    }
    ggml_backend_synchronize(im.backend);
    im.wr_changed = true;   // the rings and compressor states (no rows)
    return true;
}

int Ds41Dense::snapshot_pos() const { return p_->snap_pos; }

// larry (batch slots, patch-seqs-dense.py)
bool Ds41Dense::bind_seq(int s) {
    Impl & im = *p_;
    if (s < 0 || (size_t) s >= im.seqs.size()) { im.err = "no conversation slot " + std::to_string(s); return false; }
    if (s == im.seq) return true;
    ggml_backend_synchronize(im.backend);
    Impl::SeqT & O = im.seqs[(size_t) im.seq];
    O.snap_pos = im.snap_pos; O.wr_lo = im.wr_lo; O.wr_hi = im.wr_hi; O.wr_changed = im.wr_changed;
    const Impl::SeqT & N = im.seqs[(size_t) s];
    for (size_t l = 0; l < im.ly.size(); ++l) {
        Impl::Layer & L = im.ly[l];
        const auto & a = N.t[l];
        L.raw = a[0]; L.raw_snap = a[1]; L.comp = a[2]; L.idxk = a[3];
        L.st_kv = a[4]; L.st_sc = a[5]; L.st_kv_snap = a[6]; L.st_sc_snap = a[7];
    }
    im.snap_pos = N.snap_pos; im.wr_lo = N.wr_lo; im.wr_hi = N.wr_hi; im.wr_changed = N.wr_changed;
    im.span_memo_pos0 = -1;   // the span inputs' memo is the other conversation's
    im.engram.bind_seq(s);
    im.seq = s;
    return true;
}

int Ds41Dense::n_seqs() const { return (int) std::max<size_t>(1, p_->seqs.size()); }
int Ds41Dense::seq() const { return p_->seq; }

void Ds41Dense::kv_bytes_for(int64_t positions, std::vector<size_t> & bytes) const {
    const Impl & im = *p_;
    bytes.clear();
    for (size_t i = 0; i < im.kv_units.size(); ++i) {   // row r of a ratio-k source holds a position >= r * k: rows [0, p / k] + a margin
        const auto & u = im.kv_units[i];
        if (i < im.kv_unit_seq.size() && im.kv_unit_seq[i] != im.seq) { bytes.push_back(0); continue; }   // (batch slots)
        const int64_t k = u.second == 2 ? 2 : 1;
        // larry, patch-kvbucket.py: the readers' cap bucket, not the positions - every scan reads next_pow2 rows
        // (span_pass_caps, the TP core's caps and its replica growth / state sync from card 0); positions + 32K died at
        // 524,288 where both ratios' buckets became the whole tensor (DEVICE_LOST, 2026-10-24)
        const int64_t rows = std::min<int64_t>(u.first->ne[1] - 1, next_pow2(std::max<int64_t>(1024, positions / k + 2)) + 4);
        bytes.push_back((size_t) rows * u.first->nb[1]);
    }
}

bool Ds41Dense::kv_lazy() const { return p_->kv_buf != nullptr; }

// larry (patch-parkpool.py): the rows of a ratio-r source that hold positions [0, positions): row k holds positions
// [k r, k r + r), so positions / r + 1 rows; + 1 more as a margin (kv_bytes_for's rule).  Was positions + 1 for both.
static int64_t park_rows(const ggml_tensor * t, int64_t ratio, int64_t positions) {
    const int64_t r = ratio == 2 ? 2 : 1;
    return std::min<int64_t>(t->ne[1] - 1, positions / r + 2);
}

bool Ds41Dense::park(int64_t positions, int slot) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1 || slot < 0 || slot >= 64) return false;
    if ((size_t) slot >= im.pk.size()) im.pk.resize((size_t) slot + 1);
    ggml_backend_synchronize(im.backend);
    Impl::ParkSlot & S = im.pk[slot];
    S.rows.clear();
    S.bytes = 0;
    int pl = 0, pkind = 0;   // (larry, batch slots: the region's layer and kind)
    auto put = [&](ggml_tensor * t, size_t off, size_t n) {
        Impl::ParkedRows r{ t, off, std::vector<uint8_t>(n), pl, pkind };
        ggml_backend_tensor_get(t, r.b.data(), off, n);
        S.rows.push_back(std::move(r));
        S.bytes += n;
    };
    for (auto & L : im.ly) {
        pkind = 0;
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap }) {
            if (t) put(t, 0, ggml_nbytes(t));
            ++pkind;
        }
        for (ggml_tensor * t : { L.comp, L.idxk }) {
            if (t) {
                const int64_t rows = t->ne[1];
                const int64_t n = park_rows(t, L.ratio, positions);
                put(t, 0, (size_t) n * t->nb[1]);
                put(t, (size_t) (rows - 1) * t->nb[1], t->nb[1]);   // the spare row (open compressor groups write it)
            }
            ++pkind;
        }
        ++pl;
    }
    S.pos = im.snap_pos;
    S.n = positions;
    return true;
}

size_t Ds41Dense::park_bytes(int slot) const {
    return slot >= 0 && (size_t) slot < p_->pk.size() && p_->pk[slot].pos >= 0 ? p_->pk[slot].bytes : 0;
}

size_t Ds41Dense::park_bytes_for(int64_t positions) const {
    const Impl & im = *p_;
    size_t b = 0;
    for (const auto & L : im.ly) {
        for (const ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) b += ggml_nbytes(t);
        for (const ggml_tensor * t : { L.comp, L.idxk })
            if (t) b += (size_t) (park_rows(t, L.ratio, positions) + 1) * t->nb[1];
    }
    return b;
}

bool Ds41Dense::unpark(int slot) {
    Impl & im = *p_;
    if (slot < 0 || (size_t) slot >= im.pk.size() || im.pk[slot].pos < 0) return false;
    Impl::ParkSlot & S = im.pk[slot];
    ggml_backend_synchronize(im.backend);
    for (auto & r : S.rows) {   // (larry, batch slots: into the BOUND conversation's tensor of that layer and kind)
        ggml_tensor * t = r.t;
        if (r.l >= 0 && (size_t) r.l < im.ly.size()) {
            const Impl::Layer & L = im.ly[(size_t) r.l];
            ggml_tensor * const by_kind[5] = { L.raw_snap, L.st_kv_snap, L.st_sc_snap, L.comp, L.idxk };
            if (r.k >= 0 && r.k < 5 && by_kind[r.k]) t = by_kind[r.k];
        }
        ggml_backend_tensor_set(t, r.b.data(), r.off, r.b.size());
    }
    ggml_backend_synchronize(im.backend);
    im.wrote(0, S.n + 4);   // the parked positions (patch-kvbucket.py: was 2 * (positions + 1) - past card 0's lazy map)
    im.snap_pos = S.pos;
    drop_park(slot);
    return true;
}

void Ds41Dense::drop_park(int slot) {
    if (slot < 0 || (size_t) slot >= p_->pk.size()) return;
    p_->pk[slot].rows.clear();
    p_->pk[slot].rows.shrink_to_fit();
    p_->pk[slot].pos = -1;
    p_->pk[slot].bytes = 0;
}

bool Ds41Dense::parked(int slot) const { return slot >= 0 && (size_t) slot < p_->pk.size() && p_->pk[slot].pos >= 0; }

bool Ds41Dense::state_ptrs(int il, void ** raw, void ** comp, void ** idxk, void ** st_kv, void ** st_sc) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || im.cpu) return false;
    const Impl::Layer & L = im.ly[(size_t) il];
    if (!L.raw || L.raw->type != GGML_TYPE_F32) return false;
    if ((L.comp && L.comp->type != GGML_TYPE_BF16) || (L.idxk && L.idxk->type != GGML_TYPE_BF16)) return false;
    *raw = L.raw->data;
    *comp = L.comp ? L.comp->data : nullptr;
    *idxk = L.idxk ? L.idxk->data : nullptr;
    *st_kv = L.st_kv ? L.st_kv->data : nullptr;
    *st_sc = L.st_sc ? L.st_sc->data : nullptr;
    return true;
}

void Ds41Dense::take_written(int64_t * lo, int64_t * hi, bool * changed) {
    Impl & im = *p_;
    *lo = im.wr_lo;
    *hi = im.wr_hi;
    *changed = im.wr_changed;
    im.wr_lo = INT64_MAX;
    im.wr_hi = -1;
    im.wr_changed = false;
}

bool Ds41Dense::engram_rows(const int * tids, int n, int pos0, float * const * out) {
    Impl & im = *p_;
    if (!im.engram.enabled()) return true;
    if (!engram_ready_all(im)) return false;   // a dense pass's batch still pending: flushed first
    if (im.eng_pf) {                       // the prompt prefetch's rows land in the row cache first
        std::string e2;
        im.engram.wait(im.eng_pf, e2);
        im.eng_pf.reset();
    }
    for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);
    auto b = im.engram.start_into(pos0, n, out, im.err);
    return b && im.engram.wait(b, im.err);
}

bool Ds41Dense::engram_rows_slot(const int * tids, int n, int pos0, int slot, float * out) {
    Impl & im = *p_;
    if (!im.engram.enabled() || slot < 0 || slot >= im.engram.n_slots()) return true;
    if (slot == 0) {
        if (!engram_ready_all(im)) return false;
        if (im.eng_pf) {
            std::string e2;
            im.engram.wait(im.eng_pf, e2);
            im.eng_pf.reset();
        }
        for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);
    }
    float * outp[2] = { slot == 0 ? out : nullptr, slot == 1 ? out : nullptr };
    auto b = im.engram.start_into(pos0, n, outp, im.err);
    return b && im.engram.wait(b, im.err);
}

bool Ds41Dense::engram_rows_slot_seq(int s, const int * tids, int n, int pos0, int slot, float * out) {
    Impl & im = *p_;
    if (s == im.seq) return engram_rows_slot(tids, n, pos0, slot, out);
    if (!im.engram.enabled() || slot < 0 || slot >= im.engram.n_slots()) return true;
    if (s < 0 || (size_t) s >= im.seqs.size()) { im.err = "Engram rows: no conversation slot " + std::to_string(s); return false; }
    if (slot == 0) {
        if (!engram_ready_all(im)) return false;
        if (im.eng_pf) {
            std::string e2;
            im.engram.wait(im.eng_pf, e2);
            im.eng_pf.reset();
        }
    }
    im.engram.bind_seq(s);   // its history: the tokens' record and the hashes (start_into hashes before it returns)
    if (slot == 0)
        for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);
    float * outp[2] = { slot == 0 ? out : nullptr, slot == 1 ? out : nullptr };
    auto b = im.engram.start_into(pos0, n, outp, im.err);
    im.engram.bind_seq(im.seq);
    return b && im.engram.wait(b, im.err);
}

bool Ds41Dense::read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
    Impl & im = *p_;
    const auto & L = im.ly[(size_t) il];
    sel.clear();
    mask.clear();
    *K = 0;
    if (!L.idx_src || !L.ratio) return true;
    const int64_t cap = im.pass_cap[L.ratio];
    if (cap == 0) return true;
    const int64_t k = std::min<int64_t>(cap, im.ITOP);
    std::vector<int32_t> s((size_t) (im.ITOP * im.n));
    std::vector<float> m((size_t) (im.ITOP * im.n));
    ggml_backend_tensor_get(L.sel, s.data(), 0, s.size() * 4);
    ggml_backend_tensor_get(L.selm, m.data(), 0, m.size() * 4);
    for (int t = 0; t < im.n; ++t)
        for (int64_t j = 0; j < k; ++j) {
            sel.push_back(s[(size_t) (t * im.ITOP + j)]);
            mask.push_back(m[(size_t) (t * im.ITOP + j)]);
        }
    *K = (int) k;
    return true;
}
