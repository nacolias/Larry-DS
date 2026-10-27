// tools/v41/engine/ds41_dense.hpp - DeepSeek-V4.1-Flash (`deepseek41`): the dense (non-routed-expert) half.
//
// Math = DeepSeek's inference/model.py (tools/v41/engine/ds41_golden.py runs it on the GGUF's own weights; that is
// this file's gate).  The structure follows tools/ds4/ds4_dense (ggml graphs, CPU or CUDA backend, per-layer graphs
// built once and reused), but V4.1 differs enough that it is its own class:
//
//   - one PASS = n consecutive tokens at positions pos0..pos0+n-1 (n = 1: decode; n = a prompt chunk: prefill);
//     every graph is the same code for both (state rings are read as [ring | the pass's rows] and written back)
//   - hyper-connections single pass: a sublayer's hc_mixes give the NEXT sublayer's pre-mix (pre_state carries the
//     attention pre-mix from layer l's FFN to layer l+1)
//   - no per-head q norm; window K/V ring of 128; compressed KV (ratio 2 / 1) only on kv source layers, read by the
//     later layers of the same ratio; indexers on index source layers, their top-k reused by the layers after them;
//     layer `cand_layer` (20) picks 2048 candidate blocks of 8 that the later indexers rank inside
//   - attention = each query's own key set (its window + its top-k compressed rows) gathered and attended in one
//     flash-attn with the query as the batch axis (sinks added)
//   - Engram at layers 1 / 14: n-gram hash ids (host, from the token history) -> rows read from the HF shards
//     (Ds41Engram) -> wkv -> per-hc gated add, before the block
//
// Per layer per pass:  attn_router(l) -> [routed experts on fn] -> finish(l).  Then logits(...).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "ds41_geometry.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>

struct Ds41DenseConfig {
    ggml_backend_t backend = nullptr;   ///< NULL = create and own a CPU backend
    int     n_threads = 8;
    int64_t ctx_max = 65536;            ///< positions the caches hold (compressed rows = ctx_max / ratio)
    int n_seqs = 1;                     ///< larry (batch slots): conversations held side by side (bind_seq)
    int     max_pass = 1;               ///< tokens a pass may carry (>= 1; prefill chunk size = the sub-batch size S)
    /// Layer-major prompt batch (0 = off): the state tensors hold this many token columns and a prompt of up to
    /// `prompt_batch` tokens is walked layer by layer, one dense sub-batch of `max_pass` tokens at a time, with the
    /// experts run ONCE per layer over the whole batch (begin_batch / attn_router_span / finish_span).
    int     prompt_batch = 0;
    bool    skip_routed_experts = true; ///< do not load the routed experts (the tier owns them)
    /// Directories holding the HF shards with the Engram tables (model.safetensors.index.json in the first).
    /// Empty = no Engram (its add is skipped: a debugging mode, the model is then not the model).
    std::vector<std::string> engram_dirs;
    int     engram_threads = 32;        ///< parallel row reads (the disk's queue depth; cold peaks at 32-64)
    /// KV caches (window ring, compressed rows, indexer keys) element type: GGML_TYPE_F32 / BF16, or -1 = auto: BF16
    /// with act_quant (every cached value is then an FP8/FP4 grid point times a power-of-two / E4M3 scale: <= 7
    /// significant bits, exact in BF16 - half the VRAM of F32 at 512K), F32 without.
    int     kv_type = -1;
    /// model.py's activation rounding (ds41_ops.hpp): window K FP8, compressed KV FP4/E4M3, indexer q/k FP4.
    bool    act_quant = true;
    /// --kv-fp4 (needs act_quant, CUDA): the compressed KV as NVFP4 and the indexer keys as MXFP4 - their fake-quant
    /// grids exactly (e2m1 x E4M3 per 16 / x 2^k per 32), packed bit for bit by the vendored ggml-cuda's set_rows
    /// (cpy-utils.cuh *_ds41_block): 4.5 / 4.25 bits a value instead of BF16's 16 - ~3.5x less KV VRAM at long ctx.
    bool    kv_fp4 = false;
    /// --serve: start with max_pass (S) state columns and let set_columns() grow them for a layer-major batch and give
    /// them back after it (the prompt_batch columns are ~2.6 GiB at 8192 - expert slots while decoding).
    bool    elastic_cols = false;
    /// larry, engine1 (STRATA_TP_PAGE): each layer's q_b, wo_a, wo_b, shared expert (gate | up, down) and Engram
    /// projection - the dense weights the TP cards hold between them - in memory the caller provides and pages.
    /// Called once at load with every layer's bytes (a unit, 0 = none); returns a backend buffer whose base is the
    /// units' region (unit_off[l] filled in) or null (the normal buffer).  Q6_K tensors only; the work on a layer's unit
    /// is bracketed by the page hooks (set_page_hooks).
    std::function<ggml_backend_buffer_t(const std::vector<size_t> & unit_bytes, std::vector<size_t> & unit_off)> paged_alloc;
    /// larry (STRATA_DENSE_LAZY_KV): the KV-source layers' compressed KV and indexer keys in memory the caller provides
    /// and maps as the context grows (kv_bytes_for).  Called once at init with each tensor's bytes and its spare row's
    /// offset; returns a backend buffer over the units (unit_off filled in) or null (the state buffer holds them).
    std::function<ggml_backend_buffer_t(const std::vector<size_t> & unit_bytes, const std::vector<size_t> & spare_off,
                                        std::vector<size_t> & unit_off)> kv_alloc;
};

/// larry, engine1: a weight in the paged units (cfg.paged_alloc): its layer, address, Q6_K shape
struct Ds41DensePaged {
    std::string name;
    int layer = -1;
    void * data = nullptr;
    int64_t K = 0, N = 0;
    size_t bytes = 0;
};

// larry, STRATA_GPU_TL: called at a decode layer's stage boundaries (0 attention queued, 1 readbacks done, 2 shared
// expert done, 4 finish done) - the generator points it at the Xe tier's profiling tags
extern void (*ds41_dense_tl_hook)(int id);
extern void (*ds41_dense_ptl_hook)(int tag, int il, int col0, double t0);   // larry, STRATA_GPU_PTL: a prompt span queued

class Ds41Dense {
public:
    struct Impl;
    Ds41Dense();
    ~Ds41Dense();
    Ds41Dense(const Ds41Dense &) = delete;
    Ds41Dense & operator=(const Ds41Dense &) = delete;

    bool init(const std::string & gguf, const Ds41DenseConfig & cfg, std::string & err);
    const strata::v41::Ds41Geometry & geom() const;
    /// Blocks present in the file (all 40 for the real model; a prefix for a partial test file).
    int64_t n_layer() const;
    ggml_backend_t backend() const;

    /// Start a pass: tokens tids[0..n) at positions pos0.., n <= max_pass.  Embeddings + Engram rows (the hash
    /// needs the 3 tokens before pos0: the history is kept by position, so a re-decoded position overwrites it).
    bool begin(const int * tids, int n, int pos0);
    /// Layer-major prompt batch (cfg.prompt_batch > 0): the whole batch's inputs are prepared once - embeddings,
    /// positions, window / compressed / indexer index maps, Engram rows and the initial x/pre state - into the state
    /// tensors' columns 0..B-1; then attn_router_span / finish_span walk layer il's sub-batches in order.  B <=
    /// cfg.prompt_batch (the state columns) and B <= ctx_max - pos0.  Every cross-layer dependency of the sequential
    /// chunk path (the per-layer KV ring / compressed rows / indexer state advances in sub-batch order; layer l's
    /// residual and pre-mix are the columns the sub-batch used) is preserved, so the outputs match --prefill-chunk.
    bool begin_batch(const int * tids, int B, int pos0);
    /// larry (vision, patch-vision-engine.py): the prompt's image spans - absolute positions [p0, p0 + n) take their
    /// input embeddings from `rows` (the vision encoder's F32 [n][n_embd], reading order; valid until clear_images),
    /// select their experts with the model's VL bias (exp_probs_b_vl) and get no Engram (model.py image spans).  Set
    /// around a prompt's feed; begin_batch / begin read them.
    struct ImageSpan { int64_t p0; int n; const float * rows; };
    void set_images(const std::vector<ImageSpan> & spans);
    void clear_images();
    /// larry (vision): the Engram's token history at positions [pos0, pos0 + n) again (an image position as DEAD)
    void engram_tokens(int64_t pos0, const int * tids, int n);
    /// larry: the NEXT layer-major batch's Engram rows (tokens tids[0..n) at positions pos0..) read into the Engram row
    /// cache while this batch computes (STRATA_ENG_PREFETCH, default 1); the next pass's inputs wait for it first.
    void engram_prefetch(const int * tids, int n, int pos0);
    /// Dense half of layer il for the batch's sub-batch starting at column col0 (n tokens, col0 % max_pass == 0):
    /// attention, hc, router, shared expert + the Engram add.  ids / wts / fn land in the batch buffers (batch_ids /
    /// batch_wts / batch_fn) at column col0.  Uses (and advances) layer il's own KV state in sub-batch order.
    bool attn_router_span(int il, int col0, int n);
    /// Encoder-only skip (enc-skip): build and run ONLY the persistent-KV half of layer il for the batch's sub-batch
    /// at column col0 - hc_pre / attn_norm, the raw window K/V ring write and, on a kv source layer, the compressed-KV
    /// row + indexer-key row + ratio-2 compressor state (all the state a LATER query can read across positions).  No
    /// attention output, no router, no shared expert, no hc_post / finish, no indexer selection / candidates: nothing
    /// that only ever flows into this layer's own output columns.  For prompt tokens whose layers past the replay
    /// window are never read (the model's cross-position reach is the 128-token window per layer, so the last
    /// `enc_replay` tokens are all that matters).  Writes no host ids/wts/fn, so batch_ids()/batch_wts()/batch_fn()
    /// are left untouched at these columns.
    bool attn_router_span_kv(int il, int col0, int n);
    /// The batch's host buffers filled by attn_router_span: ids / wts are n_used * B (token-major), fn is n_embd * B.
    const int32_t * batch_ids() const;
    const float * batch_wts() const;
    const float * batch_fn() const;
    /// larry: the fn rows on the device (o_fn, [D, NC] floats on card 0; null on the CPU backend), and whether the
    /// batch spans read them back into batch_fn() at all (set_fn_host(false): a GPU tier copies them card-to-card)
    const float * batch_fn_dev() const;
    void set_fn_host(bool on);
    /// larry (pipelined batch, with the submit lock set): span graphs alternate two allocators guarded by events, so
    /// a span no longer waits for the device; pass_sync() waits for everything (the tier reads the spans' outputs)
    void set_pass_ring(bool on);
    /// larry, STRATA_SHEXP_TIER (patch-shexpdense.py): the following prompt spans leave the shared expert to the expert
    /// tier (their graphs write zeros into its columns; the tier adds it into the routed sum)
    void set_shexp_skip(bool on);
    /// larry: called right before a decode / verify layer's host waits for its router readback (layer, width) - the
    /// tier queues its cards' share then (Ds41XeTier::arm)
    void set_pre_wait_hook(std::function<void(int, int)> h);
    /// larry, STRATA_XE_GROUTE: called right after a decode / verify layer's fn readback is queued (layer, width, the
    /// router's ids / wts and fn on the device, the readback's host mirror) - the tier queues its route kernel there
    /// (Ds41XeTier::route)
    void set_route_hook(std::function<void(int, int, const void *, const void *, const void *, const float *)> h);
    /// larry, STRATA_TP_OPROJ=2: the attention output projection of a decode / verify layer, queued between the two
    /// halves of its attention graph (layer, width, wo_a, wo_b, o: OGD * OG floats a column, dst: n_embd a column)
    void set_oproj_hook(std::function<bool(int, int, const ggml_tensor *, const ggml_tensor *, const float *, float *)> h);
    void pass_sync();
    int batch_size() const;
    /// finish for the batch's sub-batch at column col0: x_state[.., col0+n) = hc_post(shexp + routed, hap, post_f,
    /// comb_f).  `routed_host` is n_embd * B (token-major); call once per layer, before the sub-batches' finishes.
    bool upload_routed(const float * routed_host, int B);
    /// The routed-sum input tensor's device memory (columns of n_embd floats; null on the CPU backend): the tier can
    /// write a layer-major batch's routed sums straight into it (run_chunk_dev) instead of upload_routed's host trip.
    float * routed_dev();
    bool finish_span(int il, int col0, int n);
    /// larry (pipelined prompt batch): while set, the span calls (attn_router_span[_kv], finish_span) hold this lock
    /// around their enqueues (not their waits) - the tier's card threads enqueue concurrently
    void set_submit_lock(std::mutex * m, std::atomic<int> * prio = nullptr);
    /// STRATA_DENSE_PROF=1: print + reset the batch dense split (build / alloc / compute / Engram / readback) - diagnostic
    void prof_report(const char * tag);
    /// Give the prompt passes' compute buffer back to the device (rebuilt by the next multi-token pass): the deferred
    /// VRAM expert cache opens into it after the prompt.
    void release_pass();
    /// Called when a graph's buffer cannot be allocated; return true if it freed device memory (the allocation is
    /// then retried once).  E.g. the expert tier's prestage region.
    void set_oom_hook(std::function<bool()> hook);
    /// Selection-only routing bias for layer il (n_expert floats): added to the top-k score, not to the mixing weights.
    /// Soft pruning (-penalty on cold experts) / cache-aware routing (+bonus on VRAM-resident ones).  Default 0.
    void set_route_bias(int il, const float * bias);
    /// Layer il of the current pass: Engram (layers 1/14), attention, hc, ffn_norm, router, shared expert.
    /// ids / wts: n * n_used (token-major); *fn_host: n * n_embd floats (the routed experts' input, valid until the
    /// next call).  `fn_dev` (optional): the same activations on the backend.
    bool attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev = nullptr);
    /// Prefetch hint for layer il of a one-token pass (call before attn_router(il)): the layer's router applied to
    /// rms_norm(hc_pre(state, pre)) * ffn_norm.weight, + the selection bias, ranked; top_ids gets n_top (<= 16).
    /// A hint only: the result does not depend on it.
    bool predict(int il, int * top_ids, int n_top);
    /// predict() for EVERY column of the current pass (call before attn_router(il)): top_ids[t * 16 + i] = token t's
    /// i-th predicted expert of layer il (16 per token).  For the multi-token verify pass's expert prefetch; a hint only.
    bool predict_multi(int il, int * top_ids);
    /// routed_sum: n * n_embd floats, the routed experts' weighted sum (host).
    bool finish(int il, const float * routed_sum);
    /// Decode (n == 1) fusion: finish(il) and predict(next_il) in ONE cached graph - the router prediction for the next
    /// layer reads the state finish just wrote, so the result is identical to finish() then predict(), with one graph
    /// launch / sync / readback less per layer.  top_ids gets the next layer's top n_top (<= 16) predicted experts.
    bool finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top);
    /// finish_predict + the routing predictions of the `ahead` layers after next_il, from the same state, in the same
    /// graph: ahead_ids[a * 16 ..] = layer next_il + 1 + a's top 16 (-1 past the last layer).  For speculative
    /// file-tier reads; a hint only.
    bool finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                              int * ahead_ids);
    /// Logits of rows row0..row0+nrows-1 of the pass after the last layer's finish: nrows * n_vocab floats.
    bool logits(int row0, int nrows, float * out);
    /// Conversation reuse: every cache is position-indexed except the window ring and the ratio-2 compressor state
    /// ring, so a snapshot is those rings (~11 MB) + its position; restore() rewinds to it (decoding continues at
    /// snapshot_pos()).  Positions past a rewind point are overwritten when decoded again (rows past a query's
    /// visible count are never attended).
    bool snapshot(int pos);
    bool restore();
    int snapshot_pos() const;
    /// Serve: keep a conversation through an UNRELATED request.  An unrelated request writes its own positions from 0, so
    /// it overwrites the start of every position-indexed cache (compressed KV, indexer keys) and the snapshot rings.
    /// park(n) copies the snapshot rings + the first n positions' rows of those caches (and their spare rows) to host
    /// memory; unpark() writes them back and re-arms the snapshot, so restore() resumes the parked conversation.  Valid
    /// only while no request has written past position n since park (the caller checks).
    /// larry: `slot` 0 .. 63: a pool of parked conversations (STRATA_SERVE_PARK_SLOTS; patch-parkpool.py)
    bool park(int64_t positions, int slot = 0);
    /// larry (cfg.kv_alloc): per KV unit (kv_alloc's order) the bytes from its start that `positions` positions use
    void kv_bytes_for(int64_t positions, std::vector<size_t> & bytes) const;
    bool kv_lazy() const;
    bool unpark(int slot = 0);
    void drop_park(int slot = 0);
    bool parked(int slot = 0) const;
    /// larry (patch-parkpool.py): the host bytes parked slot `slot` holds / a park of `positions` would take
    size_t park_bytes(int slot) const;
    size_t park_bytes_for(int64_t positions) const;
    /// larry (batch slots, patch-seqs-dense.py): cfg.n_seqs conversations side by side - each its own window ring,
    /// compressed KV, indexer keys, compressor states, snapshot and written range.  bind_seq(s): every later call (prompt
    /// batches, snapshot / restore, park / unpark, state_ptrs, kv_bytes_for, take_written, the Engram history) works on
    /// conversation s.  Not inside a pass or a batch.
    bool bind_seq(int s);
    int n_seqs() const;
    int seq() const;
    /// Debug / gate taps: the hc state after the last finish (n * hc * n_embd floats, token-major) and the pre mix.
    bool read_state(std::vector<float> & x, std::vector<float> & pre);
    /// The hc state of state columns [c0, c0 + n) (token-major, n * hc * n_embd floats) - a layer-major batch's columns
    /// after finish_span (the serve DSpark seed reads its main_hidden there instead of re-running the tail as passes).
    bool read_state_cols(int c0, int n, std::vector<float> & x);
    /// Resize the state columns to max(S, min(nc, prompt_batch)) between passes (drops every cached graph); columns()
    /// is the current count.  A no-op when unchanged.
    bool set_columns(int nc);
    int columns() const;
    /// Gate tap: an index source layer's top-k rows of the current pass (n * K ints, token-major; K = min(cap, top-k))
    /// and their visibility mask; empty for other layers.
    bool read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K);
    /// The x state after begin (+ the Engram add of the layer `il`, when il is an Engram layer) - gate tap.
    const std::string & last_error() const;
    /// A loaded weight on the backend (nullptr if absent) - for sharing the trunk head with the DSpark draft stack.
    ggml_tensor * weight(const std::string & name) const;
    /// larry: every 2-D Q6_K weight a decode reads through the reordered layout - for ggml_backend_sycl_prereorder at load
    /// (the shared expert's gate / up halves are left out when they are fused: only the fused weight may be reordered)
    std::vector<ggml_tensor *> reorder_weights() const;
    /// larry: the 2-D Q6_K weights read only through per-group slice views (wo_a) - for ggml_backend_sycl_prereorder_slices
    std::vector<ggml_tensor *> slice_reorder_weights() const;
    /// larry, engine1 (STRATA_TP): layer il's persistent state tensors on the backend (device addresses; null where
    /// the layer has none) - the window ring [RAW][DH] f32, the compressed KV / indexer keys ([ctx_max / ratio + 1][DH]
    /// / [..][ID] bf16) and the ratio-2 compressor state [2][DH] f32 (values, scores).  False: another layout (--kv-fp4,
    /// an f32 cache, the CPU backend).
    bool state_ptrs(int il, void ** raw, void ** comp, void ** idxk, void ** st_kv, void ** st_sc) const;
    /// larry, engine1: what the dense half wrote to that state since the last call - the positions [*lo, *hi) of its
    /// passes / batches / unpark, and *changed (any write at all: the rings and compressor states too, e.g. restore).
    /// An external decode core re-syncs its copies from it.  Every state write is behind a pass_sync() the caller does.
    void take_written(int64_t * lo, int64_t * hi, bool * changed);
    /// larry, engine1: the weights in the paged units (cfg.paged_alloc; the shared expert's pair under its fused name)
    std::vector<Ds41DensePaged> paged_weights() const;
    /// larry, engine1: need(il) before any work on layer il's weights is queued, touch(il) after (the caller pages them)
    void set_page_hooks(std::function<void(int)> need, std::function<void(int)> touch);
    /// larry, engine1: the Engram rows of tokens tids[0..n) at pos0.. into out[e] (n * cols * dim floats, slot e =
    /// the e-th Engram layer), the tokens recorded in the n-gram history first; blocking.  No Engram: true, untouched.
    bool engram_rows(const int * tids, int n, int pos0, float * const * out);
    /// larry, engine1: one slot's rows (slot 0 first: it records the tokens) - the first Engram layer can start before
    /// the second's rows are read
    bool engram_rows_slot(const int * tids, int n, int pos0, int slot, float * out);

private:
    std::unique_ptr<Impl> p_;
};
