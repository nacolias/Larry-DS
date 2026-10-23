// tools/v41/engine/xe/ds41_xe_tier.hpp - DeepSeek-V4.1's routed-expert tier for SEVERAL Intel Arc GPUs (SYCL).
//
// The CUDA tier (ds41_moe.hpp) is built around ONE 24 GB card: a VRAM LRU, a pinned arena fed over PCIe, the CPU pool
// and an NVMe file tier, all racing per layer.  A box with four 32 GB B70s (larry) can hold ~90 GiB of the 153 GiB of
// experts in VRAM, so this tier is laid out differently:
//
//   * PLACEMENT, once: the experts in hotness order (rank.bin) are dealt over the "permanent" cards (every card but the
//     dense half's, by default) until their budgets are full; the rest go to a pinned host arena (SYCL host USM, so a
//     DMA to any card reads it directly).  When the engine opens its VRAM cache after a prompt (seed_from_ranked with
//     set_slot_gib), the next ranked arena experts are also copied onto the DENSE card, and release_cache() gives
//     that VRAM back for the next prompt's buffers - the protocol ds41_generate already follows for the CUDA tier.
//   * DECODE / VERIFY (<= 8 tokens): each card computes the routed experts it holds (two kernels: gate/up + SwiGLU,
//     down + weighted sum), the CPU pool computes the arena ones at the same time, the partial sums are added up.
//   * PROMPT CHUNKS (any number of tokens): expert-major.  Resident experts run where they live; the arena experts the
//     chunk routes to are streamed (DMA) onto the cards round-robin and run there; per-card partial sums are added.
//
// The API is the subset of Ds41MoeTier's that ds41_generate calls, with the same meaning, so the engine switches tiers
// with one typedef (DS41_XE).  Hints the CUDA tier needs (prefetch, spec_read, advise_file, rebalance_*) are no-ops.
//
// THE INVARIANT is the CUDA tier's: run()'s output equals the weighted sum of the dequantised experts in F32, up to the
// CPU pool's 8-bit activation rounding for the experts it computes (the cards compute in F32 activations).
#pragma once

#include <cstdlib>

#include "ds41_moe.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <mutex>
#include <atomic>

namespace strata::v41 {

struct Ds41XeImpl;

class Ds41XeTier {
public:
    static constexpr int kMaxMulti = 8;

    Ds41XeTier();
    ~Ds41XeTier();
    Ds41XeTier(const Ds41XeTier&) = delete;
    Ds41XeTier& operator=(const Ds41XeTier&) = delete;

    /// `blobs` must outlive the tier.  Opens the SYCL devices (every Level Zero GPU, or STRATA_XE_DEVICES=0,1,2,3),
    /// sizes the per-card budgets and allocates the scratch.  `dense_dev` is the address of a buffer of the dense
    /// half's backend (its routed tensor): the card that holds it is the dense card.
    bool init(const Ds41MoeGeom& geom, Ds41BlobSource* blobs, const Ds41MoeConfig& cfg, const void* dense_dev,
              std::string& err);

    bool build_arena_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);
    bool seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);
    void set_slot_gib(double gib);
    /// larry: the dense half's own in-order queue (ggml_backend_sycl_get_queue).  When set, a decode / verify call's
    /// final sum runs on it and the call returns without waiting: the dense half's next graph (on that queue) orders
    /// after it, and the host goes on submitting while the cards compute.
    void set_dense_queue(void* q);
    /// larry, STRATA_GPU_TL=1 (diagnostic): device timestamps (profiling tags) on the dense queue at the decode layer's
    /// stage boundaries (tl_tag ids: 0 attention queued, 1 readbacks done, 2 shared expert done, 4 finish done; the
    /// tier tags 3 after its sum) and on each card around its share; tl_flush (once per token) folds them into
    /// per-transition averages, printed every 64 tokens
    static bool tl_on();
    /// larry (ggml-sycl's mat-vec hook, STRATA_HEAD_SPLIT): the LM head split by rows over every card - each expert card
    /// keeps its rows' R8 planes, gets the int8 activations by a card-to-card copy and sends its rows of y back the same
    /// way; the dense card computes the first rows.  `user` is the tier.
    static bool mv_hook(void* queue, const void* w, int64_t K, int64_t N, int ncols, const int8_t* xq, const float* xd,
                        float* y, int64_t y_col_stride, void* user);
    /// larry (ggml-sycl's DSV41_OPROJ hook, STRATA_TP_OPROJ): the attention output split by head groups - expert card k
    /// keeps wo_a's slices of its groups and wo_b's matching columns, gets its groups' int8 o by a card-to-card copy, sends
    /// its partial sum back; the dense card computes the first groups and adds the parts.  `user` is the tier.
    static bool oproj_hook(void* queue, const void* wo_a, size_t wo_a_slice_bytes, const void* wo_b, int64_t OGD,
                           int64_t OL, int64_t OG, int64_t D, int ncols, const float* o, int64_t o_col_stride, float* dst,
                           int64_t dst_col_stride, void* user);
    void tl_tag(int id);
    void tl_flush();
    void set_cache_enabled(bool on);
    void release_cache();
    int64_t shrink_cache_for(int64_t need_bytes, std::string& err);
    int64_t regrow_cache(std::string& err);
    /// larry, engine1: the dense card's expert replicas (STRATA_XE_REPLICA_GIB) given back for an allocation that found
    /// the card full - ggml-sycl's reclaim hook (`gpu`: the device's Level Zero index).  True when there were any (the
    /// allocation is tried once more); the next seed / regrow uploads them again where there is room.
    bool reclaim(int gpu, int64_t bytes);
    /// larry, STRATA_SERVE_KEEP_CACHE (default 1): a prompt batch of <= STRATA_SERVE_KEEP_MAX tokens gives back only the
    /// dense card's tail slabs it needs (shrink_cache_for); the decode after it refills just those (regrow_cache)
    static bool keep_cache_mode() { const char* e = std::getenv("STRATA_SERVE_KEEP_CACHE"); return !e || std::atoi(e) != 0; }
    bool release_prestage() { return false; }
    void release_chunk() {}
    /// larry, STRATA_XE_PREFETCH (default 1): a layer-major prompt batch whose layers first..last each run one
    /// run_chunk_dev of >= STRATA_XE_PREFETCH_MIN (2048) tokens declares them here; the tier then streams every arena
    /// expert of layer l + 1 onto the expert cards (a copy queue per card; pageable experts staged into pinned memory
    /// by a helper thread first) while layer l computes and the dense card runs layer l + 1, and run_chunk_dev(l + 1)
    /// reads them in place.  prefetch_end() after the batch (frees the cards' buffers).  Returns false = off.
    bool prefetch_begin(int64_t first, int64_t last, int n_call);
    void prefetch_end();
    /// larry (pipelined prompt batch, STRATA_PIPE): run_chunk_dev on a tier thread - returns at once, run_chunk_wait()
    /// joins it (false = the call failed).  `last_of_layer`: the layer's last call (its prefetch slot is recycled then).
    bool run_chunk_dev_async(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, void* out_dev,
                             bool last_of_layer, const float* x_dev = nullptr);   ///< x_dev: X on the dense card (copied card-to-card)
    bool run_chunk_wait();
    /// The lock around every SYCL enqueue of the tier's threads; the dense half takes it too while a pipelined batch
    /// has a tier call in flight (the Level Zero driver has deadlocked / faulted under concurrent enqueues).
    static std::mutex& submit_mutex();
    /// While > 0 the tier's threads hold off taking the submit lock (the dense half increments it while it waits).
    static std::atomic<int>& submit_priority();

    void prefetch(int64_t, const int32_t*, int = 16) {}
    void prefetch_multi(int64_t, const int32_t*, int, int) {}
    void advise_file(int64_t, const int32_t*, int) {}
    void spec_read(int64_t, const int32_t*, int, bool = false) {}
    void rebalance_begin_prompt() {}
    void rebalance_plan() {}
    void rebalance_tick(bool = false) {}
    void trace_open(const std::string&) {}
    void trace_mark(const float*, int) {}
    static void profiler(bool) {}

    bool run(int64_t layer, const int32_t* ids6, const float* w6, const float* x, float* out);
    bool run_to_dev(int64_t layer, const int32_t* ids6, const float* w6, const float* x, void* out_dev);
    bool run_multi(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, float* out);
    bool run_multi_to_dev(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, void* out_dev);
    /// larry, STRATA_XE_ARM: queue the expert cards' share of the coming decode / verify call of `layer` (nt tokens)
    /// before its routing is known - they wait on a release that run_to_dev / run_multi_to_dev give them once the
    /// routing is in (anything else releases them with null jobs).  The dense half calls it before its readback wait.
    void arm(int64_t layer, int nt);
    /// larry, STRATA_XE_GROUTE (default 0): the dense card routes the call itself - the dense half calls this right
    /// after queueing the layer's fn readback (ids / wts / fn: the router's outputs on the dense card); a route kernel
    /// on its queue writes every card's jobs and the CPU pool's entries, the expert cards are armed and released by the
    /// dense card, and the matching run_to_dev / run_multi_to_dev only runs the CPU part and the sum.  Only for the call
    /// right after groute_next() (the generate loop's decode / verify paths: their tier call matches the pass).
    void route(int64_t layer, int nt, const void* ids_dev, const void* wts_dev, const void* fn_dev, const float* fn_host);
    void groute_next();
    /// larry, engine1 (STRATA_TP): the residency table for the tensor-parallel decode core - pinned RtEnt
    /// [n_layer * n_expert] (the expert's blob on its card, offsets, types, card or -1), allocated on the first call and
    /// kept up to date with every residency change; *version moves with each change.  Null: it cannot be allocated.
    const void* tp_route_table(const volatile uint64_t** version);
    /// larry, engine1: the CPU pool's share of a decode / verify call the TP core routes - the picks of ids / w (nt
    /// tokens x top_k) whose expert is on no card, weighted into out [nt][n_embd] (zeroed first when there are any).
    /// Returns how many CPU-pool experts ran (0: out untouched).  Synchronous (the pool's threads).
    int tp_cpu_part(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, float* out);
    /// larry, engine1: between two TP passes (no TP kernel in flight) - the promotion cache (STRATA_XE_PROMOTE): set
    /// up once, finished copies published (the residency table moves: the next pass refreshes its copies), up to
    /// `chunks` more copy chunks queued on the tier's card queues (they run beside the next pass; the slots they fill
    /// were taken off the table before that pass)
    void tp_between_passes(int chunks);
    /// larry, engine1: the queues (sycl::queue*, by card) the promotion copies go to - the TP core's own: in order with
    /// its passes (a second queue on a card whose TP queue spins between its cards barely ran: promotions stalled)
    void tp_set_promote_queues(const std::vector<void*>& qs);
    /// larry, engine1: the table keys (layer * n_expert + expert) changed since the last call; true = take every key
    bool tp_take_dirty(std::vector<uint32_t>& keys);
    /// larry, engine1 (STRATA_TP_LAZY_KV): `bytes` of the TP core's VRAM on GPU `gpu` are not mapped yet - the card's
    /// flex region (promotion slots) holds them meanwhile.  Before placement.
    void tp_defer_vram(int gpu, size_t bytes);
    /// larry, engine1: room for `bytes` more on GPU `gpu` (the TP core maps KV rows): the flex region shrinks when the
    /// card's free VRAM would drop below STRATA_XE_FLEX_MARGIN_GIB.  Between passes.  False: no room.
    bool tp_make_room(int gpu, size_t bytes);
    /// larry, engine1: rows [r0, r1) of weight `w` (the dense card's reordered tensor) are on GPU `gpu` already as one
    /// planar slice at `p` (the TP core's shard): the LM head split (mv_hook) reads it instead of a copy.  Before use.
    void tp_head_slice(const void* w, int gpu, int64_t r0, int64_t r1, const void* p);
    /// larry, engine1: the dense card's expert cap (STRATA_XE_DENSE_GIB) raised by `bytes` (card 0's dense weights
    /// the pager gave back).  Before the dense card's seed.
    void tp_dense_bonus(size_t bytes);
    /// debug (STRATA_DUMP_COL): a copy of `bytes` of dense-card memory at `src` into pinned capture `slot`, queued on the
    /// dense queue (no wait); dbg_take waits for that queue and returns the bytes
    void dbg_capture(int slot, const void* src, size_t bytes);
    std::vector<uint8_t> dbg_take(int slot);
    bool run_chunk(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, float* out);
    bool run_chunk_dev(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, void* out_dev);

    void close();
    const Ds41MoeStats& stats() const;
    void reset_stats();
    void add_stats(const Ds41MoeStats& s);
    const Ds41MoeGeom& geom() const;
    const Ds41MoeConfig& config() const;
    const char* mode() const;
    int64_t resident() const;
    int64_t arena_experts() const;
    int64_t file_tier() const;
    double arena_gib() const;
    int64_t blob_bytes(int64_t layer) const;
    bool resident(int64_t layer, int64_t expert) const;
    bool in_arena(int64_t layer, int64_t expert) const;
    /// One line per card: experts, GiB, free VRAM (the startup report).
    std::string report() const;

private:
    std::unique_ptr<Ds41XeImpl> im_;
};

}  // namespace strata::v41
