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

#include "ds41_moe.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

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
    void tl_tag(int id);
    void tl_flush();
    void set_cache_enabled(bool on);
    void release_cache();
    int64_t shrink_cache_for(int64_t need_bytes, std::string& err);
    int64_t regrow_cache(std::string& err);
    static bool keep_cache_mode() { return false; }
    bool release_prestage() { return false; }
    void release_chunk() {}

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
