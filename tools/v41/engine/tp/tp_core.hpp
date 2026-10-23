// tools/v41/engine/tp/tp_core.hpp - larry (DeepSeek-V4.1 on N Intel Arc B70s): engine1's tensor-parallel decode core.
//
// One decode / verify pass (n <= 7 tokens) of the whole 40-layer trunk with the dense half split over N cards (any N:
// the splits are computed from N and the shapes) and the host out of the loop: per card the pass is one SYCL graph,
// replayed (N graph launches a pass); the cards meet twice a layer (the attention output and the FFN output are sums
// of the cards' partials: copy-engine P2P copies + sequence flags + bounded spins, the consumer adds the N partials in
// rank order, so every card's replicated state stays bit-identical).  See engine1/DESIGN.md.
//
//   rank k: q_b rows / attention / wo_a of its o-groups, wo_b's matching K columns, the shared expert's n_ff blocks
//           (gate / up rows, down columns), its rows of the Engram projection (all-gathered) and of the LM head;
//           everything else (hc mixes, norms, q_a + kv, the window ring, the compressor, the indexer, the router)
//           replicated - same kernels on the same bits, so the same results on every card.
//
// Phase A (this file's first form): the core owns all its state (a bench fills it); routed experts not yet.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::v41 {

/// Rank 0's persistent state of one layer: the dense half's tensors on card 0 (ggml-sycl buffers, same context), so
/// the prompt path, park / snapshot and the next pass see what a TP pass wrote.  Null members: the layer has none.
struct Ds41TpLayerState {
    float * raw = nullptr;          ///< [RAW][DH] f32 window ring
    uint16_t * comp = nullptr;      ///< [cmax + 1][DH] bf16 compressed KV (kv sources)
    uint16_t * idxk = nullptr;      ///< [cmax + 1][ID] bf16 indexer keys (kv sources)
    float * st_kv = nullptr, * st_sc = nullptr;   ///< [2][DH] f32 the ratio-2 compressor state
};

/// What the core takes from the engine (all optional: a bench runs without them)
struct Ds41TpHooks {
    /// rank 0's state, one entry a layer (empty: the core owns rank 0's state too)
    std::vector<Ds41TpLayerState> state;
    /// a pass's Engram rows (tids[0..n) at pos0..; out[e] = n * cols * dim floats, slot e = the e-th Engram layer),
    /// blocking - called right after the pass is launched; the cards wait for it at the Engram layers
    std::function<bool(const int * tids, int n, int pos0, float * const * out)> engram;
    /// optional: the same for one slot (0 / 1) - the first Engram layer's gate opens before the second's rows are read
    /// (engram is then only the fallback); the tokens are recorded by the slot 0 call
    std::function<bool(const int * tids, int n, int pos0, int slot, float * out)> engram_slot;
    /// the expert tier's pinned residency table (RtEnt [n_layer * n_expert]: blob on its card, offsets, types, card or
    /// -1) and its change counter: the cards' copies are refreshed when it moved (between passes)
    const void * rtab = nullptr;
    const volatile uint64_t * rtab_version = nullptr;
    /// optional: the keys changed since the last call (true: every key) - the refresh copies only those
    std::function<bool(std::vector<uint32_t> & keys)> rtab_dirty;
    /// rank 0's view of a dense weight the engine already holds on card 0 (the GGUF name -> its device address), in the
    /// core's layout: Q6_K planar over the whole tensor (wo_a: per group slice), F16 rows - or null (the core loads its
    /// own).  Rank 0's slices start at row 0, so they are prefixes of the engine's planes: no second copy on card 0.
    std::function<const void *(const std::string & name)> shared_weight;
    /// the CPU pool's share of layer l (the picks on no card) weighted into out [n][D] (zeroed by the callee when it
    /// runs any); returns how many experts it ran (0: out untouched)
    std::function<int(int l, int n, const int32_t * ids, const float * w, const float * x, float * out)> cpu_part;
};

struct Ds41TpConfig {
    std::string gguf;               ///< the trunk GGUF (V41-GSQ-1017.gguf)
    int n_cards = 1;                ///< the tensor-parallel degree: ranks 0..N-1
    std::vector<int> devices;       ///< their Level Zero GPU indices (empty: 0..N-1)
    int max_n = 7;                  ///< tokens a pass (the fused attention's widths: < 8)
    int64_t ctx_max = 65536;        ///< positions the caches hold (a multiple of 256)
    std::string cache_dir;          ///< the Q8_0 -> Q6_K requantisation cache ("" = requantise at every load)
    bool act_quant = true;          ///< model.py's activation roundings (window K FP8, compressed KV / indexer FP4)
    int n_layer = -1;               ///< blocks to run (-1 = all; a prefix for quick tests)
    bool shexp = true;              ///< the shared expert (off: timing experiments only)
    int bench_experts = 0;          ///< phase A: routed experts from this many real blobs (every card the same set;
                                    ///< expert (l, e) uses blob (l * NE + e) % count, held by rank (e + l) % N); 0 = none
};

class Ds41Tp {
public:
    Ds41Tp();
    ~Ds41Tp();
    Ds41Tp(const Ds41Tp &) = delete;
    Ds41Tp & operator=(const Ds41Tp &) = delete;

    bool init(const Ds41TpConfig & cfg, std::string & err) { return init(cfg, Ds41TpHooks(), err); }
    bool init(const Ds41TpConfig & cfg, const Ds41TpHooks & hooks, std::string & err);
    /// the expert tier, once it is placed (hooks.rtab / rtab_version / cpu_part; see Ds41TpHooks): every card then
    /// computes the selected experts it holds and rank 0 adds the CPU pool's share
    bool attach_experts(const void * rtab, const volatile uint64_t * rtab_version,
                        std::function<int(int, int, const int32_t *, const float *, const float *, float *)> cpu_part,
                        std::function<bool(std::vector<uint32_t> &)> rtab_dirty = nullptr);
    /// ranks >= 1 pull rank 0's state after something else wrote it (a prompt, restore, unpark): every window ring and
    /// compressor state, and the compressed / indexer rows of positions [lo, hi) (+ the spare rows); all = every row
    bool sync_state(int64_t lo, int64_t hi, bool all);
    /// the selection-only routing bias of layer il (n_expert floats; zeros = the model's routing), every card
    void set_route_bias(int il, const float * bias);
    int64_t vocab() const;
    int64_t n_embd() const;
    int n_layer() const;

    /// bench: random persistent state (window rings, compressed rows, indexer keys, compressor state) as if positions
    /// 0..pos-1 had been processed - the same bits on every card
    void fill_state(int64_t pos, uint32_t seed);
    /// one pass: tokens tids[0..n) at positions pos0..pos0+n-1.  logits: n * vocab floats (host) or null.  graph: replay
    /// the pass's recorded SYCL graphs (recorded at the first pass of its (n, cap bucket)); else eager submission.
    /// Engram rows: zeros in phase A (the bench has no Engram tables).
    bool pass(const int * tids, int n, int pos0, float * logits, bool graph) { return pass(tids, n, pos0, logits, nullptr, graph); }
    /// main_hidden (optional): n * 3 * n_embd floats - DSpark's taps, the HC-mean of the state leaving layers 36 / 37 / 38
    bool pass(const int * tids, int n, int pos0, float * logits, float * main_hidden, bool graph);
    /// debug: every card's intermediate buffers of the last pass (its last layer) -> dir/r<k>_<name>.bin (floats)
    bool dump(const std::string & dir, int n);
    /// the last pass's per-card GPU time (us, from the first to the last command; eager mode with profiling off: 0)
    std::string report() const;
    const std::string & last_error() const;
    /// the ranks' in-order queues (sycl::queue*): work queued there runs between passes, in order with them
    std::vector<void *> queues() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::v41
