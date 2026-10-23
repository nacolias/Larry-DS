// tools/v41/engine/tp/tp_pager.hpp - larry (DeepSeek-V4.1 on N Intel Arc B70s), engine1: card 0's copy of the dense
// weights the TP cards also hold, paged.
//
// With the TP core running decode / verify, only the prompt path (layer-major batches) reads card 0's full copies of
// each layer's q_b, wo_a, wo_b, shared expert (gate | up, down) and Engram projection - ~5.3 GB, which the TP ranks
// hold between them already.  The dense half allocates those tensors through alloc(): per layer one unit of Level
// Zero virtual memory, every page mapped while the weights load and reorder.  enable() then gives the pages back; a
// unit is mapped into one of a few slots when the dense half is about to queue work on its layer (need()) and filled
// from the ranks' shards - card 0 pulls them (card-to-card copies on its own copy queue: the other cards' copies into a
// remapped address land in its old pages) straight into the rows (row / slice splits), K splits through a staging
// buffer and a local scatter kernel - and the next layer's copies start meanwhile.  Its pages go to another layer once
// the work queued on it (touch()) is done.  See engine1/DESIGN.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::v41 {

class Ds41Tp;

/// A dense weight the pager holds: its layer (unit), the engine's tensor (address, Q6_K shape).  The shared expert's
/// gate and up are one entry under the fused name ("ffn_gate_up_shexp.weight", 2 n_ff rows).
struct Ds41PagedTensor {
    std::string name;
    int layer = -1;
    void * data = nullptr;
    int64_t K = 0, N = 0;
    size_t bytes = 0;
};

class Ds41Pager {
public:
    Ds41Pager();
    ~Ds41Pager();
    Ds41Pager(const Ds41Pager &) = delete;
    Ds41Pager & operator=(const Ds41Pager &) = delete;

    /// queue: the dense backend's sycl::queue * (card 0)
    bool init(void * queue, std::string & err);
    /// the dense half's allocation: unit_bytes[l] = layer l's bytes (0: none) -> unit_off[l] (filled in); returns the
    /// units' region (*total bytes; the caller puts its backend buffer over it), every page mapped until enable()
    void * alloc(const std::vector<size_t> & unit_bytes, std::vector<size_t> & unit_off, size_t & total);
    /// after the TP core is up (its ranks own every listed tensor's shards): the fill plans, the load-time pages given
    /// back, the slots made.  STRATA_TP_PAGE_CHECK=1 first refills every unit in place and compares it with the load.
    bool enable(const Ds41Tp & tp, const std::vector<Ds41PagedTensor> & tensors, std::string & err);
    /// layer's unit mapped and its fill queued on the dense queue (before the work that reads it); the next layer's
    /// copies start
    void need(int layer);
    /// after the dense half queued work on layer's unit: its pages are free once that work is done
    void touch(int layer);
    bool enabled() const;
    size_t freed() const;          ///< VRAM given back at enable() (bytes)
    std::string report() const;
    const std::string & last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::v41
