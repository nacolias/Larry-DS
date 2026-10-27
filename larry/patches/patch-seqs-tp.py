#!/usr/bin/env python3
# patch-seqs-tp.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, stage 1): the TP core holds several
# conversations side by side (Ds41TpConfig::n_seqs, default 1 = as before, bit for bit).  Each conversation has its own
# KV replicas on ranks >= 1 (window rings, lazily mapped compressed rows / indexer keys, compressor states, rows_done),
# rank 0's state from the dense half's per-conversation tensors (hooks.seq_state), its own draft rings and draft-window
# map, and its own recorded graphs (they bake the state addresses).  bind_seq(s) swaps them in between passes - no copy,
# no re-recording; the per-pass scratch (sel / selm, the taps, the input block) is shared.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
T = os.path.join(X, "tools/v41/engine/tp")


def patch(path, pairs, mark):
    s = open(path).read()
    if mark in s:
        print("already patched", path)
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit("%s: anchor x%d: %s" % (path, s.count(old), old[:90]))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


patch(os.path.join(T, "tp_core.hpp"), [
("""    /// rank 0's state, one entry a layer (empty: the core owns rank 0's state too)
    std::vector<Ds41TpLayerState> state;
""",
"""    /// rank 0's state, one entry a layer (empty: the core owns rank 0's state too)
    std::vector<Ds41TpLayerState> state;
    /// larry (batch slots, patch-seqs-tp.py): rank 0's state of conversations 1 .. n_seqs - 1, as `state` (conversation 0's)
    std::vector<std::vector<Ds41TpLayerState>> seq_state;
"""),
("""    int bench_experts = 0;          ///< phase A: routed experts from this many real blobs (every card the same set;
""",
"""    int n_seqs = 1;                 ///< larry (batch slots): conversations held side by side (bind_seq)
    int bench_experts = 0;          ///< phase A: routed experts from this many real blobs (every card the same set;
"""),
("""    int n_cards() const;

""",
"""    int n_cards() const;
    /// larry (batch slots, patch-seqs-tp.py): the conversation pass / sync_state / the draft work on from here on - its
    /// KV replicas, rings, draft window and recorded graphs are its own (no copy, no re-recording).  Between passes only.
    bool bind_seq(int s);
    int n_seqs() const;
    int seq() const;

"""),
], "bind_seq")

patch(os.path.join(T, "tp_core.cpp"), [
# Card: the other conversations' state and graphs
("""    // graphs: (n, cap1, cap2) -> the recorded pass
    std::map<std::tuple<int, int64_t, int64_t>, sx::command_graph<sx::graph_state::executable>> graphs;
};
""",
"""    // graphs: (n, cap1, cap2) -> the recorded pass
    std::map<std::tuple<int, int64_t, int64_t>, sx::command_graph<sx::graph_state::executable>> graphs;
    // larry (batch slots, patch-seqs-tp.py): the other conversations' state and graphs - bind_seq swaps them with ls /
    // graphs (entry [seq] is empty while that conversation is bound)
    std::vector<std::vector<LS>> ls_seq;
    std::vector<std::map<std::tuple<int, int64_t, int64_t>, sx::command_graph<sx::graph_state::executable>>> graphs_seq;
};
"""),
# Impl members
("""    bool lazy = false;
    int64_t rows_done[3] = { 0, 0, 0 };
    std::function<bool(int, size_t)> make_room;   // the tier: room for that many more bytes on that GPU
""",
"""    bool lazy = false;
    int64_t rows_done[3] = { 0, 0, 0 };
    std::function<bool(int, size_t)> make_room;   // the tier: room for that many more bytes on that GPU
    // larry (batch slots, patch-seqs-tp.py): NS conversations, `seq` bound; the others' rows_done / draft maps here
    int NS = 1, seq = 0;
    std::vector<std::array<int64_t, 3>> rows_done_seq;
    std::vector<std::vector<int>> d_filled_seq;
"""),
# the destructor: every conversation's lazy buffers
("""            for (auto & s : c->ls) { s.vcomp.release(zc); s.vidxk.release(zc); }
""",
"""            for (auto & s : c->ls) { s.vcomp.release(zc); s.vidxk.release(zc); }
            for (auto & v : c->ls_seq) for (auto & s : v) { s.vcomp.release(zc); s.vidxk.release(zc); }
            for (auto & g : c->graphs_seq) g.clear();
"""),
# the persistent state: a function of the conversation
("""        // persistent state
        c.ls.assign((size_t) NL, {});
        for (int l = 0; l < NL; ++l) {
            LS & s = c.ls[(size_t) l];
            const LInfo & L = li[(size_t) l];
            if (c.rank == 0 && !hooks.state.empty()) {   // the dense half's own tensors on card 0
                const Ds41TpLayerState & e = hooks.state[(size_t) l];
""",
"""        // persistent state (larry, batch slots: one set a conversation - conversation 0's in c.ls, the others' in
        // c.ls_seq; the per-pass scratch sel / selm is conversation 0's for all)
        c.ls_seq.assign((size_t) NS, {});
        c.graphs_seq.assign((size_t) NS, {});
        for (int q = 0; q < NS; ++q) {
        std::vector<LS> & lsq = q == 0 ? c.ls : c.ls_seq[(size_t) q];
        const std::vector<Ds41TpLayerState> * stq = q == 0 ? &hooks.state :
            (size_t) q - 1 < hooks.seq_state.size() ? &hooks.seq_state[(size_t) q - 1] : nullptr;
        if (c.rank == 0 && !hooks.state.empty() && (!stq || stq->size() < (size_t) NL)) {
            err = "tp: conversation " + std::to_string(q) + ": the engine gave no state for it";
            return false;
        }
        lsq.assign((size_t) NL, {});
        for (int l = 0; l < NL; ++l) {
            LS & s = lsq[(size_t) l];
            const LInfo & L = li[(size_t) l];
            if (c.rank == 0 && !hooks.state.empty()) {   // the dense half's own tensors on card 0
                const Ds41TpLayerState & e = (*stq)[(size_t) l];
"""),
("""            if (L.idx_src) { s.sel = dalloc<int32_t>(c, n * (size_t) ITOP); s.selm = dalloc<float>(c, n * (size_t) ITOP); }
        }
        if (!err.empty()) return false;
""",
"""            if (L.idx_src) {
                if (q == 0) { s.sel = dalloc<int32_t>(c, n * (size_t) ITOP); s.selm = dalloc<float>(c, n * (size_t) ITOP); }
                else { s.sel = c.ls[(size_t) l].sel; s.selm = c.ls[(size_t) l].selm; }
            }
        }
        if (!err.empty()) return false;
        }
"""),
# deferred / mapped: every conversation's
("""    size_t deferred(const Card & c) const {
        size_t b = 0;
        for (const LS & s : c.ls) {
            if (s.vcomp.va) b += s.vcomp.cap - s.vcomp.mapped();
            if (s.vidxk.va) b += s.vidxk.cap - s.vidxk.mapped();
        }
        return b;
    }
""",
"""    size_t deferred(const Card & c) const {
        size_t b = 0;
        auto add = [&](const std::vector<LS> & v) {
            for (const LS & s : v) {
                if (s.vcomp.va) b += s.vcomp.cap - s.vcomp.mapped();
                if (s.vidxk.va) b += s.vidxk.cap - s.vidxk.mapped();
            }
        };
        add(c.ls);
        for (const auto & v : c.ls_seq) add(v);   // (larry, batch slots: every conversation's)
        return b;
    }
    // larry (batch slots, patch-seqs-tp.py): conversation s's state, graphs, rows_done and draft rings / window in
    // place of the bound one's (swaps: the bound one's go to its own entries)
    bool bind_seq(int s) {
        if (s < 0 || s >= NS) { err = "tp: no conversation " + std::to_string(s); return false; }
        if (s == seq) return true;
        for (auto & cp : cards) {
            Card & c = *cp;
            std::swap(c.ls, c.ls_seq[(size_t) seq]);
            std::swap(c.ls, c.ls_seq[(size_t) s]);
            std::swap(c.graphs, c.graphs_seq[(size_t) seq]);
            std::swap(c.graphs, c.graphs_seq[(size_t) s]);
        }
        for (auto & d : dc) {
            if (d.ring_seq.size() != (size_t) NS) continue;
            for (int b = 0; b < 3; ++b) {
                std::swap(d.ring[b], d.ring_seq[(size_t) seq][(size_t) b]);
                std::swap(d.ring[b], d.ring_seq[(size_t) s][(size_t) b]);
            }
            std::swap(d.graphs, d.graphs_seq[(size_t) seq]);
            std::swap(d.graphs, d.graphs_seq[(size_t) s]);
        }
        rows_done_seq[(size_t) seq] = { rows_done[0], rows_done[1], rows_done[2] };
        for (int r = 0; r < 3; ++r) rows_done[r] = rows_done_seq[(size_t) s][(size_t) r];
        std::swap(d_filled, d_filled_seq[(size_t) seq]);
        std::swap(d_filled, d_filled_seq[(size_t) s]);
        if (d_filled.empty() && !d_filled_seq[(size_t) seq].empty()) d_filled.assign(DWIN, -1);   // its first draft
        seq = s;
        return true;
    }
"""),
("""    size_t lazy_mapped(const Card & c) const {
        size_t b = 0;
        for (const LS & s : c.ls) b += s.vcomp.mapped() + s.vidxk.mapped();
        return b;
    }
""",
"""    size_t lazy_mapped(const Card & c) const {
        size_t b = 0;
        for (const LS & s : c.ls) b += s.vcomp.mapped() + s.vidxk.mapped();
        for (const auto & v : c.ls_seq) for (const LS & s : v) b += s.vcomp.mapped() + s.vidxk.mapped();   // (batch slots)
        return b;
    }
"""),
# the draft card: rings and graphs a conversation
("""        uint64_t * stamps = nullptr;                       // STRATA_TP_DRAFT_STAMP: the clock after each phase
        std::map<int, sx::command_graph<sx::graph_state::executable>> graphs;
    };
""",
"""        uint64_t * stamps = nullptr;                       // STRATA_TP_DRAFT_STAMP: the clock after each phase
        std::map<int, sx::command_graph<sx::graph_state::executable>> graphs;
        // larry (batch slots, patch-seqs-tp.py): the other conversations' rings and graphs (swapped by bind_seq)
        std::vector<std::array<float *, 3>> ring_seq;
        std::vector<std::map<int, sx::command_graph<sx::graph_state::executable>>> graphs_seq;
    };
"""),
("""        for (int b = 0; b < 3; ++b) d.ring[b] = dalloc<float>(c, (size_t) (DWIN * DH));
""",
"""        for (int b = 0; b < 3; ++b) d.ring[b] = dalloc<float>(c, (size_t) (DWIN * DH));
        d.ring_seq.assign((size_t) NS, { nullptr, nullptr, nullptr });   // (larry, batch slots: a window a conversation)
        d.graphs_seq.assign((size_t) NS, {});
        for (int q = 0; q < NS; ++q)
            if (q != seq)
                for (int b = 0; b < 3; ++b) d.ring_seq[(size_t) q][(size_t) b] = dalloc<float>(c, (size_t) (DWIN * DH));
"""),
# init: the conversation count and every conversation's rows_done
("""    for (int r = 1; r <= 2; ++r) im.rows_done[r] = im.lazy ? std::min<int64_t>(im.cmax[r], 1024) : im.cmax[r];
""",
"""    for (int r = 1; r <= 2; ++r) im.rows_done[r] = im.lazy ? std::min<int64_t>(im.cmax[r], 1024) : im.cmax[r];
    im.NS = std::max(1, std::min(8, cfg.n_seqs));   // larry (batch slots, patch-seqs-tp.py)
    im.rows_done_seq.assign((size_t) im.NS, { im.rows_done[0], im.rows_done[1], im.rows_done[2] });
    im.d_filled_seq.assign((size_t) im.NS, {});
"""),
# attach: every conversation's graphs go, and are pre-recorded for each
("""    for (auto & c : im.cards) c->graphs.clear();   // recorded without the expert part
""",
"""    for (auto & c : im.cards) {   // recorded without the expert part (larry, batch slots: every conversation's)
        c->graphs.clear();
        for (auto & g : c->graphs_seq) g.clear();
    }
"""),
("""    if (env_on("STRATA_TP_PRERECORD", true)) {
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
""",
"""    if (env_on("STRATA_TP_PRERECORD", true)) {
        const double t0 = now_us();
        int nk = 0;
        const int s0 = im.seq;
        for (int q = 0; q < im.NS; ++q) {   // (larry, batch slots: each conversation's own - they bake its state)
            if (!im.bind_seq(q)) return false;
            for (int n = 1; n <= im.NMAX; ++n)
                for (int taps = 0; taps < 2; ++taps) {
                    Impl::PassK P;
                    P.n = n;
                    im.caps(0, n, P.cap);
                    if (!im.record(P, std::make_tuple(n * 2 + taps, P.cap[1], P.cap[2]))) return false;
                    ++nk;
                }
        }
        if (!im.bind_seq(s0)) return false;
        im.pf_rec = 0;
"""),
# draft_init: the trunk's graphs of every conversation go
("""    for (auto & c : im.cards) c->graphs.clear();
""",
"""    for (auto & c : im.cards) {   // (larry, batch slots: every conversation's)
        c->graphs.clear();
        for (auto & g : c->graphs_seq) g.clear();
    }
"""),
# the public API
("""Ds41Tp::Ds41Tp() : p_(std::make_unique<Impl>()) {}
""",
"""Ds41Tp::Ds41Tp() : p_(std::make_unique<Impl>()) {}
bool Ds41Tp::bind_seq(int s) { return p_->bind_seq(s); }
int Ds41Tp::n_seqs() const { return p_->NS; }
int Ds41Tp::seq() const { return p_->seq; }
"""),
], "bind_seq")
