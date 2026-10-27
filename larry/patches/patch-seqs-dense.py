#!/usr/bin/env python3
# patch-seqs-dense.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, stage 1): the dense half (card 0) and the
# Engram history hold several conversations side by side (Ds41DenseConfig::n_seqs, default 1 = as before, bit for bit).
# Each conversation has its own window ring, snapshot rings, compressed KV / indexer keys (in the lazy KV map when it is
# on: one set of units a conversation, mapped as that conversation grows), compressor states, snapshot position and
# written range; bind_seq(s) points the layers' state members at conversation s's tensors (the prompt path builds its
# graphs per span; the kept decode graphs are keyed by the conversation too).  A park holds (layer, kind) regions, so a
# parked conversation comes back into whichever conversation slot is bound.  The Engram history is one region a
# conversation (sized at once: a prefetch's planner thread reads it - it keeps the region it started with).  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
E = os.path.join(X, "tools/v41/engine")


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


patch(os.path.join(E, "ds41_engram.hpp"), [
("""    /// Record token `tid` at position `pos` (history for the hashes; positions may be rewritten).
    void set_token(int64_t pos, int tid);
""",
"""    /// Record token `tid` at position `pos` (history for the hashes; positions may be rewritten).
    void set_token(int64_t pos, int tid);
    /// larry (batch slots, patch-seqs-dense.py): `n` conversations' histories, `ctx` positions each, sized now (a
    /// prefetch's planner thread reads the history: no resize under it); bind_seq(s) = the one set_token / hash use
    void set_seqs(int n, int64_t ctx);
    void bind_seq(int s);
"""),
("""    std::vector<int64_t> hist_;      // compressed ids by position
""",
"""    std::vector<int64_t> hist_;      // compressed ids by position
    int64_t hbase_ = 0, hstride_ = 0;   // larry (batch slots): the bound conversation's region of hist_ (stride 0: one)
    int nseq_ = 1;
    void hash_at(int64_t base, int64_t pos, int e, int64_t * out) const;
"""),
("""    void prefetch_run(Batch * b, int64_t pos0, int n);   // larry: prefetch()'s planning + reads, on its own thread
""",
"""    void prefetch_run(Batch * b, int64_t pos0, int n, int64_t hbase);   // larry: prefetch()'s planning + reads, on its own thread
"""),
], "set_seqs")

patch(os.path.join(E, "ds41_engram.cpp"), [
("""void Ds41Engram::set_token(int64_t pos, int tid) {
    if (pos >= (int64_t) hist_.size()) hist_.resize((size_t) pos + 1024, 0);
    // larry (vision, patch-vision-engine.py): tid < 0 = an image position - model.py's DEAD: no n-gram crosses it
    hist_[(size_t) pos] = tid < 0 ? (int64_t) -1 : (int64_t) g_->engram_token_map[(size_t) tid];
}

void Ds41Engram::hash(int64_t pos, int e, int64_t * out) const {
    const int64_t N = g_->engram_ngram, H = g_->engram_heads;
    int64_t tok[8];
    // the look-back stops at the sequence's start and (larry, vision) at an image position: pad from there on
    bool blocked = false;
    for (int64_t k = 0; k < N; ++k) {
        if (pos - k < 0 || hist_[(size_t) (pos - k)] < 0) blocked = true;
        tok[k] = blocked ? g_->engram_pad : hist_[(size_t) (pos - k)];
    }
""",
"""void Ds41Engram::set_seqs(int n, int64_t ctx) {
    nseq_ = std::max(1, n);
    hstride_ = nseq_ > 1 ? ctx : 0;
    hbase_ = 0;
    if (nseq_ > 1) hist_.assign((size_t) (nseq_ * ctx), 0);
}

void Ds41Engram::bind_seq(int s) {
    if (s >= 0 && s < nseq_) hbase_ = (int64_t) s * hstride_;
}

void Ds41Engram::set_token(int64_t pos, int tid) {
    if (hstride_ > 0) {   // larry (batch slots): the bound conversation's region, sized at set_seqs
        if (pos < 0 || pos >= hstride_) return;
    } else if (pos >= (int64_t) hist_.size()) {
        hist_.resize((size_t) pos + 1024, 0);
    }
    // larry (vision, patch-vision-engine.py): tid < 0 = an image position - model.py's DEAD: no n-gram crosses it
    hist_[(size_t) (hbase_ + pos)] = tid < 0 ? (int64_t) -1 : (int64_t) g_->engram_token_map[(size_t) tid];
}

void Ds41Engram::hash(int64_t pos, int e, int64_t * out) const { hash_at(hbase_, pos, e, out); }

void Ds41Engram::hash_at(int64_t base, int64_t pos, int e, int64_t * out) const {
    const int64_t N = g_->engram_ngram, H = g_->engram_heads;
    int64_t tok[8];
    // the look-back stops at the sequence's start and (larry, vision) at an image position: pad from there on
    bool blocked = false;
    for (int64_t k = 0; k < N; ++k) {
        if (pos - k < 0 || hist_[(size_t) (base + pos - k)] < 0) blocked = true;
        tok[k] = blocked ? g_->engram_pad : hist_[(size_t) (base + pos - k)];
    }
"""),
("""    Batch * bp = b.get();
    b->th = std::thread([this, bp, pos0, n] { prefetch_run(bp, pos0, n); });
""",
"""    Batch * bp = b.get();
    const int64_t hb = hbase_;   // larry (batch slots): the planner hashes this conversation's history, whatever is bound later
    b->th = std::thread([this, bp, pos0, n, hb] { prefetch_run(bp, pos0, n, hb); });
"""),
("""void Ds41Engram::prefetch_run(Batch * b, int64_t pos0, int n) {
""",
"""void Ds41Engram::prefetch_run(Batch * b, int64_t pos0, int n, int64_t hbase) {
"""),
("""        for (int64_t t = 0; t < n; ++t) hash(pos0 + t, (int) e, &ids[(size_t) (t * cols)]);
        for (int64_t i = 0; i < m; ++i) r[(size_t) i] = (uint32_t) ids[(size_t) i];
""",
"""        for (int64_t t = 0; t < n; ++t) hash_at(hbase, pos0 + t, (int) e, &ids[(size_t) (t * cols)]);
        for (int64_t i = 0; i < m; ++i) r[(size_t) i] = (uint32_t) ids[(size_t) i];
"""),
], "set_seqs")

patch(os.path.join(E, "ds41_dense.hpp"), [
("""    int64_t ctx_max = 65536;            ///< positions the caches hold (compressed rows = ctx_max / ratio)
""",
"""    int64_t ctx_max = 65536;            ///< positions the caches hold (compressed rows = ctx_max / ratio)
    int n_seqs = 1;                     ///< larry (batch slots): conversations held side by side (bind_seq)
"""),
("""    size_t park_bytes_for(int64_t positions) const;
""",
"""    size_t park_bytes_for(int64_t positions) const;
    /// larry (batch slots, patch-seqs-dense.py): cfg.n_seqs conversations side by side - each its own window ring,
    /// compressed KV, indexer keys, compressor states, snapshot and written range.  bind_seq(s): every later call (prompt
    /// batches, snapshot / restore, park / unpark, state_ptrs, kv_bytes_for, take_written, the Engram history) works on
    /// conversation s.  Not inside a pass or a batch.
    bool bind_seq(int s);
    int n_seqs() const;
    int seq() const;
"""),
], "bind_seq")

patch(os.path.join(E, "ds41_dense.cpp"), [
# park regions by (layer, kind); the conversations' state
("""    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; };
""",
"""    // (larry, batch slots: by layer l and kind k - 0 raw_snap, 1 st_kv_snap, 2 st_sc_snap, 3 comp, 4 idxk - so a parked
    // conversation comes back into whichever conversation is bound)
    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; int l = -1, k = -1; };
"""),
("""    std::vector<ParkSlot> pk;
    std::vector<Layer> ly;
""",
"""    std::vector<ParkSlot> pk;
    std::vector<Layer> ly;
    // larry (batch slots, patch-seqs-dense.py): every conversation's state tensors (per layer: raw, raw_snap, comp, idxk,
    // st_kv, st_sc, st_kv_snap, st_sc_snap - the Layer members are the bound one's), snapshot position, written range
    struct SeqT { std::vector<std::array<ggml_tensor *, 8>> t; int snap_pos = -1; int64_t wr_lo = 0, wr_hi = INT64_MAX; bool wr_changed = true; };
    std::vector<SeqT> seqs;
    int seq = 0;
    std::vector<int> kv_unit_seq;   // the conversation of each kv_units entry
"""),
# the kept graphs: by conversation too
("""    struct GKey { int il, kind; int64_t cap, n; bool operator<(const GKey & o) const {
        return std::tie(il, kind, cap, n) < std::tie(o.il, o.kind, o.cap, o.n); } };
""",
"""    struct GKey { int il, kind; int64_t cap, n; int seq; bool operator<(const GKey & o) const {   // (larry: + the conversation)
        return std::tie(il, kind, cap, n, seq) < std::tie(o.il, o.kind, o.cap, o.n, o.seq); } };
"""),
("""        const Impl::GKey key { -1, 0, 0, (int64_t) n };
""",
"""        const Impl::GKey key { -1, 0, 0, (int64_t) n, im.seq };
"""),
("""        const Impl::GKey ka { il, 11, cap, (int64_t) n }, kb { il, 12, cap, (int64_t) n };
""",
"""        const Impl::GKey ka { il, 11, cap, (int64_t) n, im.seq }, kb { il, 12, cap, (int64_t) n, im.seq };
"""),
("""        const Impl::GKey key { il, im.fin_pending ? 13 : 1, cap, (int64_t) n };
""",
"""        const Impl::GKey key { il, im.fin_pending ? 13 : 1, cap, (int64_t) n, im.seq };
"""),
("""        const Impl::GKey key { il, 7, 0, (int64_t) n };
""",
"""        const Impl::GKey key { il, 7, 0, (int64_t) n, im.seq };
"""),
("""    const Impl::GKey key { il, 3, 0, 1 };
""",
"""    const Impl::GKey key { il, 3, 0, 1, im.seq };
"""),
# init: the Engram histories, more tensor slots, the other conversations' tensors, their KV units, their scores
("""    if (!cfg.engram_dirs.empty() && !im.engram.open(G, cfg.engram_dirs, eng_threads, err)) return false;
""",
"""    if (!cfg.engram_dirs.empty() && !im.engram.open(G, cfg.engram_dirs, eng_threads, err)) return false;
    if (cfg.n_seqs > 1) im.engram.set_seqs(cfg.n_seqs, cfg.ctx_max);   // larry (batch slots): a history a conversation
"""),
("""    ggml_init_params sp = { ggml_tensor_overhead() * 512, nullptr, true };
""",
"""    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (512 + 384 * std::max(0, cfg.n_seqs - 1)), nullptr, true };
"""),
("""    // larry (cfg.kv_alloc, STRATA_DENSE_LAZY_KV): the compressed KV and indexer keys in the caller's virtual memory,
    // mapped as the context grows - a 1M context's rows stay out of card 0's VRAM until a conversation reaches them
    if (im.sycl && im.cfg.kv_alloc) {
        const ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(im.backend);
        std::vector<size_t> ub, spare, uo;
        for (auto & L : im.ly)
            for (ggml_tensor * t : { L.comp, L.idxk })
                if (t) {
                    im.kv_units.push_back({ t, (int) L.ratio });
                    ub.push_back(ggml_backend_buft_get_alloc_size(bt, t));
                    spare.push_back((size_t) (t->ne[1] - 1) * t->nb[1]);   // the spare row (open compressor groups)
                }
""",
"""    // larry (batch slots, patch-seqs-dense.py): conversation 0's tensors are the Layer members; the others' the same
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
"""),
("""        if (!im.kv_buf) im.kv_units.clear();
""",
"""        if (!im.kv_buf) { im.kv_units.clear(); im.kv_unit_seq.clear(); }
"""),
("""    for (auto & L : im.ly)
        if (L.st_sc) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); ggml_backend_tensor_set(L.st_sc, v.data(), 0, v.size() * 4); }
""",
"""    for (auto & S : im.seqs)   // (larry, batch slots: every conversation's)
        for (auto & a : S.t)
            if (ggml_tensor * sc = a[5]) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); ggml_backend_tensor_set(sc, v.data(), 0, v.size() * 4); }
"""),
# kv_bytes_for: the bound conversation's units
("""    for (const auto & u : im.kv_units) {   // row r of a ratio-k source holds a position >= r * k: rows [0, p / k] + a margin
        const int64_t k = u.second == 2 ? 2 : 1;
""",
"""    for (size_t i = 0; i < im.kv_units.size(); ++i) {   // row r of a ratio-k source holds a position >= r * k: rows [0, p / k] + a margin
        const auto & u = im.kv_units[i];
        if (i < im.kv_unit_seq.size() && im.kv_unit_seq[i] != im.seq) { bytes.push_back(0); continue; }   // (batch slots)
        const int64_t k = u.second == 2 ? 2 : 1;
"""),
# park: (layer, kind) regions; unpark: into the bound conversation
("""    auto put = [&](ggml_tensor * t, size_t off, size_t n) {
        Impl::ParkedRows r{ t, off, std::vector<uint8_t>(n) };
        ggml_backend_tensor_get(t, r.b.data(), off, n);
        S.rows.push_back(std::move(r));
        S.bytes += n;
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) put(t, 0, ggml_nbytes(t));
        for (ggml_tensor * t : { L.comp, L.idxk }) {
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = park_rows(t, L.ratio, positions);
            put(t, 0, (size_t) n * t->nb[1]);
            put(t, (size_t) (rows - 1) * t->nb[1], t->nb[1]);   // the spare row (open compressor groups write it)
        }
    }
""",
"""    int pl = 0, pkind = 0;   // (larry, batch slots: the region's layer and kind)
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
"""),
("""    for (auto & r : S.rows) ggml_backend_tensor_set(r.t, r.b.data(), r.off, r.b.size());
""",
"""    for (auto & r : S.rows) {   // (larry, batch slots: into the BOUND conversation's tensor of that layer and kind)
        ggml_tensor * t = r.t;
        if (r.l >= 0 && (size_t) r.l < im.ly.size()) {
            const Impl::Layer & L = im.ly[(size_t) r.l];
            ggml_tensor * const by_kind[5] = { L.raw_snap, L.st_kv_snap, L.st_sc_snap, L.comp, L.idxk };
            if (r.k >= 0 && r.k < 5 && by_kind[r.k]) t = by_kind[r.k];
        }
        ggml_backend_tensor_set(t, r.b.data(), r.off, r.b.size());
    }
"""),
# bind_seq
("""int Ds41Dense::snapshot_pos() const { return p_->snap_pos; }
""",
"""int Ds41Dense::snapshot_pos() const { return p_->snap_pos; }

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
"""),
], "bind_seq")
