#!/usr/bin/env python3
# patch-batch-tp.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, stage 2: batched decode): the TP core runs ONE
# pass over the rows of several conversations - Ds41Tp::pass_multi(runs): run u = consecutive positions of conversation
# seqs[u], rows back to back (<= 7 in all).  The weights' work (mat-vecs, the shared and routed experts, the CPU pool,
# the LM head) takes every row at once - the decode cost that two conversations taking turns pay twice; the state kernels
# (the window row, the compressor, the indexer's scan + top-k, the attention) run once a run on offset views of the
# pass's rows, in that run's conversation's state (attn_dec's grouped q8 output: patch-batch-kern.py).  A plain pass is
# the one-run case of the same code (the bound conversation, offsets 0: the same kernels with the same arguments, the
# same graphs).  Passes of runs: their own recorded graphs, keyed by the runs' conversations, widths and scan buckets
# (recorded at first use).  Engram rows: each run's from its own conversation's history (hooks.engram_slot_seq).
# Measured first (patch-widthbench.py, 2026-10-27 19:46): two conversations' rows in one pass cost what one
# conversation's rows of that width cost.  Idempotent.
import hashlib
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
H = os.path.join(X, "tools/v41/engine/tp/tp_core.hpp")
C = os.path.join(X, "tools/v41/engine/tp/tp_core.cpp")
h = open(H).read()
c = open(C).read()
if "pass_multi" in h and "pass_multi" in c:
    print("already patched", H, C)
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


# ---------------------------------------------------------------- tp_core.hpp
h = rep(h, """    std::function<bool(const int * tids, int n, int pos0, int slot, float * out)> engram_slot;
""", """    std::function<bool(const int * tids, int n, int pos0, int slot, float * out)> engram_slot;
    /// larry (batched decode, patch-batch-tp.py): engram_slot for conversation `seq`'s history - a pass of several
    /// conversations reads each run's rows from its own (required by pass_multi when the Engram is on)
    std::function<bool(int seq, const int * tids, int n, int pos0, int slot, float * out)> engram_slot_seq;
""", "hpp hooks")
h = rep(h, """    bool pass(const int * tids, int n, int pos0, float * logits, float * main_hidden, bool graph);
""", """    bool pass(const int * tids, int n, int pos0, float * logits, float * main_hidden, bool graph);
    /// larry (batched decode, patch-batch-tp.py): one pass over several conversations - run u: ns[u] tokens of
    /// conversation seqs[u] at positions pos0s[u].. (tids: the runs' tokens back to back, sum ns <= 7, a conversation at
    /// most once).  Each run reads and writes its own conversation's state (window ring, compressor, indexer); every
    /// row shares the weights' work.  Logits / main_hidden rows in tids' order (DSpark's draft_step takes its taps by row
    /// as after a plain pass).  Its own graphs (recorded at the first pass of its shape).  The bound conversation stays bound.
    bool pass_multi(const int * tids, const int * seqs, const int * ns, const int * pos0s, int nrun, float * logits, float * main_hidden);
""", "hpp pass_multi")

# ---------------------------------------------------------------- tp_core.cpp: buffers, layout, the pass's runs
c = rep(c, """    std::vector<std::map<std::tuple<int, int64_t, int64_t>, sx::command_graph<sx::graph_state::executable>>> graphs_seq;
};
""", """    std::vector<std::map<std::tuple<int, int64_t, int64_t>, sx::command_graph<sx::graph_state::executable>>> graphs_seq;
    // larry (batched decode, patch-batch-tp.py): passes of several conversations' runs - (taps, runs, then per run its
    // conversation, width and scan buckets) -> the recorded pass (they bake every run's state: not swapped by bind_seq)
    std::map<std::vector<int64_t>, sx::command_graph<sx::graph_state::executable>> graphs_m;
};
""", "Card graphs_m")
c = rep(c, """        sidx = take(2 * (size_t) NMAX); sslot = take(2); lastblk = take(NMAX);
""", """        sidx = take(2 * (size_t) NMAX); sslot = take(2 * (size_t) NMAX); lastblk = take(NMAX);   // (larry: sslot a pair a run)
""", "InLayout sslot")
c = rep(c, """    struct PassK { int n; int64_t cap[3]; };
""", """    // larry (batched decode, patch-batch-tp.py): a run of a pass - n consecutive positions p0.. of conversation seq in
    // rows r0 .. r0 + n - 1; cap its indexer scan buckets.  A plain pass has none (nrun 0: one run, the bound conversation)
    struct Run { int seq = 0, r0 = 0, n = 0; int64_t p0 = 0; int64_t cap[3] = { 0, 0, 0 }; };
    struct PassK { int n; int64_t cap[3]; int nrun = 0; Run run[8]; };
""", "PassK")
# the destructor's and the re-record points' graph drops: the runs' graphs too
c = rep(c, """            c->graphs.clear();
            if (c->peer_blk) sycl::free(c->peer_blk, c->q);
""", """            c->graphs.clear();
            c->graphs_m.clear();
            if (c->peer_blk) sycl::free(c->peer_blk, c->q);
""", "dtor graphs_m")
old = """        for (auto & g : c->graphs_seq) g.clear();
    }
"""
if c.count(old) != 2:
    sys.exit("graph clears: anchor x%d" % c.count(old))
c = c.replace(old, """        for (auto & g : c->graphs_seq) g.clear();
        c->graphs_m.clear();   // (larry, batched decode)
    }
""")

# ---------------------------------------------------------------- emit_layer: the state kernels once a run
c = rep(c, """        const int64_t r = L.ratio, cap = r ? P.cap[r] : 0;
        const size_t ar_slot = (size_t) NMAX * D;
        ST(c, l, 0);
""", """        const int64_t r = L.ratio;
        const size_t ar_slot = (size_t) NMAX * D;
        ST(c, l, 0);
""", "emit_layer head")
a = c.index("        {\n            tpdec::KvFinArgs a;\n            a.x = c.qkv + RQ;")
end = "            tpdec::attn_dec(c.q, a);\n        }\n"
b = c.index(end, a) + len(end)
blk = c[a:b]
if hashlib.md5(blk.encode()).hexdigest() != "48129e85cedbf0a0dfacc72bf817cfea":
    sys.exit("emit_layer's state block is not the one this patch was written against")
new = r"""        // larry (batched decode, patch-batch-tp.py): the state kernels once a run (a plain pass: one run, the bound
        // conversation, offsets 0) - each in its run's conversation's state, on views from the run's first row; the
        // mat-vecs between them take every row of the pass at once.  The transient buffers (scores, top-k ids, block
        // maxima, the indexer's packed q) are reused run after run (one in-order queue); sel / selm / cand keep each row.
        Run one;
        const Run * runs = P.run;
        int nrun = P.nrun;
        if (nrun == 0) {
            one.seq = seq; one.r0 = 0; one.n = n;
            for (int i = 0; i < 3; ++i) one.cap[i] = P.cap[i];
            runs = &one;
            nrun = 1;
        }
        for (int u = 0; u < nrun; ++u) {
            const Run & R = runs[u];
            LS & S = ls_of(c, R.seq)[(size_t) l];
            tpdec::KvFinArgs a;
            a.x = c.qkv + RQ + (size_t) R.r0 * (RQ + DH); a.x_cs = RQ + DH; a.w = w.kva_norm; a.pos = inp<int32_t>(c, IL.pos) + R.r0;
            a.eps = (float) G.rms_eps;
            a.rope = rope_p(r); a.fq = cfg.act_quant; a.y = c.kvn + (size_t) R.r0 * DH; a.y_cs = DH;
            a.ring = S.raw; a.ring_cs = DH; a.slot = inp<int32_t>(c, IL.wslot) + R.r0;
            a.slot_from = R.n - (int) std::min<int64_t>(R.n, RAW); a.tab = c.rtab + (size_t) R.r0 * ND; a.D = (int) DH; a.n = R.n;
            tpdec::kv_fin(c.q, a);
        }
        ST(c, l, 5);
        // ---- the compressor (kv sources) and the indexer (index sources)
        const uint16_t * comp[8] = {};
        const int32_t * sel[8] = {};
        const float * selm[8] = {};
        int K[8] = {};
        if (r) {
            if (L.kv_src) {
                mv_f16(c, w.comp, c.xn, D, n, c.cy, w.comp.N);
                for (int u = 0; u < nrun; ++u) {
                    const Run & R = runs[u];
                    LS & S = ls_of(c, R.seq)[(size_t) l];
                    float * cy = c.cy + (size_t) R.r0 * w.comp.N;
                    tpk::CompArgs a;
                    a.y = cy; a.ratio = (int) r; a.st_kv = S.st_kv; a.st_sc = S.st_sc; a.sidx = inp<int32_t>(c, IL.sidx) + 2 * R.r0;
                    a.cnorm = w.comp_norm; a.attn_k = w.idx_k; a.knorm = w.idx_knorm;
                    a.cpos = inp<int32_t>(c, IL.cpos[r]) + R.r0; a.cslot = inp<int32_t>(c, IL.cslot[r]) + R.r0;
                    a.rope = rope_p(r); a.eps = (float) G.rms_eps; a.fq = cfg.act_quant;
                    a.comp = S.comp; a.idxk = S.idxk; a.n = R.n;
                    tpk::comp_fin(c.q, a);
                    if (r == 2) tpk::comp_state(c.q, cy, R.n, inp<int32_t>(c, IL.sslot) + 2 * u, S.st_kv, S.st_sc);
                }
            }
            ST(c, l, 6);
            bool scan = false;
            for (int u = 0; u < nrun; ++u) scan = scan || runs[u].cap[r] > 0;
            if (scan && L.idx_src) {
                mv_f16(c, w.iqb, c.qr, RQ, n, c.iq, IH * ID);
                mv_f16(c, w.iproj, c.xn, D, n, c.iw, IH);
            }
            for (int u = 0; u < nrun; ++u) {
                const Run & R = runs[u];
                const int64_t cap = R.cap[r];
                if (cap <= 0) continue;
                const LS & KS = ls_of(c, R.seq)[(size_t) L.kvl];
                comp[u] = KS.comp;
                K[u] = (int) std::min<int64_t>(cap, ITOP);
                if (L.idx_src) {
                    LS & S = ls_of(c, R.seq)[(size_t) l];
                    tpdec::LiArgs a;
                    a.n = R.n; a.q = c.iq + (size_t) R.r0 * IH * ID; a.q_h = ID; a.q_b = IH * ID; a.tab = c.rtab + (size_t) R.r0 * ND;
                    a.n_dims = (int) ND; a.fq = cfg.act_quant;
                    a.w = c.iw + (size_t) R.r0 * IH; a.w_b = IH; a.wscale = 1.0f / std::sqrt((float) (ID * IH));
                    a.k = (const tpdec::li_bf16 *) KS.idxk; a.ld_k = ID; a.NKV = cap;
                    a.nvis = inp<float>(c, IL.nvis[r]) + R.r0;
                    const int64_t nblk = cap / CBS, ncb = std::min<int64_t>(CB, nblk);
                    if (L.uses_cand) { a.cand = c.cand + (size_t) R.r0 * CB; a.cand_b = (size_t) CB; a.ncb = (int) ncb; }
                    a.o = c.score; a.ld_o = cap;
                    if (L.cand_src) { a.lastblk = inp<float>(c, IL.lastblk) + R.r0; a.bmax = c.bmax; a.ld_bm = nblk; }
                    a.qp = c.qp; a.ws = c.ws; a.bits = c.bits;
                    tpdec::li_prep(c.q, a);
                    tpdec::li_run(c.q, a);
                    if (L.cand_src) topk(c, c.bmax, nblk, (int) nblk, R.n, (int) ncb, c.cand + (size_t) R.r0 * CB, CB);
                    topk(c, c.score, cap, (int) cap, R.n, K[u], c.tkid, K[u]);
                    tpdec::SelArgs s;
                    s.ids = c.tkid; s.ids_b = (size_t) K[u]; s.nvis = inp<float>(c, IL.nvis[r]) + R.r0;
                    s.sel = S.sel + (size_t) R.r0 * ITOP; s.sel_b = (size_t) ITOP; s.selm = S.selm + (size_t) R.r0 * ITOP; s.selm_b = (size_t) ITOP;
                    s.K = K[u]; s.n = R.n;
                    tpdec::sel_fin(c.q, s);
                    sel[u] = s.sel; selm[u] = s.selm;
                } else {
                    const LS & IS = ls_of(c, R.seq)[(size_t) L.idxl];
                    sel[u] = IS.sel + (size_t) R.r0 * ITOP; selm[u] = IS.selm + (size_t) R.r0 * ITOP;
                }
            }
            ST(c, l, 7);
        } else {
            ST(c, l, 6);
            ST(c, l, 7);
        }
        // ---- attention over this card's heads (+ its q8 for wo_a: the pass's rows), the output projection -> partial A
        for (int u = 0; u < nrun; ++u) {
            const Run & R = runs[u];
            const LS & S = ls_of(c, R.seq)[(size_t) l];
            tpdec::AttnArgs a;
            a.q = c.qh + (size_t) R.r0 * c.nhk * DH; a.q_h = DH; a.q_b = (size_t) (c.nhk * DH); a.tab = c.rtab + (size_t) R.r0 * ND;
            a.n_dims = (int) ND;
            a.ring = S.raw; a.ring_cs = DH; a.RAW = (int) RAW; a.rows = c.kvn + (size_t) R.r0 * DH; a.rows_cs = DH;
            a.widx = inp<int32_t>(c, IL.widx) + (size_t) R.r0 * SWA; a.wi_b = SWA;
            a.wmask = inp<float>(c, IL.wmask) + (size_t) R.r0 * SWA; a.wm_b = SWA;
            if (comp[u] && sel[u]) {
                a.comp = comp[u]; a.comp_t = 1; a.comp_rb = (size_t) DH * 2;
                a.sel = sel[u]; a.sel_b = ITOP; a.selm = selm[u]; a.sm_b = ITOP; a.K = K[u];
            }
            a.swa = (int) SWA; a.sinks = w.sinks; a.o = c.o + (size_t) R.r0 * c.nhk * DH; a.o_h = DH; a.o_b = (size_t) (c.nhk * DH);
            a.NH = (int) c.nhk; a.n = R.n; a.scale = 1.0f / std::sqrt((float) DH); a.part = c.apart;
            a.hpg = (int) (OGD / DH); a.oq = c.oq; a.od = c.od; a.row0 = R.r0; a.n_rows = n;
            if (xmx(w.woa, n)) { a.oT = c.oT; a.odT = c.odT; }
            tpdec::attn_dec(c.q, a);
        }
"""
c = c[:a] + new + c[b:]

# ---------------------------------------------------------------- the inputs of a pass of runs; record; ls_of; run_pass
a = c.index("    void make_inputs(const int * tids, int n, int pos0) {\n")
end = """            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(embd) + (size_t) tids[t] * (size_t) D, emb + (size_t) t * D, D);
    }
"""
b = c.index(end, a) + len(end)
blk = c[a:b]
if hashlib.md5(blk.encode()).hexdigest() != "1ed89ad45ca23c163bcaa48af203232c":
    sys.exit("make_inputs is not the one this patch was written against")
c = c[:a] + r"""    // larry (batched decode, patch-batch-tp.py): the inputs of a pass of runs (consecutive positions of one conversation
    // each, rows back to back; a plain pass is one run).  Within a run the maps index its own rows (its state kernels
    // take views from its first row) and its conversation's ring / state rows; the ratio-2 state slots: a pair a run.
    void make_inputs(const int * tids, int n, int pos0) {
        Run one;
        one.n = n;
        one.p0 = pos0;
        make_inputs_runs(tids, &one, 1);
    }
    void make_inputs_runs(const int * tids, const Run * runs, int nrun) {
        std::memset(hin, 0, IL.emb * 4);   // the maps (the embedding rows are all written below)
        if (++pid >= (1u << 19)) pid = 1;
        hin[IL.pid] = (int32_t) pid;
        int ntot = 0;
        for (int u = 0; u < nrun; ++u) {
            const int r0 = runs[u].r0, n = runs[u].n;
            const int64_t pos0 = runs[u].p0;
            ntot = std::max(ntot, r0 + n);
            int32_t * pos = hin + IL.pos + r0;
            for (int t = 0; t < n; ++t) pos[t] = (int32_t) (pos0 + t);
            int32_t * widx = hin + IL.widx + (size_t) r0 * SWA;
            float * wm = (float *) (hin + IL.wmask) + (size_t) r0 * SWA;
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
                for (int64_t j = 0; j < k; ++j) hin[IL.wslot + r0 + j] = (int32_t) ((pos0 + n - k + j) % RAW);
            }
            for (int r = 1; r <= 2; ++r) {
                float * nv = (float *) (hin + IL.nvis[r]) + r0;
                for (int t = 0; t < n; ++t) {
                    const int64_t p = pos0 + t;
                    nv[t] = (float) ((p + 1) / r);
                    const bool done = (p + 1) % r == 0;
                    hin[IL.cslot[r] + r0 + t] = (int32_t) (done ? p / r : cmax[r]);
                    hin[IL.cpos[r] + r0 + t] = (int32_t) ((p / r) * r);
                }
                if (r == 1) {
                    float * lb = (float *) (hin + IL.lastblk) + r0;
                    for (int t = 0; t < n; ++t) lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / CBS : -1);
                }
            }
            {
                auto row = [&](int64_t q) { return (int32_t) (q >= pos0 ? 2 + (q - pos0) : q % 2); };
                for (int t = 0; t < n; ++t) {
                    const int64_t p = pos0 + t, g0 = (p / 2) * 2;
                    hin[IL.sidx + 2 * (r0 + t)] = row(g0);
                    hin[IL.sidx + 2 * (r0 + t) + 1] = row(p >= g0 + 1 ? g0 + 1 : g0);
                }
                const int64_t k = std::min<int64_t>(n, 2);
                for (int64_t j = 0; j < k; ++j) hin[IL.sslot + 2 * u + j] = (int32_t) ((pos0 + n - k + j) % 2);
            }
        }
        float * emb = (float *) (hin + IL.emb);
        for (int t = 0; t < ntot; ++t)   // (the row converter: per-value calls cost ~100 us a token)
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(embd) + (size_t) tids[t] * (size_t) D, emb + (size_t) t * D, D);
    }
    // larry (batched decode): conversation s's state on card c (the bound one's is c.ls; the others' wait in ls_seq)
    std::vector<LS> & ls_of(Card & c, int s) { return s == seq ? c.ls : c.ls_seq[(size_t) s]; }
    // larry (batched decode): every card's graph of a pass of runs (key: taps, runs, per run conversation / width / caps)
    bool record_m(const PassK & P, const std::vector<int64_t> & key) {
        tap_on = key[0] != 0;
        for (auto & cp : cards) {
            Card & c = *cp;
            if (c.graphs_m.count(key)) continue;
            if (&c == cards[0].get()) ++pf_rec;
            sx::command_graph<sx::graph_state::modifiable> g(ctx, c.dev);
            g.begin_recording(c.q);
            sync_idx = 0;
            emit_prologue(c, P);
            for (int l = 0; l < NL; ++l) emit_layer(c, l, P);
            emit_epilogue(c, P);
            g.end_recording(c.q);
            if (!err.empty()) return false;
            c.graphs_m.emplace(key, g.finalize());
        }
        return true;
    }
    // larry (batched decode): a pass of runs - each run's scan buckets and its conversation's lazy replicas mapped for
    // them (each bound in turn; the bound conversation bound again after)
    bool runs_ready(PassK & P, const Run * runs, int nrun) {
        P.cap[0] = P.cap[1] = P.cap[2] = 0;
        P.nrun = nrun;
        const int s0 = seq;
        bool ok = true;
        for (int u = 0; u < nrun && ok; ++u) {
            P.run[u] = runs[u];
            ok = bind_seq(runs[u].seq);
            if (ok) {
                caps((int) runs[u].p0, runs[u].n, P.run[u].cap);
                ok = ensure_rows(P.run[u].cap);
            }
        }
        const bool back = bind_seq(s0);
        return ok && back;
    }
    // a pass: a plain one (runs null: tids at pos0.. in the bound conversation) or one of runs (Ds41Tp::pass_multi)
    bool run_pass(const int * tids, int n, int pos0, const Run * runs, int nrun, float * logits, float * main_hidden, bool graph);
""" + c[b:]

# ---------------------------------------------------------------- Ds41Tp::pass -> Impl::run_pass (+ pass_multi)
c = rep(c, """bool Ds41Tp::pass(const int * tids, int n, int pos0, float * logits, float * main_hidden, bool graph) {
    Impl & im = *p_;
    im.err.clear();
""", """bool Ds41Tp::pass(const int * tids, int n, int pos0, float * logits, float * main_hidden, bool graph) {
    return p_->run_pass(tids, n, pos0, nullptr, 0, logits, main_hidden, graph);
}

// larry (batched decode, patch-batch-tp.py)
bool Ds41Tp::pass_multi(const int * tids, const int * seqs, const int * ns, const int * pos0s, int nrun, float * logits, float * main_hidden) {
    Impl & im = *p_;
    im.err.clear();
    if (nrun < 1 || nrun > 8) { im.err = "tp: pass runs"; return false; }
    if ((im.hooks.engram || im.hooks.engram_slot) && !im.hooks.engram_slot_seq) {
        im.err = "tp: a pass of several conversations needs their Engram rows (engram_slot_seq)";
        return false;
    }
    Impl::Run runs[8];
    int n = 0;
    for (int u = 0; u < nrun; ++u) {
        if (seqs[u] < 0 || seqs[u] >= im.NS) { im.err = "tp: no conversation " + std::to_string(seqs[u]); return false; }
        for (int v = 0; v < u; ++v)
            if (seqs[v] == seqs[u]) { im.err = "tp: a conversation twice in one pass"; return false; }
        if (ns[u] < 1 || pos0s[u] < 0 || (int64_t) pos0s[u] + ns[u] > im.cfg.ctx_max) { im.err = "tp: a run past the context"; return false; }
        runs[u].seq = seqs[u];
        runs[u].r0 = n;
        runs[u].n = ns[u];
        runs[u].p0 = pos0s[u];
        n += ns[u];
    }
    return im.run_pass(tids, n, -1, runs, nrun, logits, main_hidden, true);
}

bool Ds41Tp::Impl::run_pass(const int * tids, int n, int pos0, const Run * runs, int nrun, float * logits, float * main_hidden, bool graph) {
    Impl & im = *this;
    im.err.clear();
""", "pass header")
c = rep(c, """    if (pos0 < 0 || (int64_t) pos0 + n > im.cfg.ctx_max) { im.err = "tp: pass past the context"; return false; }
""", """    if (nrun == 0 && (pos0 < 0 || (int64_t) pos0 + n > im.cfg.ctx_max)) { im.err = "tp: pass past the context"; return false; }
""", "pass bounds")
c = rep(c, """    Impl::PassK P;
    P.n = n;
    im.caps(pos0, n, P.cap);
    if (!im.ensure_rows(P.cap)) return false;   // lazy replicas: the rows the indexer scans (a cap bucket)
""", """    Impl::PassK P;
    P.n = n;
    if (nrun == 0) {
        im.caps(pos0, n, P.cap);
        if (!im.ensure_rows(P.cap)) return false;   // lazy replicas: the rows the indexer scans (a cap bucket)
    } else if (!im.runs_ready(P, runs, nrun)) {
        return false;
    }
""", "pass caps")
c = rep(c, """    im.make_inputs(tids, n, pos0);
    im.tap_on = main_hidden != nullptr;
""", """    if (nrun == 0) im.make_inputs(tids, n, pos0);
    else im.make_inputs_runs(tids, P.run, nrun);
    im.tap_on = main_hidden != nullptr;
""", "pass inputs")
c = rep(c, """    if (graph) {
        const auto key = std::make_tuple(n * 2 + (im.tap_on ? 1 : 0), P.cap[1], P.cap[2]);
""", """    if (graph && nrun > 0) {   // larry (batched decode): a pass of runs - its own graphs
        std::vector<int64_t> key = { im.tap_on ? 1 : 0, nrun };
        for (int u = 0; u < nrun; ++u) key.insert(key.end(), { (int64_t) P.run[u].seq, (int64_t) P.run[u].n, P.run[u].cap[1], P.run[u].cap[2] });
        if (!im.record_m(P, key)) return false;
        for (auto & cp : im.cards) ev.push_back(cp->q.ext_oneapi_graph(cp->graphs_m.at(key)));
    } else if (graph) {
        const auto key = std::make_tuple(n * 2 + (im.tap_on ? 1 : 0), P.cap[1], P.cap[2]);
""", "pass graph")
c = rep(c, """    auto engram_rows = [&im, tids, n, pos0, want0, &eng_ok, &t_eng] {
        const double te = now_us();
        float * outp[2] = { im.heng, im.heng + (size_t) im.NMAX * im.EC * im.ED };
        const bool split = im.hooks.engram_slot != nullptr;
        if (split) {
            if (!im.hooks.engram_slot(tids, n, pos0, 0, outp[0])) eng_ok = false;
        } else if (!im.hooks.engram(tids, n, pos0, outp)) eng_ok = false;
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<volatile int32_t *>(im.hflag)[0] = want0 + Impl::S_ENG;
        if (split && !im.hooks.engram_slot(tids, n, pos0, 1, outp[1])) eng_ok = false;
""", """    auto engram_rows = [&im, tids, n, pos0, want0, &eng_ok, &t_eng, &P] {
        const double te = now_us();
        float * outp[2] = { im.heng, im.heng + (size_t) im.NMAX * im.EC * im.ED };
        const bool split = im.hooks.engram_slot != nullptr;
        // larry (batched decode): a pass of runs - each run's rows from its own conversation's history
        auto runs_slot = [&](int e) {
            for (int u = 0; u < P.nrun; ++u) {
                const Run & R = P.run[u];
                if (!im.hooks.engram_slot_seq(R.seq, tids + R.r0, R.n, (int) R.p0, e, outp[e] + (size_t) R.r0 * im.EC * im.ED)) eng_ok = false;
            }
        };
        if (P.nrun > 0) {
            runs_slot(0);
        } else if (split) {
            if (!im.hooks.engram_slot(tids, n, pos0, 0, outp[0])) eng_ok = false;
        } else if (!im.hooks.engram(tids, n, pos0, outp)) eng_ok = false;
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<volatile int32_t *>(im.hflag)[0] = want0 + Impl::S_ENG;
        if (P.nrun > 0) runs_slot(1);
        else if (split && !im.hooks.engram_slot(tids, n, pos0, 1, outp[1])) eng_ok = false;
""", "pass engram")

open(H, "w").write(h)
open(C, "w").write(c)
print("patched", H, C)
