#!/usr/bin/env python3
# patch-shexptier.py [XE_DIR] - larry, STRATA_SHEXP_TIER (the engine's switch; with patch-shexpdense.py and
# patch-shexpgen.py): a prompt call's shared expert on the expert cards instead of card 0.  Card 0 bounds layers 0-19
# of a fresh prompt from 8K up (its compute engine ~100 % busy, the expert cards ~60 %: abwT99 / abwT100, a 16K
# batch's dense spans 7.62 ms each), and the shared expert (gate | up 5120 -> 2 x 2048, SwiGLU, down) is ~0.6 ms of a
# span.  Its input is the routed experts' (each expert card has the call's X already) and its output adds to the
# routed sum (the combine adds the cards' parts anyway), so: each expert card takes the TP core's shards it holds
# (rank k's n_ff rows of gate | up, the matching K blocks of down; planar Q6_K), rank 0's slice copied to the first
# expert card at load; per layer the slices dequantised to fp16 once (fastmv's dequant_q6k_planar_f16), per call
# X -> f16, gate | up GEMM, the clamped SwiGLU, down GEMM added into the card's part - every token, weight 1.
# Idempotent.
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe"
P, H = D + "/ds41_xe_tier.cpp", D + "/ds41_xe_tier.hpp"
s, h = open(P).read(), open(H).read()
if "STRATA_SHEXP_TIER" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1, which="cpp"):
    global s, h
    t = s if which == "cpp" else h
    if t.count(old) != n:
        sys.exit("%s anchor x%d (want %d): %s" % (which, t.count(old), n, old[:100]))
    t = t.replace(old, new)
    if which == "cpp": s = t
    else: h = t


rep("""    void tp_head_slice(const void* w, int gpu, int64_t r0, int64_t r1, const void* p);
""", """    void tp_head_slice(const void* w, int gpu, int64_t r0, int64_t r1, const void* p);
    /// larry, STRATA_SHEXP_TIER (patch-shexptier.py): layer l's shared-expert slice of the TP core on GPU `gpu` - gate |
    /// up rows (2 nf x n_embd) at gu, down (n_embd x nf) at dn, planar Q6_K; a slice on the dense card is copied to the
    /// first expert card.  Before serving.
    void tp_shexp_slice(int64_t l, int gpu, const void* gu, const void* dn, int64_t nf);
    /// every layer has its shared-expert slices on the expert cards
    bool shexp_ready() const;
    /// the next run_chunk_dev_async computes the shared expert for all its tokens too (added into the routed sums)
    void set_shexp_next(bool on);
""", which="hpp")
rep("""    float* cOUT = nullptr; size_t cOUT_cap = 0;      ///< floats
""", """    float* cOUT = nullptr; size_t cOUT_cap = 0;      ///< floats
    // larry, STRATA_SHEXP_TIER (patch-shexptier.py): this card's shared-expert slices of layer sh_layer in fp16 (per
    // slice gate | up rows, then down) and a call's scratch for them (X rows, gate | up, SwiGLU), the row indices 0..
    sycl::half* sh_w = nullptr; size_t sh_w_cap = 0; int64_t sh_layer = -1;
    uint8_t* sh_buf = nullptr; sycl::half* sh_x = nullptr; float* sh_g = nullptr; sycl::half* sh_h = nullptr;
    size_t sh_rows = 0, sh_nf = 0;
    int32_t* sh_iota = nullptr; size_t sh_iota_n = 0;
""")
rep("""    std::map<const void*, std::vector<GivenSlice>> head_given;
""", """    std::map<const void*, std::vector<GivenSlice>> head_given;
    // larry, STRATA_SHEXP_TIER: per layer the shared expert's TP shards on the expert cards (card index, gate | up
    // 2 nf x H, down H x nf - planar Q6_K, nf); copies made here (rank 0's) in sh_own
    struct ShSlice { int card; const uint8_t* gu; const uint8_t* dn; int64_t nf; };
    std::vector<std::vector<ShSlice>> sh_slices;
    std::vector<std::pair<XeCard*, uint8_t*>> sh_own;
    bool sh_next = false, sh_call = false;
""")
# the API
rep("""void Ds41XeTier::tp_head_slice(const void* w, int gpu, int64_t r0, int64_t r1, const void* p) {
    if (w && p) im_->head_given[w].push_back({gpu, r0, r1, p});
}
""", """void Ds41XeTier::tp_head_slice(const void* w, int gpu, int64_t r0, int64_t r1, const void* p) {
    if (w && p) im_->head_given[w].push_back({gpu, r0, r1, p});
}

void Ds41XeTier::tp_shexp_slice(int64_t l, int gpu, const void* gu, const void* dn, int64_t nf) {
    Ds41XeImpl& im = *im_;
    const int64_t H = im.g.n_embd;
    if (l < 0 || !gu || !dn || nf <= 0 || nf % 256 || H % 256) return;
    if ((size_t) l >= im.sh_slices.size()) im.sh_slices.resize((size_t) l + 1);
    XeCard* own = nullptr;
    for (auto& cp : im.cards) if (cp->gpu == gpu) own = cp.get();
    if (!own) return;
    if (!own->dense) { im.sh_slices[(size_t) l].push_back({own->idx, (const uint8_t*) gu, (const uint8_t*) dn, nf}); return; }
    // rank 0's slice: onto the first expert card, through the host (once, at load)
    XeCard* dst = nullptr;
    for (auto& cp : im.cards) if (!cp->dense) { dst = cp.get(); break; }
    if (!dst) return;
    const size_t bgu = (size_t) (2 * nf * H / 256) * 210, bdn = (size_t) (H * nf / 256) * 210;
    uint8_t* p = local_alloc(dst->q, bgu + bdn);
    if (!p) {
        std::fprintf(stderr, "xe tier: shared expert: no room on card %d for layer %lld's rank-0 slice\\n", dst->idx, (long long) l);
        return;
    }
    std::vector<uint8_t> tmp(bgu + bdn);
    {
        std::lock_guard<PrioMutex> lk(g_submit);
        own->q.memcpy(tmp.data(), gu, bgu);
        own->q.memcpy(tmp.data() + bgu, dn, bdn);
        own->q.wait();
        dst->q.memcpy(p, tmp.data(), bgu + bdn).wait();
    }
    im.sh_own.emplace_back(dst, p);
    im.sh_slices[(size_t) l].push_back({dst->idx, p, p + bgu, nf});
}

bool Ds41XeTier::shexp_ready() const {
    const Ds41XeImpl& im = *im_;
    if ((int64_t) im.sh_slices.size() < im.g.n_layers) return false;
    for (const auto& v : im.sh_slices) if (v.empty()) return false;
    return true;
}

void Ds41XeTier::set_shexp_next(bool on) { im_->sh_next = on; }
""")
# run_chunk_dev_async hands the flag to its call
rep("""    if (im.run_thr.joinable()) return false;   // one call in flight
    im.run_ok = true;
    im.run_thr = std::thread([&im, l, n, ids, w, x, out_dev, last_of_layer, x_dev] {""",
    """    if (im.run_thr.joinable()) return false;   // one call in flight
    im.run_ok = true;
    im.sh_call = im.sh_next && n > kMaxJobTok;   // larry, STRATA_SHEXP_TIER: this call's (run_big's) shared expert
    im.sh_next = false;
    im.run_thr = std::thread([&im, l, n, ids, w, x, out_dev, last_of_layer, x_dev] {""")
# run_big: the flag, the cards that must run, the work
rep("""    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.busy = !cl[(size_t) c.idx].empty();
        if (!c.busy) continue;
        const double tg0 = prof ? now_ms() : 0;""", """    // larry, STRATA_SHEXP_TIER: the shared expert on the expert cards holding its slices (they run even with no routed
    // expert of their own)
    const bool shx = im.sh_call && (size_t) l < im.sh_slices.size() && !im.sh_slices[(size_t) l].empty();
    im.sh_call = false;
    auto sh_has = [&](int idx) {
        if (!shx) return false;
        for (const auto& sl : im.sh_slices[(size_t) l]) if (sl.card == idx) return true;
        return false;
    };
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        c.busy = !cl[(size_t) c.idx].empty() || (!c.dense && sh_has(c.idx));
        if (!c.busy) continue;
        const double tg0 = prof ? now_ms() : 0;""")
rep("""        if (!out_dev) {
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.memcpy(c.cOUT_host, c.cOUT, (size_t) n * H * 4);
        }
        if (!out_init) {   // larry, STRATA_XE_GATHER_INIT: no gather ran - the part is zero""", """        // larry, STRATA_SHEXP_TIER (patch-shexptier.py): this card's shared-expert slices for every token of the call,
        // added into its part: per slice its layer's fp16 copy (at the layer's first call), then X rows -> f16, the
        // gate | up GEMM, the clamped SwiGLU, the down GEMM into the part (beta 1 once anything wrote it)
        if (shx && !c.dense && sh_has(c.idx)) {
            const auto& sls = im.sh_slices[(size_t) l];
            size_t wneed = 0, nfm = 0;
            for (const auto& sl : sls) if (sl.card == c.idx) { wneed += (size_t) (3 * sl.nf * H); nfm = std::max(nfm, (size_t) sl.nf); }
            const int64_t RBs = 1024;
            bool ok_sh;
            {
                std::lock_guard<PrioMutex> lk(g_submit);
                const sycl::half* w_was = c.sh_w;
                ok_sh = grow_loc(c.sh_w, c.sh_w_cap, wneed, c.q);
                if (c.sh_w != w_was) c.sh_layer = -1;   // a new buffer: its layer's copy is gone
                if (ok_sh && (c.sh_rows < (size_t) RBs || c.sh_nf < nfm)) {
                    if (c.sh_buf) { c.q.wait(); local_free(c.q, c.sh_buf); c.sh_buf = nullptr; c.sh_rows = c.sh_nf = 0; }
                    const size_t bx = (size_t) RBs * H * 2, bg = (size_t) RBs * 2 * nfm * 4, bh = (size_t) RBs * nfm * 2;
                    c.sh_buf = local_alloc(c.q, bx + bg + bh);
                    ok_sh = c.sh_buf != nullptr;
                    if (ok_sh) {
                        c.sh_x = (sycl::half*) c.sh_buf;
                        c.sh_g = (float*) (c.sh_buf + bx);
                        c.sh_h = (sycl::half*) (c.sh_buf + bx + bg);
                        c.sh_rows = (size_t) RBs;
                        c.sh_nf = nfm;
                    }
                }
                if (ok_sh && c.sh_iota_n < (size_t) n) {
                    if (c.sh_iota) { c.q.wait(); local_free(c.q, c.sh_iota); c.sh_iota = nullptr; c.sh_iota_n = 0; }
                    const size_t ni = std::max<size_t>((size_t) n, 16384);
                    c.sh_iota = (int32_t*) local_alloc(c.q, ni * 4);
                    ok_sh = c.sh_iota != nullptr;
                    if (ok_sh) {
                        int32_t* io = c.sh_iota;
                        c.q.parallel_for(sycl::range<1>(ni), [=](sycl::id<1> i) { io[i] = (int32_t) i[0]; });
                        c.sh_iota_n = ni;
                    }
                }
                if (ok_sh && c.sh_layer != l) {
                    size_t off = 0;
                    for (const auto& sl : sls) {
                        if (sl.card != c.idx) continue;
                        fastmv::dequant_q6k_planar_f16(c.q, sl.gu, 2 * sl.nf, H, c.sh_w + off);
                        fastmv::dequant_q6k_planar_f16(c.q, sl.dn, H, sl.nf, c.sh_w + off + (size_t) (2 * sl.nf * H));
                        off += (size_t) (3 * sl.nf * H);
                    }
                    c.sh_layer = l;
                }
            }
            if (!ok_sh) { std::fprintf(stderr, "xe tier: card %d: shared-expert buffers failed\\n", c.idx); return false; }
            size_t off = 0;
            for (const auto& sl : sls) {
                if (sl.card != c.idx) continue;
                const int64_t nf = sl.nf;
                const sycl::half* Wgu = c.sh_w + off;
                const sycl::half* Wdn = c.sh_w + off + (size_t) (2 * nf * H);
                for (int64_t r0 = 0; r0 < (int64_t) n; r0 += RBs) {
                    const int64_t m = std::min<int64_t>(RBs, (int64_t) n - r0);
                    std::lock_guard<PrioMutex> lk(g_submit);
                    xe::xe_gather_rows_f16(c.q, xin, c.sh_iota + r0, (int) m, (int) H, c.sh_x);
                    mkl::blas::row_major::gemm(c.q, mkl::transpose::nontrans, mkl::transpose::trans, m, 2 * nf, H, 1.0f, c.sh_x, H,
                                               Wgu, H, 0.0f, c.sh_g, 2 * nf);
                    xe::xe_swiglu_f16(c.q, c.sh_g, (int) m, (int) nf, im.L, c.sh_h);
                    mkl::blas::row_major::gemm(c.q, mkl::transpose::nontrans, mkl::transpose::trans, m, H, nf, 1.0f, c.sh_h, nf,
                                               Wdn, nf, out_init ? 1.0f : 0.0f, c.cOUT + (size_t) r0 * H, H);
                }
                out_init = true;
                off += (size_t) (3 * nf * H);
            }
        }
        if (!out_dev) {
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.memcpy(c.cOUT_host, c.cOUT, (size_t) n * H * 4);
        }
        if (!out_init) {   // larry, STRATA_XE_GATHER_INIT: no gather ran - the part is zero""")
# the copies and buffers go at close (before the cards)
rep("""    for (auto& kv : im.heads)   // the head split's slices and scratch, on their cards (before the cards go)""",
    """    for (auto& cs : im.sh_own) { cs.first->q.wait(); local_free(cs.first->q, cs.second); }   // larry, STRATA_SHEXP_TIER
    im.sh_own.clear();
    im.sh_slices.clear();
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.sh_w && !c.sh_buf && !c.sh_iota) continue;
        c.q.wait();
        if (c.sh_w) local_free(c.q, c.sh_w);
        if (c.sh_buf) local_free(c.q, c.sh_buf);
        if (c.sh_iota) local_free(c.q, c.sh_iota);
        c.sh_w = nullptr; c.sh_buf = nullptr; c.sh_iota = nullptr;
    }
    for (auto& kv : im.heads)   // the head split's slices and scratch, on their cards (before the cards go)""")
open(P, "w").write(s)
open(H, "w").write(h)
print("patched", P, H)
