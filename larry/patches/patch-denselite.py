#!/usr/bin/env python3
# patch-denselite.py [TIER_CPP] - larry, two prompt-call options (default 0 until measured; needs patch-calltl.py first):
#
# STRATA_XE_DENSE_LITE=1: in a pipelined prompt call (x_dev: the dense half runs beside it on the same card) the dense
#   card's part of the tier waits for that card's GPU at every step - the 2026-10-25 call timelines (STRATA_XE_CALL_TL):
#   one streamed expert (15 ring chunks, a copy each, each waited for) took ~120 ms on card 0 against 15-25 ms on the
#   expert cards, and card 0's GEMM experts + jobs ended 45-110 ms into calls whose expert cards were done at 55-75 ms:
#   card 0 was the last card of most calls.  So, in such a call: (1) no streamed experts on the dense card (the expert
#   cards take its share); (2) the dense card reads X where the dense half wrote it (no copy - a part's fn stays put
#   until its call is joined); (3) its part goes on its queue without one host wait (the GEMM / job tables copied from
#   pinned per-call staging, not from vectors the next step refills; the combine follows on the same queue and waits).
# STRATA_XE_GATHER_INIT=1: a card's first gather of a call writes its part (no memset of it first; same sums).
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_DENSE_LITE" in s:
    print("already patched"); sys.exit(0)
if "STRATA_XE_CALL_TL" not in s:
    sys.exit("needs patch-calltl.py first")


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


# the dense card's pinned per-call staging
rep("""    uint8_t* dq_host = nullptr; size_t dq_host_cap = 0;   ///< pinned: xe_dqgemm's job / tile tables of a call
""", """    uint8_t* dq_host = nullptr; size_t dq_host_cap = 0;   ///< pinned: xe_dqgemm's job / tile tables of a call
    uint8_t* nw_host = nullptr; size_t nw_cap = 0;   ///< larry, STRATA_XE_DENSE_LITE: pinned per-call staging (no host waits)
""")
rep("""fr(c.cjobs_host); fr(c.dq_host); fr(c.cH); fr(c.cY);""",
    """fr(c.cjobs_host); fr(c.dq_host); fr(c.nw_host); fr(c.cH); fr(c.cY);""")

# the switches
rep("""    const int gemm_min = env_i("STRATA_XE_GEMM_MIN", 8);
    const bool prof = xe_prof();
""", """    const int gemm_min = env_i("STRATA_XE_GEMM_MIN", 8);
    const bool prof = xe_prof();
    // larry, STRATA_XE_DENSE_LITE / STRATA_XE_GATHER_INIT: see patch-denselite.py (the dense card's part of a pipelined
    // call: no streamed experts, X read in place, no host waits; the first gather writes the part)
    static const int dense_lite = env_i("STRATA_XE_DENSE_LITE", 0);
    static const int gather_init = env_i("STRATA_XE_GATHER_INIT", 0);
    const bool lite = dense_lite && x_dev && out_dev && im.dense >= 0 && NC > 1;
""")

# (1) no streamed experts on the dense card
rep("""        size_t best = 0;
        for (size_t k = 1; k < NC; ++k) if (sbytes[k] < sbytes[best]) best = k;
        sbytes[best] += (int64_t) ((im.bbytes(l, e) + 255) & ~(size_t) 255);
""", """        size_t best = SIZE_MAX;
        for (size_t k = 0; k < NC; ++k) {
            if (lite && im.cards[k]->dense) continue;   // larry, STRATA_XE_DENSE_LITE
            if (best == SIZE_MAX || sbytes[k] < sbytes[best]) best = k;
        }
        sbytes[best] += (int64_t) ((im.bbytes(l, e) + 255) & ~(size_t) 255);
""")

# (2) no X copy onto the dense card; GATHER_INIT: no memset
rep("""        } else {
            c.q.memcpy(c.cX, x_dev ? x_dev : x, (size_t) n * H * 4);   // x_dev: card-to-card from the dense card
        }
""", """        } else if (!(lite && c.dense)) {   // larry, STRATA_XE_DENSE_LITE: the dense card reads x_dev where it is
            c.q.memcpy(c.cX, x_dev ? x_dev : x, (size_t) n * H * 4);   // x_dev: card-to-card from the dense card
        }
""")
rep("""        c.q.memset(c.cOUT, 0, (size_t) n * H * 4);
        if (prof) {
            const double tq3 = now_ms();""", """        if (!gather_init) c.q.memset(c.cOUT, 0, (size_t) n * H * 4);   // larry, STRATA_XE_GATHER_INIT: the first gather writes
        if (prof) {
            const double tq3 = now_ms();""")

# (3) card_work: X source, the no-wait mode and its staging
rep("""        CallTl& tl = ctl[(size_t) c.idx];
        if (call_tl) tl.s = now_ms() - t0;
        std::vector<int32_t> S((size_t) n * K);
""", """        CallTl& tl = ctl[(size_t) c.idx];
        if (call_tl) tl.s = now_ms() - t0;
        // larry, STRATA_XE_DENSE_LITE: X where the dense half wrote it; the part queued without host waits (with the
        // combine on this queue after it).  The staging: GEMM rows + slot map, each job batch's jobs + slot map.
        const float* xin = lite && c.dense ? x_dev : c.cX;
        const bool nowait = lite && c.dense && cmb > 0 && streamed[(size_t) c.idx].empty() && kept[(size_t) c.idx].empty();
        size_t nw_off = 0;
        bool out_init = !gather_init;   // larry, STRATA_XE_GATHER_INIT: false = the next gather writes the part
        if (nowait) {
            int64_t rg = 0, njp = 0, njr = 0;
            for (int32_t e : cl[(size_t) c.idx]) {
                const int64_t m = (int64_t) by_e[(size_t) e].size();
                if (m >= gemm_min) rg += (m + 7) & ~(int64_t) 7;
                else if (im.planar && im.card_of[(size_t) im.key(l, e)] >= 0) njp += (m + kMaxJobTok - 1) / kMaxJobTok;
                else njr += (m + kMaxJobTok - 1) / kMaxJobTok;
            }
            const int64_t nbat = (njp + BJ - 1) / BJ + (njr + BJ - 1) / BJ;
            const size_t need = 4 * 256 + (size_t) rg * 4 + (size_t) n * K * 4 +
                                (size_t) nbat * ((size_t) BJ * sizeof(XeJob) + (size_t) n * K * 4 + 512);
            if (c.nw_cap < need) {
                std::lock_guard<PrioMutex> lk(g_submit);
                c.q.wait();   // nothing of this call reads the old staging yet; the previous call's combine waited
                if (c.nw_host) sycl::free(c.nw_host, c.q);
                c.nw_host = sycl::malloc_host<uint8_t>(need + need / 4, c.q);
                c.nw_cap = c.nw_host ? need + need / 4 : 0;
            }
            if (!c.nw_host) { std::fprintf(stderr, "xe tier: card %d: staging allocation failed\\n", c.idx); return false; }
        }
        auto nw_take = [&](size_t bytes) -> uint8_t* {
            uint8_t* p = c.nw_host + nw_off;
            nw_off += (bytes + 255) & ~(size_t) 255;
            return p;
        };
        std::vector<int32_t> S((size_t) n * K);
""")
# the GEMM tables (both GEMM paths)
rep("""                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    c.q.memcpy(dtok, ptok.data(), ptok.size() * 4);
                    c.q.memcpy(c.cS, S.data(), S.size() * 4);
                }
                c.q.wait();   // S and ptok are reused below
""", """                if (nowait) {   // larry, STRATA_XE_DENSE_LITE: from pinned per-call copies, no wait
                    uint8_t* hp = nw_take(ptok.size() * 4);
                    uint8_t* hs = nw_take(S.size() * 4);
                    std::memcpy(hp, ptok.data(), ptok.size() * 4);
                    std::memcpy(hs, S.data(), S.size() * 4);
                    std::lock_guard<PrioMutex> lk(g_submit);
                    c.q.memcpy(dtok, hp, ptok.size() * 4);
                    c.q.memcpy(c.cS, hs, S.size() * 4);
                } else {
                    {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(dtok, ptok.data(), ptok.size() * 4);
                        c.q.memcpy(c.cS, S.data(), S.size() * 4);
                    }
                    c.q.wait();   // S and ptok are reused below
                }
""", 2)
rep("""                    xe::xe_gather_rows_f16(c.q, c.cX, dtok + r0, (int) rows, (int) H, Xg);
""", """                    xe::xe_gather_rows_f16(c.q, xin, dtok + r0, (int) rows, (int) H, Xg);
""")
rep("""                    xe::xe_gather_rows_f16(c.q, c.cX, dtok + base, (int) m, (int) H, Xg);
""", """                    xe::xe_gather_rows_f16(c.q, xin, dtok + base, (int) m, (int) H, Xg);
""")
rep("""                    xe::xe_gather(c.q, Yg, c.cS, c.cW, n, K, (int) H, c.cOUT, true);
""", """                    xe::xe_gather(c.q, Yg, c.cS, c.cW, n, K, (int) H, c.cOUT, out_init);
                    out_init = true;
""", 2)
# the job batches
rep("""                    c.q.wait();   // cjobs / cjobs_host and cS are read by the previous batch / the GEMM gather
                    if (call_tl) { if (tl.b == 0) tl.b = now_ms() - t0; tl.nj += (int64_t) nb; }
                    std::fill(S.begin(), S.end(), -1);
                    int max_nt = 1;
                    for (size_t q = 0; q < nb; ++q) {
                        c.cjobs_host[q] = js[b0 + q];
""", """                    if (!nowait) c.q.wait();   // cjobs / cjobs_host and cS are read by the previous batch / the GEMM gather
                    if (call_tl) { if (tl.b == 0) tl.b = now_ms() - t0; tl.nj += (int64_t) nb; }
                    XeJob* jh = nowait ? (XeJob*) nw_take(nb * sizeof(XeJob)) : c.cjobs_host;   // larry, STRATA_XE_DENSE_LITE
                    std::fill(S.begin(), S.end(), -1);
                    int max_nt = 1;
                    for (size_t q = 0; q < nb; ++q) {
                        jh[q] = js[b0 + q];
""")
rep("""                        xe::xe_quant_rows(c.q, c.cX, n, (int) H, c.cxq);
""", """                        xe::xe_quant_rows(c.q, xin, n, (int) H, c.cxq);
""")
rep("""                    {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.cjobs, c.cjobs_host, nb * sizeof(XeJob));
                        c.q.memcpy(c.cS, S.data(), S.size() * 4);
                    }
                    c.q.wait();
""", """                    {
                        const int32_t* sp = S.data();
                        if (nowait) {   // larry, STRATA_XE_DENSE_LITE: a per-batch pinned copy, no wait
                            uint8_t* hs = nw_take(S.size() * 4);
                            std::memcpy(hs, S.data(), S.size() * 4);
                            sp = (const int32_t*) hs;
                        }
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.cjobs, jh, nb * sizeof(XeJob));
                        c.q.memcpy(c.cS, sp, S.size() * 4);
                    }
                    if (!nowait) c.q.wait();
""")
rep("""                    else xe::xe_experts_y(c.q, c.cjobs, (int) nb, c.cX, c.cH, c.cY, (int) H, (int) F, im.L);
                    xe::xe_gather(c.q, c.cY, c.cS, c.cW, n, K, (int) H, c.cOUT, true);
""", """                    else xe::xe_experts_y(c.q, c.cjobs, (int) nb, xin, c.cH, c.cY, (int) H, (int) F, im.L);
                    xe::xe_gather(c.q, c.cY, c.cS, c.cW, n, K, (int) H, c.cOUT, out_init);
                    out_init = true;
""")
# a card without a gather (cannot happen: a busy card has an expert) still leaves a zero part; the end without a wait
rep("""        if (out_dev && !c.dense && cmb > 0 && cdc) {   // STRATA_XE_COMBINE: this card's part onto the dense card now
""", """        if (!out_init) {   // larry, STRATA_XE_GATHER_INIT: no gather ran - the part is zero
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.memset(c.cOUT, 0, (size_t) n * H * 4);
            out_init = true;
        }
        if (out_dev && !c.dense && cmb > 0 && cdc) {   // STRATA_XE_COMBINE: this card's part onto the dense card now
""")
rep("""        c.q.wait();
        if (call_tl) tl.e = now_ms() - t0;
""", """        if (!nowait) c.q.wait();   // larry, STRATA_XE_DENSE_LITE: the combine on this queue waits for the part
        if (call_tl) tl.e = now_ms() - t0;
""")
open(P, "w").write(s)
print("patched", P)
