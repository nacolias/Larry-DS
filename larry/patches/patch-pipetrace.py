#!/usr/bin/env python3
# patch-pipetrace.py [ENGINE_DIR] - larry, two diagnostics (off by default) for the pipelined prompt batch's per-layer
# timeline (a fresh 16K prompt's lower layers ran ~350 ms each against ~235 ms of card-0 GPU work):
#  - STRATA_PIPE_TRACE=1 (ds41_generate.cpp): a line per dense part (D: its spans queued), finish part (F), tier join
#    (J: the dense half waiting for the tier call in flight) and tier launch (T: pass_sync + the call's thread start),
#    with its start (ms from the batch's layer loop) and duration;
#  - STRATA_XE_PF_TRACE=1 (xe tier, after patch-pftrace.py) also prints every chunk call's wall split.
# Idempotent.
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine"
G, T = D + "/ds41_generate.cpp", D + "/xe/ds41_xe_tier.cpp"
g, t = open(G).read(), open(T).read()


def rep(s, old, new, n=1):
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:90]))
    return s.replace(old, new)


if "STRATA_PIPE_TRACE" not in g:
    g = rep(g, """                    bool inflight = false;
                    int fl = -1, fa = 0, fb = 0;   // the part whose finish waits for the tier call in flight
                    auto PD = [&](int l, int a0, int a1) -> bool {
                        const double t = now_ms();
                        for (int s = a0; s < a1; s += SB)
                            if (!dense.attn_router_span(l, s, std::min(SB, a1 - s))) { ferr = dense.last_error(); return false; }
                        fd_dense += now_ms() - t;
                        return true;
                    };
                    auto PF = [&](int l, int a0, int a1) -> bool {
                        const double t = now_ms();
                        for (int s = a0; s < a1; s += SB)
                            if (!dense.finish_span(l, s, std::min(SB, a1 - s))) { ferr = dense.last_error(); return false; }
                        fd_dense += now_ms() - t;
                        return true;
                    };
""", """                    bool inflight = false;
                    int fl = -1, fa = 0, fb = 0;   // the part whose finish waits for the tier call in flight
                    // larry, STRATA_PIPE_TRACE: each step's start (ms into the layer loop) and duration
                    static const bool pipe_trace = std::getenv("STRATA_PIPE_TRACE") != nullptr;
                    const double pt0 = now_ms();
                    auto ptr = [&](const char * what, int l, int a0, int a1, double t) {
                        if (pipe_trace) std::fprintf(stderr, "pipe trace: %8.1f  L%-2d %s [%5d,%5d) %7.1f ms\\n", t - pt0, l, what, a0, a1, now_ms() - t);
                    };
                    auto PD = [&](int l, int a0, int a1) -> bool {
                        const double t = now_ms();
                        for (int s = a0; s < a1; s += SB)
                            if (!dense.attn_router_span(l, s, std::min(SB, a1 - s))) { ferr = dense.last_error(); return false; }
                        fd_dense += now_ms() - t;
                        ptr("D", l, a0, a1, t);
                        return true;
                    };
                    auto PF = [&](int l, int a0, int a1) -> bool {
                        const double t = now_ms();
                        for (int s = a0; s < a1; s += SB)
                            if (!dense.finish_span(l, s, std::min(SB, a1 - s))) { ferr = dense.last_error(); return false; }
                        fd_dense += now_ms() - t;
                        ptr("F", l, a0, a1, t);
                        return true;
                    };
""")
    g = rep(g, """                        const double t = now_ms();
                        const bool ok = tier.run_chunk_wait();
                        fd_tier += now_ms() - t;
                        if (!ok) { ferr = "tier run refused"; return false; }
""", """                        const double t = now_ms();
                        const bool ok = tier.run_chunk_wait();
                        fd_tier += now_ms() - t;
                        ptr("J", fl, fa, fb, t);
                        if (!ok) { ferr = "tier run refused"; return false; }
""")
    g = rep(g, """                    auto PT = [&](int l, int a0, int a1, bool last) -> bool {
                        dense.pass_sync();   // the part's ids / weights / fn are final (the pass ring does not wait)
""", """                    auto PT = [&](int l, int a0, int a1, bool last) -> bool {
                        const double tt = now_ms();
                        dense.pass_sync();   // the part's ids / weights / fn are final (the pass ring does not wait)
""")
    g = rep(g, """                        if (!inflight) { ferr = "tier run refused"; return false; }
                        fl = l; fa = a0; fb = a1;
                        return true;
                    };
""", """                        if (!inflight) { ferr = "tier run refused"; return false; }
                        fl = l; fa = a0; fb = a1;
                        ptr("T", l, a0, a1, tt);
                        return true;
                    };
""")
    open(G, "w").write(g)
    print("patched", G)
else:
    print("already patched", G)

if "call trace:" not in t:
    t = rep(t, """    if (prof) {
        g_bp.prep += t_prep - t0;
""", """    {   // larry, STRATA_XE_PF_TRACE: the call's wall split
        static const bool ct = env_i("STRATA_XE_PF_TRACE", 0) != 0;
        if (ct)
            std::fprintf(stderr, "call trace: layer %lld n %d: %.1f ms = prefetch %.1f + plan %.1f + lock %.1f + prep %.1f + cards %.1f + "
                         "combine %.1f\\n", (long long) l, n, now_ms() - t0, t_pf - t0, t_plan - t_pf, t_lk - t_plan, t_prep - t_lk,
                         t_cards - t_prep, now_ms() - t_cards);
    }
    if (prof) {
        g_bp.prep += t_prep - t0;
""")
    open(T, "w").write(t)
    print("patched", T)
else:
    print("already patched", T)
