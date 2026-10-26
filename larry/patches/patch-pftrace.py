#!/usr/bin/env python3
# patch-pftrace.py [TIER_CPP] - larry, STRATA_XE_PF_TRACE=1 (diagnostic, off by default): a line per prompt-prefetch
# layer at its first tier call - when the layer's slot was planned, when its last copy went on the queues (staged),
# how long its DMAs still ran once the call started (the host waits for them in this mode only), and the bytes:
#   pf trace: layer L: planned -P ms, staged -S ms (or "+S after the call began"), DMAs done +D ms, B GB, N experts
# (times relative to the call's start).  The profile said each layer's first call waited for its prefetch copies
# (274-527 MB per card still streaming) though the slot had the previous layer's ~300 ms to fill.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_PF_TRACE" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:90]))
    s = s.replace(old, new)


rep("""        double t_enq = 0;              ///< when its copies went on the queues (STRATA_XE_PROF)
        bool timed = false;
    };
""", """        double t_enq = 0;              ///< when its copies went on the queues (STRATA_XE_PROF)
        bool timed = false;
        double t_plan = 0, t_staged = 0;   ///< larry, STRATA_XE_PF_TRACE: planned / last copy submitted (now_ms)
        bool traced = false;
    };
""")
# pf_plan: the plan time
rep("""    ps.layer = l;
    ps.enqueued = false;
    ps.timed = false;
""", """    ps.layer = l;
    ps.enqueued = false;
    ps.timed = false;
    ps.t_plan = now_ms();   // larry, STRATA_XE_PF_TRACE
    ps.t_staged = 0;
    ps.traced = false;
""")
# pf_stager: the slot's last copy submitted
rep("""        std::lock_guard<std::mutex> lk(im.pf_mu);
        --ps.active;
        if (ps.active == 0 && ps.next >= E) { ps.staged = 0; im.pf_cv.notify_all(); }
""", """        std::lock_guard<std::mutex> lk(im.pf_mu);
        --ps.active;
        if (ps.active == 0 && ps.next >= E) { ps.staged = 0; ps.t_staged = now_ms(); im.pf_cv.notify_all(); }
""")
# run_big: at the layer's first call, after its copies are enqueued
rep("""        if (ps) pf_enqueue(im, pslot, true);
        if (ps && prof && !ps->timed) {""", """        // larry, STRATA_XE_PF_TRACE: the slot's timeline at its first call (the host waits for its DMAs here)
        static const bool pf_trace = env_i("STRATA_XE_PF_TRACE", 0) != 0;
        const double t_call = now_ms();
        const double t_st0 = ps ? ps->t_staged : 0;
        if (ps) pf_enqueue(im, pslot, true);
        if (ps && pf_trace && !ps->traced) {
            ps->traced = true;
            const double t_enq_done = now_ms();
            for (auto& cp : im.cards) if (cp->pbuf_cap && cp->pbuf[pslot]) cp->pev[pslot].wait();
            const double t_dma = now_ms();
            const double tst = ps->t_staged > 0 ? ps->t_staged : t_enq_done;
            std::fprintf(stderr, "pf trace: layer %lld (call n %d): planned %.0f ms before, staged %s%.0f ms, enqueue wait %.0f ms, "
                         "DMAs done +%.0f ms; %.2f GB, %lld experts\\n", (long long) l, n, t_call - ps->t_plan,
                         t_st0 > 0 ? "-" : "+", t_st0 > 0 ? t_call - t_st0 : tst - t_call, t_enq_done - t_call, t_dma - t_call,
                         ps->bytes / 1e9, (long long) ps->n);
        }
        if (ps && prof && !ps->timed) {""")
open(P, "w").write(s)
print("patched", P)
