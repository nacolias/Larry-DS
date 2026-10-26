#!/usr/bin/env python3
# patch-pfnb.py [TIER_CPP] - larry, STRATA_XE_PF_NOBLOCK (default 0 = the old form; measured no gain, abwT64): a layer's last tier call waited
# for the NEXT layer's prefetch to be staged (pf_enqueue(l + 1, block)) before it queued its own combine -
# STRATA_GPU_PTL (abwT63, fresh 8K): the second calls' parts were all on card 0 20-50 ms before their combine was
# queued, card 0's queue empty meanwhile.  With the hot rings the stagers queue every copy themselves; now the stager
# that queues a slot's last copy also records the cards' barriers for it (under the submit lock, so they cover exactly
# the copies queued so far: not the next layer's, which a barrier recorded later would also wait for), and the last
# call only marks the next slot enqueued when it is staged already - that layer's first call waits for its staging
# itself (on the tier thread, card 0 running its spans meanwhile).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_PF_NOBLOCK" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""        double t_plan = 0, t_staged = 0;   ///< larry, STRATA_XE_PF_TRACE: planned / last copy submitted (now_ms)
""", """        double t_plan = 0, t_staged = 0;   ///< larry, STRATA_XE_PF_TRACE: planned / last copy submitted (now_ms)
        bool rec = false;              ///< larry, STRATA_XE_PF_NOBLOCK: its last stager recorded the cards' barriers
""")
rep("""    ps.layer = l;
    ps.enqueued = false;
""", """    ps.layer = l;
    ps.enqueued = false;
    ps.rec = false;   // larry, STRATA_XE_PF_NOBLOCK
""")
rep("""        if (xe_prof()) { g_bp.st_us += (int64_t) ((now_ms() - ts) * 1000.0); g_bp.st_bytes += (int64_t) im.bbytes(l, e); }
        std::lock_guard<std::mutex> lk(im.pf_mu);
        --ps.active;
        if (ps.active == 0 && ps.next >= E) { ps.staged = 0; ps.t_staged = now_ms(); im.pf_cv.notify_all(); }
    }
}
""", """        if (xe_prof()) { g_bp.st_us += (int64_t) ((now_ms() - ts) * 1000.0); g_bp.st_bytes += (int64_t) im.bbytes(l, e); }
        // larry, STRATA_XE_PF_NOBLOCK (patch-pfnb.py): the stager that queued the slot's last copy records its barriers
        static const int pf_nb = env_i("STRATA_XE_PF_NOBLOCK", 0);
        bool last = false;
        {
            std::lock_guard<std::mutex> lk(im.pf_mu);
            --ps.active;
            last = ps.active == 0 && ps.next >= E;
            if (last && !(hot && pf_nb)) { ps.staged = 0; ps.t_staged = now_ms(); im.pf_cv.notify_all(); }
        }
        if (last && hot && pf_nb) {
            const double tr = now_ms();
            {
                std::lock_guard<PrioMutex> lk(g_submit);
                for (auto& cp : im.cards)
                    if (cp->pbuf_cap && cp->pbuf[s]) {
                        cp->pev[s] = cp->pq.ext_oneapi_submit_barrier();
                        ptl_stamp(im.ctx, 9 + cp->idx, cp->pq, 30, (int) l, cp->idx, tr);   // larry, STRATA_GPU_PTL
                    }
            }
            std::lock_guard<std::mutex> lk(im.pf_mu);
            ps.rec = true;
            ps.staged = 0;
            ps.t_staged = now_ms();
            im.pf_cv.notify_all();
        }
    }
}
""")
rep("""    if (hot_on() && !im.pf_rings.empty()) {   // STRATA_XE_HOTRING: the stagers put every copy on the queues already
        std::lock_guard<PrioMutex> lk(g_submit);
        for (auto& cp : im.cards)
            if (cp->pbuf_cap && cp->pbuf[s]) {
""", """    if (hot_on() && !im.pf_rings.empty()) {   // STRATA_XE_HOTRING: the stagers put every copy on the queues already
        std::lock_guard<PrioMutex> lk(g_submit);
        for (auto& cp : im.cards)
            if (!ps.rec && cp->pbuf_cap && cp->pbuf[s]) {   // larry, PF_NOBLOCK: rec = its last stager recorded them
""")
rep("""        if (last_call) {
            if (im.pf[pslot ^ 1].layer == l + 1) pf_enqueue(im, pslot ^ 1, true);
""", """        if (last_call) {
            // larry, STRATA_XE_PF_NOBLOCK: with the hot rings the next layer's copies are queued by the stagers (and its
            // barriers recorded by them); its first call waits for its staging - this call's combine does not
            static const int pf_nb = env_i("STRATA_XE_PF_NOBLOCK", 0);
            const bool pf_block = !(pf_nb && hot_on() && !im.pf_rings.empty());
            if (im.pf[pslot ^ 1].layer == l + 1) pf_enqueue(im, pslot ^ 1, pf_block);
""")
open(P, "w").write(s)
print("patched", P)
