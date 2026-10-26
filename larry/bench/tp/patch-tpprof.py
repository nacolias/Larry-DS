#!/usr/bin/env python3
# patch-tpprof.py - STRATA_TP_PROF=1: host time of each step of the attention output split's submissions (the dense
# card's quant_x, each expert card's part, the dense card's own quarter, the sum), printed every 4000 calls.  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "tp_prof" in s:
    print("already"); sys.exit(0)
def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"anchor x{s.count(old)}: {old[:80]!r}")
    s = s.replace(old, new)
rep("""        std::lock_guard<PrioMutex> lk(g_submit);
        const bool tl = Ds41XeTier::tl_on();""", """        static const bool tp_prof = env_i("STRATA_TP_PROF", 0) != 0;
        static double tpp[6] = {};   // lock, quant_x, cards (sum over cards), own quarter, sum, calls
        double tq = tp_prof ? now_ms() : 0;
        auto tick = [&](int k) { if (tp_prof) { const double t = now_ms(); tpp[k] += t - tq; tq = t; } };
        std::lock_guard<PrioMutex> lk(g_submit);
        tick(0);
        const bool tl = Ds41XeTier::tl_on();""")
rep("""        std::vector<sycl::event> ev;
        static const bool rec = env_i("STRATA_TP_GRAPH", 1) != 0;""", """        tick(1);
        std::vector<sycl::event> ev;
        static const bool rec = env_i("STRATA_TP_GRAPH", 1) != 0;""")
rep("""        os_mv(q0, s.wa[0], OGD, OL, gp, sb, n, im.os_xq, im.os_xd, im.os_oa, KB, OL);   // its groups: o's first rows""",
    """        tick(2);
        os_mv(q0, s.wa[0], OGD, OL, gp, sb, n, im.os_xq, im.os_xd, im.os_oa, KB, OL);   // its groups: o's first rows""")
rep("""        if (tl) tl_stamp(im, 0, q0, 6);
        const float* sl = im.os_slots;""", """        if (tl) tl_stamp(im, 0, q0, 6);
        tick(3);
        const float* sl = im.os_slots;""")
rep("""        if (tl) tl_stamp(im, 0, q0, 7);
        ++im.os_calls;""", """        if (tl) tl_stamp(im, 0, q0, 7);
        tick(4);
        if (tp_prof && ++tpp[5] >= 1 && (long) tpp[5] % 4000 == 0)
            std::fprintf(stderr, "xe tp prof (%.0f calls): host us a call - lock %.1f, quant_x %.1f, cards %.1f, own quarter %.1f, sum %.1f\\n",
                         tpp[5], 1e3 * tpp[0] / tpp[5], 1e3 * tpp[1] / tpp[5], 1e3 * tpp[2] / tpp[5], 1e3 * tpp[3] / tpp[5], 1e3 * tpp[4] / tpp[5]);
        ++im.os_calls;""")
open(P, "w").write(s)
print("patched")
