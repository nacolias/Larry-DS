#!/usr/bin/env python3
# patch-ltprof.py [TIER_CPP] - larry (diagnostic): STRATA_XE_LT_PROF=1 splits a prompt call's CPU-light part - the
# buffers + row list, X read back, the jobs' setup (quantisation included), the pool, the sum - summed over calls and
# printed with the CPU-light line.  Its "quantise" share (~7 ms a 512-token call, abwT68) did not drop with the
# quantisation on 4 threads (abwT69): which step is it?  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_LT_PROF" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""        if (!light.empty()) try {
            const double tq = now_ms();
""", """        if (!light.empty()) try {
            const double tq = now_ms();
            // larry, STRATA_XE_LT_PROF (patch-ltprof.py): the light part's steps, ms summed over calls
            static const bool lt_prof = env_i("STRATA_XE_LT_PROF", 0) != 0;
            static double lp[6] = {};   // setup, readback, jobs, pool, sum, calls
            double lpt = lt_prof ? now_ms() : 0;
            auto lpl = [&](int i) { if (lt_prof) { const double t = now_ms(); lp[i] += t - lpt; lpt = t; } };
""")
rep("""            const bool xl_compact = light_ok && lt_compact && x_dev;
            const float* xl = x;
""", """            const bool xl_compact = light_ok && lt_compact && x_dev;
            lpl(0);   // larry, STRATA_XE_LT_PROF: buffers + row list
            const float* xl = x;
""")
rep("""            double t_run = 0, t_sum = 0;
            size_t done = 0;
""", """            lpl(1);   // larry, STRATA_XE_LT_PROF: X read back
            double t_run = 0, t_sum = 0;
            size_t done = 0;
""")
rep("""                const double tr = now_ms();
                im.pool->run_split_multi_native(*im.lt_jobs[0].fmt, im.lt_jobs.data(), (int) nb);
                t_run += now_ms() - tr;
""", """                lpl(2);   // larry, STRATA_XE_LT_PROF: the jobs' setup (quantisation)
                const double tr = now_ms();
                im.pool->run_split_multi_native(*im.lt_jobs[0].fmt, im.lt_jobs.data(), (int) nb);
                t_run += now_ms() - tr;
                lpl(3);   // the pool
""")
rep("""            im.lt_q_ms += now_ms() - tq - t_run - t_sum;
""", """            im.lt_q_ms += now_ms() - tq - t_run - t_sum;
            lpl(4);   // larry, STRATA_XE_LT_PROF: the sum
            if (lt_prof && ++lp[5] >= 0 && (int64_t) lp[5] % 100 == 0)
                std::fprintf(stderr, "xe tier: CPU-light prof (%d calls): setup %.0f, readback %.0f, jobs %.0f, pool %.0f, sum %.0f ms\\n",
                             (int) lp[5], lp[0], lp[1], lp[2], lp[3], lp[4]);
""")
open(P, "w").write(s)
print("patched", P)
