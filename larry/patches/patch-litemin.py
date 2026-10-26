#!/usr/bin/env python3
# patch-litemin.py [TIER_CPP] - larry, STRATA_XE_DENSE_LITE_MIN (default 2048): DENSE_LITE keeps streamed experts off
# the dense card only in calls of >= this many tokens.  A short pipelined batch's dense parts are short too (a
# 1.5-2.2K-token turn: ~2 spans a part) and its calls are bound by their on-demand streaming - there card 0 is idle and
# its quarter of the copies is worth more than it costs (abwT19: those turns 0.07-0.13 s slower without it).  X in
# place and the no-wait part stay for every DENSE_LITE call.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_DENSE_LITE_MIN" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    const bool lite = dense_lite && x_dev && out_dev && im.dense >= 0 && NC > 1;
""", """    const bool lite = dense_lite && x_dev && out_dev && im.dense >= 0 && NC > 1;
    // larry, STRATA_XE_DENSE_LITE_MIN: the streamed experts kept off the dense card only in calls this long (see
    // patch-litemin.py); shorter calls stream on every card as before
    static const int lite_min = env_i("STRATA_XE_DENSE_LITE_MIN", 2048);
    const bool lite_nostream = lite && n >= lite_min;
""")
rep("""            if (lite && im.cards[k]->dense) continue;   // larry, STRATA_XE_DENSE_LITE
""", """            if (lite_nostream && im.cards[k]->dense) continue;   // larry, STRATA_XE_DENSE_LITE (_MIN)
""")
open(P, "w").write(s)
print("patched", P)
