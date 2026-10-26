#!/usr/bin/env python3
# patch-fn16q.py [TIER_CPP] - larry: with STRATA_XE_DENSE_INLINE the dense card's queue is the dense half's for the
# call, so STRATA_XE_FN16's conversion (the expert cards wait for it before their X copies) would queue behind the
# dense half's backlog (up to two spans, ~15 ms) - it goes on the dense card's own tier queue.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "patch-fn16q" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    if (fn16 && grow_dev(im.fn16, im.fn16_cap, (size_t) n * H, cd16->q)) {
        const float* xs = x_dev;
        sycl::half* hd = im.fn16;
        fn16_ev = cd16->q.parallel_for(""", """    // larry, patch-fn16q: on the dense card's own tier queue (DENSE_INLINE swapped its queue for the dense half's)
    sycl::queue& q16 = qswap.c && qswap.saved ? *qswap.saved : (cd16 ? cd16->q : im.cards[0]->q);
    if (fn16 && grow_dev(im.fn16, im.fn16_cap, (size_t) n * H, q16)) {
        const float* xs = x_dev;
        sycl::half* hd = im.fn16;
        fn16_ev = q16.parallel_for(""")
open(P, "w").write(s)
print("patched", P)
