#!/usr/bin/env python3
# patch-denseinline.py [TIER_CPP] - larry, STRATA_XE_DENSE_INLINE=1 (default 0 until measured; with STRATA_XE_DENSE_LITE):
# a pipelined prompt call's dense-card part - its GEMM experts, jobs and the combine - goes on the dense half's own
# queue (the ggml-sycl queue the tier already sums decode calls on; same default platform context) instead of the
# tier's second queue on that card.  Measured with DENSE_LITE alone (2026-10-25 abwT12 b1): the waits went (16K expert
# waits 2.1 -> 1.3 s) but the dense half got as much slower (+0.6 s): two queues competing for one card cost more than
# the work itself; in the dense half's stream it runs between its spans.  The card's queue is swapped for the call
# (restored when run_big returns; nothing else submits to the dense card's tier queue during a prompt batch).
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_DENSE_INLINE" in s:
    print("already patched"); sys.exit(0)
if "STRATA_XE_DENSE_LITE" not in s:
    sys.exit("needs patch-denselite.py first")


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    const bool lite = dense_lite && x_dev && out_dev && im.dense >= 0 && NC > 1;
""", """    const bool lite = dense_lite && x_dev && out_dev && im.dense >= 0 && NC > 1;
    // larry, STRATA_XE_DENSE_INLINE: the dense card's part on the dense half's queue for this call (see patch-denseinline.py)
    static const int dense_inline = env_i("STRATA_XE_DENSE_INLINE", 0);
    struct QSwap {
        XeCard* c = nullptr;
        std::optional<sycl::queue> saved;
        ~QSwap() { if (c && saved) c->q = *saved; }
    } qswap;
""")
# the swap: before the call's first enqueue on the dense card (the FN16 block and the per-card prep come after it)
rep("""    std::unique_lock<PrioMutex> prep_lk(g_submit);
    const double t_lk = now_ms();
""", """    std::unique_lock<PrioMutex> prep_lk(g_submit);
    const double t_lk = now_ms();
    if (lite && dense_inline && im.dense_q) {   // larry, STRATA_XE_DENSE_INLINE
        XeCard& dcq = *im.cards[(size_t) im.dense];
        qswap.saved = dcq.q;
        qswap.c = &dcq;
        dcq.q = *im.dense_q;
    }
""")
if "#include <optional>" not in s:
    rep("#include <mutex>\n", "#include <mutex>\n#include <optional>\n")
open(P, "w").write(s)
print("patched", P)
