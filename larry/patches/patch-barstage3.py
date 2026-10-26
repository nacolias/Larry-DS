#!/usr/bin/env python3
# patch-barstage3.py [TIER_CPP] - larry, after patch-barstage2.py / patch-pfbtrace.py: a BAR flex region was re-made in
# nearly every batch (abwT85: 1.80 -> 1.81, 1.81 -> 1.71, 1.82 -> 1.95 GiB ... - the computed slot size moves by 10-130
# MiB from prompt to prompt), and a new region's CPU pages cost 260-420 ms a card to fault in (prefetch_begin 481-699
# ms a batch vs 32-124 without BAR).  Now a BAR region stays while it is at most 256 MiB too small (the slots shrink to
# what it holds) or 160 MiB too large (inside the prefetch reserve's margin); real changes (the TP core's KV rows
# growing on the expert cards, 8K -> 16K: ~0.5 GiB) still re-make it.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "patch-barstage3" in s:
    print("already patched"); sys.exit(0)
if "patch-pfbtrace" not in s:
    sys.exit("patch-pfbtrace.py first")


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""        if (!c.flex || c.flex_cap < 2 * cap || c.flex_cap > 2 * cap + ((size_t) 64 << 20)) {
            const double ta = now_ms();""", """        // larry, patch-barstage3: a BAR region stays unless it is > 256 MiB too small or > 160 MiB too large (its CPU
        // pages cost ~0.25 s a GiB to fault in again); the slots then take what it holds
        const bool bar_f = c.flex && bar_cpu(c.flex);
        const size_t f_lo = bar_f ? ((size_t) 256 << 20) : 0, f_hi = (size_t) (bar_f ? 160 : 64) << 20;
        if (!c.flex || c.flex_cap + f_lo < 2 * cap || c.flex_cap > 2 * cap + f_hi) {
            const double ta = now_ms();""")
rep("""        if (pfb_log && keep_st) pfb += " (stream kept)";
        c.pbuf[0] = c.flex;
        c.pbuf[1] = c.flex + cap;
        c.pbuf_cap = cap;
""", """        if (pfb_log && keep_st) pfb += " (stream kept)";
        const size_t use = std::min(cap, (c.flex_cap / 2) & ~(size_t) 255);   // larry, patch-barstage3
        if (use < (size_t) im.blob_max) { c.pbuf_cap = 0; continue; }
        c.pbuf[0] = c.flex;
        c.pbuf[1] = c.flex + use;
        c.pbuf_cap = use;
""")
open(P, "w").write(s)
print("patched", P)
