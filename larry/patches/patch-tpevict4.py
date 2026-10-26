#!/usr/bin/env python3
# patch-tpevict4.py [TIER_CPP] - larry, after patch-tpevict3.py: evict only when the TP core's KV rows do not fit beside
# the free VRAM (STRATA_XE_TP_EVICT_KEEP_GIB default 1.5 -> STRATA_TP_ROOM_KEEP_GIB's 0.25).  After any prompt the
# expert cards' run_big buffers (kept for the next one) hold most of the reserve, so free VRAM sat below 1.5 GiB and the
# first decode after the first prompt evicted 1-2 slabs a card for 36 MiB of rows (~570 experts: the 1M replay's decode
# ran 14 CPU-pool experts a pass instead of 8).  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
old = """    // larry, STRATA_XE_TP_EVICT: the free VRAM down to STRATA_XE_TP_EVICT_KEEP_GIB (most of the prompt reserve), then
    // the card's evictable slabs, the flex region last
    if (tp_evict_on() && !c.pbuf_cap) {
        const int64_t ekeep = (int64_t) (env_d("STRATA_XE_TP_EVICT_KEEP_GIB", 1.5) * kGiB);
"""
new = """    // larry, STRATA_XE_TP_EVICT: the free VRAM down to STRATA_XE_TP_EVICT_KEEP_GIB (as the flex logic: 0.25 - the
    // run_big buffers kept after a prompt already hold its reserve; 1.5 evicted 570 experts for 36 MiB of rows), then
    // the card's evictable slabs, the flex region last
    if (tp_evict_on() && !c.pbuf_cap) {
        const int64_t ekeep = (int64_t) (env_d("STRATA_XE_TP_EVICT_KEEP_GIB", 0.25) * kGiB);
"""
if new in s:
    print("already patched"); sys.exit(0)
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
