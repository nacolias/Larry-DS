#!/usr/bin/env python3
# svc-keeplearn.py [ENGINE_SH] - the service: STRATA_SERVE_KEEP_LEARN=1 (patch-keeplearn.py + patch-keeplearn2.py) after
# the keep-cache size, and STRATA_XE_CPU_LIGHT_PINNED=0 after the CPU-light threshold, in ds41-xe-engine.sh.  Backup
# ds41-xe-engine.sh.pre-keeplearn-20261025.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_SERVE_KEEP_LEARN" in s:
    print("already patched"); sys.exit(0)
anchor = 'export STRATA_SERVE_KEEP_MAX="${STRATA_SERVE_KEEP_MAX-32768}"\n'
if s.count(anchor) != 1:
    sys.exit("anchor x%d" % s.count(anchor))
shutil.copy2(P, P + ".pre-keeplearn-20261025")
s = s.replace(anchor, anchor + """# the keep-cache shrink sized from what the last kept batch took on card 0 (its low point of free VRAM, sampled during it)
# instead of the calibrated 3.5 GiB + 0.45 MiB a column + 1 GiB: after a start's first batch the pools exist, and an
# 8K batch took 3.3 of the 8.0 GiB freed for it - the rest was card 0's experts evicted, streamed and re-uploaded for
# nothing.  Turns no deeper than the learned one (or 16K) only, 0.6 MiB a column more + 1.5 GiB margin; a low point
# under 1 GiB goes back to the calibrated size.  Regrow 8K 517 -> 266 experts, 16K 839 -> 582; fresh 8K / 16K -0.1 s
# (abwT67).  0 = the calibrated size always
export STRATA_SERVE_KEEP_LEARN="${STRATA_SERVE_KEEP_LEARN-1}"
""")
a2 = 'export STRATA_XE_CPU_LIGHT="${STRATA_XE_CPU_LIGHT-4}"\n'
if s.count(a2) != 1:
    sys.exit("anchor x%d (CPU_LIGHT)" % s.count(a2))
s = s.replace(a2, a2 + """# ... but not the pinned arena's light experts: they reach the cards by DMA from where they are (one DRAM read, no CPU),
# so the pool takes only pageable ones (a pageable expert streamed costs ~3 DRAM reads: copy into a ring + the DMA) -
# fresh 512 / 1K / 2K / 3K prompts 2.19 / 2.70 / 3.67 / 3.92 -> 2.08 / 2.43 / 3.33 / 3.79 s, 4.5K-16K the same or a
# little faster (abwT69, abwT70).  1 = the pool takes pinned ones too
export STRATA_XE_CPU_LIGHT_PINNED="${STRATA_XE_CPU_LIGHT_PINNED-0}"
""")
open(P, "w").write(s)
print("patched", P)
