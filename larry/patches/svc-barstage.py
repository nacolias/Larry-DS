#!/usr/bin/env python3
# svc-barstage.py [ENGINE_SH] - the service: STRATA_XE_BAR_STAGE=1 (patch-barstage.py + patch-barstage2.py) after the
# hot ring settings in ds41-xe-engine.sh.  Backup ds41-xe-engine.sh.pre-barstage-20261025.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_XE_BAR_STAGE" in s:
    print("already patched"); sys.exit(0)
anchor = 'export STRATA_XE_HOTRING_N="${STRATA_XE_HOTRING_N-2}"\n'
if s.count(anchor) != 1:
    sys.exit("anchor x%d" % s.count(anchor))
shutil.copy2(P, P + ".pre-barstage-20261025")
s = s.replace(anchor, anchor + """# the prompt path's pageable arena experts written by the CPU straight into the expert cards' VRAM through their 32 GiB
# BARs (flex regions / stream buffers made through the xe uAPI, imported into Level Zero) instead of the hot rings'
# copy + a DMA a chunk: no ring DMAs queued behind the pinned chunks in the cards' one copy engine, the pinned DMAs and
# the CPU's writes side by side on the x16 uplink (~27 GB/s).  One stager (its default with this).  abwT83 against the
# base twice, fresh prompts: 512 2.00 -> 1.94 s, 2K 2.97 -> 2.90, 4.5K 4.14 -> 3.84, 8K 5.36 -> 4.94, 16K 7.83 ->
# 7.43, 31K 13.57 -> 13.19 (layers 20-39 1.4-2.2 -> 0.9-1.2 s); HOTRING_CHECK 40,000 experts, 0 differ (abwT80)
export STRATA_XE_BAR_STAGE="${STRATA_XE_BAR_STAGE-1}"
""")
open(P, "w").write(s)
print("patched", P)
