#!/usr/bin/env python3
# svc-dq2.py [ENGINE_SH] - the service (#3, 2026-10-28): xe_dqgemm v2 on the expert cards (STRATA_XE_DQ2=1, patch-dqv2.py:
# the weights decoded once a chunk into local memory VNNI-packed, a sub-group 32 tokens x 64 rows - the same bits, 1.4-1.8x
# alone at 128 tokens an expert).  abwT164 (same binary off / on): fresh 16K 6.14 -> 5.84 s, 31K 9.87 / 11.19 -> 9.67 /
# 10.12, 64K 19.80 -> 18.82 / 19.15, a 131K needle 40.75 -> 38.45.  Backup ds41-xe-engine.sh.pre-dq2-20261028; the file is
# replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_XE_DQ2" in s:
    print("already patched"); sys.exit(0)
anchor = 'export STRATA_XE_FN16="${STRATA_XE_FN16-2}"\n'
if s.count(anchor) != 1:
    sys.exit("anchor x%d" % s.count(anchor))
new = anchor + ('# the expert GEMMs, v2 (patch-dqv2.py: weights decoded into local memory VNNI-packed, a sub-group 32 tokens x 64 rows;\n'
                '# the same bits): fresh 16K -5 %, 64K -4 %, 131K -6 %\n'
                'export STRATA_XE_DQ2="${STRATA_XE_DQ2-1}"\n')
s = s.replace(anchor, new)
shutil.copy2(P, P + ".pre-dq2-20261028")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
