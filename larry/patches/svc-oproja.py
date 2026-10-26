#!/usr/bin/env python3
# svc-oproja.py [ENGINE_SH] - the service: STRATA_OPROJ_A=1 (patch-oproja.py + patch-oproja2.py: the prompt path's wo_a
# as one pointer gemm_batch writing wo_b's input layout) + STRATA_HC_NORM_MV_WIDE=1 (patch-normmvw.py: the prompt path's
# hc mixes as one wide norm_mv).  Numbers in the comment it adds (LIF_NOTE).  Backup ds41-xe-engine.sh.pre-oproja-20261026;
# the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_OPROJ_A" in s:
    print("already patched"); sys.exit(0)
old = 'export STRATA_LI_NVIS="${STRATA_LI_NVIS:-1}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
NOTE = os.environ.get("LIF_NOTE", "")
new = old + (
    "# card 0's 512-column prompt span: wo_a's 8 slices as one product written in wo_b's layout (768 us + a 162 us copy ->\n"
    "# 520 us) and the hc mixes' rms_norm copy + [24 x 20480] GEMM as one pass over the state (2 x 323 -> 2 x 153 us)\n"
    + "".join("# " + l + "\n" for l in NOTE.split("\n") if l) +
    'export STRATA_OPROJ_A="${STRATA_OPROJ_A:-1}"\n'
    'export STRATA_HC_NORM_MV_WIDE="${STRATA_HC_NORM_MV_WIDE:-1}"\n')
s = s.replace(old, new)
shutil.copy2(P, P + ".pre-oproja-20261026")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
