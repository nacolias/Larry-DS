#!/usr/bin/env python3
# svc-hcwide.py [ENGINE_SH] - the service: STRATA_HC_FUSE_MAXN=16384 - the fused hc ops (hc_fa / hc_fb: hc_post, the
# mixes' dots and hc_pre in one pass, the outputs written in place) at prompt widths too, with patch-hcwide.py's wide
# hc_fa.  abwT124 (on top of OPROJ_A + HC_NORM_MV_WIDE): fresh 8K 4.79 -> 4.50 s, 16K 6.45 -> 6.23 s, 31K 10.90 / 10.96
# -> 10.55 / 10.35 s; the agent replay's side prompts 15.31 -> 14.81 s.  Backup ds41-xe-engine.sh.pre-hcwide-20261026;
# the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_HC_FUSE_MAXN" in s:
    print("already patched"); sys.exit(0)
old = 'export STRATA_HC_NORM_MV_WIDE="${STRATA_HC_NORM_MV_WIDE:-1}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
new = old + (
    "# the fused hc ops at prompt widths too (patch-hcwide.py's wide hc_fa: 16 tokens a chunk's work-group, the mixes' rows\n"
    "# staged once): the separate hc_post / hc_pre / mixes and their state copies go - abwT124: fresh 16K 6.45 -> 6.23 s,\n"
    "# 31K 10.90 / 10.96 -> 10.55 / 10.35 s (here the prompt path's STRATA_HC_NORM_MV_WIDE is then unused)\n"
    'export STRATA_HC_FUSE_MAXN="${STRATA_HC_FUSE_MAXN:-16384}"\n')
s = s.replace(old, new)
shutil.copy2(P, P + ".pre-hcwide-20261026")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
