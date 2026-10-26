#!/usr/bin/env python3
# svc-hcon.py [ENGINE_SH] - the service: STRATA_HC_FUSE_MAXN back to 16384 (the fused hc ops at prompt widths).  It was
# off since 12:50 because abwT131's 1M turn died in DSV41_HC_FA with UR OUT_OF_RESOURCES - the kept batch's VRAM edge at
# 1M (abwT141 died the same way with it off), which STRATA_SERVE_KEEP_LEARN_DEPTH_GIB now covers: abwT144 with it on -
# fresh 16K / 31K -3-4 %, the 984K needle 357 -> 346 s, its turn passed with 1.66 GiB free at the low point.
# Backup ds41-xe-engine.sh.pre-hcon-20261026; the file is replaced by a rename.  Idempotent.
import shutil
import os
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
new_line = 'export STRATA_HC_FUSE_MAXN="${STRATA_HC_FUSE_MAXN:-16384}"\n'
if new_line in s:
    print("already patched"); sys.exit(0)
old = """# OFF again since 2026-10-26 12:50 (8 = decode widths only): abwT131 - an 11,684-token turn at 973K context failed in
# DSV41_HC_FA with UR_RESULT_ERROR_OUT_OF_RESOURCES (card 0 nearly full there); with 8 the same turn ran (abwT132)
export STRATA_HC_FUSE_MAXN="${STRATA_HC_FUSE_MAXN:-8}"
"""
new = """# (off 12:50-20:10: abwT131's 973K turn died in DSV41_HC_FA with UR OUT_OF_RESOURCES - the kept batch's VRAM edge at 1M,
# which failed with it off too (abwT141) and STRATA_SERVE_KEEP_LEARN_DEPTH_GIB now covers; abwT144 with it on: fresh
# 16K / 31K -3-4 %, the 984K needle 357 -> 346 s, its turn passed with 1.66 GiB free at the low point)
""" + new_line
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
shutil.copy2(P, P + ".pre-hcon-20261026")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
