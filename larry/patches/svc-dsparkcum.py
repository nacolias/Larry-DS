#!/usr/bin/env python3
# svc-dsparkcum.py [ENGINE_SH] - the service: STRATA_DSPARK_CUM=0.5 (patch-dsparkcum.py: a DSpark draft is verified only
# while the product of the drafts' confidences so far stays >= 0.5, besides the 0.3 floor on each; Project Maya's
# MAYA_DS4_DSPARK_CUM).  Numbers in the comment it adds (CUM_NOTE).  Backup ds41-xe-engine.sh.pre-dsparkcum-20261026;
# the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_DSPARK_CUM" in s:
    print("already patched"); sys.exit(0)
old = 'export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.3}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
NOTE = os.environ.get("CUM_NOTE", "")
new = old + (
    "# DSpark: stop the verify window where the product of the drafts' confidences drops below 0.5 (Project Maya's rule)\n"
    + "".join("# " + l + "\n" for l in NOTE.split("\n") if l) +
    'export STRATA_DSPARK_CUM="${STRATA_DSPARK_CUM:-0.5}"\n')
s = s.replace(old, new)
shutil.copy2(P, P + ".pre-dsparkcum-20261026")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
