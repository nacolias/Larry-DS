#!/usr/bin/env python3
# svc-keepdepth.py [ENGINE_SH] - the service: STRATA_SERVE_KEEP_LEARN_DEPTH_GIB=2.0 (patch-keepdepth.py: a kept-cache
# prompt batch gets 2.0 GiB * min(1, tokens / 1M) more room on the dense card - a 984K conversation's 11.7K-token turn
# needed all of the 10.27 GiB the shrink left, and died with 9.85).  Numbers in the comment it adds (DEPTH_NOTE).
# Backup ds41-xe-engine.sh.pre-keepdepth-20261026; the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_SERVE_KEEP_LEARN_DEPTH_GIB" in s:
    print("already patched"); sys.exit(0)
anchor = 'export STRATA_SERVE_KEEP_LEARN_HIST="${STRATA_SERVE_KEEP_LEARN_HIST-1}"'
if s.count(anchor) != 1:
    sys.exit("anchor x%d" % s.count(anchor))
i = s.index(anchor)
j = s.index("\n", i) + 1
NOTE = os.environ.get("DEPTH_NOTE", "")
new = ("# a kept batch's room on card 0 grows with the conversation: +2.0 GiB at 1M tokens (+0.06 at 31K)\n"
       + "".join("# " + l + "\n" for l in NOTE.split("\n") if l) +
       'export STRATA_SERVE_KEEP_LEARN_DEPTH_GIB="${STRATA_SERVE_KEEP_LEARN_DEPTH_GIB-2.0}"\n')
s = s[:j] + new + s[j:]
shutil.copy2(P, P + ".pre-keepdepth-20261026")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
