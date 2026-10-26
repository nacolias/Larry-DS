#!/usr/bin/env python3
# svc-engdrop.py [ENGINE_SH] - the service: STRATA_ENGRAM_DROP=1 (patch-engdrop.py: a prompt batch drops the Engram
# weight pages it read from the page cache, so a prompt's Engram reads stop filling the page cache - each new page
# costs reclaim, and reclaim runs the xe shrinker over the engine's own buffers).  Numbers in the comment it adds
# (DROP_NOTE).  Backup ds41-xe-engine.sh.pre-engdrop-20261026; the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_ENGRAM_DROP" in s:
    print("already patched"); sys.exit(0)
anchor = 'export STRATA_ENGRAM_THREADS="${STRATA_ENGRAM_THREADS-256}"'
i = s.find(anchor)
if i < 0 or s.count(anchor) != 1:
    sys.exit("anchor x%d" % s.count(anchor))
j = s.index("\n", i) + 1
NOTE = os.environ.get("DROP_NOTE", "")
new = ("# a prompt batch drops the Engram weight pages it read from the page cache (each new page costs reclaim, which\n"
       "# runs the xe shrinker over the engine's own buffers)\n"
       + "".join("# " + l + "\n" for l in NOTE.split("\n") if l) +
       'export STRATA_ENGRAM_DROP="${STRATA_ENGRAM_DROP-1}"\n')
s = s[:j] + new + s[j:]
shutil.copy2(P, P + ".pre-engdrop-20261026")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
