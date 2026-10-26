#!/usr/bin/env python3
# svc-sdpao16.py [ENGINE_SH] - the service: STRATA_SDPA_O16=1 (patch-sdpao16.py: the prompt attention's output stays
# fp16 through the in-place de-RoPE into wo_a's op - no f32 o and no conversion back).  The comment's numbers: LIF_NOTE.
# Backup ds41-xe-engine.sh.pre-sdpao16-20261026; the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_SDPA_O16" in s:
    print("already patched"); sys.exit(0)
old = 'export STRATA_HC_FUSE_MAXN="${STRATA_HC_FUSE_MAXN:-16384}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
NOTE = os.environ.get("LIF_NOTE", "")
new = old + (
    "# the prompt attention's output in fp16 straight into wo_a's op (with STRATA_OPROJ_A): no f32 copy of it, no\n"
    "# conversion back\n" + "".join("# " + l + "\n" for l in NOTE.split("\n") if l) +
    'export STRATA_SDPA_O16="${STRATA_SDPA_O16:-1}"\n')
s = s.replace(old, new)
shutil.copy2(P, P + ".pre-sdpao16-20261026")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
