#!/usr/bin/env python3
# svc-parkpool.py [ENGINE_SH] - the service: STRATA_SERVE_PARK_SLOTS=8 + STRATA_SERVE_PARK_GIB=4 (patch-parkpool.py:
# the parked conversations are a pool - up to 8 kept beside the live one in at most 4 GiB of host memory, ~3.2 KB a
# token: a 100K-token conversation ~0.33 GB; the oldest go first).  2026-10-27 s0test (four ~3-4K conversations
# rotated, each second turn taken back from the pool: 0.4-0.5 s, ~24 tokens read, against 3.5-4.6 s fresh).
# Backup ds41-xe-engine.sh.pre-parkpool-20261027; the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_SERVE_PARK_SLOTS" in s:
    print("already patched"); sys.exit(0)
anchor = 'export STRATA_SERVE_PARK_MAX="${STRATA_SERVE_PARK_MAX-262144}"'
if s.count(anchor) != 1:
    sys.exit("anchor x%d" % s.count(anchor))
i = s.index(anchor)
j = s.index("\n", i) + 1
new = ("# the parked conversations: a pool of 8 beside the live one, at most 4 GiB of host memory together (~3.2 KB a token;\n"
       "# the oldest go first; patch-parkpool.py) - rotating several agents / chats no longer re-reads their prompts\n"
       'export STRATA_SERVE_PARK_SLOTS="${STRATA_SERVE_PARK_SLOTS-8}"\n'
       'export STRATA_SERVE_PARK_GIB="${STRATA_SERVE_PARK_GIB-4}"\n')
s = s[:j] + new + s[j:]
shutil.copy2(P, P + ".pre-parkpool-20261027")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
