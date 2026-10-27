#!/usr/bin/env python3
# svc-lazykv.py [ENGINE_SH] - the service: card 0's KV state mapped as the conversations grow (STRATA_DENSE_LAZY_KV=1,
# the first 65536 positions of each conversation slot at load: STRATA_DENSE_KV_INIT=65536) instead of both slots' 1M
# allocated at load - 0.84 instead of 6.28 GiB, the rest for card 0's expert cache.  Safe since patch-tpstage.py (the
# other cards pull those pages through a staging buffer; read directly they came back other data).  2026-10-27 A/B
# (abw-lazyspeed, same engine): the agent replay 98.8 -> 90.5 s (prompts 66.7 -> 58.7 s, CPU-pool experts a decode pass
# 7.5 -> 5.7), fresh 2K / 4.4K -5 %, 8K / 16K even, decode: essay 76.3 -> 84.6 tok/s, code even, math 115.0 -> 121.8;
# r1m-stage.sh: a 131K needle in one slot then a 984K one in the other, both found, the 984K's decode 68.5 tok/s.
# Backup ds41-xe-engine.sh.pre-lazykv-20261027; the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_DENSE_LAZY_KV" in s:
    print("already patched"); sys.exit(0)
anchor = 'export STRATA_SERVE_PARK_GIB="${STRATA_SERVE_PARK_GIB-4}"'
if s.count(anchor) != 1:
    sys.exit("anchor x%d" % s.count(anchor))
i = s.index(anchor)
j = s.index("\n", i) + 1
new = ("# card 0's KV state mapped as the conversations grow (patch-kvbucket.py / patch-kvresident.py / patch-tpstage.py):\n"
       "# 65536 positions of each conversation slot at load, the rest by the readers' power-of-two buckets when a prompt or a\n"
       "# pass reaches them (card 0's expert cache gives the VRAM back) - both slots' 1M at load took 6.28 GiB of card 0\n"
       'export STRATA_DENSE_LAZY_KV="${STRATA_DENSE_LAZY_KV-1}"\n'
       'export STRATA_DENSE_KV_INIT="${STRATA_DENSE_KV_INIT-65536}"\n')
s = s[:j] + new + s[j:]
shutil.copy2(P, P + ".pre-lazykv-20261027")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
