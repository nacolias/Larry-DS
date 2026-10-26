#!/usr/bin/env python3
# svc-engpg.py [ENGINE_SH] [CONFIG_JSON] - larry, 2026-10-25: (1) the first Engram layer's table read from its paged copy
# on the WD SN850X (/srv/dsv41/engram-epg/engram-L1.epg: 4 KiB pages of 15 rows, one read a row; the shard lives on the
# Sabrent, ~230K random reads/s vs ~850K) - its directory first in --engram, the repo's shards stay as they are;
# (2) keep-cache for every prompt batch (STRATA_SERVE_KEEP_MAX 8192 -> 32768: a 16K / 31K batch gives back only what it
# needs from card 0's expert cache - sized with the xe constants since patch-keepneed.py - instead of all 19.9 GiB).
# Backups .pre-engpg-20261025.  Idempotent.
import json
import shutil
import sys

SH = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
CJ = sys.argv[2] if len(sys.argv) > 2 else "/opt/ai/dsv41/larry-ds41.json"
EPG = "/srv/dsv41/engram-epg"

s = open(SH).read()
old = """# keep-cache up to 8192-token batches (was 4096): the batch gives back only the VRAM it needs from card 0's expert
# cache, and the decode after it regrows the tail instead of re-seeding the whole cache (~0.9 s)
export STRATA_SERVE_KEEP_MAX="${STRATA_SERVE_KEEP_MAX-8192}"
"""
new = """# keep-cache for every prompt batch (8192 -> 32768 on 2026-10-25, with patch-keepneed.py's xe sizing): the batch gives
# back only the VRAM it needs from card 0's expert cache, and the decode after it regrows the tail instead of
# re-seeding the whole cache (~0.9 s); a 16K batch keeps ~9 GiB of the 19.9 GiB cache (less streaming)
export STRATA_SERVE_KEEP_MAX="${STRATA_SERVE_KEEP_MAX-32768}"
"""
if new not in s:
    if s.count(old) != 1:
        sys.exit("engine sh anchor x%d" % s.count(old))
    shutil.copy(SH, SH + ".pre-engpg-20261025")
    open(SH, "w").write(s.replace(old, new))
    print("patched", SH)
else:
    print("already patched", SH)

c = json.load(open(CJ))
a = c["args"]
i = a.index("--engram") + 1
dirs = a[i].split(":")
if dirs[0] != EPG:
    shutil.copy(CJ, CJ + ".pre-engpg-20261025")
    a[i] = ":".join([EPG] + [d for d in dirs if d != EPG])
    json.dump(c, open(CJ, "w"), indent=2)
    open(CJ, "a").write("\n")
    print("patched", CJ, "->", a[i])
else:
    print("already patched", CJ)
