#!/usr/bin/env python3
# svc-slots.py [CONFIG_JSON] [ENGINE_SH] - the service: two concurrent sessions of up to 1M each (the owner's choice,
# 2026-10-27).  larry-ds41.json "parallel": 2 (serve/server.py passes --batch 2 and answers two requests at once through
# its batch client; patch-slots-serve.py).  Both conversation slots' card-0 KV state is allocated at load (+3.1 GiB on
# card 0); SLOTS_LAZY=1 instead maps it as the conversations grow (STRATA_DENSE_LAZY_KV=1, KV_INIT 65536 a slot) - NOT
# for the service yet: r1m (2026-10-27 18:41) garbled a 970K needle's decode in the slot that had held another
# conversation, then DEVICE_LOST; the same sequence with the state allocated at load passed (18:52).  Backups
# *.pre-slots-20261027; each file replaced by a rename.  Idempotent.
import json
import os
import shutil
import sys

J = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/larry-ds41.json"
E = sys.argv[2] if len(sys.argv) > 2 else "/opt/ai/dsv41/ds41-xe-engine.sh"


def put(path, text):
    shutil.copy2(path, path + ".pre-slots-20261027")
    tmp = path + ".new"
    open(tmp, "w").write(text)
    shutil.copymode(path, tmp)
    os.rename(tmp, path)
    print("patched", path)


c = json.load(open(J))
if c.get("parallel") != 2:
    c["parallel"] = 2
    put(J, json.dumps(c, indent=2) + "\n")
else:
    print("already", J)
s = open(E).read()
if os.environ.get("SLOTS_LAZY") != "1":
    print("card 0's KV state allocated at load (SLOTS_LAZY=1 maps it lazily)")
elif "STRATA_DENSE_LAZY_KV" not in s:
    anchor = 'export STRATA_SERVE_PARK_GIB="${STRATA_SERVE_PARK_GIB-4}"\n'
    if s.count(anchor) != 1:
        sys.exit("anchor x%d" % s.count(anchor))
    s = s.replace(anchor, anchor +
                  "# card 0's KV state mapped as each conversation grows (the readers' power-of-two buckets, resident on the\n"
                  "# other cards as mapped: patch-kvbucket.py + patch-kvresident.py) - two slots of up to 1M cost VRAM only when used\n"
                  'export STRATA_DENSE_LAZY_KV="${STRATA_DENSE_LAZY_KV-1}"\n'
                  'export STRATA_DENSE_KV_INIT="${STRATA_DENSE_KV_INIT-65536}"\n')
    put(E, s)
else:
    print("already", E)
