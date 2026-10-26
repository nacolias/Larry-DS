#!/usr/bin/env python3
# svc-ctx1m.py - larry: the service's context 262,144 -> 1,048,576 tokens (larry-ds41.json --ctx) and the park's limit
# 65,536 -> 262,144 positions (ds41-xe-engine.sh STRATA_SERVE_PARK_MAX: a side request of a long conversation may
# write that far; ~0.8 GB of host memory at ~3 KB a position) and STRATA_XE_TP_EVICT=1.  Backups .pre-ctx1m-20261024.  Idempotent.
import json
import shutil
import sys

J = "/opt/ai/dsv41/larry-ds41.json"
E = "/opt/ai/dsv41/ds41-xe-engine.sh"
c = json.load(open(J))
a = c["args"]
i = a.index("--ctx")
if a[i + 1] != "1048576":
    shutil.copy(J, J + ".pre-ctx1m-20261024")
    a[i + 1] = "1048576"
    open(J, "w").write(json.dumps(c, indent=2) + "\n")
    print("ctx ->", a[i + 1])
s = open(E).read()
if "STRATA_SERVE_PARK_MAX" not in s:
    anchor = 'exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"\n'
    if s.count(anchor) != 1:
        sys.exit("exec anchor x%d" % s.count(anchor))
    shutil.copy(E, E + ".pre-ctx1m-20261024")
    block = """# the context is 1M tokens (larry-ds41.json; KV ~3 KB a token: the TP cards map ~3.1 GiB each as it fills): a side
# request inside a long conversation may write up to 262,144 positions and still park the conversation (~0.8 GB of host
# memory; was 65,536)
export STRATA_SERVE_PARK_MAX="${STRATA_SERVE_PARK_MAX-262144}"
# the TP cards' KV room beyond what 262K needs (0.8 GiB a card) holds experts until a long context needs it: then
# their slabs go back (copies read from the model file) - without it a 1M context cost 757 experts on the cards
export STRATA_XE_TP_EVICT="${STRATA_XE_TP_EVICT-1}"
"""
    open(E, "w").write(s.replace(anchor, block + anchor))
    print("park max -> 262144")
