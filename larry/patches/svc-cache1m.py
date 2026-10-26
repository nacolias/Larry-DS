#!/usr/bin/env python3
# svc-cache1m.py [ENGINE_SH] - larry: with the 1M context card 0's KV state takes 2.3 GiB more of its VRAM at load, and
# its expert cache gave it up (1,850 -> 1,669 experts) while the replicas kept 2.5 GiB.  The cache is worth more:
# STRATA_XE_DENSE_RESERVE_GIB 4 -> 1.75 (the replicas used to take ~2.5 of that reserve anyway) and STRATA_XE_REPLICA_GIB
# 2.5 -> 0.25.  Agent replay at 1M (2026-10-24 22:20): prompts 103.3 -> 99.0 s (the 262K config's 99.0), decode 112.6 ->
# 115.6 tok/s, CPU-pool experts a pass 12.4 -> 10.2.  Backup .pre-cache1m-20261024.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if 'STRATA_XE_DENSE_RESERVE_GIB:-1.75' in s:
    print("already patched"); sys.exit(0)
old1 = """# the dense card's expert share: 14 GiB (decode DSpark +7.5% vs 10: fewer CPU-pool experts), 4 GiB kept free
export STRATA_XE_DENSE_GIB="${STRATA_XE_DENSE_GIB:-14}"
export STRATA_XE_DENSE_RESERVE_GIB="${STRATA_XE_DENSE_RESERVE_GIB:-4}"
"""
new1 = """# the dense card's expert share: 14 GiB (decode DSpark +7.5% vs 10: fewer CPU-pool experts), 1.75 GiB kept free (was 4:
# the replicas below took ~2.5 of it; with the 1M context's KV state the cache is worth more than they are - 1M agent
# replay prompts 103.3 -> 99.0 s, decode 112.6 -> 115.6 tok/s with 0.25 GiB of replicas, 2026-10-24 22:20)
export STRATA_XE_DENSE_GIB="${STRATA_XE_DENSE_GIB:-14}"
export STRATA_XE_DENSE_RESERVE_GIB="${STRATA_XE_DENSE_RESERVE_GIB:-1.75}"
"""
old2 = 'export STRATA_XE_REPLICA_GIB="${STRATA_XE_REPLICA_GIB-2.5}"\n'
new2 = '# (0.25 since the 1M context: card 0\'s VRAM goes to its expert cache first - see STRATA_XE_DENSE_RESERVE_GIB)\nexport STRATA_XE_REPLICA_GIB="${STRATA_XE_REPLICA_GIB-0.25}"\n'
for o in (old1, old2):
    if s.count(o) != 1:
        sys.exit("anchor x%d: %s" % (s.count(o), o[:60]))
shutil.copy(P, P + ".pre-cache1m-20261024")
open(P, "w").write(s.replace(old1, new1).replace(old2, new2))
print("patched", P)
