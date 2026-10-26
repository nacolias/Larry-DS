#!/usr/bin/env python3
# svc-hotring.py [ENGINE_SH] - larry: STRATA_XE_HOTRING=1 / STRATA_XE_HOTRING_N=2 into ds41-xe-engine.sh (before the
# exec line, with its comment).  Backs the file up first (.pre-hotring-20261024).  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_XE_HOTRING" in s:
    print("already patched"); sys.exit(0)
anchor = 'exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"\n'
if s.count(anchor) != 1:
    sys.exit("exec anchor x%d" % s.count(anchor))
shutil.copy(P, P + ".pre-hotring-20261024")
block = """# a prompt's pageable arena experts reach the cards through small pinned rings that stay in the CPU's L3: each 1 MiB
# chunk is DMA'd as soon as it is copied, so the card reads it from the cache instead of DRAM (bench/dqb/stage_ring:
# 13.0 -> 21.7 GB/s onto cards 1-3); the prefetch's two 1.5 GiB pinned stages are gone, their budget is pinned arena.
# Agent replay prompts 127 -> 118.5 s, 16K / 31K from empty -2 / -1.4 s (2026-10-24 A/B: see the README)
export STRATA_XE_HOTRING="${STRATA_XE_HOTRING-1}"
export STRATA_XE_HOTRING_N="${STRATA_XE_HOTRING_N-2}"
"""
open(P, "w").write(s.replace(anchor, block + anchor))
print("patched", P)
