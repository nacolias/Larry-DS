#!/usr/bin/env python3
# svc-pinzc.py [ENGINE_SH] - larry, 2026-10-25 ~11:00: STRATA_XE_PIN_STREAM=1 (patch-pinstream.py) and STRATA_XE_ZCOPY=1
# (patch-zcopy.py) into the service: the pinned arena for what prompts stream (phase B 14.4 -> 16.5 GB/s) and a call's
# small uploads by kernels from pinned per-call copies (no copy-engine FIFO, no host waits on the expert cards).  Fresh
# 8K -0.22 s, 16K -0.3 s, 31K -0.2..-0.7 s (abwT50); tpg pass (abwT51).  Backup .pre-pinzc-20261025.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_XE_PIN_STREAM" in s:
    print("already patched"); sys.exit(0)
old = """exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
new = """# the pinned arena re-targeted at what prompts stream: card 0's cache below layer 20 (phase A streams it, the regrow
# refills it) + the next-ranked experts of layers 20.. (phase B's); the cache's layers 20.. (never streamed) pageable.
# A tier call's small uploads (row lists, slot maps, GEMM job / tile tables) by kernels reading pinned per-call copies:
# a memcpy waited in the card's one copy engine behind the staging DMAs (2026-10-25 abwT39-T50)
export STRATA_XE_PIN_STREAM="${STRATA_XE_PIN_STREAM-1}"
export STRATA_XE_ZCOPY="${STRATA_XE_ZCOPY-1}"
exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
shutil.copy(P, P + ".pre-pinzc-20261025")
open(P, "w").write(s.replace(old, new))
print("patched", P)
