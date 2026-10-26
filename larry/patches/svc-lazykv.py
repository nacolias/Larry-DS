#!/usr/bin/env python3
# svc-lazykv.py [ENGINE_SH] - larry: STRATA_DENSE_LAZY_KV=1 into ds41-xe-engine.sh (before the exec line, with its
# comment).  Backs the file up first (.pre-lazykv-20261024).  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_DENSE_LAZY_KV" in s:
    print("already patched"); sys.exit(0)
anchor = 'exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"\n'
if s.count(anchor) != 1:
    sys.exit("exec anchor x%d" % s.count(anchor))
shutil.copy(P, P + ".pre-lazykv-20261024")
block = """# card 0's own KV state in virtual memory, mapped as the context grows (the first 262,144 positions at load, then 32K
# at a time, card 0's expert cache giving the VRAM back): a 1M context allocated 2.3 GiB of it at load out of that cache
export STRATA_DENSE_LAZY_KV="${STRATA_DENSE_LAZY_KV-1}"
"""
open(P, "w").write(s.replace(anchor, block + anchor))
print("patched", P)
