#!/usr/bin/env python3
# svc-sdpa.py [ENGINE_SH] - larry, 2026-10-25 ~12:05: STRATA_ATTN_SDPA=1 into the service (patch-sdpa.py +
# patch-sdpa-dense.py): a full 512-query span of the fixed-key layers (0-19) through one oneDNN graph SDPA.  Fresh 16K
# -0.2 s, 31K -0.4..-0.6 s (abwT54); KL p512 2.8e-3 / p2k 6.3e-3 against floors 1.9e-3 / 5.9e-3 (tlppl-sdpa.sh); tpg
# pass (abwT55).  Backup .pre-sdpa-20261025.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_ATTN_SDPA" in s:
    print("already patched"); sys.exit(0)
old = """exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
new = """# the prompt attention of layers 0-19 (128 or 640 keys a query) as one oneDNN graph SDPA a 512-query span: keys 576 wide
# with an extra sink key (2026-10-25 abwT54; partial spans and layers 20-39 keep the oneMKL path)
export STRATA_ATTN_SDPA="${STRATA_ATTN_SDPA-1}"
exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
shutil.copy(P, P + ".pre-sdpa-20261025")
open(P, "w").write(s.replace(old, new))
print("patched", P)
