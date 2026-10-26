#!/usr/bin/env python3
# svc-embpin.py [ENGINE_SH] - larry, 2026-10-25 ~07:30: STRATA_EMB_PIN=1 into the service (patch-embpin.py): a batch's
# embeddings from a pinned token table, uploaded async (begin_batch 0.4-0.6 s -> ~10 ms of host work; fresh 16K -0.3..
# -0.5 s, 31K -0.9..-1.3 s, abwT31).  Backup .pre-embpin-20261025.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_EMB_PIN" in s:
    print("already patched"); sys.exit(0)
old = """exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
new = """# a prompt batch's embeddings from the token table pinned at load (1.3 GB), staged f16 and uploaded async - the host
# converted them one thread from the mmap (0.4-0.6 s a fresh batch; 2026-10-25 abwT31)
export STRATA_EMB_PIN="${STRATA_EMB_PIN-1}"
exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
shutil.copy(P, P + ".pre-embpin-20261025")
open(P, "w").write(s.replace(old, new))
print("patched", P)
