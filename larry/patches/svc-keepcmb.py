#!/usr/bin/env python3
# svc-keepcmb.py [ENGINE_SH] - larry, 2026-10-25 02:30: (1) keep-cache for every prompt batch again (STRATA_SERVE_KEEP_MAX
# 8192 -> 32768): the card-0 faults it exposed are fixed (patch-pcrace.py, commit 1d9e8d3f; the reproduction - forced
# KV-room evictions with keep-cache to 32K - ran 12 prompts clean); (2) STRATA_XE_COMBINE 1 -> 2: the expert cards push
# their prompt-chunk parts to card 0 as fp16 (half the bytes over card 0's link; fp16 has more mantissa than the model's
# bf16): 16K 10.54 -> 10.17 s, 31K 17.69 -> 17.41 s (abwT08 v1 / v3).  Backup .pre-keepcmb-20261025.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if 'STRATA_XE_COMBINE:-2' in s:
    print("already patched"); sys.exit(0)
old1 = """# (back to 8192 at 01:40: a CAT error on card 0 at the first tier call right after a 16K batch shrank the cache, abwT05
# q2 01:24 - the 16K shrink path is under investigation; ds41-xe-engine.sh.keep32k-20261025 has the 32768 setting)
export STRATA_SERVE_KEEP_MAX="${STRATA_SERVE_KEEP_MAX-8192}"
"""
new1 = """# (8192 from 01:40 to 02:30 on 2026-10-25: card-0 CAT errors after keep-cache shrinks - late promotions moved cached
# experts and the shrink freed their slabs; fixed in patch-pcrace.py / commit 1d9e8d3f, the reproduction ran clean)
export STRATA_SERVE_KEEP_MAX="${STRATA_SERVE_KEEP_MAX-32768}"
"""
old2 = 'export STRATA_XE_COMBINE="${STRATA_XE_COMBINE:-1}"\n'
new2 = """# prompt chunks: the expert cards push their parts to card 0 as fp16 (2; 1 = f32): half the bytes over card 0's link -
# 16K 10.54 -> 10.17 s, 31K 17.69 -> 17.41 s (2026-10-25 abwT08)
export STRATA_XE_COMBINE="${STRATA_XE_COMBINE:-2}"
"""
for o in (old1, old2):
    if s.count(o) != 1:
        sys.exit("anchor x%d: %s" % (s.count(o), o[:70]))
shutil.copy(P, P + ".pre-keepcmb-20261025")
open(P, "w").write(s.replace(old1, new1).replace(old2, new2))
print("patched", P)
