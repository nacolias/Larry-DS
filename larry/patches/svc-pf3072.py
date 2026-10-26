#!/usr/bin/env python3
# svc-pf3072.py [ENGINE_SH] - the service: STRATA_XE_PREFETCH_MIN 4096 -> 3072.  abwT103 (agent replay, two runs each):
# at 2048, the 3.6-3.8K-token follow-ups at 7.9-34K went 4.57-4.99 -> 3.89-4.34 s (their layers 0-19 streamed on
# demand at 4096: 2048-column calls of 70-75 ms) but a 2,197-token one 3.12-3.20 -> 3.56 s and a 1,711-token one
# 2.92-2.95 -> 3.01-3.10 s; follow-ups 45.05 / 45.23 -> 42.88 / 42.60 s.  3072 keeps the first and leaves the others
# (fresh prompts ending before STRATA_XE_PREFETCH_DEPTH take none either way).
# Backup ds41-xe-engine.sh.pre-pf3072-20261026.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
new = 'export STRATA_XE_PREFETCH_MIN="${STRATA_XE_PREFETCH_MIN-3072}"'
if new in s:
    print("already patched"); sys.exit(0)
old = 'export STRATA_XE_PREFETCH_MIN="${STRATA_XE_PREFETCH_MIN-4096}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
shutil.copy2(P, P + ".pre-pf3072-20261026")
s = s.replace(old, "# 3072 since 2026-10-26 (was 4096): a chat's 3.6-3.8K-token follow-ups at depth -0.65..-0.87 s (their layers 0-19 no\n"
              "# longer stream on demand); 2048 made 1.7-2.2K ones 0.1-0.45 s slower (abwT103)\n" + new + "\n")
open(P, "w").write(s)
print("patched", P)
