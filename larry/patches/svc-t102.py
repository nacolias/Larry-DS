#!/usr/bin/env python3
# svc-t102.py [ENGINE_SH] - the service after abwT102 (2026-10-26 01:00-01:20):
#   STRATA_SERVE_KEEP_LEARN_HIST=1 + STRATA_SERVE_REGROW_ASYNC=1 (patch-keephist.py, patch-regrowasync*.py): the agent
#     replay's follow-up prompts 45.93 / 45.80 -> 44.98 / 45.03 s (two runs each; 0.05-0.15 s a turn), fresh side
#     prompts and decode even;
#   STRATA_SHEXP_TIER=1 at STRATA_SHEXP_TIER_MIN=4096 (patch-shexptier.py / -dense / -gen): fresh 16K 7.26 -> 7.00 s,
#     31K 12.72 -> 12.09 s, 8K even; numerics within the run-to-run floor (p2k KL 2.4e-3 vs 2.2e-3, p8k 2.7e-3 vs 2.9e-3);
#     a 2K prompt's 1K parts were ~1 s slower with it, hence parts of >= 4096 columns only.
# Backup ds41-xe-engine.sh.pre-t102-20261026.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_SERVE_REGROW_ASYNC" in s:
    print("already patched"); sys.exit(0)
old = 'export STRATA_SERVE_KEEP_LEARN="${STRATA_SERVE_KEEP_LEARN-1}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
shutil.copy2(P, P + ".pre-t102-20261026")
add = ('# ... from the last 8 kept batches: the smallest VRAM a batch at least as wide took, x the depth ratio for a deeper\n'
       '# turn (a chat\'s follow-ups end deeper every time, so the single learned size never applied to them), and the regrow\n'
       '# after the batch left running while decode starts (published by the first TP pass after its copies): agent replay\n'
       '# follow-ups 45.9 -> 45.0 s, fresh prompts and decode even (abwT102)\n'
       'export STRATA_SERVE_KEEP_LEARN_HIST="${STRATA_SERVE_KEEP_LEARN_HIST-1}"\n'
       'export STRATA_SERVE_REGROW_ASYNC="${STRATA_SERVE_REGROW_ASYNC-1}"\n'
       '# the shared expert of a prompt call\'s parts of >= 4096 columns on the expert cards (card 0 bounds those batches):\n'
       '# fresh 16K 7.26 -> 7.00 s, 31K 12.72 -> 12.09 s; KL within the run-to-run floor; smaller parts slower (abwT102)\n'
       'export STRATA_SHEXP_TIER="${STRATA_SHEXP_TIER-1}"\n'
       'export STRATA_SHEXP_TIER_MIN="${STRATA_SHEXP_TIER_MIN-4096}"\n')
s = s.replace(old, old + add)
open(P, "w").write(s)
print("patched", P)
