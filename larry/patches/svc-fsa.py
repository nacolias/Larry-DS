#!/usr/bin/env python3
# svc-fsa.py [ENGINE_SH] - the service (#3, 2026-10-28): the prompt attention as one kernel that reads each query's keys
# where they lie (STRATA_ATTN_FSA=1, patch-fsa.py: no 387 MB gathered copy a 512-column span, no Q' / mask / output
# kernels - bench/fa/test_fsa.cpp 2.37 -> 1.44 ms a span at 30K), the expert cards' X in fp16 by peer loads again
# (STRATA_XE_FN16=2: card 0 now waits on the expert cards, so their X read counts), and the long prompts' indexer back on
# card 0 alone (STRATA_LI_REMOTE 131072 -> 0: with card 0 ~0.9 ms a span faster the expert cards' share only slowed their
# expert calls - abwT159: 984K 307.7 s alone against 339 / 357 s sharing 65 / 50 %, 131K 39.3 s).  abwT157 / T158: fresh
# 16K 6.48 -> 6.15 s, 31K 10.72 / 11.23 -> 9.96 / 10.35, 64K 21.3 -> 19.7.  Backup ds41-xe-engine.sh.pre-fsa-20261028; the
# file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_ATTN_FSA" in s:
    print("already patched"); sys.exit(0)


def rep(s, old, new):
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:80]))
    return s.replace(old, new)


s = rep(s, 'export STRATA_ATTN_SDPA="${STRATA_ATTN_SDPA-1}"\n',
        'export STRATA_ATTN_SDPA="${STRATA_ATTN_SDPA-1}"\n'
        '# the prompt attention as one kernel reading each query\'s keys where they lie (patch-fsa.py; with STRATA_SDPA_O16):\n'
        '# 2.37 -> 1.44 ms a 512-column span at 30K; and the expert cards\' X in fp16 by peer loads (card 0 waits on them now)\n'
        'export STRATA_ATTN_FSA="${STRATA_ATTN_FSA-1}"\n'
        'export STRATA_XE_FN16="${STRATA_XE_FN16-2}"\n')
s = rep(s, 'export STRATA_LI_REMOTE="${STRATA_LI_REMOTE-131072}"\n',
        '# (svc-fsa.py: 131072 -> 0 - with the fused attention card 0 is faster and the expert cards\' share slowed their calls;\n'
        '# 984K 307.7 s alone against 339 s sharing)\n'
        'export STRATA_LI_REMOTE="${STRATA_LI_REMOTE-0}"\n')
shutil.copy2(P, P + ".pre-fsa-20261028")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
