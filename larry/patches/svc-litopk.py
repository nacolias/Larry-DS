#!/usr/bin/env python3
# svc-litopk.py [ENGINE_SH] - the service: the prompt indexer of layers 2 / 8 / 14 as one op with its top-k
# (STRATA_LI_TOPK=1, patch-litopk.py + patch-litopk-eng.py) and, for prompts of >= 131072 tokens, its queries shared with
# the expert cards (STRATA_LI_REMOTE=131072; card 0 keeps GGML_SYCL_LI_REMOTE_SHARE=35 % of each call).  The ids are a
# valid top-k of card 0's own scores on every checked call (GGML_SYCL_LI_TOPK_CHECK: 0 wrong, card 0's share and the
# others').  2026-10-28 (bench/longctx/lt1m.sh, a 131K needle then a 967K one): the 967K prompt 338.9 -> 308.3 s (card 0's
# dense phase 282.4 -> 235.2 s; the expert calls' waits 38.9 -> 58.7 s - the cards now share both).  Backup
# ds41-xe-engine.sh.pre-litopk-20261028; the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_LI_TOPK" in s:
    print("already patched"); sys.exit(0)
anchor = 'export GGML_SYCL_TOPK2="${GGML_SYCL_TOPK2-1}"\n'
if s.count(anchor) != 1:
    sys.exit("anchor x%d" % s.count(anchor))
new = ('# the indexer of every layer that ranks all visible rows (2 / 8 / 14) as one op with its top-k (patch-litopk.py); a\n'
       '# prompt of >= 131072 tokens hands 65 % of each call\'s queries to the expert cards (their fp16 key copies, 657 MiB a\n'
       '# card at 1M, freed after the prompt): the 967K prompt 338.9 -> 308.3 s\n'
       'export STRATA_LI_TOPK="${STRATA_LI_TOPK-1}"\n'
       'export STRATA_LI_REMOTE="${STRATA_LI_REMOTE-131072}"\n'
       'export GGML_SYCL_LI_REMOTE_SHARE="${GGML_SYCL_LI_REMOTE_SHARE-35}"\n')
i = s.index(anchor) + len(anchor)
s = s[:i] + new + s[i:]
shutil.copy2(P, P + ".pre-litopk-20261028")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
