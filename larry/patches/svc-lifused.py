#!/usr/bin/env python3
# svc-lifused.py [ENGINE_SH] - the service: GGML_SYCL_LI_FUSED=1 (the prompt indexer as one fused XMX kernel,
# patch-lifused.py) + STRATA_LI_NVIS=1 (the indexer told each query's visible rows instead of a [cap, n] mask,
# patch-linvis.py).  Numbers in the comment it adds (abwT112).  Backup ds41-xe-engine.sh.pre-lifused-20261026; the file
# is replaced by a rename (a bash already reading it keeps the old one).  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "GGML_SYCL_LI_FUSED" in s:
    print("already patched"); sys.exit(0)
old = 'export GGML_SYCL_LI_XMX="${GGML_SYCL_LI_XMX:-1}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
NOTE = os.environ.get("LIF_NOTE", "")
new = old + (
    "# the prompt indexer (512-query spans) as one fused XMX kernel: li_gemm wrote the [B*32 x cap] f32 score matrix and\n"
    "# read it back (5.7 ms a call at a 31K prompt's second batch, ~10 % of card 0) - fused 16K keys 4.36 -> 0.63 ms,\n"
    "# 64K 17.4 -> 2.33 ms (bench/li/li_probe); and told each query's visible rows instead of a [cap, n] 0 / -inf mask\n"
    "# (its repeat / add / step / log / F16 copy, ~0.9 ms a call)" + (" - " + NOTE if NOTE else "") + "\n"
    'export GGML_SYCL_LI_FUSED="${GGML_SYCL_LI_FUSED:-1}"\n'
    'export STRATA_LI_NVIS="${STRATA_LI_NVIS:-1}"\n')
s = s.replace(old, new)
shutil.copy2(P, P + ".pre-lifused-20261026")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
