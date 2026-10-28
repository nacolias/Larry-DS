#!/usr/bin/env python3
# svc-topk2.py [ENGINE_SH] - the service: the prompt indexer's top-k in two stages (GGML_SYCL_TOPK2=1, patch-topk2.py) and
# its keys read as bf16 where they are (GGML_SYCL_LI_FUSED 1 -> 2, patch-lifbf16.py).  Both keep the outputs: the top-k
# picks the same set (bench/li/topk2_test.cpp: 1280 rows of four kinds, 64K-1M columns), the bf16 scores are the fp16
# path's bits (GGML_SYCL_LI_FUSED_CHECK16: 0 of 3.5M differ).  2026-10-28 (bench/longctx/pp-ab.sh, the same engine): a
# 512K needle 165.5 -> 157.1 s (card 0's dense phase 136.3 -> 130.5 s).  Backup ds41-xe-engine.sh.pre-topk2-20261028;
# the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "GGML_SYCL_TOPK2" in s:
    print("already patched"); sys.exit(0)
old = 'export GGML_SYCL_LI_FUSED="${GGML_SYCL_LI_FUSED:-1}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
new = ('export GGML_SYCL_LI_FUSED="${GGML_SYCL_LI_FUSED:-2}"   # 2: the BF16 keys read as they are (patch-lifbf16.py)\n'
       '# the prompt indexer keeps each query\'s 512 best rows in two stages: its blocks\' maxima, then the radix select over the\n'
       '# blocks that reach the 512th of those (patch-topk2.py: the same set, 2.6x at 512K rows)\n'
       'export GGML_SYCL_TOPK2="${GGML_SYCL_TOPK2-1}"\n')
s = s.replace(old, new)
shutil.copy2(P, P + ".pre-topk2-20261028")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
