#!/usr/bin/env python3
# svc-lif1.py [ENGINE_SH] - the service's fused prompt indexer back to mode 1 (GGML_SYCL_LI_FUSED 2 -> 1; svc-topk2.py had
# set 2).  bench/li/lifbf_probe.cpp (one card, 512 queries, the same bits either way): mode 2 (bf16 keys in place) is 2-4 %
# faster up to 64K keys, even at 128K, 3 % / 6 % slower at 256K / 512K keys - where a long prompt spends its indexer time;
# the 512K needle's gain was the two-stage top-k's (GGML_SYCL_TOPK2 stays).  Backup ds41-xe-engine.sh.pre-lif1-20261028;
# the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
old = 'export GGML_SYCL_LI_FUSED="${GGML_SYCL_LI_FUSED:-2}"   # 2: the BF16 keys read as they are (patch-lifbf16.py)\n'
if old not in s:
    print("already patched (or not svc-topk2.py's line)"); sys.exit(0)
new = ('export GGML_SYCL_LI_FUSED="${GGML_SYCL_LI_FUSED:-1}"   # (2, the BF16 keys in place: 6 % slower at 512K keys - svc-lif1.py)\n')
s = s.replace(old, new)
shutil.copy2(P, P + ".pre-lif1-20261028")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
