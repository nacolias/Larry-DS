#!/usr/bin/env python3
# dcmp.py REF_DIR TP_DIR N - compare tp_bench TP_DUMPDIR dumps: N=1 (REF) vs N cards (per card / summed / sliced)
import sys, numpy as np
ref, tp, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
def ld(d, k, nm, dt=np.float32): return np.fromfile(f"{d}/r{k}_{nm}.bin", dtype=dt)
def cmp(nm, a, b):
    d = np.abs(a - b).max() if a.size else 0
    print(f"{nm:10s} size {a.size:7d} max|diff| {d:.3g}  max|ref| {np.abs(b).max():.3g}")
for nm in ["x", "pre", "hap", "xn", "apre", "apost", "acomb", "qkv", "qr", "kvn", "fn", "lg", "wts", "z", "zn"]:
    r = ld(ref, 0, nm)
    for k in range(N): cmp(f"{nm}[{k}]", ld(tp, k, nm), r)
print("ids", ld(ref, 0, "ids", np.int32), [ld(tp, k, "ids", np.int32) for k in range(N)])
# sliced: qh / o by heads, oa by groups
for nm, unit in [("qh", 512), ("o", 512), ("oa", 1024)]:
    r = ld(ref, 0, nm)
    parts = [ld(tp, k, nm) for k in range(N)]
    cmp(f"{nm}cat", np.concatenate(parts), r)
cmp("ara_sum", sum(ld(tp, k, "ara") for k in range(N)), ld(ref, 0, "ara"))
cmp("arb_sum", sum(ld(tp, k, "arb") for k in range(N)), ld(ref, 0, "arb"))
import os
for l in range(64):
    f = f"{ref}/r0_sel{l}.bin"
    if not os.path.exists(f): continue
    a = set(np.fromfile(f, dtype=np.int32).tolist())
    for k in range(N):
        b = set(np.fromfile(f"{tp}/r{k}_sel{l}.bin", dtype=np.int32).tolist())
        print(f"sel{l}[{k}]: {len(a & b)} of {len(a)} picks shared")
