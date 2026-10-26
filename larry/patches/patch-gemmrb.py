#!/usr/bin/env python3
# patch-gemmrb.py [TIER_CPP] - larry, STRATA_XE_GEMM_RB=N (default 1024 = before): the GEMM experts' batch rows (the
# activation / gate|up / SwiGLU scratch).  A 16K call's ~18.5K rows a card ran as ~18 batches of ~5 experts, each up to
# 6 xe_dqgemm launches a phase (one per quant type x layout) of a few hundred tiles, plus the gather / SwiGLU: the
# card's GEMM phase took ~2x what bench/dqb/bench_dqgemm does for the same experts in one launch.  Bigger batches, fewer
# and fuller launches; the scratch grows ~34 MB a 1024 rows (the expert cards have ~0.4 GiB free at a prompt's peak).
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_GEMM_RB" in s:
    print("already patched"); sys.exit(0)
old = """                const int64_t RB = std::max<int64_t>(1024, (gm_max + 7) & ~(int64_t) 7);"""
new = """                static const int64_t rb_env = std::max(256, env_i("STRATA_XE_GEMM_RB", 1024));   // larry, patch-gemmrb.py
                const int64_t RB = std::max<int64_t>(rb_env, (gm_max + 7) & ~(int64_t) 7);"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
