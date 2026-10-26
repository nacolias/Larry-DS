#!/bin/bash
# ttp17.sh - the row-111 hunt (3): KEEP mode (the trigger: 2 of 3 runs bad) with STRATA_DUMP_COL=111 - per layer that
# column's routing, FFN input and routed sum; x4, each run's bad rows vs t14_off1 and its dump kept (logs/dump_col.*)
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PAGE_KEEP=1 STRATA_DUMP_COL=111"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
row() { venv/bin/python -I - "$1" <<'PY'
import sys, numpy as np
V = 129280
ref = np.memmap("logs/t14_off1.logits", dtype=np.float32, mode="r").reshape(-1, V)
x = np.memmap("logs/%s.logits" % sys.argv[1], dtype=np.float32, mode="r").reshape(-1, V)
def kl(p, q):
    p = p.astype(np.float64); q = q.astype(np.float64); p -= p.max(); q -= q.max()
    lp = p - np.log(np.exp(p).sum()); lq = q - np.log(np.exp(q).sum())
    return float((np.exp(lp) * (lp - lq)).sum())
print("row 111 KL %.3f" % kl(ref[111], x[111]))
PY
}
for i in 1 2 3 4; do
  N=t17_$i
  env $B $T RUN_TIMEOUT=1800 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --ctx 8192 --prompt-batch 2048 --route-bias 0 --route-bias-file 0 --dump-logits logs/$N.logits > /dev/null 2>&1
  d=$(grep -ao "logs/dump_col\.[0-9]*\.bin" logs/$N.err | tail -1); [ -n "$d" ] && mv $d logs/$N.dump
  echo "[$N] $(grep -ah "^rc=" logs/$N.err) $(row $N) dump $(ls -la logs/$N.dump 2>/dev/null | awk '{print $5}')"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP17 done $(date +%T)"
