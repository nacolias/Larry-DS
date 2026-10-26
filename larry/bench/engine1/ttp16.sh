#!/bin/bash
# ttp16.sh - engine1 pager, the row-111 hunt (2): e4a the region kept as loaded + hooks, no refill (the memory type);
# e4b the same + a 3 ms busy kernel on the dense queue each layer (timing only); e3 KEEP with the refill copies as
# peer reads on card 0's queue (not the source cards' queues); x3 each, logits vs t14_off1
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache"
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
ks = np.array([kl(ref[i], x[i]) for i in range(ref.shape[0])])
print("bad rows", [(int(i), round(float(ks[i]), 2)) for i in np.where(ks > 0.5)[0]], "mean KL %.5f" % ks.mean())
PY
}
for c in e4a e4b e3; do
  case $c in
    e4a) P="STRATA_TP_PAGE_DUMMY=0" ;;
    e4b) P="STRATA_TP_PAGE_DUMMY=3000" ;;
    e3) P="STRATA_TP_PAGE_KEEP=1 STRATA_TP_PAGE_ON0=1" ;;

  esac
  for i in 1 2 3; do
    N=t16_${c}$i
    env $B $T $P RUN_TIMEOUT=1800 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --ctx 8192 --prompt-batch 2048 --route-bias 0 --route-bias-file 0 --dump-logits logs/$N.logits > /dev/null 2>&1
    echo "[$N] $(grep -ah PPL logs/$N.out | tail -1 | cut -c1-40) $(grep -ah "^rc=" logs/$N.err) $(row $N)"
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP16 done $(date +%T)"
