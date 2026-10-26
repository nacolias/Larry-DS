#!/bin/bash
# tppl.sh - engine1 numerics on the real model: decode-path PPL + logits of the 2K text (p2k), every token its own pass
# (--prefill-chunk 1), exact routing; today's path twice (its run-to-run floor) vs STRATA_TP=4 twice; KL between them
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROMPT=1"
RX="--route-bias 0 --route-bias-file 0"
q() { N=$1; shift; env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 $RX --dump-logits logs/$N.logits > /dev/null 2>&1
  echo "[$N] $(grep -h PPL logs/$N.out | tail -1) $(grep -h "^rc=" logs/$N.err)"; }
kl() { cmp -s logs/$1.logits logs/$2.logits && echo "$1 == $2 (bit-identical)" || echo "KL $1 || $2: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/$1.logits logs/$2.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-300)"; }
for c in ${CASES:-off1 tp1 off2 tp2}; do
  case $c in
    off*) q ppl_$c STRATA_XE_PROMOTE=0 ;;
    tp*) q ppl_$c $T STRATA_XE_PROMOTE=0 ;;
  esac
done
kl ppl_off1 ppl_off2
kl ppl_tp1 ppl_tp2
kl ppl_off1 ppl_tp1
kl ppl_off2 ppl_tp2
echo "TPPL done $(date +%T)"
