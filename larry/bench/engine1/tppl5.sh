#!/bin/bash
# tppl5.sh - engine1 numerics of the tuned mat-vec forms: decode-path PPL + logits of the 2K text (p2k) in 5-token
# passes (--prefill-chunk 5: TP passes of width 5 take the XMX forms the tuner picks there), exact routing; fastmv's
# dispatch twice (the run-to-run floor) vs STRATA_TP_MV_TUNE=1 twice; KL between them.  Both with route_jobs a lane a
# pick and the expert kernels' job bound / grid (bit-identical).  Service held.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROMPT=1 STRATA_TP_ROUTE_PAR=1 STRATA_TP_NJ_DEV=1 STRATA_TP_JOB_GRID=8"
RX="--route-bias 0 --route-bias-file 0"
q() { N=$1; shift; env $B $T "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 5 $RX --dump-logits logs/$N.logits > /dev/null 2>&1
  echo "[$N] $(grep -h PPL logs/$N.out | tail -1) $(grep -h "^rc=" logs/$N.err)"; }
kl() { cmp -s logs/$1.logits logs/$2.logits && echo "$1 == $2 (bit-identical)" || echo "KL $1 || $2: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/$1.logits logs/$2.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-300)"; }
for c in ${CASES:-base1 tune1 base2 tune2}; do
  case $c in
    base*) q p5_$c STRATA_XE_PROMOTE=0 STRATA_TP_MV_TUNE=0 ;;
    tune*) q p5_$c STRATA_XE_PROMOTE=0 STRATA_TP_MV_TUNE=1 ;;
  esac
done
grep -h "us a launch" logs/p5_tune1.err | cut -c1-160
kl p5_base1 p5_base2
kl p5_tune1 p5_tune2
kl p5_base1 p5_tune1
kl p5_base2 p5_tune2
echo "TPPL5 done $(date +%T)"
