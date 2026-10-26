#!/bin/bash
# tlong.sh - engine1 at LONG context: the essay request behind a 95K-token prefix (p95ke), 16K prompt batches (the
# service's), DSpark x200; today's path vs STRATA_TP=4, both with the 3 GiB promotion region (does the prompt path still
# fit?), clocks pinned
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=${FLEX:-3} STRATA_XE_FLEX_MARGIN_GIB=0.3"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
RB="--route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; shift; env $B $D "$@" RUN_TIMEOUT=1800 ./run.sh $N --ids-file p95ke.i32 -n 200 --ctx 131072 --prompt-batch 16384 $RB > /dev/null 2>&1
  echo "== $N $(grep -h "^prompt:" logs/$N.err | cut -c1-60) | $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
  grep -h "refused\|OUT_OF\|error\|failed" logs/$N.err | head -3
  grep -v "^#" logs/$N.out | tr "\n" " " | /opt/ai/dsv41/venv/bin/python /opt/ai/dsv41/tok.py dec - 2>/dev/null | head -c 300; echo; }
for c in ${CASES:-off tp}; do
  case $c in
    off) run lg_off ;;
    tp) run lg_tp $T ;;
  esac
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TLONG done $(date +%T)"
