#!/bin/bash
# ttp5.sh - engine1: why the TP path runs more CPU-pool experts - (a) today's path with TP's VRAM footprint taken off
# the expert budgets, (b) TP with more promotion; DSpark essay x400, service env + bias 0.5 / 0.3, clocks pinned
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
RB="--route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; shift; env $B $D "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 $RB > /dev/null 2>&1
  echo "== $N [$*] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /\|cpu [0-9]* ms" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
  grep -h "promotion cache: [0-9]* promoted\|promotion cache: [0-9]* slots" logs/$N.err | tail -2; grep -h "experts in VRAM" logs/$N.err | tail -1; }
for c in ${CASES:-a b}; do
  case $c in
    a) run t5_offlow STRATA_XE_RESERVE_GIB=5.5 STRATA_XE_DENSE_GIB=12 ;;
    b) run t5_tppc $T STRATA_TP_PROMOTE_PAR=8 STRATA_TP_PROMOTE_CHUNKS=64 ;;
    c) run t5_tpres $T STRATA_XE_RESERVE_GIB=1.5 ;;
    d) run t5_tpd18 $T STRATA_XE_DENSE_GIB=18 ;;
    e) run t5_offd18 STRATA_XE_DENSE_GIB=18 ;;
    f) run t5_tp $T ;;
    g) run t5_tpc64 $T STRATA_TP_PROMOTE_CHUNKS=64 STRATA_TP_PROMOTE_PAR=8 ;;
    h) run t5_tpc128 $T STRATA_TP_PROMOTE_CHUNKS=128 STRATA_TP_PROMOTE_PAR=8 ;;
    i) run t5_tpflex $T STRATA_XE_FLEX_GIB=3 STRATA_XE_FLEX_MARGIN_GIB=0.3 ;;
    j) run t5_tpflex32 $T STRATA_TP_PROMOTE_CHUNKS=32 STRATA_TP_PROMOTE_PAR=8 STRATA_XE_FLEX_GIB=3 STRATA_XE_FLEX_MARGIN_GIB=0.3 ;;
    k) run t5_offflex STRATA_XE_FLEX_GIB=3 STRATA_XE_FLEX_MARGIN_GIB=0.3 ;;
  esac
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP5 done $(date +%T)"
