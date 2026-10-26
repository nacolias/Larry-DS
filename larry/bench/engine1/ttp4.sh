#!/bin/bash
# ttp4.sh - engine1 phase C, first light in the engine: one-shot essay decode, today's path vs STRATA_TP=4 (and the
# DSpark verify passes), service env + bias 0.5 / 0.3, clocks pinned; the service held down
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
RB="--route-bias 0.5 --route-bias-file 0.3"
NP=${NP:-160}
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n $NP $RB > /dev/null 2>&1
  echo "== $N [$*] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)" | sed "s|STRATA_DSPARK=[^ ]* STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K|DSpark|"; }
for c in ${CASES:-off tp}; do
  case $c in
    off) run tp4_off ;;
    tp) run tp4_tp $T ;;
    doff) run tp4_doff $D ;;
    dtp) run tp4_dtp $D $T ;;
  esac
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP4 done $(date +%T)"
