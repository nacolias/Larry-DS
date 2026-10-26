#!/bin/bash
# tcpu1.sh - engine1: where the CPU-pool experts of TP passes come from (STRATA_XE_CPU_LOG: arena pinned / pageable,
# repeats, promotions) and how fast the pool runs them cold vs with their weights hot in the caches
# (STRATA_XE_CPU_TWICE): the DSpark essay at the service's settings, clocks pinned
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in ${TCPU_CASES:-log twice}; do
  X="STRATA_XE_CPU_LOG=logs/tcpu1_$c.cpu"
  [ $c = twice ] && X="$X STRATA_XE_CPU_TWICE=1"
  env $B $D $T $X RUN_TIMEOUT=1800 ./run.sh tcpu1_$c --ids-file pe.i32 -n 400 --ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
  echo "== tcpu1_$c $(grep -ah "^decode:" logs/tcpu1_$c.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /\|cpu [0-9]* ms" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/tcpu1_$c.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/tcpu1_$c.err)"
  tr "\r" "\n" < logs/tcpu1_$c.err | grep -a "tp prof" | tail -3 | cut -c1-260
  tr "\r" "\n" < logs/tcpu1_$c.err | grep -a "promotion cache:\|dense card 0 seeded\|arena .* pinned" | tail -4 | cut -c1-200
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TCPU1 done $(date +%T)"
