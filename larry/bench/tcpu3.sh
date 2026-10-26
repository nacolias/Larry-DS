#!/bin/bash
# tcpu3.sh - how CPU-pool experts are distributed over calls (decode / verify): a histogram of CPU experts a call
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
RB="--route-bias 0.5 --route-bias-file 0.3"
for m in p d; do
  N=ch_$m; E=""; [ $m = d ] && E="$D"
  env $B $E STRATA_XE_PROF=1 RUN_TIMEOUT=1500 ./run.sh $N --ids-file pe.i32 -n 400 $RB > /dev/null 2>&1
  echo "== $N $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /\|cpu [0-9]* ms" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
  grep -h "cpu experts a call" logs/$N.err | tail -2
done
echo "TCPU3 done $(date +%T)"
