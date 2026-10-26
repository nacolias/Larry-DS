#!/bin/bash
# tctx.sh - the same essay request with and without a 95K-token prefix (p30k x3 + pe.i32 vs pe.i32), DSpark,
# GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1, armed cards on, no tracer: what long context still costs per pass
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
run() { N=$1; shift; env $B $D RUN_TIMEOUT=1800 ./run.sh $N "$@" > /dev/null 2>&1
  echo "== $N: $(grep -h "^decode:" logs/$N.err | grep -o "[0-9]* tokens in [0-9.]* s = [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass, draft [0-9.]* ms/pass" | tr "\n" " ")"; }
for i in 1 2; do
  run cx95_$i --ids-file p95ke.i32 -n 400 --ctx 131072 --prompt-batch 16384
  run cxs_$i --ids-file pe.i32 -n 400
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
