#!/bin/bash
# tgppl.sh - decode-path perplexity (2048 tokens one a step) with the kept graphs eager vs replayed as SYCL graphs
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
for cfg in "GGML_SYCL_GRAPH_CACHE=0" "GGML_SYCL_GRAPH_CACHE=1 GGML_SYCL_GRAPH_STATS=1" "GGML_SYCL_GRAPH_CACHE=0"; do
  N=gp$(echo $cfg | cut -c23)$RANDOM
  env $B $cfg RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 > /dev/null 2>&1
  echo "[$cfg] $(grep -h PPL logs/$N.out) | $(grep -h "^prompt:" logs/$N.err | cut -c1-60) | $(grep -h "graph cache" logs/$N.err | tail -1)"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
