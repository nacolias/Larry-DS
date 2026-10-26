#!/bin/bash
# ttp3.sh - where the attention output split's time goes in the engine (STRATA_GPU_TL stamps 5/6/7 on the dense card,
# 20/21 around each expert card's part): plain decode, split + graph replay / split eager
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=4.2 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
run() { N=$1; shift; env $B STRATA_DEC_PROF=1 STRATA_GPU_TL=1 "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 300 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s")"
  grep -h "dec prof" logs/$N.err | tail -1
  grep -h "gpu timeline" logs/$N.err | tail -1 | tr ";" "\n" | grep -v "promotion"; }
run tr2g1 STRATA_TP_OPROJ=2 GGML_SYCL_GRAPH_CACHE=1
run tr2g0 STRATA_TP_OPROJ=2 GGML_SYCL_GRAPH_CACHE=0
run tr2g1n STRATA_TP_OPROJ=2 GGML_SYCL_GRAPH_CACHE=1 STRATA_TP_GRAPH=0
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
