#!/bin/bash
# tgraph.sh - plain decode, kept graphs replayed as SYCL graphs (GGML_SYCL_GRAPH_CACHE=1) vs eager: host phases
# (STRATA_DEC_PROF), speed, the text
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
run() { N=$1; shift; env $B STRATA_DEC_PROF=1 "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 300 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s")"
  grep -h "dec prof" logs/$N.err | tail -1
  grep -h "graph cache\|not recordable" logs/$N.err | tail -3
  python3 tok.py dec logs/$N.out 2>/dev/null | head -c 300; echo; }
run gc0
run gc1 GGML_SYCL_GRAPH_CACHE=1 GGML_SYCL_GRAPH_STATS=1
run gc0b
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
