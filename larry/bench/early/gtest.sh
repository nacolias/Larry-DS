#!/bin/bash
# gtest.sh - greedy decode with the graph caches off / on (tokens must match bit for bit) + DSpark verify passes
cd /opt/ai/dsv41
source xe-env.sh
run() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DEC_PROF=1 STRATA_XE_PROF=1 ./run.sh $N --ids-file p1.i32 -n 128
  echo "== $N ($*)"; grep -h "decode step\|^decode:\|not recordable\|graph.*eager\|DSpark:\|pass_v split\|GGML_ASSERT\|rc=" logs/$N.err | tail -5; }
run g0 GGML_SYCL_GRAPH_CACHE=0 STRATA_XE_GRAPH=0
run g1 GGML_SYCL_GRAPH_CACHE=1 STRATA_XE_GRAPH=0
run g2 GGML_SYCL_GRAPH_CACHE=1 STRATA_XE_GRAPH=1
for n in g1 g2; do cmp -s logs/g0.out logs/$n.out && echo "$n tokens == g0" || echo "$n tokens DIFFER from g0"; done
venv/bin/python tok.py dec - < logs/g2.out | head -c 300; echo
run d0 STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 GGML_SYCL_GRAPH_CACHE=0 STRATA_XE_GRAPH=0
run d1 STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 GGML_SYCL_GRAPH_CACHE=1 STRATA_XE_GRAPH=1
cmp -s logs/d0.out logs/d1.out && echo "d1 tokens == d0" || echo "d1 tokens DIFFER from d0"
