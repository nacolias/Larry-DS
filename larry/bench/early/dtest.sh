#!/bin/bash
# dtest.sh - determinism bisect: each config run twice, tokens compared
cd /opt/ai/dsv41
OLD="STRATA_XE_PLANAR=0 STRATA_XE_SYNC=1 STRATA_SHEXP_INLINE=1 GGML_SYCL_DSV4_DEC=0 GGML_SYCL_LARRY_Q6R8=0 GGML_SYCL_GRAPH_CACHE=0 STRATA_XE_GRAPH=0"
pair() { N=$1; shift
  for k in a b; do env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N$k --ids-file p1.i32 -n 128; done
  cmp -s logs/${N}a.out logs/${N}b.out && echo "$N: deterministic ($*)" || echo "$N: DIFFERS ($*) at $(cmp logs/${N}a.out logs/${N}b.out | head -1)"; }
pair old $OLD
pair planar $OLD STRATA_XE_PLANAR=1
pair async $OLD STRATA_XE_PLANAR=1 STRATA_XE_SYNC=0
source xe-env.sh
dsp() { N=$1; shift; env "$@" STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 128
  echo "== $N ($*)"; grep -h "^decode:\|DSpark: [0-9]\|pass_v split" logs/$N.err | tail -3; }
dsp d0c GGML_SYCL_GRAPH_CACHE=0 STRATA_XE_GRAPH=0
dsp d1c GGML_SYCL_GRAPH_CACHE=1 STRATA_XE_GRAPH=1
