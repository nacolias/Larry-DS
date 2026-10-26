#!/bin/bash
# ttl2.sh - where a decode layer's time goes now: STRATA_GPU_TL (device timestamps of the layer's stages) + the tier's
# host profile, the whole fusion round on vs off (plain decode, service env, bias 0.5, clocks pinned)
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
OFF="STRATA_ATTN_FUSE=0 STRATA_DIRECT_OUT=0 STRATA_HC_MERGE=0 STRATA_FIN_DEFER=0"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; shift; env $B "$@" STRATA_GPU_TL=1 STRATA_XE_PROF=1 RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-110)"
  grep -h "xe decode prof" logs/$N.err | tail -1
  grep -h "gpu timeline" logs/$N.err | tail -2 | cut -c1-600; }
run tl_all
run tl_off $OFF
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTL2 done $(date +%T)"
