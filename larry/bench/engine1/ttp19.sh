#!/bin/bash
# ttp19.sh - the pager's refill check with the fix: every unit refilled through card 0's copy queue and compared
# with the load, bit for bit (STRATA_TP_PAGE_CHECK=1), at the service config; a short DSpark run on top
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PAGE_CHECK=1"
env $B $D $T RUN_TIMEOUT=900 ./run.sh t19_chk --ids-file pe.i32 -n 32 --ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
grep -a -E "tp pager|check|rc=" logs/t19_chk.err | cut -c1-200 | tail -8
echo "TTP19 done $(date +%T)"
