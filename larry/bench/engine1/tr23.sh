#!/bin/bash
# tr23.sh - rebuild libds41_tp (STRATA_TP_DRAFT_MK_CHECK), then the essay with the check: the old Markov chain and the
# fused one in the same draft graph on the same inputs, their tokens compared every draft.  300 tokens, the host
# placement, the service's settings.  Then the service's sequence on a private port with the placement (tpg.sh).
cd /opt/ai/dsv41/bench/engine1
TARGETS=ds41_tp ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_DRAFT=1"
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
H="STRATA_HOST_CPUS=0 STRATA_XE_POOL_WORKERS=6 STRATA_XE_HELPER_CPUS=7,15"
A="--ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3"
N=tr23_mkchk
env $B $D $T $REP $H STRATA_TP_DRAFT_MK_CHECK=1 RUN_TIMEOUT=1200 ./run.sh $N --ids-file pe.i32 -n 300 $A > /dev/null 2>&1
echo "== $N $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|draft [0-9.]* ms/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
tr "\r" "\n" < logs/$N.err | grep -a "tp draft mk check" | tail -2
dmesg -T | grep -c "CAT error"
TPG_ROUNDS=1 bench/engine1/tpg.sh pin $H 2>&1
echo "TR23 done $(date +%T)"
