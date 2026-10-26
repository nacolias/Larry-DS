#!/bin/bash
# tr15.sh - engine1's draft, step 2: rebuild (the draft's weights load right after the TP core, before the expert tier
# fills the cards; STRATA_TP_DRAFT_STAMP phase stamps), then where the TP draft's time goes: the essay with the phase
# stamps, promotions on / off (STRATA_XE_PROMOTE=0), and the ggml draft with promotions off.  Service held, clocks pinned.
cd /opt/ai/dsv41/bench/engine1
TARGETS="${TR15_TARGETS:-ds41_tp ds41_generate_xe}" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
A="--ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in ${TR15_CASES:-st nopc_st ggml_nopc}; do
  case $c in
    st) X="STRATA_TP_DRAFT=1 STRATA_TP_DRAFT_STAMP=1" ;;
    nopc_st) X="STRATA_TP_DRAFT=1 STRATA_TP_DRAFT_STAMP=1 STRATA_XE_PROMOTE=0" ;;
    ggml_nopc) X="STRATA_TP_DRAFT=0 STRATA_XE_PROMOTE=0" ;;
    tp) X="STRATA_TP_DRAFT=1" ;;
  esac
  N=tr15_$c
  env $B $D $T $REP $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file ${TR15_F:-pe.i32} -n 1000 $A > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "^tp draft:" | cut -c1-240
  tr "\r" "\n" < logs/$N.err | grep -a "tp draft stamps" | tail -4
  tr "\r" "\n" < logs/$N.err | grep -a "seeded:" | head -2 | cut -c1-200
done
python3 -I bench/engine1/cyc.py logs/tr13_essay.cpu logs/tr14_essay.cpu logs/tr15_st.cpu logs/tr15_nopc_st.cpu logs/tr15_ggml_nopc.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
dmesg -T | grep -c "CAT error"
echo "TR15 done $(date +%T)"
