#!/bin/bash
# ttp8.sh - engine1 at the service's settings (ctx 262144, 16K batches, FLEX 1 / margin 1.5): where the TP side's extra
# ~1.2 ms a DSpark draft comes from (STRATA_DSPARK_PROF: inputs / graph / enqueue / sync / readback) - the promotion
# repack helpers (4 at normal priority under TP: 1 helper, or idle class), and the expert cards' slack (TP leaves
# ~0.7 GiB a card more free than today's path: STRATA_XE_RESERVE_GIB 2.8); essay x400 DSpark, clocks pinned
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5 STRATA_DSPARK_PROF=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
RB="--route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 --ctx 262144 --prompt-batch 16384 $RB > /dev/null 2>&1
  echo "== $N $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
  grep -h "^dspark prof" logs/$N.err | tail -1
  grep -h "card [1-3]:.*VRAM free" logs/$N.err | tr "\n" " "; echo
  grep -h "refused\|OUT_OF\|failed\|no room" logs/$N.err | sort | uniq -c | head -4; }
run dp_off STRATA_XE_RESERVE_GIB=3.5 $D
run dp_tp STRATA_XE_RESERVE_GIB=3.5 $D $T
run dp_tp1 STRATA_XE_RESERVE_GIB=3.5 $D $T STRATA_TP_PC_THREADS=1
run dp_tpi STRATA_XE_RESERVE_GIB=3.5 $D $T STRATA_TP_PC_IDLE=1
run dp_tpr STRATA_XE_RESERVE_GIB=2.8 $D $T
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP8 done $(date +%T)"
