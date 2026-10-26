#!/bin/bash
# tbound.sh - upper bounds of the post-readback levers (timing only: the debug knobs make the output WRONG):
#   noshexp  STRATA_DEBUG_NO_SHEXP=1      the shared expert left out of card 0's work
#   nodense  STRATA_XE_DEBUG_SKIP_DENSE=1 card 0's own expert jobs left out
#   nocpu    STRATA_XE_CPU_SKIP=1         the CPU-pool experts left out
# plain decode (essay -n 400) and DSpark verify at a fixed width (STRATA_DSPARK_CONF=0: every pass depth + 1 = 5 rows,
# so ms/pass compares even when the draft's acceptance changes); service env, bias 0.5 / 0.3, clocks pinned.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0 STRATA_DSPARK_DENSE_TYPE=q6_K"
RB="--route-bias 0.5 --route-bias-file 0.3"
declare -A K=([base]="" [noshexp]="STRATA_DEBUG_NO_SHEXP=1" [nodense]="STRATA_XE_DEBUG_SKIP_DENSE=1" [nocpu]="STRATA_XE_CPU_SKIP=1"
              [card0free]="STRATA_DEBUG_NO_SHEXP=1 STRATA_XE_DEBUG_SKIP_DENSE=1" [floor]="STRATA_DEBUG_NO_SHEXP=1 STRATA_XE_DEBUG_SKIP_DENSE=1 STRATA_XE_CPU_SKIP=1")
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; S=$2; shift 2; env $B ${K[$S]} "$@" STRATA_GPU_TL=1 RUN_TIMEOUT=1500 ./run.sh $N --ids-file pe.i32 -n 400 $RB > /dev/null 2>&1
  echo "== $N [$S] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
  grep -h "gpu timeline" logs/$N.err | tail -1 | grep -o "dense us:.*sum submitted [0-9.]* us" | cut -c1-260; }
for s in base noshexp nodense nocpu card0free floor; do run bp_$s $s; done
for s in base noshexp nodense card0free floor; do run bv_$s $s $D; done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TBOUND done $(date +%T)"
