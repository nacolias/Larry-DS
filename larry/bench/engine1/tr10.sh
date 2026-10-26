#!/bin/bash
# tr10.sh - engine1 replicas, step 3: workload-matched replicas (STRATA_XE_REPLICA_USAGE: the set by a usage profile's
# jobs) at a 1.0 GiB margin - the upper bound before an adaptive set.  Essay: without; with its own profile (twice, the
# first byte-checked); with the code profile.  Code with its own.  Service held; clocks pinned.
cd /opt/ai/dsv41/bench/engine1
TARGETS="ds41_xe ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
R="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_MARGIN_GIB=1.0"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in ${TR10_CASES:-essay_r0b essay_ue essay_ue2 essay_uc code_uc}; do
  case $c in essay*) F=pe.i32 ;; code*) F=pcode.i32 ;; math*) F=pmath.i32 ;; esac
  case $c in
    *_r0*) X="STRATA_XE_REPLICA_GIB=0" ;;
    essay_ue) X="$R STRATA_XE_REPLICA_USAGE=logs/use_essay.txt STRATA_XE_REPLICA_CHECK=1" ;;
    *_ue*) X="$R STRATA_XE_REPLICA_USAGE=logs/use_essay.txt" ;;
    *_uc*) X="$R STRATA_XE_REPLICA_USAGE=logs/use_code.txt" ;;
  esac
  N=tr10_$c
  env $B $D $T $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n 1000 --ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "replicas:" | cut -c1-200
  tr "\r" "\n" < logs/$N.err | grep -a "tp prof" | tail -1 | cut -c1-250
done
python3 -I bench/engine1/cyc.py logs/tr9_essay_r0.cpu logs/tr10_essay_r0b.cpu logs/tr9_essay_r2.cpu logs/tr10_essay_ue.cpu logs/tr10_essay_ue2.cpu logs/tr10_essay_uc.cpu
python3 -I bench/engine1/cyc.py logs/tr9_code_r0.cpu logs/tr9_code_r2.cpu logs/tr10_code_uc.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TR10 done $(date +%T)"
