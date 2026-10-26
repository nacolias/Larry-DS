#!/bin/bash
# tr9.sh - engine1 replicas, step 2 (in the engine): rebuild libds41_xe (RtEnt 40 bytes; STRATA_XE_REPLICA_GIB: copies of
# the expert cards' hottest experts on the dense card) + libds41_tp (route_jobs balancing replicated jobs), then the
# DSpark texts at the service's settings, 1000 tokens: essay without / with 2 GiB of replicas (and once with every
# layer's stamps), code without / with.  Service held; clocks pinned.
cd /opt/ai/dsv41/bench/engine1
# the build first: build-xe.sh sources oneAPI itself (a second source returns 3 under its set -e)
TARGETS="ds41_xe ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in ${TR9_CASES:-essay_r0 essay_r2 essay_r2st code_r0 code_r2}; do
  case $c in essay*) F=pe.i32 ;; code*) F=pcode.i32 ;; math*) F=pmath.i32 ;; esac
  case $c in
    *_r0*) X="STRATA_XE_REPLICA_GIB=0" ;;
    *_r2st*) X="STRATA_XE_REPLICA_GIB=2 STRATA_TP_STAMP_LAYER=-1" ;;
    *_r2*) X="STRATA_XE_REPLICA_GIB=2" ;;
    *_r3*) X="STRATA_XE_REPLICA_GIB=3" ;;
  esac
  N=tr9_$c
  env $B $D $T $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n 1000 --ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "replicas:\|tp prof" | tail -3 | cut -c1-230
  tr "\r" "\n" < logs/$N.err | grep -a "tp stamps all layers" | tail -1
  head -c 300 logs/$N.out | tr "\n" " "; echo
done
python3 -I bench/engine1/cyc.py logs/tr9_essay_r0.cpu logs/tr9_essay_r2.cpu logs/tr5_new.cpu logs/tr5_new_2.cpu
python3 -I bench/engine1/cyc.py logs/tr9_code_r0.cpu logs/tr9_code_r2.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TR9 done $(date +%T)"
