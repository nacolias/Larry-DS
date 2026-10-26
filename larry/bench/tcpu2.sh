#!/bin/bash
# tcpu2.sh - CPU-pool experts: SMT (--threads 15: 7 pinned workers + 8 unpinned, the cores' siblings) and the CPU part
# on the main thread (STRATA_XE_CPU_ASYNC=0); per-expert time from the pool's timers.  Service env, clocks pinned.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
RB="--route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; T=$2; shift 2; env $B "$@" STRATA_XE_PROF=1 RUN_TIMEOUT=1500 ./run.sh $N --ids-file pe.i32 -n 400 $RB $T > /dev/null 2>&1
  echo "== $N [$T $*] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /\|cpu [0-9]* ms" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)" | sed "s|STRATA_DSPARK=[^ ]* STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K|DSpark|"
  grep -h "xe cpu pool" logs/$N.err | tail -1 | grep -o "[0-9]* us an expert (cpu_ents wall).*workers"; }
run ct_t15 "--threads 15"
run ct_t15s "--threads 15" STRATA_XE_CPU_ASYNC=0
run ct_t7s "" STRATA_XE_CPU_ASYNC=0
run ct_t11s "--threads 11" STRATA_XE_CPU_ASYNC=0
run ctd_t7 "" $D
run ctd_t7s "" $D STRATA_XE_CPU_ASYNC=0
run ctd_t15s "--threads 15" $D STRATA_XE_CPU_ASYNC=0
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TCPU2 done $(date +%T)"
