#!/bin/bash
# tr2.sh - engine1 verify kernels, round 3: rebuild libds41_tp + tp_bench (STRATA_TP_STAMP_LAYER: in-graph clock stamps),
# then (a) tp_bench graph passes, widths 1-5, stamps on layer 20 (route_jobs a lane a pick); (b) the DSpark essay at
# the service's exact settings (code defaults: 64 promotion chunks a pass, not the A/B's 256) - its cycle vs tcpu5;
# (c) the same with the new route_jobs and the stamps on layer 20.  Service held; clocks pinned.
cd /opt/ai/dsv41/bench/engine1
# the build first: build-xe.sh sources oneAPI itself (a second source returns 3 under its set -e)
TARGETS="ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
source xe-env.sh
BT=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
mkdir -p bench/engine1/prof5
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for n in ${WS:-1 2 3 4 5}; do
  STRATA_XE_JOBNT=1 STRATA_TP_ROUTE_PAR=1 STRATA_TP_STAMP_LAYER=20 $BT -m $M -N 4 --pos 1000 -n $n --iters 130 --graph 1 --cache /srv/dsv41/tpcache --experts 48 > bench/engine1/prof5/g_n$n.out 2>&1
  echo "== bench n $n: rc=$? $(grep -h "pass median" bench/engine1/prof5/g_n$n.out)"
  grep -h "tp stamps" bench/engine1/prof5/g_n$n.out | tail -4
done
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
for c in ${TR2_CASES:-svc stamp}; do
  case $c in
    svc*) X="" ;;
    stamp*) X="STRATA_TP_ROUTE_PAR=1 STRATA_TP_STAMP_LAYER=20" ;;
  esac
  N=tr2_$c
  env $B $D $T $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file pe.i32 -n 1000 --ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /\|cpu [0-9]* ms" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "tp prof" | tail -2 | cut -c1-230
  tr "\r" "\n" < logs/$N.err | grep -a "tp stamps" | tail -4
done
python3 -I bench/engine1/cyc.py logs/tcpu5_early2c.cpu logs/tcpu5_early2c_2.cpu logs/tr2_svc.cpu logs/tr2_stamp.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TR2 done $(date +%T)"
