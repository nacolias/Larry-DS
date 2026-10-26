#!/bin/bash
# tr3.sh - engine1 verify kernels, round 4: rebuild libds41_tp + tp_bench (per-shape mat-vec forms, STRATA_TP_MV_TUNE),
# then tp_bench on 4 cards, widths 1-5, forms tuned vs fastmv's dispatch (both with route_jobs a lane a pick): the
# first pass's logits compared (--dump / --cmp) and the graph pass times.  Service held; clocks pinned.
cd /opt/ai/dsv41/bench/engine1
# the build first: build-xe.sh sources oneAPI itself (a second source returns 3 under its set -e)
TARGETS="ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
source /opt/ai/dsv41/xe-env.sh
export STRATA_XE_JOBNT=1 STRATA_TP_ROUTE_PAR=1
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
mkdir -p prof6
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for n in ${WS:-1 2 3 4 5}; do
  STRATA_TP_MV_TUNE=0 $B -m $M -N 4 --pos 1000 -n $n --iters 40 --graph 1 --cache $C --experts 48 --dump prof6/ref_n$n.bin > prof6/g0_n$n.out 2>&1
  echo "== n $n dispatch: rc=$? $(grep -h "pass median" prof6/g0_n$n.out)"
  STRATA_TP_MV_TUNE=1 $B -m $M -N 4 --pos 1000 -n $n --iters 40 --graph 1 --cache $C --experts 48 --cmp prof6/ref_n$n.bin > prof6/g1_n$n.out 2>&1
  echo "== n $n tuned:    rc=$? $(grep -h "pass median" prof6/g1_n$n.out)"
  grep -h "^row " prof6/g1_n$n.out
done
grep -h "^tp: .*us a launch\|forms tuned" prof6/g1_n1.out
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TR3 done $(date +%T)"
