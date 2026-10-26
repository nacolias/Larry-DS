#!/bin/bash
# tr4.sh - engine1 verify kernels, round 5: the expert kernels' engine1 options.  Rebuild libds41_xe + libds41_tp +
# tp_bench; then on tp_bench (4 cards, route_jobs a lane a pick, fastmv's dispatch):
#   (a) numerics, -L 2 and -L 40, n 5: each option vs the base (NJ_DEV / JOB_GRID must be bit-identical);
#   (b) layer-20 stamps + pass medians, n 3 and 5: base | NJ_DEV | NJ_DEV + JOB_GRID 8 / 12 | + DOWN_SPLIT.
# Service held; clocks pinned.
cd /opt/ai/dsv41/bench/engine1
# the build first: build-xe.sh sources oneAPI itself (a second source returns 3 under its set -e)
TARGETS="ds41_xe ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
source /opt/ai/dsv41/xe-env.sh
export STRATA_XE_JOBNT=1 STRATA_TP_ROUTE_PAR=1
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
mkdir -p prof7
/usr/local/sbin/larry-gpu-clock pin > /dev/null
opt() {
  case $1 in
    base) echo "" ;;
    nj) echo "STRATA_TP_NJ_DEV=1" ;;
    g8) echo "STRATA_TP_NJ_DEV=1 STRATA_TP_JOB_GRID=8" ;;
    g12) echo "STRATA_TP_NJ_DEV=1 STRATA_TP_JOB_GRID=12" ;;
    split) echo "STRATA_TP_NJ_DEV=1 STRATA_XE_DOWN_SPLIT=1" ;;
    g8split) echo "STRATA_TP_NJ_DEV=1 STRATA_TP_JOB_GRID=8 STRATA_XE_DOWN_SPLIT=1" ;;
  esac
}
for L in 2 40; do
  $B -m $M -N 4 -L $L --pos 1000 -n 5 --iters 3 --graph 1 --cache $C --experts 48 --dump prof7/refL$L.bin > /dev/null 2>&1
  for o in nj g8 split g8split; do
    echo "== L $L $o vs base: $(env $(opt $o) $B -m $M -N 4 -L $L --pos 1000 -n 5 --iters 3 --graph 1 --cache $C --experts 48 --cmp prof7/refL$L.bin 2>&1 | grep '^row' | awk '{print $5}' | tr '\n' ' ')"
  done
done
for n in 3 5; do
  for o in base nj g8 g12 split g8split; do
    env $(opt $o) STRATA_TP_STAMP_LAYER=20 $B -m $M -N 4 --pos 1000 -n $n --iters 130 --graph 1 --cache $C --experts 48 > prof7/st_${o}_n$n.out 2>&1
    echo "== n $n $o: $(grep -h "pass median" prof7/st_${o}_n$n.out | grep -o 'median [0-9.]* us') | experts by card: $(grep -h "tp stamps" prof7/st_${o}_n$n.out | tail -4 | grep -o 'experts [0-9.]*' | awk '{print $2}' | tr '\n' ' ') | xB-wait: $(grep -h "tp stamps" prof7/st_${o}_n$n.out | tail -4 | grep -o 'xB-wait [0-9.]*' | awk '{print $2}' | tr '\n' ' ')"
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TR4 done $(date +%T)"
