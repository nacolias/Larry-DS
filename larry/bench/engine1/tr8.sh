#!/bin/bash
# tr8.sh - engine1 replicas, step 1 (the TP core alone): rebuild libds41_tp + tp_bench (TpExp 40 bytes with a second
# holder, route_jobs balancing replicated jobs); then tp_bench on 4 cards:
#   (a) numerics, -L 2 / -L 40, n 5: the balancing route_jobs without replicas vs the one-work-item form (must be
#       bit-identical); with STRATA_TP_BENCH_REPLICAS=2 vs without (only the sums' association moves);
#   (b) n 3 / 5: pass medians + layer-20 stamps (experts and xB-wait by card) for replicas off / every 4th / every 2nd.
# Service held; clocks pinned.
cd /opt/ai/dsv41/bench/engine1
# the build first: build-xe.sh sources oneAPI itself (a second source returns 3 under its set -e)
TARGETS="ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
source /opt/ai/dsv41/xe-env.sh
export STRATA_XE_JOBNT=1
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
mkdir -p prof10
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for L in 2 40; do
  STRATA_TP_ROUTE_PAR=0 $B -m $M -N 4 -L $L --pos 1000 -n 5 --iters 3 --graph 1 --cache $C --experts 48 --dump prof10/serL$L.bin > /dev/null 2>&1
  echo "== L $L balancing route_jobs (no replicas) vs serial: $($B -m $M -N 4 -L $L --pos 1000 -n 5 --iters 3 --graph 1 --cache $C --experts 48 --cmp prof10/serL$L.bin 2>&1 | grep '^row' | awk '{print $5}' | tr '\n' ' ')"
  echo "== L $L replicas every 2nd vs serial: $(STRATA_TP_BENCH_REPLICAS=2 $B -m $M -N 4 -L $L --pos 1000 -n 5 --iters 3 --graph 1 --cache $C --experts 48 --cmp prof10/serL$L.bin 2>&1 | grep '^row' | awk '{print $5}' | tr '\n' ' ')"
done
for n in 3 5; do
  for r in 0 4 2; do
    STRATA_TP_BENCH_REPLICAS=$r STRATA_TP_STAMP_LAYER=20 $B -m $M -N 4 --pos 1000 -n $n --iters 130 --graph 1 --cache $C --experts 48 > prof10/st_r${r}_n$n.out 2>&1
    echo "== n $n replicas 1/$r: $(grep -h "pass median" prof10/st_r${r}_n$n.out | grep -o 'median [0-9.]* us') | experts by card: $(grep -h "tp stamps" prof10/st_r${r}_n$n.out | tail -4 | grep -o 'experts [0-9.]*' | awk '{print $2}' | tr '\n' ' ') | xB-wait: $(grep -h "tp stamps" prof10/st_r${r}_n$n.out | tail -4 | grep -o 'xB-wait [0-9.]*' | awk '{print $2}' | tr '\n' ' ')"
  done
  for r in 0 4 2; do   # without stamps: the plain pass time
    echo "   n $n replicas 1/$r, no stamps: $(STRATA_TP_BENCH_REPLICAS=$r $B -m $M -N 4 --pos 1000 -n $n --iters 40 --graph 1 --cache $C --experts 48 2>&1 | grep -o 'median [0-9.]* us')"
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TR8 done $(date +%T)"
