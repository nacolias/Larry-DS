#!/bin/bash
# tr3b.sh - round 4b: (a) numerics without 40 layers of amplification: tp_bench -L 2, n 5, dispatch twice (the floor)
# and tuned vs dispatch; (b) the full stack, n 5, stamps on layer 20, dispatch vs tuned.  Service held; clocks pinned.
cd /opt/ai/dsv41/bench/engine1
source /opt/ai/dsv41/xe-env.sh
export STRATA_XE_JOBNT=1 STRATA_TP_ROUTE_PAR=1
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for L in 2 40; do
  STRATA_TP_MV_TUNE=0 $B -m $M -N 4 -L $L --pos 1000 -n 5 --iters 5 --graph 1 --cache $C --experts 48 --dump prof6/refL$L.bin > /dev/null 2>&1
  echo "== L $L dispatch vs dispatch:"; STRATA_TP_MV_TUNE=0 $B -m $M -N 4 -L $L --pos 1000 -n 5 --iters 5 --graph 1 --cache $C --experts 48 --cmp prof6/refL$L.bin 2>&1 | grep "^row"
  echo "== L $L tuned vs dispatch:"; STRATA_TP_MV_TUNE=1 $B -m $M -N 4 -L $L --pos 1000 -n 5 --iters 5 --graph 1 --cache $C --experts 48 --cmp prof6/refL$L.bin 2>&1 | grep "^row"
done
for t in 0 1; do
  STRATA_TP_MV_TUNE=$t STRATA_TP_STAMP_LAYER=20 $B -m $M -N 4 --pos 1000 -n 5 --iters 130 --graph 1 --cache $C --experts 48 > prof6/st$t.out 2>&1
  echo "== stamps tune $t: $(grep -h "pass median" prof6/st$t.out)"; grep -h "tp stamps" prof6/st$t.out | tail -4
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TR3B done $(date +%T)"
