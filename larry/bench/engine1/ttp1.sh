#!/bin/bash
# ttp1.sh - engine1 phase A, first light: tp_bench on a 4-layer prefix, N = 1 then 2 then 4 (eager, then graph),
# clocks pinned; the service must be held down (larry-profile-a.hold) before this runs
cd /opt/ai/dsv41/bench/engine1
source /opt/ai/dsv41/xe-env.sh
B=/opt/ai/dsv41/xolotl/build-xe
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
mkdir -p $C
if systemctl is-active -q larry-ds41; then echo "service is running - refusing"; exit 1; fi
/usr/local/sbin/larry-gpu-clock pin > /dev/null
echo "start $(date +%T)"
L=${L:-4}
export STRATA_TP_SPINSTAT=1
for N in 1 2 4; do
  cmpf=""; [ $N -gt 1 ] && cmpf="--cmp ref_L${L}.bin"
  dmp=""; [ $N -eq 1 ] && dmp="--dump ref_L${L}.bin"
  echo "== N $N eager"; $B/tp_bench -m $M -N $N -L $L --ctx 65536 --pos 8192 -n 1 --iters 20 --graph 0 --cache $C $dmp $cmpf 2>&1 | grep -v "^tp: rank"
  echo "rc=$?"
  echo "== N $N graph"; $B/tp_bench -m $M -N $N -L $L --ctx 65536 --pos 8192 -n 1 --iters 20 --graph 1 --cache $C 2>&1 | grep -v "^tp: rank"
  echo "rc=$?"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "done $(date +%T)"
