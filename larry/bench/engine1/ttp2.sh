#!/bin/bash
# ttp2.sh - engine1 phase A: the whole 40-layer stack on N = 1..4 cards, decode (n 1) and verify width (n 4),
# graph replay vs eager; clocks pinned; the service held down
cd /opt/ai/dsv41/bench/engine1
source /opt/ai/dsv41/xe-env.sh
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
if systemctl is-active -q larry-ds41; then echo "service is running - refusing"; exit 1; fi
/usr/local/sbin/larry-gpu-clock pin > /dev/null
echo "start $(date +%T)"
POS=${POS:-8192}
for N in 1 2 3 4; do
  for n in 1 4; do
    cmpf=""; [ $N -gt 1 ] && cmpf="--cmp full_n${n}.bin"
    dmp=""; [ $N -eq 1 ] && dmp="--dump full_n${n}.bin"
    for g in 1 0; do
      $B -m $M -N $N --ctx 65536 --pos $POS -n $n --iters 30 --graph $g --cache $C $dmp $cmpf 2>&1 | grep -v "^tp: rank\|^tp: first copy"
      dmp=""
    done
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "done $(date +%T)"
