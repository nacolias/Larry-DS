#!/bin/bash
# ttp3.sh - engine1 phase A.2: the whole stack WITH routed experts (48 real blobs, every routed expert resident),
# N = 1..4, decode (n 1) and verify width (n 4), graph replay; clocks pinned; the service held down
cd /opt/ai/dsv41/bench/engine1
source /opt/ai/dsv41/xe-env.sh
export STRATA_XE_JOBNT=1
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
if systemctl is-active -q larry-ds41; then echo "service is running - refusing"; exit 1; fi
/usr/local/sbin/larry-gpu-clock pin > /dev/null
echo "start $(date +%T)"
POS=${POS:-8192}
for N in ${NS:-1 2 3 4}; do
  for n in ${WS:-1 4}; do
    $B -m $M -N $N --ctx 65536 --pos $POS -n $n --iters 30 --graph 1 --cache $C --experts 48 2>&1 | grep -v "^tp: rank\|^tp: first copy"
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "done $(date +%T)"
