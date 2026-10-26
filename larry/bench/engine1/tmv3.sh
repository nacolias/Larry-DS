#!/bin/bash
# tmv3.sh - engine1's verify-pass kernels, round 1: (a) bench/fm3/tpmv_bench on card 0 (the TP core's Q6_K mat-vec
# shapes at 1-6 columns: dp4a variants, XMX, the streaming floor); (b) tp_bench graph passes on 4 cards, widths 1-5,
# the all-reduce by copies (today) vs STRATA_TP_PUSH=1 (P2P stores by kernels).  Service held; clocks pinned.
cd /opt/ai/dsv41/bench/engine1
source /opt/ai/dsv41/xe-env.sh
export STRATA_XE_JOBNT=1
[ -f /run/lock/larry-profile-a.hold ] || { echo "no hold file"; exit 1; }
systemctl is-active -q larry-ds41 && { echo "the service is running"; exit 1; }
pgrep -x ds41_generate_x > /dev/null && { echo "an engine is running"; exit 1; }
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
mkdir -p prof3
/usr/local/sbin/larry-gpu-clock pin > /dev/null
ONEAPI_DEVICE_SELECTOR=level_zero:0 ../fm3/tpmv_bench ${REPLAYS:-20} > prof3/tpmv.out 2>&1
echo "tpmv rc=$?"
for push in 0 1; do
  for n in ${WS:-1 2 3 4 5}; do
    STRATA_TP_PUSH=$push $B -m $M -N 4 --pos 1000 -n $n --iters 40 --graph 1 --cache $C --experts 48 > prof3/g_p${push}_n$n.out 2>&1
    echo "push $push n $n: rc=$? $(grep -h "pass median" prof3/g_p${push}_n$n.out)"
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TMV3 done $(date +%T)"
