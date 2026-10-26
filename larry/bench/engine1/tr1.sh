#!/bin/bash
# tr1.sh - engine1 verify kernels, round 2: rebuild libds41_tp + tp_bench (route_jobs a lane a pick), the device-clock
# probe on card 0, then tp_bench graph passes on 4 cards (widths 1-5) with the new route_jobs vs the serial one
# (STRATA_TP_ROUTE_SERIAL=1, same binary).  Service held; clocks pinned for the timings.
cd /opt/ai/dsv41/bench/engine1
# the build first: build-xe.sh sources oneAPI itself (a second source returns 3 under its set -e)
TARGETS="ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
source /opt/ai/dsv41/xe-env.sh
export STRATA_XE_JOBNT=1
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
mkdir -p prof4
/usr/local/sbin/larry-gpu-clock pin > /dev/null
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./clk_probe > prof4/clk.out 2>&1
echo "clk rc=$?"; cat prof4/clk.out
for rs in 0 1; do
  for n in ${WS:-1 2 3 4 5}; do
    STRATA_TP_ROUTE_SERIAL=$rs $B -m $M -N 4 --pos 1000 -n $n --iters 40 --graph 1 --cache $C --experts 48 > prof4/g_rs${rs}_n$n.out 2>&1
    echo "serial $rs n $n: rc=$? $(grep -h "pass median\|first pass" prof4/g_rs${rs}_n$n.out | tr '\n' ' ')"
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TR1 done $(date +%T)"
