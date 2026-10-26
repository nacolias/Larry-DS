#!/bin/bash
# tprof.sh - engine1: the TP pass's kernels under unitrace (eager submission: the kernel times are the GPU's, the waits
# show as the spin kernels' time), N = 1 and 4, decode n 1 and verify n 4, 48 real expert blobs; clocks pinned
cd /opt/ai/dsv41/bench/engine1
source /opt/ai/dsv41/xe-env.sh
export STRATA_XE_JOBNT=1
UT=/opt/src/pti-gpu/tools/unitrace/build/unitrace
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for N in ${NS:-1 4}; do
  for n in ${WS:-1 4}; do
    $UT -d -o ut_tp_N${N}_n${n}.txt $B -m $M -N $N --pos 8192 -n $n --iters 10 --graph 0 --cache $C --experts 48 > ut_tp_N${N}_n${n}.out 2>&1
    echo "N $N n $n: rc=$? $(grep -h "pass median" ut_tp_N${N}_n${n}.out)"
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TPROF done $(date +%T)"
