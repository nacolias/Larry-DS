#!/bin/bash
# tprof2.sh - engine1: where a verify pass's GPU time goes.  tp_bench on 4 cards at the essay's context (pos 1000: the
# 1024-row cap bucket), widths 1-5: graph replay timed first (what the engine runs), then the same pass eager under
# unitrace (-d: every kernel's device time; the spin kernels hold the waits) + one chrome timeline (n 4).
# The service must be held down; clocks pinned.
cd /opt/ai/dsv41/bench/engine1
source /opt/ai/dsv41/xe-env.sh
export STRATA_XE_JOBNT=1
[ -f /run/lock/larry-profile-a.hold ] || { echo "no hold file"; exit 1; }
systemctl is-active -q larry-ds41 && { echo "the service is running"; exit 1; }
pgrep -x ds41_generate_x > /dev/null && { echo "an engine is running"; exit 1; }
UT=/opt/src/pti-gpu/tools/unitrace/build/unitrace
B=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
C=/srv/dsv41/tpcache
P=${POS:-1000}
mkdir -p prof2
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for n in ${WS:-1 2 3 4 5}; do
  $B -m $M -N 4 --pos $P -n $n --iters 40 --graph 1 --cache $C --experts 48 > prof2/g_n$n.out 2>&1
  echo "graph n $n: rc=$? $(grep -h "pass median" prof2/g_n$n.out)"
  $UT -d -o prof2/ut_n$n.txt $B -m $M -N 4 --pos $P -n $n --iters 10 --graph 0 --cache $C --experts 48 > prof2/e_n$n.out 2>&1
  echo "eager n $n: rc=$? $(grep -h "pass median" prof2/e_n$n.out)"
done
if [ -n "${CHROME:-4}" ]; then
  rm -rf prof2/ct; mkdir -p prof2/ct
  $UT --chrome-kernel-logging --chrome-no-thread-on-device --output-dir-path prof2/ct $B -m $M -N 4 --pos $P -n ${CHROME:-4} --iters 4 --graph 0 --cache $C --experts 48 > prof2/ct.out 2>&1
  echo "chrome n ${CHROME:-4}: rc=$?"
fi
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TPROF2 done $(date +%T)"
