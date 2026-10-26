#!/bin/bash
# build-backend.sh - relink the engine (--backend gpu, the old "cuda" an alias) with the service held and no engine
# running, then check the new parser without touching a GPU (it exits before any backend starts)
cd /opt/ai/dsv41
[ -f /run/lock/larry-profile-a.hold ] || { echo "no hold file"; exit 1; }
systemctl is-active -q larry-ds41 && { echo "the service is running"; exit 1; }
pgrep -f "[d]s41_generate_x[e]" > /dev/null && { echo "an engine is running"; exit 1; }
TARGETS="ds41_generate_xe" ./build-xe.sh > logs/build-backend.log 2>&1
rc=$?
echo "build rc=$rc"
grep -E " error|Error " logs/build-backend.log | head -5
tail -1 logs/build-backend.log
ls -la --time-style=+%T xolotl/build-xe/libds41_xe.so xolotl/build-xe/libds41_tp.so xolotl/build-xe/ds41_generate_xe
[ $rc = 0 ] || exit $rc
source ./xe-env.sh > /dev/null 2>&1
B=xolotl/build-xe/ds41_generate_xe
for be in foo gpu cuda sycl cpu; do echo "--backend $be (no model):"; $B --backend $be; echo "  rc=$?"; done
