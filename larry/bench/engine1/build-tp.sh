#!/bin/bash
# build-tp.sh - rebuild engine1's library and the engine with the service held and no engine running
cd /opt/ai/dsv41
[ -f /run/lock/larry-profile-a.hold ] || { echo "no hold file"; exit 1; }
systemctl is-active -q larry-ds41 && { echo "the service is running"; exit 1; }
pgrep -f "[d]s41_generate_x[e]" > /dev/null && { echo "an engine is running"; exit 1; }
TARGETS="ds41_tp ds41_generate_xe" ./build-xe.sh > logs/build-pagefix.log 2>&1
rc=$?
echo "build rc=$rc"
grep -E " error|Error " logs/build-pagefix.log | head -5
tail -1 logs/build-pagefix.log
ls -la --time-style=+%T xolotl/build-xe/libds41_tp.so xolotl/build-xe/ds41_generate_xe
