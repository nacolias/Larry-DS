#!/bin/bash
# build-tier.sh - rebuild the xe tier's library (libds41_xe.so; TARGETS="ds41_xe ds41_tp": the TP core's too) with the service held and no engine running
cd /opt/ai/dsv41
[ -f /run/lock/larry-profile-a.hold ] || { echo "no hold file"; exit 1; }
systemctl is-active -q larry-ds41 && { echo "the service is running"; exit 1; }
pgrep -x ds41_generate_x > /dev/null && { echo "an engine is running"; exit 1; }
TARGETS="${TARGETS:-ds41_xe}" ./build-xe.sh > logs/build-tier.log 2>&1
rc=$?
echo "build rc=$rc"
grep -E " error|Error " logs/build-tier.log | head -5
tail -1 logs/build-tier.log
ls -la --time-style=+%T xolotl/build-xe/libds41_xe.so xolotl/build-xe/libds41_tp.so
exit $rc
