#!/bin/bash
# launch-lab.sh - STRATA_XE_PC_LEND's A/B (lendab.sh) in a window, once xbuild.sh pcl has staged the engine in pclbin
cd /opt/ai/dsv41
for i in $(seq 1 240); do grep -q "BUILT\|BUILD FAILED\|PATCH FAILED" logs/xbuild-pcl.out 2>/dev/null && break; sleep 15; done
grep -q "BUILT" logs/xbuild-pcl.out || { echo "no staged engine: nothing launched" > logs/abw-lab.log; exit 1; }
setsid nohup bash bench/speed/abw.sh "@bash bench/loop/lendab.sh lab" > logs/abw-lab.log 2>&1 < /dev/null &
echo "launched: logs/abw-lab.log"
