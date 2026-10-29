#!/bin/bash
# launch-kva.sh - patch-kvahead.py: xbuild.sh kva, then kvab.sh in a window
cd /opt/ai/dsv41
bash bench/longctx/xbuild.sh kva patch-kvahead > logs/xbuild-kva.out 2>&1
grep -q "BUILT" logs/xbuild-kva.out || { echo "build failed: nothing launched" > logs/abw-kva.log; exit 1; }
setsid nohup bash bench/speed/abw.sh "@bash bench/loop/kvab.sh" > logs/abw-kva.log 2>&1 < /dev/null &
echo "launched: logs/abw-kva.log"
