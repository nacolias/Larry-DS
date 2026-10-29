#!/bin/bash
# launch-kva2.sh - patch-kvahead.py (revised: one lead everywhere) + patch-pclend2.py: xbuild.sh kva2, then kvab.sh kva2 in a window
cd /opt/ai/dsv41
bash bench/longctx/xbuild.sh kva2 patch-kvahead patch-pclend2 > logs/xbuild-kva2.out 2>&1
grep -q "BUILT" logs/xbuild-kva2.out || { echo "build failed: nothing launched" > logs/abw-kva2.log; exit 1; }
KV_BIN=kva2bin setsid nohup bash bench/speed/abw.sh "@bash bench/loop/kvab.sh kva2" > logs/abw-kva2.log 2>&1 < /dev/null &
echo "launched: logs/abw-kva2.log"
