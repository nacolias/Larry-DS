#!/bin/bash
# launch-pl3.sh - patch-pclend3.py (a prompt call keeps the lent slots when it cannot grow a buffer): xbuild.sh pl3, then
# lendab.sh pl3 on pl3bin with STRATA_XE_PC_LEND_KEEP_CALLS=0 / =1 (lending on in both; call timelines on)
cd /opt/ai/dsv41
bash bench/longctx/xbuild.sh pl3 patch-pclend3 > logs/xbuild-pl3.out 2>&1
grep -q "BUILT" logs/xbuild-pl3.out || { echo "build failed: nothing launched" > logs/abw-pl3.log; exit 1; }
LAB_BIN=pl3bin LAB_OFF="STRATA_XE_PC_LEND_KEEP_CALLS=0 STRATA_XE_CALL_TL=1" LAB_ON="STRATA_XE_CALL_TL=1" \
  setsid nohup bash bench/speed/abw.sh "@bash bench/loop/lendab.sh pl3" > logs/abw-pl3.log 2>&1 < /dev/null &
echo "launched: logs/abw-pl3.log"
