#!/bin/bash
# launch-pcb.sh - decode at depth, part 2 (patch-pcbuf.py: the prompt buffers lent too): xbuild.sh pcb, then in a window
# lendab.sh pcb on pcbbin with STRATA_XE_PC_LEND_BUFS=0 / =1 (lending on in both)
cd /opt/ai/dsv41
bash bench/longctx/xbuild.sh pcb patch-pcbuf > logs/xbuild-pcb.out 2>&1
grep -q "BUILT" logs/xbuild-pcb.out || { echo "build failed: nothing launched" > logs/abw-pcb.log; exit 1; }
LAB_BIN=pcbbin LAB_OFF="STRATA_XE_PC_LEND=1 STRATA_XE_PC_LEND_BUFS=0" LAB_ON="STRATA_XE_PC_LEND=1 STRATA_XE_PC_LEND_BUFS=1" \
  setsid nohup bash bench/speed/abw.sh "@bash bench/loop/lendab.sh pcb" > logs/abw-pcb.log 2>&1 < /dev/null &
echo "launched: logs/abw-pcb.log"
