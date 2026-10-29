#!/bin/bash
# lendab.sh [PFX] - the promotion cache's lending in a window, on a staged engine (LAB_BIN, default pclbin): the deep agent
# replay (deeprun.sh: agentd.py, thinking on, temperature 1, 64K -> ~79K, the CPU-pool log) with LAB_OFF / LAB_ON / LAB_OFF
# / LAB_ON (default STRATA_XE_PC_LEND=0 / =1; part 2: STRATA_XE_PC_LEND_BUFS=0 / =1), then LAB_ON under the paths that take
# the VRAM back - needles of 32K and 131K from empty (the prefetch), their decode and a 12K-token turn at depth (KV rows,
# a batch), concurtest (two conversations at once).  The service must be stopped: an abw.sh probe ("@bash bench/loop/lendab.sh").
cd /opt/ai/dsv41
P=${1:-lab}
BIN=${LAB_BIN:-pclbin}
B="XE_BIN=/opt/ai/dsv41/$BIN LD_LIBRARY_PATH=/opt/ai/dsv41/$BIN"
OFF=${LAB_OFF:-STRATA_XE_PC_LEND=0}
ON=${LAB_ON:-STRATA_XE_PC_LEND=1}
for r in off1 on1 off2 on2; do
  E=$OFF
  [ "${r:0:2}" = on ] && E=$ON
  bash bench/loop/deeprun.sh $P-$r $B $E 2>&1 | grep -E "turns 1-|lent|promotion cache|done|STILL|exited|rror|no room"
  [ -e "${AB_FLAG:-/nonexistent}" ] && exit 3
done
DEEP_TESTS=needles,conc bash bench/loop/deeprun.sh $P-chk $B $ON 2>&1 | tail -24
