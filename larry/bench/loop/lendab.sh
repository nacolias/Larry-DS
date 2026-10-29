#!/bin/bash
# lendab.sh [PFX] - STRATA_XE_PC_LEND (patch-pclend.py, the staged engine in pclbin) in a window: the deep agent replay
# (deeprun.sh: agentd.py, thinking on, temperature 1, 64K -> ~79K, the CPU-pool log) with lending off / on / off / on,
# then lending on under the paths that take the VRAM back - needles of 32K and 131K from empty (the prefetch), their
# decode and a 12K-token turn at depth (KV rows, a batch), concurtest (two conversations at once).  The service must be
# stopped: run as an abw.sh probe ("@bash bench/loop/lendab.sh").
cd /opt/ai/dsv41
P=${1:-lab}
B="XE_BIN=/opt/ai/dsv41/pclbin LD_LIBRARY_PATH=/opt/ai/dsv41/pclbin"
for r in off1 on1 off2 on2; do
  v=0
  [ "${r:0:2}" = on ] && v=1
  bash bench/loop/deeprun.sh $P-$r $B STRATA_XE_PC_LEND=$v 2>&1 | grep -E "turns 1-|lent|promotion cache|done|STILL|exited|rror|no room"
  [ -e "${AB_FLAG:-/nonexistent}" ] && exit 3
done
DEEP_TESTS=needles,conc bash bench/loop/deeprun.sh $P-chk $B STRATA_XE_PC_LEND=1 2>&1 | tail -24
