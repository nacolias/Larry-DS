#!/bin/bash
# kvab.sh [PFX] - patch-kvahead.py's read-ahead (the staged engine in KV_BIN, default kvabin) in a window: agentd.py's deep
# session from 63K (its first reply and turns cross the 65,536-row KV bucket) with the read-ahead off (STRATA_TP_DEC_AHEAD=0)
# and on; the expert cards' evictions with their reads / read-ahead waits, and the read-aheads, from the engine logs.
# The service must be stopped: an abw.sh probe ("@bash bench/loop/kvab.sh").
cd /opt/ai/dsv41
P=${1:-kva}
K=${KV_BIN:-kvabin}
B="XE_BIN=/opt/ai/dsv41/$K LD_LIBRARY_PATH=/opt/ai/dsv41/$K"
export AGENTD_ARGS="--first 63000 --turns 10"
for r in off on; do
  E=""
  [ $r = off ] && E="STRATA_TP_DEC_AHEAD=0"
  bash bench/loop/deeprun.sh $P-$r $B $E 2>&1 | grep -E "\] t[0-9]+ |turns 1-|\] done|STILL|exited|rror|no room"
  grep -a -h "gave [0-9]* experts\|read-ahead for\|KV replicas now" logs/engine-$P-$r.log | cut -c1-230
  [ -e "${AB_FLAG:-/nonexistent}" ] && exit 3
done
