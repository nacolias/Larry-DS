#!/bin/bash
# launch-t128.sh - abwT128: the Engram tables read with O_DIRECT (STRATA_ENGRAM_DIRECT=1: no page cache) on fresh servers,
# with /proc/vmstat sampled (vmwatch.sh).  abwT127: through every big prompt kswapd reclaimed 0.1-1M pages / 2 s with
# bursts of direct reclaim and direct compaction (compact_stall up to 330 / 2 s), worst right after a start - and the
# first 8K / 16K / 31K prompts after a start stall on the tier's staging (abwT112 s2 22 s, abwT126 f0 18 s)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/vmwatch.sh logs/vmwatch-T128.log 1500 > /dev/null 2>&1 < /dev/null &
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  e0:prompts e1:prompts:STRATA_ENGRAM_DIRECT=1 e0b:prompts e1b:prompts:STRATA_ENGRAM_DIRECT=1 \
  > logs/abwT128.log 2>&1 < /dev/null &
echo "abwT128 launched: logs/abwT128.log, logs/vmwatch-T128.log"
