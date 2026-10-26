#!/bin/bash
# launch-t138.sh - abwT138: where a fresh engine's memory pressure during prompts comes from (memtrace3.bt: page-cache
# insertions by file, anonymous faults, swap-ins, TTM backups) - two fresh servers as the service runs them: swap on,
# Engram through the filesystem (STRATA_ENGRAM_RAWDEV=0: the service's user cannot open /dev/dm-0); 8K / 16K / 31K x2.
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh "@BT=memtrace3.bt bash bench/speed/memtrace-start.sh T138 25" \
  m1:prompts:STRATA_ENGRAM_RAWDEV=0 m2:prompts:STRATA_ENGRAM_RAWDEV=0 "@bash bench/speed/memtrace-stop.sh" > logs/abwT138.log 2>&1 < /dev/null &
echo "abwT138 launched: logs/abwT138.log"
