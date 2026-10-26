#!/bin/bash
# launch-t140.sh - abwT140: STRATA_ENGRAM_DROP=1 (patch-engdrop.py: a prompt batch drops the Engram weight pages it
# read from the page cache) against 0, three fresh servers each, interleaved, as the service runs them (swap on, Engram
# through the filesystem, the tier's load through O_DIRECT); memtrace3.bt + vmwatch2.py; 8K / 16K / 31K x2.
cd /opt/ai/dsv41
B=STRATA_ENGRAM_RAWDEV=0
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh "@BT=memtrace3.bt bash bench/speed/memtrace-start.sh T140 45" \
  k1:prompts:$B r1:prompts:$B,STRATA_ENGRAM_DROP=1 k2:prompts:$B r2:prompts:$B,STRATA_ENGRAM_DROP=1 \
  k3:prompts:$B r3:prompts:$B,STRATA_ENGRAM_DROP=1 "@bash bench/speed/memtrace-stop.sh" > logs/abwT140.log 2>&1 < /dev/null &
echo "abwT140 launched: logs/abwT140.log"
