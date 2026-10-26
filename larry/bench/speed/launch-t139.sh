#!/bin/bash
# launch-t139.sh - abwT139: STRATA_XE_LOAD_DIRECT (patch-loaddirect.py: the tier's bulk load through O_DIRECT, default 1)
# against 0 (the buffered reads: ~150 GB of the model file through the page cache a start), three fresh servers each,
# interleaved, as the service runs them (swap on, Engram through the filesystem); memtrace3.bt; 8K / 16K / 31K x2.
cd /opt/ai/dsv41
B=STRATA_ENGRAM_RAWDEV=0
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh "@BT=memtrace3.bt bash bench/speed/memtrace-start.sh T139 45" \
  o1:prompts:$B,STRATA_XE_LOAD_DIRECT=0 d1:prompts:$B o2:prompts:$B,STRATA_XE_LOAD_DIRECT=0 d2:prompts:$B \
  o3:prompts:$B,STRATA_XE_LOAD_DIRECT=0 d3:prompts:$B "@bash bench/speed/memtrace-stop.sh" > logs/abwT139.log 2>&1 < /dev/null &
echo "abwT139 launched: logs/abwT139.log"
