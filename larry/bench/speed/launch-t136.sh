#!/bin/bash
# launch-t136.sh - abwT136: fresh servers with swap off (no TTM backup space: the xe shrinker cannot write the engine's
# buffers out) against swap on, interleaved, three each, memtrace2.bt over the window; fresh 8K / 16K / 31K prompts twice.
# swapctl.sh guard turns swap back on when the window ends, however it ends.  The test servers run the patched
# serve_ds41.py (patch-literals.py), and so does the service the window brings back.
cd /opt/ai/dsv41
SW="@bash bench/speed/swapctl.sh"
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh "@BT=memtrace2.bt bash bench/speed/memtrace-start.sh T136 45" \
  "$SW off" s1:prompts "$SW on" n1:prompts "$SW off" s2:prompts "$SW on" n2:prompts "$SW off" s3:prompts "$SW on" n3:prompts \
  "@bash bench/speed/memtrace-stop.sh" > logs/abwT136.log 2>&1 < /dev/null &
P=$!
setsid nohup bash bench/speed/swapctl.sh guard $P > logs/swapguard-T136.log 2>&1 < /dev/null &
echo "abwT136 launched: logs/abwT136.log (abw pid $P, swap guard on it)"
