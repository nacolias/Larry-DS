#!/bin/bash
# stallhunt.sh - the slow batch heads (0.8-7.7 s, ~1 prompt in 10-30): two identical A/B configs of fresh 8K / 16K
# prompts with the stall watch's perf sampling on (STRATA_STALL_PERF=1); during the second one a one-shot kernel
# compaction is triggered every 6 s (/proc/sys/vm/compact_memory) - does page migration cause them (xe userptr
# repins under the VM lock)?  Logs: logs/abwT71.log, logs/stallhunt.log.
cd /opt/ai/dsv41
L=logs/stallhunt.log
: > $L
E=STRATA_STALL_PERF=1,STRATA_XE_CALL_TL=1
AB_PROMPTS=8000,8000,8000,16000,16000 setsid nohup bash bench/speed/abw.sh cA:prompts:$E cB:prompts:$E > logs/abwT71.log 2>&1 < /dev/null &
for i in $(seq 1 900); do grep -q "^\[cB\] up" logs/abwT71.log && break; grep -q "service health\|NEW CAT" logs/abwT71.log && exit 0; sleep 2; done
echo "$(date +%T) cB up: compaction every 6 s" >> $L
while ! grep -q "^\[cB\] done\|service health\|NEW CAT" logs/abwT71.log; do
  sleep 6
  echo 1 > /proc/sys/vm/compact_memory
  echo "$(date +%T.%N | cut -c1-12) compact_memory ($(grep -E "^pgmigrate_success" /proc/vmstat))" >> $L
done
echo "$(date +%T) done" >> $L
