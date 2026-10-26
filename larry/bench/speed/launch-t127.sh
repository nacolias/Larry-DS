#!/bin/bash
# launch-t127.sh - abwT127: a fresh server's first big prompts with /proc/vmstat sampled (vmwatch.sh) - the first 8K / 31K
# prompts after a start stalled on the tier in 3 of ~8 servers today (abwT112 s2, abwT117, abwT126 f0: the calls waited
# on their layer's staging, the regrow's slabs took 245 ms): host memory compaction / migration after the arena load?
cd /opt/ai/dsv41
setsid nohup bash bench/speed/vmwatch.sh logs/vmwatch-T127.log 900 > /dev/null 2>&1 < /dev/null &
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  m0:prompts:STRATA_XE_CALL_TL=1 m1:prompts:STRATA_XE_CALL_TL=1 \
  > logs/abwT127.log 2>&1 < /dev/null &
echo "abwT127 launched: logs/abwT127.log, logs/vmwatch-T127.log"
