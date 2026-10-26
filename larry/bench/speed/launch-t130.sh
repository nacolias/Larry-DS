#!/bin/bash
# launch-t130.sh - abwT130: long-context needles (131K, 262K) on the 12:11 service config - today's changes (the fused
# prompt indexer to 256K+ keys and its nvis form's column slices, wo_a's op, the fused hc steps, the fp16 attention
# output) were measured to 31K only
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_NEEDLES=131072,262000 setsid nohup bash bench/speed/abw.sh lc0:needles > logs/abwT130.log 2>&1 < /dev/null &
echo "abwT130 launched: logs/abwT130.log"
