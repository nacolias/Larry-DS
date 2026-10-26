#!/bin/bash
# launch-t133.sh - abwT133: fresh 8K / 16K / 31K prompts on the final config of 12:49 (the fused indexer, wo_a as one
# product, the wide hc mixes, the fp16 attention output; the fused hc path off again) - two servers
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh z0:prompts z0b:prompts > logs/abwT133.log 2>&1 < /dev/null &
echo "abwT133 launched: logs/abwT133.log"
