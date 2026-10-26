#!/bin/bash
# abwT95: a flat perf profile of short fresh prompts (perfshort.sh) - 512 / 700 / 1K, the service's settings
cd /opt/ai/dsv41
setsid nohup bash bench/speed/perfshort.sh logs/abwT95.log pf > /dev/null 2>&1 < /dev/null &
AB_PROMPTS=512,700,1000 setsid nohup bash bench/speed/abw.sh pf:prompts > logs/abwT95.log 2>&1 < /dev/null &
