#!/bin/bash
# launch-t145.sh - abwT145: the agent replay (agentx.py: a coding session growing to ~43K tokens, follow-ups of 17-6,500
# new tokens) on the service's config of 20:10, against STRATA_ENGRAM_DROP=0 (what the Engram drop costs incremental
# turns), two fresh servers each, interleaved
cd /opt/ai/dsv41
AB_IDLE_MIN=0 setsid nohup bash bench/speed/abw.sh a1:agent a0:agent:STRATA_ENGRAM_DROP=0 a1b:agent a0b:agent:STRATA_ENGRAM_DROP=0 \
  > logs/abwT145.log 2>&1 < /dev/null &
echo "abwT145 launched: logs/abwT145.log"
