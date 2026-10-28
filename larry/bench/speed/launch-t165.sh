#!/bin/bash
# launch-t165.sh - abwT165: the front page's tables on the service's config of 2026-10-28 18:57 (#3 parts 1 + 2): fresh
# prompts from empty at its sizes (530 .. 31,008 tokens, two each) and a 256K needle with its decode and 12K turn
cd /opt/ai/dsv41
AB_PROMPTS=530,1021,2003,3003,4519,8054,16099,31008 AB_NEEDLES=262000 setsid nohup bash bench/speed/abw.sh \
  now:prompts,needles > logs/abwT165.log 2>&1 < /dev/null &
echo "abwT165 launched: logs/abwT165.log"
