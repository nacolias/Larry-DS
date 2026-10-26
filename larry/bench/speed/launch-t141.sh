#!/bin/bash
# launch-t141.sh - abwT141: long contexts on the service's current config (the DSpark cut, the load through O_DIRECT,
# the Engram drop): a 131K and a 984K needle from empty, each with its follow-up turn (abwT130-132 before: 131K 41.1 s,
# 973K 348-357 s, its 11.7K turn 10.0 s)
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_NEEDLES=131072,984000 setsid nohup bash bench/speed/abw.sh lc:needles > logs/abwT141.log 2>&1 < /dev/null &
echo "abwT141 launched: logs/abwT141.log"
