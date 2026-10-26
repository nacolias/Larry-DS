#!/bin/bash
# engwin.sh LOG CFG... - for each config of an A/B window: once its warm-ups are done, the cards' engine busy ticks
# (engmon.sh, 100 ms) for 22 s into logs/eng-CFG.csv (its prompts run meanwhile)
cd /opt/ai/dsv41
A=$1; shift
for C in "$@"; do
  for i in $(seq 1 3600); do grep -q "^\[$C\] warm 2" $A 2>/dev/null && break; grep -q "^service health\|nothing run" $A 2>/dev/null && exit 0; sleep 1; done
  bash bench/speed/engmon.sh logs/eng-$C.csv 22 100 > /dev/null 2>&1
done
