#!/bin/bash
# after tattn.sh: the trace / profile comparison, then the exact-routing logits
cd /opt/ai/dsv41
while pgrep -f "[b]ench/tattn.sh" > /dev/null; do sleep 5; done
bash bench/tutfa.sh > logs/tutfa.log 2>&1
bash bench/tattn2.sh > logs/tattn2.log 2>&1
echo "TCHAIN1 done $(date +%T)" >> logs/tattn2.log
