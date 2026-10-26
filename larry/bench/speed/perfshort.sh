#!/bin/bash
# perfshort.sh LOG CFG - during an A/B window's short prompts (config CFG), a flat perf profile of the A/B engine
# (the newest ds41_generate_x, 6 s from its warm-ups' end, all threads): logs/perf-short.data
cd /opt/ai/dsv41
A=${1:-logs/abwT95.log}; C=${2:-pf}
for i in $(seq 1 3600); do grep -q "^\[$C\] warm 2" $A 2>/dev/null && break; grep -q "^service health\|nothing run" $A 2>/dev/null && exit 0; sleep 1; done
P=$(pgrep -n -x ds41_generate_x)
[ -z "$P" ] && { echo "no engine" > logs/perfshort.log; exit 1; }
echo "$(date +%T) perf on $P ($(tr "\0" " " < /proc/$P/cmdline | cut -c1-80))" > logs/perfshort.log
perf record -F 1999 -p $P -o logs/perf-short.data -- sleep 6 >> logs/perfshort.log 2>&1
echo "$(date +%T) done" >> logs/perfshort.log
