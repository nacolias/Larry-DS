#!/bin/bash
# launch-t134.sh - abwT134: the owner-approved kernel memory settings (memset.sh tuned: watermark_scale_factor 200,
# compaction_proactiveness 0, THP defrag defer) against the boot values, four fresh servers each, interleaved; fresh
# 8K / 16K / 31K prompts twice each; memtrace.bt + vmwatch2.py over the whole window (memsum.py sums them per config).
# The service comes back on the boot values.
cd /opt/ai/dsv41
M="@bash bench/speed/memset.sh"
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh "@bash bench/speed/memtrace-start.sh T134 75" \
  "$M default" d1:prompts "$M tuned" t1:prompts "$M default" d2:prompts "$M tuned" t2:prompts \
  "$M default" d3:prompts "$M tuned" t3:prompts "$M default" d4:prompts "$M tuned" t4:prompts \
  "$M default" "@bash bench/speed/memtrace-stop.sh" > logs/abwT134.log 2>&1 < /dev/null &
echo "abwT134 launched: logs/abwT134.log"
