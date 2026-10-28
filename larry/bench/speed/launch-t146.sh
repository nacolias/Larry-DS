#!/bin/bash
# launch-t146.sh - abwT146 (#3, big prompts: the baseline): card 0's GPU timeline + the tier calls' timelines
# (STRATA_GPU_PTL=1 + STRATA_XE_CALL_TL=1; ptlsum.py, calltl_stats.py) through fresh 16K / 31K / 64K prompts and a 131K
# needle on the service's config of 2026-10-28 05:06 - which side bounds a big prompt's layers 0-19 now (10-26 12:11:
# balanced) - then card 0's op profile of p30k's second batch by shape (opprof-shapes.sh).
cd /opt/ai/dsv41
AB_PROMPTS=16000,31000,64000 AB_NEEDLES=131072 setsid nohup bash bench/speed/abw.sh \
  pt3:prompts,needles:STRATA_GPU_PTL=1,STRATA_XE_CALL_TL=1 \
  "@SKIP=42000 bash bench/speed/opprof-shapes.sh p30k.i32" > logs/abwT146.log 2>&1 < /dev/null &
echo "abwT146 launched: logs/abwT146.log"
