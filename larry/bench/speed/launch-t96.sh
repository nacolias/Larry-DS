#!/bin/bash
# abwT98: the tier's submit lock per call site (STRATA_XE_LOCK_PROF=1, patch-lockprof.py) in short fresh prompts
cd /opt/ai/dsv41
AB_PROMPTS=512,700,1000,2000,512,700,1000 setsid nohup bash bench/speed/abw.sh lp:prompts:STRATA_XE_LOCK_PROF=1 > logs/abwT98.log 2>&1 < /dev/null &
