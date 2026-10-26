#!/bin/bash
# abwT89: short fresh prompts' calls with BAR staging (STRATA_XE_CALL_TL=1: per call and card - streams, GEMMs, jobs,
# the CPU-light part) - where a 512 / 2K prompt's ~37 ms a layer goes (its DRAM bound ~18)
cd /opt/ai/dsv41
AB_PROMPTS=512,2000 setsid nohup bash bench/speed/abw.sh \
  ct:prompts:STRATA_XE_CALL_TL=1,STRATA_PIPE_TRACE=1 > logs/abwT89.log 2>&1 < /dev/null &
