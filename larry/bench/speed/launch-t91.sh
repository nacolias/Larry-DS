#!/bin/bash
# abwT91: short fresh prompts - STRATA_PIPE_MIN_PART=256 on the pipelined path (patch-pipesplit2.py: a 513-767-token
# batch's layers in one part), the dense half's submit-lock priority off (STRATA_SUBMIT_PRIO=0), against the base
cd /opt/ai/dsv41
AB_PROMPTS=512,700,1000,2000,3000 setsid nohup bash bench/speed/abw.sh \
  a0:prompts a1:prompts:STRATA_PIPE_MIN_PART=256 a2:prompts:STRATA_SUBMIT_PRIO=0 \
  > logs/abwT91.log 2>&1 < /dev/null &
