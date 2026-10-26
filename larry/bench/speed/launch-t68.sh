#!/bin/bash
# abwT68: fresh short prompts with the compact CPU-light path off / on (timelines on), then card 0's span kernels traced
cd /opt/ai/dsv41
AB_PROMPTS=512,1000,2000,3000,4500 setsid nohup bash bench/speed/abw.sh \
  s0:prompts:STRATA_XE_LT_COMPACT=0,STRATA_GPU_PTL=1,STRATA_PIPE_TRACE=1,STRATA_XE_CALL_TL=1,STRATA_BATCH_TRACE=1 \
  s1:prompts:STRATA_GPU_PTL=1,STRATA_PIPE_TRACE=1,STRATA_XE_CALL_TL=1,STRATA_BATCH_TRACE=1 \
  "@/opt/ai/dsv41/venv/bin/python -I /opt/ai/dsv41/bench/speed/utdrive.py utsp 512,2000 --warm 1000" \
  > logs/abwT68.log 2>&1 < /dev/null &
