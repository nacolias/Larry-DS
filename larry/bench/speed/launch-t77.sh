#!/bin/bash
# abwT77: two fresh baseline engines, 8K / 16K / 31K prompts twice each, under xetrace.sh (kernel-side durations of
# the xe / PRIME ioctls) with the tier's allocation log - where do the slow batch heads spend their seconds?
cd /opt/ai/dsv41
setsid nohup bash bench/speed/xetrace.sh logs/abwT77.log > logs/xetrace.log 2>&1 < /dev/null &
G=STRATA_XE_ALLOC_LOG=1,STRATA_PIPE_TRACE=1
AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  t0:prompts:$G t1:prompts:$G > logs/abwT77.log 2>&1 < /dev/null &
