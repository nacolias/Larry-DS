#!/bin/bash
# after abwT75: the fp16 FFN input for the layers 0-19 calls (FN16=2, FN16_MIN=3000) at 512-column spans; gdb on stalls
cd /opt/ai/dsv41
while ! grep -q "service health\|NEW CAT" logs/abwT75.log 2>/dev/null; do sleep 10; done
grep -q "NEW CAT" logs/abwT75.log && exit 1
G=STRATA_STALL_GDB=1,STRATA_PIPE_TRACE=1
AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  f0:prompts:$G f1:prompts:STRATA_XE_FN16=2,STRATA_XE_FN16_MIN=3000,$G f0b:prompts:$G \
  > logs/abwT76.log 2>&1 < /dev/null &
