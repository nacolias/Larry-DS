#!/bin/bash
# launch-t126.sh - abwT126: the FFN input to the expert cards as fp16 again (STRATA_XE_FN16=1: card-to-card copy of half
# the bytes, widened there; =2: read by peer loads) on the 10:54 service config.  It was neutral to worse while card 0
# alone bounded layers 0-19 (2026-10-25); now card 0 waits on the combines (abwT125: idle before cb 472 ms a 16K batch)
# and each expert-card call spends ~18 ms reading X (f32)
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  f0:prompts f1:prompts:STRATA_XE_FN16=1 f2:prompts:STRATA_XE_FN16=2 \
  > logs/abwT126.log 2>&1 < /dev/null &
echo "abwT126 launched: logs/abwT126.log"
