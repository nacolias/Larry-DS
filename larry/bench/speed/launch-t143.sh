#!/bin/bash
# launch-t143.sh - abwT143: STRATA_SERVE_KEEP_LEARN_DEPTH_GIB=1.5 (patch-keepdepth.py: a kept batch gets 1.5 GiB *
# tokens/1M more room on card 0) on the 984K needle's follow-up turn - its low point had 0.00 GiB free in abwT142 and
# died in abwT141; STRATA_VRAM_TRACE=1 shows the low point
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_NEEDLES=984000 setsid nohup bash bench/speed/abw.sh lcD:needles:STRATA_VRAM_TRACE=1,STRATA_SERVE_KEEP_LEARN_DEPTH_GIB=1.5 \
  > logs/abwT143.log 2>&1 < /dev/null &
echo "abwT143 launched: logs/abwT143.log"
