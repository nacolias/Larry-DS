#!/bin/bash
# launch-t142.sh - abwT142: the 984K needle's follow-up turn (an 11.7K batch at 973K) died in abwT141 with
# UR_RESULT_ERROR_OUT_OF_RESOURCES on card 0 (DSV4_HC_PRE) - its kept-cache batch had 9.85 GiB free vs 10.27 in abwT132's
# pass.  The service's settings again (reproducible?) against STRATA_SERVE_KEEP_LEARN_MARGIN_GIB=2.5 (1 GiB more room
# for a kept batch), both with STRATA_VRAM_TRACE=1.
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_NEEDLES=984000 setsid nohup bash bench/speed/abw.sh lcA:needles:STRATA_VRAM_TRACE=1 \
  lcB:needles:STRATA_VRAM_TRACE=1,STRATA_SERVE_KEEP_LEARN_MARGIN_GIB=2.5 > logs/abwT142.log 2>&1 < /dev/null &
echo "abwT142 launched: logs/abwT142.log"
