#!/bin/bash
# launch-t132.sh - abwT132: the 984K needle + its turn again with the fused hc path off (STRATA_HC_FUSE_MAXN=8) - abwT131's
# turn (11,684 new tokens at 973K, a 13,139-column batch fit to card 0's 10.27 GiB) failed in DSV41_HC_FA with
# UR_RESULT_ERROR_OUT_OF_RESOURCES: the fused path, or card 0's VRAM at 1M with any of today's changes?
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_NEEDLES=984000 setsid nohup bash bench/speed/abw.sh lc2:needles:STRATA_HC_FUSE_MAXN=8 > logs/abwT132.log 2>&1 < /dev/null &
echo "abwT132 launched: logs/abwT132.log"
