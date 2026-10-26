#!/bin/bash
# launch-t123.sh - abwT123: STRATA_OPROJ_A=1 (wo_a as one pointer gemm_batch writing wo_b's layout: 768 + 162 (CONT) ->
# 520 us a layer-span, abwT122) + STRATA_HC_NORM_MV_WIDE=1 (v3: 2 x 323 -> 2 x 153 us):
#   1. numerics (tlppl-ab.sh)   2. fresh 8K / 16K / 31K prompts without / with   3. the agent replay without / with
cd /opt/ai/dsv41
B=STRATA_OPROJ_A=1,STRATA_HC_NORM_MV_WIDE=1
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/speed/tlppl-ab.sh opb $B" \
  u0:prompts u1:prompts:$B k0:agent k1:agent:$B \
  > logs/abwT123.log 2>&1 < /dev/null &
echo "abwT123 launched: logs/abwT123.log"
