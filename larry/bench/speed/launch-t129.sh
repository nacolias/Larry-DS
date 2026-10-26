#!/bin/bash
# launch-t129.sh - abwT129: STRATA_SDPA_O16=1 (patch-sdpao16.py: the prompt attention's output stays fp16 through the
# in-place de-RoPE into wo_a's op - no f32 o, no conversion back) on the 10:54 service config:
#   1. numerics (tlppl-ab.sh)   2. card 0's op profile, by shape   3. fresh prompts without / with
cd /opt/ai/dsv41
cp -n logs/opps_p30k.err logs/opps_p30k-T124.err 2>/dev/null
B=STRATA_SDPA_O16=1
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/speed/tlppl-ab.sh o16 $B" \
  "@SKIP=42000 STRATA_SDPA_O16=1 bash bench/speed/opprof-shapes.sh p30k.i32" \
  h0:prompts h1:prompts:$B h0b:prompts h1b:prompts:$B \
  > logs/abwT129.log 2>&1 < /dev/null &
echo "abwT129 launched: logs/abwT129.log"
