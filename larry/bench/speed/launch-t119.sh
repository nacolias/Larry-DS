#!/bin/bash
# launch-t119.sh - abwT119: STRATA_HC_NORM_MV_WIDE=1 (patch-normmvw.py: the prompt path's hc mixes as one wide norm_mv,
# the 4-stream state read once - was rms_norm's copy + a [24 x 20480] GEMM, ~0.65 ms a layer-span):
#   1. numerics (tlppl-ab.sh: p8k KL against the run-to-run floor, p30k PPL)
#   2. card 0's op profile of p30k's second batch with it, by shape (opprof-shapes.sh)
#   3. fresh 8K / 16K / 31K prompts through the server without and with it
cd /opt/ai/dsv41
cp -n logs/opps_p30k.err logs/opps_p30k-T118.err 2>/dev/null
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/speed/tlppl-ab.sh nmv STRATA_HC_NORM_MV_WIDE=1" \
  "@SKIP=42000 STRATA_HC_NORM_MV_WIDE=1 bash bench/speed/opprof-shapes.sh p30k.i32" \
  n0:prompts n1:prompts:STRATA_HC_NORM_MV_WIDE=1 \
  > logs/abwT119.log 2>&1 < /dev/null &
echo "abwT119 launched: logs/abwT119.log"
