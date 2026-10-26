#!/bin/bash
# launch-t120.sh - abwT120: STRATA_HC_NORM_MV_WIDE=1 (the reworked wide norm_mv: the weight slice staged in local memory,
# 2 tokens a sub-group) + GGML_SYCL_LARRY_SLICES_BATCH=1 (wo_a's 8 slices as one gemm_batch, patch-slicesbatch.py):
#   1. numerics of both (tlppl-ab.sh)   2. card 0's op profile with both, by shape   3. fresh prompts without / with both
cd /opt/ai/dsv41
cp -n logs/opps_p30k.err logs/opps_p30k-T119.err 2>/dev/null
ON=STRATA_HC_NORM_MV_WIDE=1,GGML_SYCL_LARRY_SLICES_BATCH=1
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/speed/tlppl-ab.sh nsb $ON" \
  "@SKIP=42000 STRATA_HC_NORM_MV_WIDE=1 GGML_SYCL_LARRY_SLICES_BATCH=1 bash bench/speed/opprof-shapes.sh p30k.i32" \
  q0:prompts q1:prompts:$ON \
  > logs/abwT120.log 2>&1 < /dev/null &
echo "abwT120 launched: logs/abwT120.log"
