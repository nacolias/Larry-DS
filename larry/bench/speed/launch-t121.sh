#!/bin/bash
# launch-t121.sh - abwT121: STRATA_OPROJ_A=1 (patch-oproja.py: wo_a as one gemm_batch writing wo_b's layout - no CONT;
# abwT120's GGML_SYCL_LARRY_SLICES_BATCH form took wo_a 768 -> 490 us) and STRATA_HC_NORM_MV_WIDE=1 (the wide norm_mv v3:
# 64 tokens a work-group, float4 staging - v2 was 604 us a call against the chain's 323):
#   1. numerics of both (tlppl-ab.sh)   2. card 0's op profile with both, by shape
#   3. fresh prompts: neither / OPROJ_A / both
cd /opt/ai/dsv41
cp -n logs/opps_p30k.err logs/opps_p30k-T120.err 2>/dev/null
B=STRATA_OPROJ_A=1,STRATA_HC_NORM_MV_WIDE=1
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/speed/tlppl-ab.sh opa $B" \
  "@SKIP=42000 STRATA_OPROJ_A=1 STRATA_HC_NORM_MV_WIDE=1 bash bench/speed/opprof-shapes.sh p30k.i32" \
  r0:prompts r1:prompts:STRATA_OPROJ_A=1 r2:prompts:$B \
  > logs/abwT121.log 2>&1 < /dev/null &
echo "abwT121 launched: logs/abwT121.log"
