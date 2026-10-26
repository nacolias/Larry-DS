#!/bin/bash
# launch-t122.sh - abwT122: card 0's op profile (p30k's second batch, by shape) with STRATA_OPROJ_A=1 (patch-oproja2.py:
# the pointer gemm_batch for wo_b's layout) + STRATA_HC_NORM_MV_WIDE=1 (v3) - does it run, and what do the ops cost
cd /opt/ai/dsv41
AB_IDLE_MIN=0 setsid nohup bash bench/speed/abw.sh \
  "@SKIP=42000 STRATA_OPROJ_A=1 STRATA_HC_NORM_MV_WIDE=1 bash bench/speed/opprof-shapes.sh p30k.i32" \
  > logs/abwT122.log 2>&1 < /dev/null &
echo "abwT122 launched: logs/abwT122.log"
