#!/bin/bash
# launch-t116.sh - abwT116: xe_dqgemm (the prompt path's expert GEMMs, weights decoded inside the XMX GEMM) in isolation
# at a big prompt's tier-call shapes (bench_dqgemm NE M card layout): a layer-0-19 call of ~8K columns gives an expert
# card ~100-130 experts x ~100-130 tokens; the tier calls' GEMM phase runs ~50 ms (abwT113) for ~1 TFLOP - ~20 TFLOPS
cd /opt/ai/dsv41
AB_IDLE_MIN=0 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/dqb/dqb-shapes.sh" \
  > logs/abwT116.log 2>&1 < /dev/null &
echo "abwT116 launched: logs/abwT116.log"
