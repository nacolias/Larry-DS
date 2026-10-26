#!/bin/bash
# launch-t103.sh - abwT103: (1) bench/dqb/mkl_gemm_probe on card 1 (does oneMKL's gemm block the host; its speed at the
# shared expert's shapes); (2) the agent replay with the service's config against STRATA_XE_PREFETCH_MIN=2048 (layers
# 0-19 of 2-4K-token follow-ups stream on demand at 4096), twice each, interleaved.
cd /opt/ai/dsv41
AB_IDLE_MIN=0 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/dqb/mkl_probe.sh" \
  p0:agent p1:agent:STRATA_XE_PREFETCH_MIN=2048 p0b:agent p1b:agent:STRATA_XE_PREFETCH_MIN=2048 \
  > logs/abwT103.log 2>&1 < /dev/null &
echo "abwT103 launched: logs/abwT103.log"
