#!/bin/bash
# launch-t162.sh - abwT162 (#3, part 2): the expert cards bound big prompts now; abwT161's trace: an 8K call runs ~81
# xe_dqgemm launches (batches of STRATA_XE_GEMM_RB = 1024 rows x quant groups x 2 phases), ~35 ms, ~29 TFLOPS against
# 40-50 alone.  Bigger batches (fewer, fuller launches): RB 1024 / 2048 / 4096 on fresh 16K / 31K / 64K prompts
cd /opt/ai/dsv41
AB_PROMPTS=16000,31000,64000 setsid nohup bash bench/speed/abw.sh rb1k:prompts rb2k:prompts:STRATA_XE_GEMM_RB=2048 \
  rb4k:prompts:STRATA_XE_GEMM_RB=4096 > logs/abwT162.log 2>&1 < /dev/null &
echo "abwT162 launched: logs/abwT162.log"
