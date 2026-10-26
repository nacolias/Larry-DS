#!/bin/bash
# launch-t115.sh - abwT115: 1024-column spans (--prefill-chunk 1024 + GGML_SYCL_DSV4_SDPA_N=1024) against 512 again, now
# that the fused indexer took ~0.4 ms a span off card 0 (abwT88 before it: card 0's dense part -17 %, but the tier's
# share grew nearly as much - 16K 7.22 -> 7.15, 31K 12.70 -> 12.64, 4.5K +0.35 s)
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=4500,8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  w5:prompts w10:prompts:GGML_SYCL_DSV4_SDPA_N=1024,AB_ARGS=--prefill-chunk:1024 \
  > logs/abwT115.log 2>&1 < /dev/null &
echo "abwT115 launched: logs/abwT115.log"
