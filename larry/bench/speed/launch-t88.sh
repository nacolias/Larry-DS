#!/bin/bash
# abwT88: with BAR staging card 0's spans bound layers 0-19 from 8K up - 1024-column spans (--prefill-chunk 1024 +
# GGML_SYCL_DSV4_SDPA_N=1024) against 512 again; and STRATA_XE_CPU_LIGHT=3 (abwT87: 1K-3K -0.05..-0.1 s) on big ones
cd /opt/ai/dsv41
AB_PROMPTS=4500,8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  k5:prompts k10:prompts:GGML_SYCL_DSV4_SDPA_N=1024,AB_ARGS=--prefill-chunk:1024 k3:prompts:STRATA_XE_CPU_LIGHT=3 \
  > logs/abwT88.log 2>&1 < /dev/null &
