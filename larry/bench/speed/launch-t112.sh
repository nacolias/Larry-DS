#!/bin/bash
# launch-t112.sh - GGML_SYCL_LI_FUSED=1 (the prompt indexer as one fused XMX kernel) and STRATA_LI_NVIS=1 (the indexer
# told each query's visible rows) through the server, both default off in the 07:08 build:
#   1. fresh 8K / 16K / 31K prompts: without, the fused kernel, both
#   2. the agent replay (agentx.py: a coding session growing to ~43K, follow-ups of 17-6,500 tokens) without / with both
cd /opt/ai/dsv41
OFF=GGML_SYCL_LI_FUSED=0,STRATA_LI_NVIS=0
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  s0:prompts:$OFF s1:prompts:GGML_SYCL_LI_FUSED=1,STRATA_LI_NVIS=0 s2:prompts:GGML_SYCL_LI_FUSED=1,STRATA_LI_NVIS=1 \
  k0:agent:$OFF k2:agent:GGML_SYCL_LI_FUSED=1,STRATA_LI_NVIS=1 \
  > logs/abwT112.log 2>&1 < /dev/null &
echo "abwT112 launched: logs/abwT112.log"
