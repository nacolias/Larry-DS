#!/bin/bash
# after abwT74: 1024-column spans with / without the fp16 FFN input for big calls (FN16=2, FN16_MIN=4096), timelines on
cd /opt/ai/dsv41
while ! grep -q "service health\|NEW CAT" logs/abwT74.log; do sleep 10; done
grep -q "NEW CAT" logs/abwT74.log && exit 1
S1K=AB_ARGS=--prefill-chunk:1024,GGML_SYCL_DSV4_SDPA_N=1024,STRATA_SERVE_BATCH_FIXED_GIB=5.7
AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  u1k:prompts:$S1K,STRATA_GPU_PTL=1,STRATA_PIPE_TRACE=1 \
  u1kf:prompts:$S1K,STRATA_XE_FN16=2,STRATA_XE_FN16_MIN=4096,STRATA_GPU_PTL=1,STRATA_PIPE_TRACE=1 \
  > logs/abwT75.log 2>&1 < /dev/null &
