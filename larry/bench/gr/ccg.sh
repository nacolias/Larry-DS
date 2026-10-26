#!/bin/bash
# ccg.sh FILE - syntax/semantic check of one ggml-sycl source with Larry's flags (local DPC++ + oneMKL 2026.1)
f=$1; shift
args=("$@")
set --
source /opt/intel/oneapi/compiler/2026.1/env/vars.sh >/dev/null 2>&1
source /opt/intel/oneapi/mkl/2026.1/env/vars.sh >/dev/null 2>&1
G=/tmp/claude-0/-opt-ai-deepseek-intel/6459cc39-7b1c-52fa-9fd2-b98c23e5cc62/scratchpad/gs
icpx -DGGML_SCHED_MAX_COPIES=4 -DGGML_SYCL_DNNL=0 -DGGML_SYCL_GRAPH -DGGML_SYCL_HOST_MEM_FALLBACK -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API \
  -DGGML_SYCL_WARP_SIZE=16 -DSTRATA_Q6K_EXPERTS=1 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -I$G/src -I$G/include -O2 -std=c++17 -fPIC -fsycl \
  -DMKL_ILP64 -fsyntax-only "${args[@]}" $f
