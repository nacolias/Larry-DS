#!/bin/bash
# cc.sh FILE [extra flags] - syntax/semantic check of one engine source with Larry's flags (host + SYCL device front end)
f=$1; shift
args=("$@")
set --
source /opt/intel/oneapi/compiler/2026.1/env/vars.sh >/dev/null 2>&1
R=/tmp/claude-0/-opt-ai-deepseek-intel/6459cc39-7b1c-52fa-9fd2-b98c23e5cc62/scratchpad/dr/xolotl
icpx -DGGML_USE_CPU -DGGML_USE_SYCL -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_VERSION=\"0.1.44.1\" -DDS41_HAVE_DSPARK -DDS41_XE \
  -DSTRATA_DSV41_FQ_OP -DSTRATA_DSV41_HC_MIX -I$R/tools/v41/engine/xe -I$R/include -I$R/tools/v41/engine -I$R/third_party/llama.cpp/ggml/src \
  -I$R/third_party/llama.cpp/ggml/include -O2 -std=c++20 -fPIC -Wall -Wextra -fsyntax-only "${args[@]}" $f
