#!/bin/bash
# build-xe.sh - build the DeepSeek-V4.1 engine with the Intel multi-card tier (ds41_generate_xe + libds41_xe.so)
# against ggml's SYCL backend (Intel Arc Pro B70: AOT for bmg-g31), with Xolotl's fused DSV4.1 ops on SYCL.
#   BUILD_DIR (build-xe)  TARGETS (ds41_generate_xe test_ds41_xe)  EXTRA_CMAKE (more -D flags)  NINJA_JOBS (all cores;
#   fewer when RAM is short: a SYCL compile job can take 2-3 GB)
#   XE_ARCH (bmg-g31): the AOT device of ggml-sycl and the tier (another Arc: its ocloc name; untested on larry)
# A change to the tier rebuilds only libds41_xe.so (~10 s); anything that re-links the engine pays ggml-sycl's device
# link (~10 min).
set -e
D=${DS41_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
set -- ; source "${ONEAPI_VARS:-/opt/intel/oneapi/2026.1/oneapi-vars.sh}" >/dev/null
cd "$D/xolotl"
B=${BUILD_DIR:-build-xe}
A=${XE_ARCH:-bmg-g31}
cmake -S . -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DEVICE_ARCH=$A -DSTRATA_XE_AOT=$A \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON ${EXTRA_CMAKE}
ninja -C $B ${NINJA_JOBS:+-j$NINJA_JOBS} ${TARGETS:-ds41_generate_xe test_ds41_xe}
