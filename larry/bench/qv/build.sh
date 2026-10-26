#!/bin/bash
# build.sh - tp_layer (step 0 of the new engine) against the engine's fastmv.cpp, AOT for the B70 (bmg-g31)
cd "$(dirname "$(readlink -f "$0")")"
source /opt/ai/dsv41/xe-env.sh
F=/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl
cp $F/fastmv.cpp $F/fastmv.hpp .
icpx -fsycl -O3 -std=c++17 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" tp_layer.cpp fastmv.cpp -o tp_layer > build.log 2>&1
echo "build rc=$? $(grep -c -i error build.log) error lines"
