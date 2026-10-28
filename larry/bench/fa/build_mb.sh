#!/bin/bash
# build_mb.sh - mqa_mb (the fused attention's inner-loop micro-benchmarks), AOT for bmg-g31
cd "$(dirname "$(readlink -f "$0")")"
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
icpx -fsycl -O3 -std=c++17 -Wno-deprecated-declarations -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
  mqa_mb.cpp -o mqa_mb > build_mb.log 2>&1
echo "build rc=$? $(grep -c -i ' error' build_mb.log) error lines"
