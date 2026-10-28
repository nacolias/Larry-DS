#!/bin/bash
# build_blk.sh - test_sdpa_blk (block-union SDPA timing), AOT for bmg-g31
cd "$(dirname "$(readlink -f "$0")")"
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
icpx -fsycl -O3 -std=c++17 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
  test_sdpa_blk.cpp -o test_sdpa_blk -ldnnl > build_blk.log 2>&1
echo "build rc=$? $(grep -c -i ' error' build_blk.log) error lines"
