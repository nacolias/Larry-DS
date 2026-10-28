#!/bin/bash
# build_fsa.sh - test_fsa (the fused prompt attention vs the service's KGATHER + oneDNN SDPA path), AOT for bmg-g31
cd "$(dirname "$(readlink -f "$0")")"
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
icpx -fsycl -O3 -std=c++17 -Wno-deprecated-declarations -Wno-ignored-attributes -fsycl-targets=spir64_gen \
  -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
  test_fsa.cpp -o test_fsa -ldnnl > build_fsa.log 2>&1
echo "build rc=$? $(grep -c -i ' error' build_fsa.log) error lines"
grep -i -E "spill|regusage|GRF" build_fsa.log | head -20
