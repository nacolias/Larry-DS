#!/bin/bash
# build_sdpa.sh - test_sdpa_dsv4 (the prompt attention as one oneDNN graph SDPA vs the oneMKL path), AOT for bmg-g31
cd "$(dirname "$(readlink -f "$0")")"
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
M=/opt/intel/oneapi/mkl/latest
icpx -fsycl -O3 -std=c++17 -DMKL_ILP64 -I$M/include -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
  test_sdpa_dsv4.cpp -o test_sdpa_dsv4 -L$M/lib -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -ldnnl > build_sdpa.log 2>&1
echo "build rc=$? $(grep -c -i ' error' build_sdpa.log) error lines"
