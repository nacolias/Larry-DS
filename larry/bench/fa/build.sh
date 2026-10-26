#!/bin/bash
# build.sh - test_mqa_pf (the fused prompt attention vs the oneMKL path), AOT for the B70 (bmg-g31)
cd "$(dirname "$(readlink -f "$0")")"
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
M=/opt/intel/oneapi/mkl/latest
icpx -fsycl -O3 -std=c++17 -DMKL_ILP64 -I$M/include -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
  -I. test_mqa_pf.cpp -o test_mqa_pf -L$M/lib -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core > build.log 2>&1
echo "build rc=$? $(grep -c -i ' error' build.log) error lines"
