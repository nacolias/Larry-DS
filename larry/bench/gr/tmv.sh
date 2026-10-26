#!/bin/bash
# tmv.sh - the latency-bound decode / verify mat-vecs under the split-K knobs (patch-mvsplit.py: GGML_SYCL_FASTMV_UPL,
# _SG): bench/fm2's cold (weights rotated past the L2) and multi-column benches, built here against the ENGINE's
# fastmv.cpp (patched), with two shapes added - q_a+kv fused 5120x1792 (STRATA_QKV_FUSE) and the F16 router 5120x384.
# One card, GPU clocks pinned; the service must be stopped.
cd "$(dirname "$(readlink -f "$0")")"
source /opt/ai/dsv41/xe-env.sh
F=/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl
AOT='-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31"'
for b in fastmv_bench_cold_x fastmv_bench_nc_x; do
  eval icpx -fsycl -O3 -std=c++17 $AOT -I$F $b.cpp $F/fastmv.cpp -o $b > $b.build.log 2>&1 || { echo "build $b failed"; tail -5 $b.build.log; exit 1; }
done
/usr/local/sbin/larry-gpu-clock pin > /dev/null
SH="shexp d R8|q_a R8|kv R8|qkv R8|router F16|shexp gu R8"
for cfg in "2 4096" "3 4096" "4 4096" "2 2048" "3 2048" "2 8192"; do
  set -- $cfg
  echo "== GGML_SYCL_FASTMV_UPL=$1 GGML_SYCL_FASTMV_SG=$2"
  GGML_SYCL_FASTMV_UPL=$1 GGML_SYCL_FASTMV_SG=$2 ./fastmv_bench_cold_x 200 2>&1 | grep -E "$SH"
  GGML_SYCL_FASTMV_UPL=$1 GGML_SYCL_FASTMV_SG=$2 ./fastmv_bench_nc_x 200 2>&1 | grep -E "$SH" | grep -E "ncols [2-5]"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
