#!/bin/bash
# topk2.sh build|run - bench/li/topk2_test.cpp (patch-topk2.py's two-stage top-k against ggml-sycl's one-group radix
# select: the same sets, and their times): build = AOT for the B70 at nice 19 (outside a window); run = on card 1 with
# the clocks as the window has them (an abw.sh "@" step).  Output: bench/li/topk2.out.
MODE=${1:-run}
set --   # (oneapi-vars.sh reads "$@")
source /opt/intel/oneapi/2026.1/oneapi-vars.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/li
case "$MODE" in
build)
  nice -n 19 icpx -fsycl -O2 -std=c++20 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -I/opt/ai/dsv41/patches topk2_test.cpp -o topk2_test > topk2_test.build.log 2>&1
  echo "topk2_test rc=$? $(date +%T)"; grep -i "error\|spill" topk2_test.build.log | head -8 ;;
run)
  { echo "== topk2_test $(date +%T)"; ONEAPI_DEVICE_SELECTOR=level_zero:1 timeout 600 ./topk2_test 0; echo "rc=$?"; } > topk2.out 2>&1
  cat topk2.out ;;
esac
