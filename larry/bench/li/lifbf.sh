#!/bin/bash
# lifbf.sh build|run - bench/li/lifbf_probe.cpp (the fused prompt indexer, mode 1 = fp16 copies of the keys a call, against
# mode 2 = bf16 in place: the same bits, and their times at 4K..512K keys): build = AOT for the B70 at nice 19 (outside a
# window); run = on card 1 (an abw.sh "@" step).  Output: bench/li/lifbf.out.
MODE=${1:-run}
set --   # (oneapi-vars.sh reads "$@")
source /opt/intel/oneapi/2026.1/oneapi-vars.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/li
case "$MODE" in
build)
  nice -n 19 icpx -fsycl -O2 -std=c++20 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    lifbf_probe.cpp -o lifbf_probe > lifbf_probe.build.log 2>&1
  echo "lifbf_probe rc=$? $(date +%T)"; grep -i "error\|spill" lifbf_probe.build.log | head -8 ;;
run)
  { echo "== lifbf_probe $(date +%T)"; ONEAPI_DEVICE_SELECTOR=level_zero:1 timeout 600 ./lifbf_probe; echo "rc=$?"; } > lifbf.out 2>&1
  cat lifbf.out ;;
esac
