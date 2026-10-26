#!/bin/bash
# build.sh - li_probe (B packed) and li_probe_cm (B col_major), AOT for the B70, nice 19
set --
source /opt/intel/oneapi/2026.1/oneapi-vars.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/li
for V in "1 li_probe" "0 li_probe_cm"; do
  set -- $V
  nice -n 19 icpx -fsycl -O2 -Wno-deprecated-declarations -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -DLI_PACKED=$1 li_probe.cpp -DMKL_ILP64 -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -ltbb -o $2 > $2.build.log 2>&1
  echo "$2 rc=$? $(date +%T)"; grep -i "error\|spill\|warning: in kernel" $2.build.log | head -8
done
