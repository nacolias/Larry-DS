#!/bin/bash
# tb.sh - service off; fastmv Q6K_R8 kernel variants (generic / row-blocked R=2,4 / 256-register mode) by columns; on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:3
for v in "0 0 0" "2 2 0" "2 2 1" "4 4 1"; do set -- $v
  echo "== R1=$1 Rm=$2 BIG=$3"
  (cd bench/fm2 && GGML_SYCL_FASTMV_R8M1_R=$1 GGML_SYCL_FASTMV_R8M_R=$2 GGML_SYCL_FASTMV_R8M_BIG=$3 ./fastmv_bench_nc2 200 | grep "R8" | grep "ncols [1258]:")
done
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
