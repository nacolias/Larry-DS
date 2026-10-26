#!/bin/bash
# pp.sh - service off; decode host-time profile with/without card-0 graph replay; fastmv 32- vs 64-value R8 units;
# submit costs; service back on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
for c in "p0 GGML_SYCL_GRAPH_CACHE=0" "p1 GGML_SYCL_GRAPH_CACHE=1"; do
  set -- $c; N=$1; shift
  env "$@" STRATA_DEC_PROF=1 STRATA_XE_PROF=1 STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 192
  echo "== $N ($*)"; grep -h "decode step\|^decode:\|dec prof\|xe decode prof\|not recordable\|rc=" logs/$N.err | tail -6
done
source xe-env.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:3
echo "== fastmv 32-value units"; (cd bench/fm && ./fastmv_bench 300 | grep R8)
echo "== fastmv 64-value units"; (cd bench/fm && ./fastmv_bench64 300 | grep R8)
echo "== submit"; (cd bench && ./submit_bench 2000; ./launch_probe 2>&1 | tail -4)
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
