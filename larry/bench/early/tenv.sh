#!/bin/bash
# tenv.sh - service off; plain decode under driver / runtime env variants (copy engine use, graph replay); service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
pl() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 160; echo "== $N ($*)"; grep -h "^decode:\|rc=[^0]\|rror\|abort" logs/$N.err | cut -c1-70 | tail -2; }
pl e0
pl e1 UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE=0
pl e2 UR_L0_USE_COPY_ENGINE=0
pl e3 GGML_SYCL_GRAPH_CACHE=1
pl e4 GGML_SYCL_GRAPH_CACHE=1 UR_L0_USE_COPY_ENGINE=0
pl e5
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
