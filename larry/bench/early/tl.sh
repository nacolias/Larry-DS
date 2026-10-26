#!/bin/bash
# tl.sh - service off; the decode layer's device timeline (STRATA_GPU_TL) eager vs graph replay; service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
run() { N=$1; shift; env "$@" STRATA_GPU_TL=1 STRATA_DEC_PROF=1 STRATA_XE_PROF=1 STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 160
  echo "== $N ($*)"; grep -h "gpu timeline\|dec prof\|xe decode prof\|^decode:\|rc=\|abort\|rror" logs/$N.err | tail -6; }
run t1
run t2 GGML_SYCL_GRAPH_CACHE=1
run t3 STRATA_DENSE_ASYNC=0
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
