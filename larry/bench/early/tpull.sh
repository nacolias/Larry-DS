#!/bin/bash
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_GPU_TL=1 STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0 STRATA_DSPARK_DENSE_TYPE=q6_K STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 300 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-46) | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass") $(grep -h "pass_v split" logs/$N.err | grep -o "[0-9.]* tokens each")"
  grep -h "gpu timeline" logs/$N.err | tail -1 | grep -o "8->9 [0-9.]*\|9->10 [0-9.]*\|10->16 [0-9.]*\|12->8 [0-9.]*\|16->11 [0-9.]*" | tr "\n" " "; echo; }
run pl0 STRATA_XE_PULL=0
run pl1 STRATA_XE_PULL=1
/usr/local/sbin/larry-gpu-clock restore > /dev/null
