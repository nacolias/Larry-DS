#!/bin/bash
# tmem.sh NAME [env...] - a one-shot run with the service env (DSpark, card-0 14 GiB, reserves), hidden memory sampled
cd /opt/ai/dsv41
N=$1; shift
ARGS=("$@")
source xe-env.sh
./memsample.sh > logs/$N.mem 2>&1 &
MP=$!
env STRATA_XE_PINNED_GIB=16 STRATA_XE_DENSE_GIB=${DG:-14} STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_DENSE_REQUANT=q6_K STRATA_ENGRAM_REQUANT=q6_K ${DSP-STRATA_DSPARK=$SIDECAR} STRATA_DSPARK_DEPTH=4 "${ARGS[@]}" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
kill $MP
echo "== $N ${ARGS[*]} $(grep -h "^decode:" logs/$N.err | cut -c1-60) | see logs/$N.mem"
grep -h "VRAM free\|GiB VRAM free" logs/$N.err | tail -4 | cut -c1-100
