#!/bin/bash
# tvp.sh - DSpark verify passes: the tier's host phases (STRATA_XE_PROF) + the timeline, no split, eager graphs
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
N=tvp1
env $B $D STRATA_XE_PROF=1 STRATA_GPU_TL=1 STRATA_DSPARK_PROF=1 RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
echo "== $N $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | tail -1)"
grep -h "xe decode prep\|xe small\|small prof\|decode phases\|verify phases\|dspark prof\|DSpark prof" logs/$N.err | tail -8
grep -h "gpu timeline" logs/$N.err | tail -1 | tr ";" "\n" | grep -v promotion
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
