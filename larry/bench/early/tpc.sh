#!/bin/bash
# tpc.sh - promotion chunk size (STRATA_XE_PROMOTE_CHUNK_MB 1 / 4 / 8): CPU-pool experts and speed, DSpark cut 0.3,
# essay prompt x2 each, clocks pinned
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PROF=1 STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* hit / [0-9]* cpu") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ")"
  grep -h "promotion cache: [0-9]* promoted\|xe verify prof" logs/$N.err | tail -2 | cut -c1-170 | sed "s/^/     /"; }
for i in 1 2; do
  run pc1$i STRATA_XE_PROMOTE_CHUNK_MB=1
  run pc4$i STRATA_XE_PROMOTE_CHUNK_MB=4
  run pc8$i STRATA_XE_PROMOTE_CHUNK_MB=8
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
