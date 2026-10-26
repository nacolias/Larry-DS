#!/bin/bash
# tpc2.sh - the promotion cache, repeated runs: plain / DSpark decode, PROMOTE 0 / MIN 1 / MIN 2
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_ENGRAM_REQUANT=q6_K STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4"
run() { N=$1; P=$2; T=$3; shift 3; env $B "$@" STRATA_DENSE_REQUANT=q6_K RUN_TIMEOUT=900 ./run.sh $N --ids-file $P -n $T
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-110) $(grep -h "verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") | $(grep -h "promotion cache:" logs/$N.err | tail -1 | cut -c25-120)"; }
for i in 1 2; do
  run a0$i pe.i32 400 STRATA_XE_PROMOTE=0
  run a1$i pe.i32 400 STRATA_XE_PROMOTE=1 STRATA_XE_PROMOTE_MIN=1
  run a2$i pe.i32 400 STRATA_XE_PROMOTE=1 STRATA_XE_PROMOTE_MIN=2
  run b0$i pe.i32 400 STRATA_XE_PROMOTE=0 STRATA_DSPARK=$SIDECAR
  run b1$i pe.i32 400 STRATA_XE_PROMOTE=1 STRATA_XE_PROMOTE_MIN=1 STRATA_DSPARK=$SIDECAR
  run b2$i pe.i32 400 STRATA_XE_PROMOTE=1 STRATA_XE_PROMOTE_MIN=2 STRATA_DSPARK=$SIDECAR
done
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
