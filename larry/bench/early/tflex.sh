#!/bin/bash
# tflex.sh - promotion cache on the flex regions: plain / DSpark decode on vs off (x2), the 8.5K prompt
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_ENGRAM_REQUANT=q6_K STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_PROMOTE_MIN=2"
run() { N=$1; P=$2; T=$3; shift 3; env $B "$@" STRATA_DENSE_REQUANT=q6_K RUN_TIMEOUT=900 ./run.sh $N --ids-file $P -n $T
  echo "== $N ($*) $(grep -h "^prompt:" logs/$N.err | cut -c1-48) | $(grep -h "^decode:" logs/$N.err | cut -c1-60) $(grep -h "verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") | $(grep -h "set up in" logs/$N.err | tail -1 | cut -c10-60) $(grep -h "promotion cache:" logs/$N.err | tail -1 | cut -c25-90)"; }
for i in 1 2; do
  run f1$i pe.i32 400 STRATA_XE_PROMOTE=1
  run f0$i pe.i32 400 STRATA_XE_PROMOTE=0
  run g1$i pe.i32 400 STRATA_XE_PROMOTE=1 STRATA_DSPARK=$SIDECAR
  run g0$i pe.i32 400 STRATA_XE_PROMOTE=0 STRATA_DSPARK=$SIDECAR
done
run h1 p8k.i32 16 STRATA_XE_PROMOTE=1 STRATA_XE_PROF=1
grep -h "xe prefetch: layers\|xe prof (80" logs/h1.err | cut -c1-200
run h0 p8k.i32 16 STRATA_XE_PROMOTE=0
for n in f11 f01 g11 h1; do echo "-- $n: $(tr "\n" " " < logs/$n.out | python3 tok.py dec - 2>&1 | head -c 120 | tr "\n" " ")"; done
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
