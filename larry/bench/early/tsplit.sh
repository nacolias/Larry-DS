#!/bin/bash
# tsplit.sh - the Engram KV (Q6_K) split over the cards (STRATA_HEAD_SPLIT_MB=100) vs not; CPU-pool experts skipped
# (STRATA_XE_CPU_SKIP=1, DEBUG: wrong output) as the bound on keeping them on cards - plain and DSpark decode
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_ROUTER_F16=1 STRATA_ENGRAM_REQUANT=q6_K STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4"
run() { N=$1; P=$2; T=$3; shift 3; env $B "$@" STRATA_DENSE_REQUANT=q6_K RUN_TIMEOUT=900 ./run.sh $N --ids-file $P -n $T
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-48) $(grep -h "verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -h "rc=" logs/$N.err | tail -1)"; }
run s2 pe.i32 400 STRATA_HEAD_SPLIT_MB=200
run s1 pe.i32 400 STRATA_HEAD_SPLIT_MB=100
run s1b pe.i32 400 STRATA_HEAD_SPLIT_MB=100
run c1 pe.i32 400 STRATA_HEAD_SPLIT_MB=100 STRATA_XE_CPU_SKIP=1
run d1 pe.i32 400 STRATA_HEAD_SPLIT_MB=100 STRATA_DSPARK=$SIDECAR
run d2 pe.i32 400 STRATA_HEAD_SPLIT_MB=200 STRATA_DSPARK=$SIDECAR
run dc pe.i32 400 STRATA_HEAD_SPLIT_MB=100 STRATA_DSPARK=$SIDECAR STRATA_XE_CPU_SKIP=1
for n in s2 s1 d1; do echo "-- $n: $(tr "\n" " " < logs/$n.out | python3 tok.py dec - 2>&1 | head -c 120 | tr "\n" " ")"; done
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
