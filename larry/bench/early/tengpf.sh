#!/bin/bash
# tengpf.sh - Engram next-batch prefetch on / off (STRATA_ENG_PREFETCH) on the 31.6K prompt x3 each, with the Engram
# phase times (STRATA_ENG_PROF) and the dense half's profile; the outputs must match
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_DENSE_PROF=1 STRATA_ENG_PROF=1"
run() { N=$1; P=$2; shift 2; env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file $P -n 8 --prompt-batch 16384 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^prompt:" logs/$N.err | cut -c1-60) | $(grep -h "prompt split" logs/$N.err | tail -1 | cut -c1-40) | $(grep -h "rc=" logs/$N.err | tail -1)"
  grep -h "engram prompt batch\|engram wait\|submit-lock" logs/$N.err | sed "s/^/     /"; }
for i in 1 2 3; do
  run ep31a$i p30k.i32 STRATA_ENG_PREFETCH=1
  run ep31b$i p30k.i32 STRATA_ENG_PREFETCH=0
done
for i in 1 2 3; do diff -q <(grep -v "^#" logs/ep31a$i.out) <(grep -v "^#" logs/ep31b1.out) > /dev/null && echo "ep31a$i output == ep31b1" || echo "ep31a$i output DIFFERS from ep31b1"; done
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
