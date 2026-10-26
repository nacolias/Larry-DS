#!/bin/bash
# twoa.sh - wo_a per-slice reorder at load on / off (STRATA_PREREORDER_SLICES): the 31.6K prompt x3 each, then 8.5K x2
# each, with the dense half's own GPU time (STRATA_DENSE_PROF)
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_DENSE_PROF=1"
run() { N=$1; P=$2; shift 2; env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file $P -n 8 --prompt-batch 16384 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^prompt:" logs/$N.err | cut -c1-60) | $(grep -h "prompt split" logs/$N.err | tail -1 | cut -c1-40) | $(grep -h "dense prof prompt:" logs/$N.err | tail -1 | grep -o "gpu compute [0-9]*") $(grep -h "rc=" logs/$N.err | tail -1)"; }
for i in 1 2 3; do
  run wo31a$i p30k.i32 STRATA_PREREORDER_SLICES=1
  run wo31b$i p30k.i32 STRATA_PREREORDER_SLICES=0
done
for i in 1 2; do
  run wo8a$i p8k.i32 STRATA_PREREORDER_SLICES=1
  run wo8b$i p8k.i32 STRATA_PREREORDER_SLICES=0
done
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
