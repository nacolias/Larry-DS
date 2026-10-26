#!/bin/bash
# tcomb.sh - STRATA_XE_COMBINE 0 / 1 / 2 on the 31.6K prompt (warm, Engram pool 256), then prompt-path PPL (2048
# tokens, one layer-major batch) for 0 / 2: fp16 parts must stay in the noise (+-0.002)
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_XE_PROF=1 STRATA_ENGRAM_THREADS=256"
for m in 0 2 1 0 2; do
  N=cm$m$RANDOM
  env $B STRATA_XE_COMBINE=$m RUN_TIMEOUT=1500 ./run.sh $N --ids-file p30k.i32 -n 8 --prompt-batch 16384 > /dev/null 2>&1
  echo "== COMBINE=$m: $(grep -h "^prompt:" logs/$N.err | cut -c1-60) | $(grep -h "xe prof (.*chunk calls" logs/$N.err | tail -1 | grep -o "wall [0-9]* ms = prep [0-9]*\|cards [0-9]* + combine [0-9]*" | tr "\n" " ") $(grep -h "combine slots" logs/$N.err | head -1)"
  grep -v "^#" logs/$N.out | head -c 200 | tr "\n" " "; echo
done
for m in 0 2; do
  N=cpl$m
  env $B STRATA_XE_COMBINE=$m RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 2048 > /dev/null 2>&1
  echo "[ppl COMBINE=$m] $(grep -h PPL logs/$N.out)"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
