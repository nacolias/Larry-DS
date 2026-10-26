#!/bin/bash
# tpb32.sh - the 31.6K prompt as one 32K batch (one pass over the RAM arena's experts) vs two 16K batches, warm
# page cache, Engram pool 256; the service's own runs have no enc-skip (--no-enc-skip here too) - and then with it
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENG_PROF=1 STRATA_XE_PROF=1 STRATA_ENGRAM_THREADS=256"
run() { N=$1; shift; env $B RUN_TIMEOUT=1800 ./run.sh $N --ids-file p30k.i32 -n 8 "$@" > /dev/null 2>&1
  echo "== $N ($*): $(grep -h "^prompt:" logs/$N.err | cut -c1-60) | $(grep -h "prompt split" logs/$N.err | tail -1 | cut -c1-50)"
  grep -h "engram prompt batch\|did not fit\|retrying\|graph allocation\|first batch" logs/$N.err | head -4 | sed "s/^/     /"
  grep -h "xe prof (.*chunk calls" logs/$N.err | tail -1 | cut -c1-160 | sed "s/^/     /"
  grep -h "xe prof: prefetch: staging" logs/$N.err | tail -1 | cut -c1-170 | sed "s/^/     /"; }
run pb16n --prompt-batch 16384 --no-enc-skip
run pb32n --prompt-batch 32768 --no-enc-skip
run pb16e --prompt-batch 16384
run pb32e --prompt-batch 32768
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
