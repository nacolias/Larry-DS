#!/bin/bash
# tengthr.sh - cold Engram reads (page cache dropped) of the 31.6K prompt's first batch vs the reader's pool size
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENG_PROF=1"
for T in 32 128 64 256; do
  N=eth$T
  sync; echo 3 > /proc/sys/vm/drop_caches
  env $B STRATA_ENGRAM_THREADS=$T RUN_TIMEOUT=1500 ./run.sh $N --ids-file p30k.i32 -n 8 --prompt-batch 16384 > /dev/null 2>&1
  echo "== threads $T: $(grep -h "^prompt:" logs/$N.err | cut -c1-60) | $(grep -h "engram prompt batch" logs/$N.err | head -1)"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
