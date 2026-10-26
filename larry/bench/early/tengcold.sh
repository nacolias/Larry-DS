#!/bin/bash
# tengcold.sh - the Engram pinned rows + deferred prefetch (default) vs both off with a COLD page cache (dropped before
# each run's prompt: the realistic case for a prompt whose n-grams were never read); the 31.6K prompt x2 each
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_DENSE_PROF=1 STRATA_ENG_PROF=1"
run() { N=$1; P=$2; shift 2
  env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file $P -n 8 --prompt-batch 16384 > /dev/null 2>&1 &
  # drop the page cache once the model is loaded (the arena is filled), right before the prompt starts
  for i in $(seq 1 900); do grep -q "^loaded in" logs/$N.err 2>/dev/null && break; sleep 0.2; done
  sync; echo 3 > /proc/sys/vm/drop_caches
  wait
  echo "== $N ($*) $(grep -h "^prompt:" logs/$N.err | cut -c1-60) | $(grep -h "prompt split" logs/$N.err | tail -1 | cut -c1-40) | $(grep -h "rc=" logs/$N.err | tail -1)"
  grep -h "engram prompt batch\|engram wait\|submit-lock" logs/$N.err | sed "s/^/     /"; }
for i in 1 2; do
  run cd31a$i p30k.i32
  run cd31b$i p30k.i32 STRATA_ENG_PIN=0 STRATA_ENG_PREFETCH=0
done
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
