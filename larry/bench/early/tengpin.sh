#!/bin/bash
# tengpin.sh - Engram pinned rows + deferred prefetch (default) vs both off: a 2K output check, then the 31.6K prompt
# x3 each with the Engram phase times; outputs must match
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_DENSE_PROF=1 STRATA_ENG_PROF=1"
run() { N=$1; P=$2; shift 2; env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file $P -n 8 --prompt-batch 16384 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^prompt:" logs/$N.err | cut -c1-60) | $(grep -h "prompt split" logs/$N.err | tail -1 | cut -c1-40) | $(grep -h "rc=" logs/$N.err | tail -1)"
  grep -h "engram prompt batch\|engram wait\|submit-lock\|Engram pinned" logs/$N.err | sed "s/^/     /"; }
env $B RUN_TIMEOUT=900 ./run.sh pin2k --ids-file p2k.i32 -n 2 > /dev/null 2>&1
diff -q <(grep -v "^#" logs/dbgc-pre.out) <(grep -v "^#" logs/pin2k.out) > /dev/null && echo "pin2k output == dbgc-pre" || echo "pin2k output DIFFERS"
grep -h "^prompt:\|engram prompt batch" logs/pin2k.err
for i in 1 2 3; do
  run pn31a$i p30k.i32
  run pn31b$i p30k.i32 STRATA_ENG_PIN=0 STRATA_ENG_PREFETCH=0
done
for i in 1 2 3; do for j in a b; do diff -q <(grep -v "^#" logs/pn31$j$i.out) <(grep -v "^#" logs/ep31b1.out) > /dev/null && echo "pn31$j$i output == ep31b1" || echo "pn31$j$i output DIFFERS from ep31b1"; done; done
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
