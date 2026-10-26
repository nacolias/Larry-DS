#!/bin/bash
# ttp5.sh - DSpark (essay, 400 tokens) with the attention output split (STRATA_TP_OPROJ=2, eager graphs) vs without,
# twice each; the split's 620 MB a card needs the 4.2 GiB reserve: the baseline at 3.5 (service) and 4.2
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
run() { N=$1; shift; env $B $D "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "experts .* hit" logs/$N.err | grep -o "[0-9]* cpu" | tail -1) $(grep -h "not split\|split [0-9]* calls" logs/$N.err | sort | uniq -c | tr "\n" " ")"; }
for i in 1 2; do
  run tv35_$i STRATA_XE_RESERVE_GIB=3.5 STRATA_TP_OPROJ=0
  run tv42_$i STRATA_XE_RESERVE_GIB=4.2 STRATA_TP_OPROJ=0
  run tvtp_$i STRATA_XE_RESERVE_GIB=4.2 STRATA_TP_OPROJ=2
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
