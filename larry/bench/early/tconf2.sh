#!/bin/bash
# tconf.sh - DSpark confidence cut (STRATA_DSPARK_CONF) with the cheaper draft: decode tok/s, tokens/pass, verify and
# draft ms/pass on two prompts (essay pe.i32 -n 400, p1.i32 -n 160), clocks pinned
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_DENSE_TYPE=q6_K STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
run() { N=$1; P=$2; T=$3; shift 3; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file $P -n $T > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ")"; }
for c in 0.1 0.2 0.3; do
  for i in 1 2; do
    run cfe$i-$c pe.i32 400 STRATA_DSPARK_CONF=$c
    run cfp$i-$c p1.i32 160 STRATA_DSPARK_CONF=$c
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
