#!/bin/bash
# ttp4.sh - the split's host cost per step (STRATA_TP_PROF), no timeline stamps: plain decode TP=2 eager (cards'
# parts as graphs / direct) and TP=1
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=4.2 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
run() { N=$1; shift; env $B STRATA_DEC_PROF=1 STRATA_TP_PROF=1 "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 300 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s")"
  grep -h "dec prof" logs/$N.err | tail -1; grep -h "xe tp prof" logs/$N.err | tail -1; }
run tw2 STRATA_TP_OPROJ=2
run tw2n STRATA_TP_OPROJ=2 STRATA_TP_GRAPH=0
run tw1 STRATA_TP_OPROJ=1
run tw0 STRATA_TP_OPROJ=0
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
