#!/bin/bash
# tdprof.sh - plain decode with the host-phase profiler (STRATA_DEC_PROF) and the GPU timeline: where the host's time
# goes per decode layer (attention submit / readback wait / shexp / finish)
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
N=dprof1
env $B STRATA_DEC_PROF=1 RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 300 > /dev/null 2>&1
echo "== $N $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s")"
grep -h "dec prof\|decode prof\|dprof" logs/$N.err | tail -4
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
