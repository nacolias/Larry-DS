#!/bin/bash
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
env STRATA_DEC_PROF=1 STRATA_XE_PROF=1 STRATA_ENG_PROF=1 STRATA_DENSE_REQUANT=q6_K ./run.sh bud --ids-file p1.i32 -n 192
grep -h "decode step\|dec prof\|xe decode prof\|engram decode\|^decode:" logs/bud.err | tail -6
env STRATA_GPU_TL=1 STRATA_DENSE_REQUANT=q6_K ./run.sh budt --ids-file p1.i32 -n 192
grep -h "gpu timeline\|^decode:" logs/budt.err | tail -2
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
