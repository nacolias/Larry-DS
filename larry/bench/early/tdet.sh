#!/bin/bash
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
pl() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 160; grep -h "^decode:" logs/$N.err | cut -c1-60; }
pl c0a STRATA_XE_CPU_ASYNC=0
pl c0b STRATA_XE_CPU_ASYNC=0
pl c1a STRATA_XE_CPU_ASYNC=1
pl c1b STRATA_XE_CPU_ASYNC=1
for p in "c0a c0b" "c1a c1b" "c0a c1a"; do set -- $p; cmp -s logs/$1.out logs/$2.out && echo "$1 == $2" || echo "$1 != $2 ($(cmp logs/$1.out logs/$2.out | head -1))"; done
for n in c0 c1; do cmp -s logs/$n.out logs/${n}a.out && echo "$n == ${n}a" || echo "$n != ${n}a"; done
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
