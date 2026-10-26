#!/bin/bash
# tl16.sh - service off; the decode expert kernels with 4 vs 16 lanes per row (STRATA_XE_LPR); service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
pl() { N=$1; P=$2; T=$3; shift 3; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file $P -n $T; echo "== $N $P ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "rc=" logs/$N.err | tail -1)"; }
for r in a b c; do pl l4$r p1.i32 160 STRATA_XE_LPR=4; pl l16$r p1.i32 160 STRATA_XE_LPR=16; done
pl l4e pe.i32 400 STRATA_XE_LPR=4; pl l16e pe.i32 400 STRATA_XE_LPR=16
env STRATA_GPU_TL=1 STRATA_XE_LPR=16 STRATA_DENSE_REQUANT=q6_K ./run.sh l16t --ids-file pe.i32 -n 300; grep -h "gpu timeline" logs/l16t.err | tail -1
env STRATA_GPU_TL=1 STRATA_XE_LPR=4 STRATA_DENSE_REQUANT=q6_K ./run.sh l4t --ids-file pe.i32 -n 300; grep -h "gpu timeline" logs/l4t.err | tail -1
tr "\n" " " < logs/l16a.out | python3 tok.py dec - 2>&1 | head -c 400; echo
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
