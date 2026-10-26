#!/bin/bash
# tg.sh - service off; the shared expert's fused gate+up (STRATA_SHEXP_GU) on the reference prompt (p1, x3) and the
# essay (pe), plain and DSpark; service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
pl() { N=$1; P=$2; T=$3; shift 3; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file $P -n $T; echo "== $N $P ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "rc=" logs/$N.err | tail -1)"; }
for r in a b c; do pl g0$r p1.i32 160 STRATA_SHEXP_GU=0; pl g1$r p1.i32 160 STRATA_SHEXP_GU=1; done
pl g0e pe.i32 400 STRATA_SHEXP_GU=0; pl g1e pe.i32 400 STRATA_SHEXP_GU=1
grep -h "shared expert:" logs/g1a.err | head -1
dsp() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 ./run.sh $N --ids-file pe.i32 -n 400; echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -h "rc=" logs/$N.err | tail -1)"; }
dsp dg0 STRATA_SHEXP_GU=0
dsp dg1 STRATA_SHEXP_GU=1
tr "\n" " " < logs/g1e.out | python3 tok.py dec - 2>&1 | head -c 500; echo
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
