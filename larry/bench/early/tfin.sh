#!/bin/bash
# tfin.sh - service off; the current defaults: reference prompt (p1) x4, essay x2, DSpark essay x2; service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
pl() { N=$1; P=$2; T=$3; shift 3; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file $P -n $T; echo "== $N $P ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "rc=" logs/$N.err | tail -1)"; }
for r in a b c d; do pl f$r p1.i32 160; done
pl fe1 pe.i32 400; pl fe2 pe.i32 400
dsp() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 ./run.sh $N --ids-file pe.i32 -n 400; echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -h "rc=" logs/$N.err | tail -1)"; }
dsp fd1; dsp fd2
tr "\n" " " < logs/fa.out | python3 tok.py dec - 2>&1 | head -c 300; echo
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
