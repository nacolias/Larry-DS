#!/bin/bash
# ths.sh - service off; host slots (STRATA_XE_HSLOTS) and the 256-register multi-token gate/up (STRATA_XE_GU_BIG); on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
pl() { N=$1; P=$2; T=$3; shift 3; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file $P -n $T; echo "== $N $P ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "rc=" logs/$N.err | tail -1)"; }
for r in a b c; do pl r0$r p1.i32 160 STRATA_PREREORDER=0; pl r1$r p1.i32 160 STRATA_PREREORDER=1; done
pl r0e pe.i32 400 STRATA_PREREORDER=0; pl r1e pe.i32 400 STRATA_PREREORDER=1
dsp() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 ./run.sh $N --ids-file pe.i32 -n 400; echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -h "pass_v" logs/$N.err | grep -o "tier [0-9.]* ms") $(grep -h "rc=" logs/$N.err | tail -1)"; }
dsp dr0 STRATA_PREREORDER=0

dsp dr1 STRATA_PREREORDER=1
env STRATA_GPU_TL=1 STRATA_DENSE_REQUANT=q6_K ./run.sh r1t --ids-file pe.i32 -n 300; grep -h "gpu timeline" logs/r1t.err | tail -1
tr "\n" " " < logs/r1a.out | python3 tok.py dec - 2>&1 | head -c 300; echo
tr "\n" " " < logs/dr1.out | python3 tok.py dec - 2>&1 | head -c 300; echo
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
