#!/bin/bash
# tn.sh - service off; plain decode with the RMS norm work-group at 1024 (old) / 256 / 128, twice each; service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41 2>/dev/null; sleep 3
f0=$(dmesg | grep -c "CAT error")
pl() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 160; echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "rc=" logs/$N.err | tail -1)"; }
for r in a b; do
  pl m0$r GGML_SYCL_LARRY_NORM_SMALL=0
  pl m1$r GGML_SYCL_LARRY_NORM_SMALL=1
  pl m2$r GGML_SYCL_LARRY_NORM_SMALL=1 GGML_SYCL_DSV4_DEC_XMX=0
done
dsp() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 ./run.sh $N --ids-file p1.i32 -n 300; echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass") $(grep -h "rc=" logs/$N.err | tail -1)"; }
source xe-env.sh
dsp dm1 GGML_SYCL_LARRY_NORM_SMALL=1
dsp dm2 GGML_SYCL_LARRY_NORM_SMALL=1 GGML_SYCL_DSV4_DEC_XMX=0
echo "CAT errors since start: $(dmesg -T | grep "CAT error" | awk -v t="$(date -d "-40 min" "+%H:%M")" "{print \$4}" | sort -u | tail -3 | tr "\n" " ")"
tr "\n" " " < logs/m1a.out | python3 tok.py dec - 2>&1 | head -c 300; echo
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
