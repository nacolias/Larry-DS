#!/bin/bash
# tn.sh - service off; plain decode with the RMS norm work-group at 1024 (old) / 256 / 128, twice each; service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41 2>/dev/null; sleep 3
f0=$(dmesg | grep -c "CAT error")
pl() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 160; echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "rc=" logs/$N.err | tail -1)"; }
for r in a b; do
  pl n0$r GGML_SYCL_LARRY_NORM_WG=0
  pl n2$r GGML_SYCL_LARRY_NORM_WG=256
  pl n1$r GGML_SYCL_LARRY_NORM_WG=128
done
echo "new GPU CAT errors: $(( $(dmesg | grep -c "CAT error") - f0 ))"
tr "\n" " " < logs/n2a.out | python3 tok.py dec - 2>&1 | head -c 300; echo
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
