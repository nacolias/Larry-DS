#!/bin/bash
# tx4.sh - service off; interleaved A/B on the long essay prompt (pe.i32, 400 tokens) and DSpark verify; service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
pl() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file pe.i32 -n 400; echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "rc=" logs/$N.err | tail -1)"; }
X0="GGML_SYCL_LARRY_NORM_SMALL=0 GGML_SYCL_DSV4_DEC_XMX=1 STRATA_RAW_F32=0"
X1="GGML_SYCL_LARRY_NORM_SMALL=1 GGML_SYCL_DSV4_DEC_XMX=1 STRATA_RAW_F32=1"
X2="GGML_SYCL_LARRY_NORM_SMALL=1 GGML_SYCL_DSV4_DEC_XMX=0 STRATA_RAW_F32=1"
X3="GGML_SYCL_LARRY_NORM_SMALL=1 GGML_SYCL_DSV4_DEC_XMX=0 STRATA_RAW_F32=0"
for r in a b; do pl x0$r $X0; pl x1$r $X1; pl x2$r $X2; pl x3$r $X3; done
dsp() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 ./run.sh $N --ids-file pe.i32 -n 400; echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -h "rc=" logs/$N.err | tail -1)"; }
dsp dx1 $X1
dsp dx2 $X2
for p in "x0a x3a" "x2a x3a"; do set -- $p; cmp -s logs/$1.out logs/$2.out && echo "$1 == $2" || echo "$1 != $2 ($(cmp logs/$1.out logs/$2.out | head -1))"; done
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
