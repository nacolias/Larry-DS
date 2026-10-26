#!/bin/bash
# tli2.sh - after the build with patch-lixmx.py: the fused XMX indexer for decode / verify (GGML_SYCL_LI_XMX=1, on top
# of GGML_SYCL_LI_GEMM_MIN=1).  1) a check run (GGML_SYCL_LI_XMX_CHECK=1: 32 calls compared with li_gemm - the
# differences must be f32-summation-order small, no inf mismatch); 2) ~95K prompt, DSpark, 200 tokens: gemm / xmx.
cd /opt/ai/dsv41
source xe-env.sh
[ -f p95k.i32 ] || cat p30k.i32 p30k.i32 p30k.i32 > p95k.i32
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
N=lixck
env $B $D GGML_SYCL_LI_XMX=1 GGML_SYCL_LI_XMX_CHECK=1 RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 60 > /dev/null 2>&1
echo "[check] $(grep -h "^rc=" logs/$N.err)"; grep -h "li_xmx check" logs/$N.err | head -32
for x in 0 1 0 1; do
  N=lix$x$RANDOM
  env $B $D GGML_SYCL_LI_XMX=$x RUN_TIMEOUT=1800 ./run.sh $N --ids-file p95k.i32 -n 200 --ctx 131072 --prompt-batch 16384 > /dev/null 2>&1
  echo "== LI_XMX=$x: $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
