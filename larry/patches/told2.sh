#!/bin/bash
# told2.sh - service off; one binary, two configurations: OLD (every switch added since the goal started turned off,
# the goal-start defaults --pf-b 1.43 / D3 prediction on / --prompt-batch 8192) vs NEW (today's defaults); plain + DSpark
# decode, an 8.5K and a 31.6K prompt
cd /opt/ai/dsv41
source xe-env.sh
OLD="STRATA_XE_PLANAR=0 STRATA_XE_SYNC=1 STRATA_SHEXP_INLINE=1 GGML_SYCL_DSV4_DEC=0 GGML_SYCL_LARRY_Q6R8=0 GGML_SYCL_LARRY_WOA=0 GGML_SYCL_LARRY_FASTMV=0 GGML_SYCL_GRAPH_CACHE=0 STRATA_XE_GRAPH=0 STRATA_ROUTER_FUSED=0 STRATA_PASS_CACHE=0 STRATA_DENSE_ASYNC=0 STRATA_XE_XDIRECT=0 STRATA_XE_CPU_ASYNC=0 GGML_SYCL_FASTMV_R8M=0 GGML_SYCL_LARRY_QREUSE=0 GGML_SYCL_LARRY_CPYBATCH=0 STRATA_ENG_ASYNC=0 GGML_SYCL_LARRY_NORM_SMALL=0 STRATA_RAW_F32=0 STRATA_SHEXP_GU=0 STRATA_HEAD_SPLIT=0 STRATA_PREREORDER=0 STRATA_XE_GU_BIG=0 STRATA_D3_PF=1 STRATA_KGATHER=0 GGML_SYCL_LARRY_F16GEMM=0 STRATA_XE_PREFETCH=0 STRATA_PIPE=0"
OLDARGS="--pf-b 1.43 --prompt-batch 8192"
NEW="STRATA_D3_PF=0"
run() { N=$1; P=$2; T=$3; CFG=$4; A=$5; shift 5; env $CFG "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file $P -n $T $A
  echo "== $N $P $(grep -h "^prompt:" logs/$N.err | cut -c1-48) | $(grep -h "^decode:" logs/$N.err | cut -c1-45) $(grep -h "verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -h "rc=" logs/$N.err | tail -1)"; }
for r in a b; do run o$r p1.i32 160 "$OLD" "$OLDARGS"; run n$r p1.i32 160 "$NEW" ""; done
run oe pe.i32 400 "$OLD" "$OLDARGS"; run ne pe.i32 400 "$NEW" ""
run od p1.i32 160 "$OLD" "$OLDARGS" STRATA_DSPARK=$SIDECAR; run nd p1.i32 160 "$NEW" "" STRATA_DSPARK=$SIDECAR
run ode pe.i32 400 "$OLD" "$OLDARGS" STRATA_DSPARK=$SIDECAR; run nde pe.i32 400 "$NEW" "" STRATA_DSPARK=$SIDECAR
run o8 p8k.i32 8 "$OLD" "$OLDARGS"; run n8 p8k.i32 8 "$NEW" ""
run o30 p30k.i32 8 "$OLD" "$OLDARGS"; run n30 p30k.i32 8 "$NEW" ""
for n in oa na ode nde o8 n8 o30 n30; do echo "-- $n: $(tr "\n" " " < logs/$n.out | python3 tok.py dec - 2>&1 | head -c 120 | tr "\n" " ")"; done
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
