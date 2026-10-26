#!/bin/bash
# trq.sh - F16 routers (STRATA_ROUTER_F16) and the Engram KV re-quantized (STRATA_ENGRAM_REQUANT): decode speed + text,
# prompt-path PPL (8.5K), decode-path PPL (2K, one token a step)
cd /opt/ai/dsv41
source xe-env.sh
run() { N=$1; P=$2; T=$3; shift 3; env "$@" STRATA_DENSE_REQUANT=q6_K RUN_TIMEOUT=900 ./run.sh $N --ids-file $P -n $T
  echo "== $N $P ($*) $(grep -h "^prompt:" logs/$N.err | cut -c1-40) | $(grep -h "^decode:" logs/$N.err | cut -c1-48) $(grep -h "dense requant" logs/$N.err | cut -c15-120) $(grep -h "rc=" logs/$N.err | tail -1)"; }
run e0 pe.i32 400 STRATA_ROUTER_F16=0
run e1 pe.i32 400 STRATA_ROUTER_F16=1
run e2 pe.i32 400 STRATA_ROUTER_F16=1 STRATA_ENGRAM_REQUANT=q6_K
run e3 pe.i32 400 STRATA_ROUTER_F16=1 STRATA_ENGRAM_REQUANT=q8_0
for n in e0 e1 e2 e3; do echo "-- $n: $(tr "\n" " " < logs/$n.out | python3 tok.py dec - 2>&1 | head -c 140 | tr "\n" " ")"; done
ppl() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K RUN_TIMEOUT=900 ./run.sh $N --ids-file p8k.i32 -n 1 --ppl
  echo "[$N $*] $(grep -h PPL logs/$N.out) $(grep -h "rc=" logs/$N.err | tail -1)"; }
ppl pr1 STRATA_ROUTER_F16=1
ppl pr2 STRATA_ROUTER_F16=1 STRATA_ENGRAM_REQUANT=q6_K
dppl() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1
  echo "[$N decode path $*] $(grep -h PPL logs/$N.out) $(grep -h "rc=" logs/$N.err | tail -1)"; }
dppl dr0 STRATA_ROUTER_F16=0
dppl dr1 STRATA_ROUTER_F16=1
dppl dr2 STRATA_ROUTER_F16=1 STRATA_ENGRAM_REQUANT=q6_K
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
