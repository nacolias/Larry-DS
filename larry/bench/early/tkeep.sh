#!/bin/bash
# tkeep.sh - xe_dqgemm v3 + STRATA_XE_KEEP_STREAM: 8.5K prompt (keep on / off, profiled), logits, perplexity
cd /opt/ai/dsv41
run() { N=$1; A=$2; shift 2; env "$@" STRATA_DENSE_REQUANT=q6_K RUN_TIMEOUT=900 ./run.sh $N --ids-file p8k.i32 -n 1 $A
  echo "== $N ($A $*) $(grep -h "^prompt:" logs/$N.err | cut -c1-48) | $(grep -h "^prompt split" logs/$N.err) $(grep -h PPL logs/$N.out) $(grep -h "rc=" logs/$N.err | tail -1)"; }
run k1 "--dump-last logs/k1.f32" STRATA_XE_KEEP_STREAM=1 STRATA_XE_PROF=1
grep -h "xe prof (" logs/k1.err | cut -c1-330
run k0 "--dump-last logs/k0.f32" STRATA_XE_KEEP_STREAM=0 STRATA_XE_PROF=1
grep -h "xe prof (" logs/k0.err | cut -c1-330
run k2 "--dump-last logs/k2.f32" STRATA_XE_KEEP_STREAM=1
run k3 "--dump-last logs/k3.f32" STRATA_XE_KEEP_STREAM=1
python3 lcmp.py logs/la.f32 logs/lb.f32 logs/r2.f32 logs/k1.f32 logs/k0.f32 logs/k2.f32 logs/k3.f32
run kq "--ppl" STRATA_XE_KEEP_STREAM=1
grep -h "error\|fail\|moved" logs/k1.err logs/k2.err | head -3
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
