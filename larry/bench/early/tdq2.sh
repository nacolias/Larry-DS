#!/bin/bash
# tdq2.sh - xe_dqgemm v2 (B in registers, wide-load decoders, transposed A) in the engine: 8.5K prompt speed + tier
# profile, last-row logits vs the references, perplexity (v1 PPL 1.6783, oneMKL path 1.6794)
cd /opt/ai/dsv41
run() { N=$1; A=$2; shift 2; env "$@" STRATA_DENSE_REQUANT=q6_K RUN_TIMEOUT=900 ./run.sh $N --ids-file p8k.i32 -n 1 $A
  echo "== $N ($A $*) $(grep -h "^prompt:" logs/$N.err | cut -c1-48) | $(grep -h "^prompt split" logs/$N.err) $(grep -h PPL logs/$N.out) $(grep -h "rc=" logs/$N.err | tail -1)"; }
run r1 "--dump-last logs/r1.f32" STRATA_XE_DQGEMM=1
run r2 "--dump-last logs/r2.f32" STRATA_XE_DQGEMM=1
run rp "--dump-last logs/rp.f32" STRATA_XE_DQGEMM=1 STRATA_XE_PROF=1
grep -h "xe prof (" logs/rp.err | cut -c1-260
python3 lcmp.py logs/la.f32 logs/lb.f32 logs/q1.f32 logs/r1.f32 logs/r2.f32 logs/rp.f32
run rq "--ppl" STRATA_XE_DQGEMM=1
grep -h "error\|fail" logs/r1.err | head -3
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
