#!/bin/bash
# tlppl-shexp.sh - numerics of STRATA_SHEXP_TIER=1 (the shared expert of prompt calls on the expert cards, patch-
# shexptier.py / -dense / -gen): one-shot prompt-path runs with the service's environment (its exports: the TP core
# on) - p2k at --prompt-batch 2048 (two 1K parts) and p8k at 16384 (one 8.5K batch, two parts), twice without (the
# run-to-run floor) and once with; lkl.py: PPL, per-row KL against the first run, top-1 agreement.  The service must be
# stopped (abw.sh probe).
cd /opt/ai/dsv41
source xe-env.sh
eval "$(grep '^export ' ds41-xe-engine.sh)"
unset STRATA_SERVE_DSPARK
for P in p2k p8k; do
  PBS=2048; [ $P = p8k ] && PBS=16384
  for R in a b s; do
    N=sh_${P}_$R
    E="STRATA_SHEXP_TIER=0"; [ $R = s ] && E="STRATA_SHEXP_TIER=1"
    env $E RUN_TIMEOUT=1200 ./run.sh $N --ids-file $P.i32 -n 1 --ppl --prompt-batch $PBS --dump-logits logs/$N.logits
    echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-90) $(grep -h "shared expert on the expert tier" logs/$N.err | tail -1) $(grep -h "^rc=" logs/$N.err)"
  done
  /opt/ai/dsv41/venv/bin/python -I bench/speed/lkl.py $P.i32 logs/sh_${P}_a.logits logs/sh_${P}_b.logits logs/sh_${P}_s.logits
  rm -f logs/sh_${P}_*.logits
done
