#!/bin/bash
# tlppl-ab.sh TAG VAR=v[,VAR=v...] - numerics of an engine switch: one-shot prompt-path runs with the service's
# environment - p8k at --prompt-batch 16384 twice without (the run-to-run floor) and once with; lkl.py: PPL, per-row KL
# against the first run, top-1 agreement.  Then p30k's PPL without / with.  The service must be stopped.
TAG=$1; ON=${2//,/ }
cd /opt/ai/dsv41
source xe-env.sh
eval "$(grep '^export ' ds41-xe-engine.sh)"
unset STRATA_SERVE_DSPARK
for R in a b x; do
  N=${TAG}_p8k_$R
  E="AB_NONE=1"; [ $R = x ] && E="$ON"
  env $E RUN_TIMEOUT=1200 ./run.sh $N --ids-file p8k.i32 -n 1 --ppl --prompt-batch 16384 --dump-logits logs/$N.logits
  echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-70) $(grep -h "^rc=" logs/$N.err)"
done
/opt/ai/dsv41/venv/bin/python -I bench/speed/lkl.py p8k.i32 logs/${TAG}_p8k_a.logits logs/${TAG}_p8k_b.logits logs/${TAG}_p8k_x.logits
rm -f logs/${TAG}_p8k_*.logits
for R in a x; do
  N=${TAG}_p30k_$R
  E="AB_NONE=1"; [ $R = x ] && E="$ON"
  env $E RUN_TIMEOUT=1500 ./run.sh $N --ids-file p30k.i32 -n 1 --ppl --prompt-batch 16384
  echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-70) $(grep -h "^rc=" logs/$N.err)"
done
