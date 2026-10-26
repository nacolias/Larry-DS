#!/bin/bash
# tlppl-lif.sh - numerics of GGML_SYCL_LI_FUSED=1 (the prompt indexer as one fused XMX kernel, patch-lifused.py) and
# + STRATA_LI_NVIS=1 (the indexer told each query's visible rows, no [cap, n] mask: patch-linvis.py): one-shot
# prompt-path runs with the service's environment - p8k at --prompt-batch 16384 twice without (the run-to-run floor),
# once with each (f: its first 8 indexer calls also through li_gemm, "li_fused check" lines); lkl.py: PPL, per-row KL
# against the first run, top-1 agreement.  Then p30k's PPL without / with each.  The service must be stopped.
cd /opt/ai/dsv41
source xe-env.sh
eval "$(grep '^export ' ds41-xe-engine.sh)"
unset STRATA_SERVE_DSPARK
envof() { case $1 in a|b) echo "GGML_SYCL_LI_FUSED=0" ;; f) echo "GGML_SYCL_LI_FUSED=1 GGML_SYCL_LI_FUSED_CHECK=$2" ;;
                     n) echo "GGML_SYCL_LI_FUSED=1 STRATA_LI_NVIS=1" ;; esac; }
for R in a b f n; do
  N=lif_p8k_$R
  env $(envof $R 8) RUN_TIMEOUT=1200 ./run.sh $N --ids-file p8k.i32 -n 1 --ppl --prompt-batch 16384 --dump-logits logs/$N.logits
  echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-70) $(grep -h "^rc=" logs/$N.err)"
  grep -h "li_fused check" logs/$N.err | head -8
done
/opt/ai/dsv41/venv/bin/python -I bench/speed/lkl.py p8k.i32 logs/lif_p8k_a.logits logs/lif_p8k_b.logits logs/lif_p8k_f.logits logs/lif_p8k_n.logits
rm -f logs/lif_p8k_*.logits
for R in a f n; do
  N=lif_p30k_$R
  env $(envof $R 4) RUN_TIMEOUT=1500 ./run.sh $N --ids-file p30k.i32 -n 1 --ppl --prompt-batch 16384
  echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-70) $(grep -h "^rc=" logs/$N.err)"
  grep -h "li_fused check" logs/$N.err | head -4
done
