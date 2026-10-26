#!/bin/bash
# opprof.sh [IDS] - abw.sh probe: one-shot prompt (default p20k: a 16384-token batch + a 3.6K one) with the service's
# environment and GGML_SYCL_OP_PROF=1 - card 0's ggml-sycl time op by op (patch-opprof.py), printed at exit.
IDS_ARG=${1:-}
cd /opt/ai/dsv41
source xe-env.sh
eval "$(grep '^export ' ds41-xe-engine.sh)"
unset STRATA_SERVE_DSPARK
IDS=${IDS_ARG:-p20k.i32}
N=opp_${IDS%.i32}
GGML_SYCL_OP_PROF_SKIP=${SKIP:-0} GGML_SYCL_SDPA_PROF=1 GGML_SYCL_OP_PROF=1 GGML_SYCL_OP_PROF_EVERY=0 GGML_SYCL_OP_PROF_TOP=60 RUN_TIMEOUT=1200 ./run.sh $N --ids-file $IDS -n 1 --prompt-batch 16384
grep -h "^prompt" logs/$N.err | cut -c1-120; grep -h "larry sdpa prof" logs/$N.err | tail -2
sed -n '/larry op prof (exit)/,$p' logs/$N.err | head -64
grep -h "^rc=" logs/$N.err
