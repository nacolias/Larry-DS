#!/bin/bash
# opprof-shapes.sh [IDS] - abw.sh probe: opprof.sh with GGML_SYCL_OP_PROF_SHAPES=1 (patch-opprof5.py: each label also
# carries the node's shape and its first source's name / type) and the top 160 labels; SKIP as opprof.sh
IDS_ARG=${1:-}
cd /opt/ai/dsv41
source xe-env.sh
eval "$(grep '^export ' ds41-xe-engine.sh)"
unset STRATA_SERVE_DSPARK
IDS=${IDS_ARG:-p20k.i32}
N=opps_${IDS%.i32}
GGML_SYCL_OP_PROF_SKIP=${SKIP:-0} GGML_SYCL_OP_PROF_SHAPES=1 GGML_SYCL_OP_PROF=1 GGML_SYCL_OP_PROF_EVERY=0 GGML_SYCL_OP_PROF_TOP=160 RUN_TIMEOUT=1200 ./run.sh $N --ids-file $IDS -n 1 --prompt-batch 16384
grep -h "^prompt" logs/$N.err | cut -c1-120
grep -h "^rc=" logs/$N.err
