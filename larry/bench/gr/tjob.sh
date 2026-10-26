#!/bin/bash
# tjob.sh - STRATA_XE_JOBNT 0 / 1 / 2: the planar dp4a expert kernels at decode / verify shapes (test_ds41_xe
# --bench-q8: layer 20, 24 experts rotated past the L2) - time per (experts, tokens a job, call width, form) and the
# output's bits against the same call in the jobs' own form ("bits ok").  One card, GPU clocks pinned.
cd /opt/ai/dsv41/xolotl
source ../xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for j in 0 1 2 0 2; do
  echo "== STRATA_XE_JOBNT=$j"
  STRATA_XE_JOBNT=$j build-xe/test_ds41_xe $GGUF --layers 20 --per-class 0 --bench 0 --bench-q8 200 2>&1 | grep -A 12 "^bench-q8"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
