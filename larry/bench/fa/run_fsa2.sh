#!/bin/bash
# run_fsa2.sh - an abw.sh probe: test_fsa at 30K with each phase skipped in turn (fsa::Args::skip: 1 the rows' loads,
# 2 QK^T, 4 the softmax, 8 PV, 16 the output) - where the fused kernel's time goes
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/fa
for k in 0 1 8 10 31; do timeout 300 ./test_fsa 512 30000 16000 64 1 $k | grep "skip"; done
for ns in 32 48 96 128; do echo "nslot $ns"; timeout 300 ./test_fsa 512 30000 16000 $ns 1 0 | grep "skip"; done
