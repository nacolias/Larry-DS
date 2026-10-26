#!/bin/bash
# run_bench_dqgemm.sh - bench_dqgemm at the tier calls' shapes (an abw.sh @ step): 16K phase A, 8K phase A, phase B
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/dqb
for s in "96 200" "96 100" "120 30"; do timeout 300 ./bench_dqgemm $s 1; done
