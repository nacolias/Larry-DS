#!/bin/bash
# run_bench_dqgemm2.sh - bench_dqgemm, planar vs the file layout (prefetched experts), at the 16K phase-A shape
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/dqb
timeout 300 ./bench_dqgemm 96 200 1 1
timeout 300 ./bench_dqgemm 96 200 1 0
timeout 300 ./bench_dqgemm 96 100 1 0
