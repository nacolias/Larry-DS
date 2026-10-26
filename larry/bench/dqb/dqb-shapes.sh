#!/bin/bash
# dqb-shapes.sh - an abw.sh probe: bench_dqgemm at a big prompt's tier-call shapes (NE experts x M tokens, card 1,
# planar = resident experts / file = prefetched), with the oneAPI environment
set --
source /opt/intel/oneapi/2026.1/oneapi-vars.sh > /dev/null 2>&1
B=/opt/ai/dsv41/bench/dqb/bench_dqgemm
for a in "128 128 1 1" "128 128 1 0" "128 64 1 1" "256 32 1 1" "64 256 1 1" "32 512 1 1"; do
  echo "== bench_dqgemm $a"; $B $a
done
