#!/bin/bash
# dqv2.sh build|run - xe_dqgemm v1 against v2 (bench/dqb/xe_dqgemm_v2.cpp): bit-for-bit outputs and times at a phase-A
# call's expert sizes, planar (resident) and the file layout (prefetched).  run: an abw.sh @ step.
MODE=${1:-run}   # (xe-env.sh clears "$@")
cd /opt/ai/dsv41/bench/dqb
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
if [ "$MODE" = build ]; then
  icpx -fsycl -O3 -std=c++17 -Wno-deprecated-declarations -Wno-ignored-attributes -I/opt/ai/dsv41/xolotl/tools/v41/engine/xe \
    -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" bench_dqgemm_v2.cpp xe_dqgemm_v2.cpp -o bench_dqgemm_v2 > build_dqv2.log 2>&1
  echo "rc=$? $(grep -c -i ' error' build_dqv2.log) errors; $(grep -i -o 'spilled around [0-9]*' build_dqv2.log | sort | uniq -c | tr '\n' ' ')"
  exit
fi
for M in 32 64 128 160 256; do echo "== 96 experts x $M tokens, planar"; timeout 300 ./bench_dqgemm_v2 96 $M 1 1 | grep -v "^Intel\|layout"; done
echo "== 96 experts x 128 tokens, file layout"; timeout 300 ./bench_dqgemm_v2 96 128 1 0 | grep -v "^Intel\|layout"
