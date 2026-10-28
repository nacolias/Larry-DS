#!/bin/bash
# dqmt.sh build|run - xe_dqgemm with taller tiles (XE_DQ_MT 16 = 128 tokens, the tier's; 20 = 160; 24 = 192): bench_dqgemm at
# expert sizes a phase-A call has (~130-160 rows an expert a card at 8K), each quant type planar.  run: an abw.sh @ step.
MODE=${1:-run}   # (xe-env.sh clears "$@")
cd /opt/ai/dsv41/bench/dqb
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
if [ "$MODE" = build ]; then
  for mt in 16 20 24; do
    icpx -fsycl -O3 -std=c++17 -Wno-deprecated-declarations -Wno-ignored-attributes -DXE_DQ_MT=$mt -I/opt/ai/dsv41/xolotl/tools/v41/engine/xe \
      -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" bench_dqgemm_mt.cpp xe_dqgemm_mt.cpp -o bench_dqgemm_mt$mt > build_dqmt$mt.log 2>&1
    echo "MT $mt: rc=$? $(grep -c -i ' error' build_dqmt$mt.log) errors; $(grep -i -o 'spilled around [0-9]*' build_dqmt$mt.log | sort -u | tr '\n' ' ')"
  done
  exit
fi
for M in 96 128 144 160 192 256; do
  for mt in 16 20 24; do echo "== MT $mt, 96 experts x $M tokens"; timeout 300 ./bench_dqgemm_mt$mt 96 $M 1 | grep -v "^Intel\|layout"; done
done
