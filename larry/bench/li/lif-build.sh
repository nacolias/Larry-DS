#!/bin/bash
# lif-build.sh - patch-lifused.py (GGML_SYCL_LI_FUSED) + patch-linvis.py (STRATA_LI_NVIS), both default off, then
# ds41_generate_xe rebuilt (nicebuild.sh; ggml.h changes: most of ggml-sycl recompiles).  Backups first: the binaries
# the service runs (02:55 build, LI_S16 compiled in, off) -> logs/tier-rebuild/bin-20261026-0255, every touched source
# -> logs/tier-rebuild/pre-lifused-*.
cd /opt/ai/dsv41
X=xolotl; G=$X/third_party/llama.cpp/ggml
BK=logs/tier-rebuild/bin-20261026-0255
mkdir -p $BK
[ -e $BK/ds41_generate_xe ] || cp -p $X/build-xe/ds41_generate_xe $X/build-xe/libds41_xe.so $BK/
for f in $G/include/ggml.h $G/src/ggml.c $G/src/ggml-sycl/ggml-sycl.cpp $G/src/ggml-sycl/lightning-indexer.cpp $X/tools/v41/engine/ds41_dense.cpp; do
  b=logs/tier-rebuild/pre-lifused-$(basename $f)
  [ -e $b ] || cp -p $f $b
done
python3 logs/tier-rebuild/patch-lifused.py $G/src/ggml-sycl/lightning-indexer.cpp || exit 1
python3 logs/tier-rebuild/patch-linvis.py $X || exit 1
bash bench/speed/nicebuild.sh
ls -la --time-style=+%T $X/build-xe/ds41_generate_xe $BK/ds41_generate_xe
