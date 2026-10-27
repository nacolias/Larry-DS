#!/bin/bash
# s2build.sh - concurrent sessions, stage 2 (batched decode): apply the patches (patch-batch-*.py), build at low priority
# (bench/speed/nicebuild.sh), stage the new engine in /opt/ai/dsv41/s2bin (ds41_generate_xe + its two libraries: a test
# server runs it with XE_BIN / LD_LIBRARY_PATH there) and put the service's build outputs back - by renames, so the
# running engine keeps its files and the service starts the same engine as before the build.  build-xe/pre-build: the
# service's files as this run found them (put back at the end); build-xe/pre-s2: the engine before stage 2 (the first
# run's, kept for a rollback).  Log: logs/s2build.log.
cd /opt/ai/dsv41
B=xolotl/build-xe
F="ds41_generate_xe libds41_tp.so libds41_xe.so"
mkdir -p $B/pre-s2 $B/pre-build s2bin
for f in $F; do [ -f $B/pre-s2/$f ] || cp -p $B/$f $B/pre-s2/$f || exit 1; done
for f in $F; do cp -p $B/$f $B/pre-build/$f || exit 1; done   # what the service runs now
# from the committed engine sources (the ggml header's patch is idempotent in its own tree)
git -C xolotl checkout -- tools/v41/engine/ds41_generate.cpp tools/v41/engine/ds41_dense.cpp tools/v41/engine/ds41_dense.hpp \
  tools/v41/engine/tp/tp_core.cpp tools/v41/engine/tp/tp_core.hpp src/kernels/cpu/ptq1_0.cpp || exit 1
for p in patch-widthbench patch-batch-kern patch-batch-tp patch-batch-dense patch-batch-serve patch-batch-check patch-tpsynclog patch-tpstage patch-ptq1rows; do
  python3 -I patches/$p.py || { echo "PATCH FAILED: $p"; exit 1; }
done
bash bench/speed/nicebuild.sh ds41_generate_xe > logs/nicebuild-s2.log 2>&1
ok=0
grep -q "build end .* rc=0" logs/nicebuild-s2.log && ok=1
if [ $ok = 1 ]; then for f in $F; do cp -p $B/$f s2bin/$f; done; fi
for f in $F; do cp -p $B/pre-build/$f $B/.$f.s2tmp && mv -f $B/.$f.s2tmp $B/$f; done   # the service's again (new inodes)
if [ $ok = 1 ]; then echo "BUILT: staged in s2bin; build-xe holds the service's engine again"; else echo "BUILD FAILED (build-xe restored)"; grep -a -m5 "error" logs/nicebuild-s2.log; fi
