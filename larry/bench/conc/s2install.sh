#!/bin/bash
# s2install.sh - concurrent sessions, stage 2 into the service: the staged engine (s2bin: s2build.sh) into build-xe by
# renames - an abw.sh "@" step (the service stopped; the window's restore starts the new engine).  The engine before it
# stays in build-xe/pre-s2 (back: the same renames from there).
cd /opt/ai/dsv41
B=xolotl/build-xe
for f in ds41_generate_xe libds41_tp.so libds41_xe.so; do
  cp -p s2bin/$f $B/.$f.dep && mv -f $B/.$f.dep $B/$f || { echo "install failed: $f"; exit 1; }
done
md5sum $B/ds41_generate_xe $B/libds41_tp.so s2bin/ds41_generate_xe s2bin/libds41_tp.so | cut -c1-12,34-
