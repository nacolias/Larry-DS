#!/bin/bash
# xinstall.sh NAME - the engine xbuild.sh staged in /opt/ai/dsv41/<NAME>bin into build-xe by renames (an abw.sh "@"
# step: the service is stopped; the window's restore starts the new engine).  The engine before it stays in
# build-xe/pre-<NAME> (back: XI_FROM=xolotl/build-xe/pre-<NAME> bash xinstall.sh NAME).
cd /opt/ai/dsv41
N=$1
B=xolotl/build-xe
S=${XI_FROM:-${N}bin}
for f in ds41_generate_xe libds41_tp.so libds41_xe.so; do
  cp -p $S/$f $B/.$f.dep && mv -f $B/.$f.dep $B/$f || { echo "install failed: $f"; exit 1; }
done
md5sum $B/ds41_generate_xe $B/libds41_tp.so $B/libds41_xe.so $S/ds41_generate_xe $S/libds41_tp.so $S/libds41_xe.so | cut -c1-12,34-
