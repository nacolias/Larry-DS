#!/bin/bash
# xbuild.sh NAME PATCH... - an engine change on top of the committed sources: the files the patches touch checked out
# (XB_FILES), the patches applied (patches/<PATCH>.py, in order), built at low priority (bench/speed/nicebuild.sh: frozen
# while the service has a request), the new engine staged in /opt/ai/dsv41/<NAME>bin (ds41_generate_xe + its two
# libraries: a test server runs it with XE_BIN / LD_LIBRARY_PATH there), and the service's build outputs put back - by
# renames, so the running engine keeps its files and the service starts the same engine as before.  build-xe/pre-build:
# the service's files as this run found them; build-xe/pre-<NAME>: the same, kept for a rollback after xinstall.sh.
# The sources stay patched (the next xbuild.sh checks them out again).  Log: logs/xbuild-<NAME>.log.
set -u
N=$1; shift
cd /opt/ai/dsv41
B=xolotl/build-xe
F="ds41_generate_xe libds41_tp.so libds41_xe.so"
XB_FILES=${XB_FILES:-"tools/v41/engine/ds41_generate.cpp tools/v41/engine/tp/tp_core.cpp tools/v41/engine/tp/tp_core.hpp tools/v41/engine/xe/ds41_xe_tier.cpp tools/v41/engine/xe/ds41_xe_tier.hpp"}
mkdir -p $B/pre-build $B/pre-$N ${N}bin
for f in $F; do cp -p $B/$f $B/pre-build/$f || exit 1; done   # what the service runs now
for f in $F; do [ -f $B/pre-$N/$f ] || cp -p $B/$f $B/pre-$N/$f || exit 1; done
git -C xolotl checkout -- $XB_FILES || exit 1
for p in "$@"; do
  python3 -I patches/$p.py || { echo "PATCH FAILED: $p"; exit 1; }
done
bash bench/speed/nicebuild.sh ds41_generate_xe > logs/xbuild-$N.log 2>&1
ok=0
grep -q "build end .* rc=0" logs/xbuild-$N.log && ok=1
if [ $ok = 1 ]; then for f in $F; do cp -p $B/$f ${N}bin/$f; done; fi
for f in $F; do cp -p $B/pre-build/$f $B/.$f.xbtmp && mv -f $B/.$f.xbtmp $B/$f; done   # the service's again (new inodes)
md5sum $B/ds41_generate_xe $B/libds41_tp.so $B/libds41_xe.so ${N}bin/* 2>/dev/null | cut -c1-12,34-
if [ $ok = 1 ]; then echo "BUILT: staged in ${N}bin; build-xe holds the service's engine again"
else echo "BUILD FAILED (build-xe restored)"; grep -a -m8 "error" logs/nicebuild-ninja.log; fi
