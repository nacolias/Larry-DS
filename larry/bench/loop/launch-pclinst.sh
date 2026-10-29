#!/bin/bash
# launch-pclinst.sh - STRATA_XE_PC_LEND into the service: the pclbin engine (xbuild.sh pcl: patch-pclend.py; only
# libds41_xe.so differs) into build-xe (xinstall.sh; build-xe/pre-pcl keeps the engine before it) + svc-pclend.py
# (backup ds41-xe-engine.sh.pre-pclend-20261029) in a window; then post-lw.sh pclinst: tsvc2 + concurtest + loopprobe
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/longctx/xinstall.sh pcl" "@python3 patches/svc-pclend.py" > logs/abw-pclinst.log 2>&1 < /dev/null &
setsid nohup bash bench/loop/post-lw.sh pclinst > /dev/null 2>&1 < /dev/null &
echo "launched: logs/abw-pclinst.log, then logs/tsvc2-pclinst.log"
