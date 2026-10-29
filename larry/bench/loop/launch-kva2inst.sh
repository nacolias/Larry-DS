#!/bin/bash
# launch-kva2inst.sh - patch-kvahead.py + patch-pclend2.py into the service: the kva2bin engine (xbuild.sh kva2) into
# build-xe (xinstall.sh; build-xe/pre-kva2 keeps the engine before it) in a window - STRATA_TP_DEC_AHEAD defaults to 2048,
# no environment change; then post-lw.sh kva2inst: tsvc2 + concurtest + loopprobe (logs/tsvc2-kva2inst.log)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/longctx/xinstall.sh kva2" > logs/abw-kva2inst.log 2>&1 < /dev/null &
setsid nohup bash bench/loop/post-lw.sh kva2inst > /dev/null 2>&1 < /dev/null &
echo "launched: logs/abw-kva2inst.log, then logs/tsvc2-kva2inst.log"
