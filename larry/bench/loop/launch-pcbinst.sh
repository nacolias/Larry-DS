#!/bin/bash
# launch-pcbinst.sh - decode at depth, part 2 into the service: the pcbbin engine (xbuild.sh pcb: patch-pcbuf.py on top of
# the committed part 1; only libds41_xe.so differs) into build-xe (xinstall.sh; build-xe/pre-pcb keeps the engine before
# it) in a window - STRATA_XE_PC_LEND_BUFS defaults to 1, so no environment change; then post-lw.sh pcbinst: tsvc2 +
# concurtest + loopprobe (logs/tsvc2-pcbinst.log)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/longctx/xinstall.sh pcb" > logs/abw-pcbinst.log 2>&1 < /dev/null &
setsid nohup bash bench/loop/post-lw.sh pcbinst > /dev/null 2>&1 < /dev/null &
echo "launched: logs/abw-pcbinst.log, then logs/tsvc2-pcbinst.log"
