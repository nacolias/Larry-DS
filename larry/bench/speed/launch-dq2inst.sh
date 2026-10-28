#!/bin/bash
# launch-dq2inst.sh - xe_dqgemm v2 into the service: the dq2bin engine (the service + patch-dqv2.py) into build-xe
# (xinstall.sh; build-xe/pre-dq2 keeps the engine before it) + svc-dq2.py (STRATA_XE_DQ2=1; backup
# ds41-xe-engine.sh.pre-dq2-20261028) in a window, then tsvc2 + concurtest (tsvc2-ra.sh dq2)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/longctx/xinstall.sh dq2" "@python3 patches/svc-dq2.py" > logs/abw-dq2inst.log 2>&1 < /dev/null &
setsid nohup bash bench/longctx/tsvc2-ra.sh dq2 > /dev/null 2>&1 < /dev/null &
echo "dq2 install launched: logs/abw-dq2inst.log, then logs/tsvc2-dq2.log"
