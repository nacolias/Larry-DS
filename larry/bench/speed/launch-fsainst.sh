#!/bin/bash
# launch-fsainst.sh - #3 into the service: the fsabin engine (patch-fsa.py) into build-xe (xinstall.sh; build-xe/pre-fsa
# keeps the engine before it) + svc-fsa.py (STRATA_ATTN_FSA=1, STRATA_XE_FN16=2, STRATA_LI_REMOTE=0; backup
# ds41-xe-engine.sh.pre-fsa-20261028) in a window, then tsvc2 + concurtest on the service (tsvc2-ra.sh fsa).
# Rollback: XI_FROM=xolotl/build-xe/pre-fsa bash bench/longctx/xinstall.sh fsa + the .pre-fsa-20261028 engine script.
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/longctx/xinstall.sh fsa" "@python3 patches/svc-fsa.py" > logs/abw-fsainst.log 2>&1 < /dev/null &
setsid nohup bash bench/longctx/tsvc2-ra.sh fsa > /dev/null 2>&1 < /dev/null &
echo "fsa install launched: logs/abw-fsainst.log, then logs/tsvc2-fsa.log"
