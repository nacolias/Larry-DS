#!/bin/bash
# run.sh - test_mqa_pf on card 0 (abw.sh probe: the service stopped, clocks pinned)
cd /opt/ai/dsv41/bench/fa
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 300 ./test_mqa_pf ${1:-512} 2>&1
echo "test rc=$?"
