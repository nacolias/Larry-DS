#!/bin/bash
# run_mb.sh - an abw.sh probe: mqa_mb on card 1 (the fused attention's inner loops)
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/fa
timeout 300 ./mqa_mb 1
