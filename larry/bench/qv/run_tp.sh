#!/bin/bash
# run_tp.sh - after the qv engine dumps (the cards free): tp_layer with the GPU clocks pinned (step 0 of the new engine)
cd /opt/ai/dsv41/bench/engine0
until grep -q "ENGINE DONE" /opt/ai/dsv41/qv/engine.log 2>/dev/null; do sleep 20; done
while pgrep -f "[d]s41_generate_xe" > /dev/null; do sleep 5; done
source /opt/ai/dsv41/xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
echo "start $(date +%T)" > tp_layer.log
./tp_layer 5 >> tp_layer.log 2>&1; echo "rc=$? $(date +%T)" >> tp_layer.log
/usr/local/sbin/larry-gpu-clock restore > /dev/null
