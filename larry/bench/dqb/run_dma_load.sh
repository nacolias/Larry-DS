#!/bin/bash
# run_dma_load.sh - the dma_load probe with the engine's environment (an abw.sh @ step)
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/dqb && timeout 300 ./dma_load 2
