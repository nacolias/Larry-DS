#!/bin/bash
# run_p2p_kread.sh - the p2p_kread probe with the engine's environment (an abw.sh @ step)
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/dqb && timeout 300 ./p2p_kread 2
