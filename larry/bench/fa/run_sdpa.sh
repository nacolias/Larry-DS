#!/bin/bash
# run_sdpa.sh - test_sdpa_dsv4 (an abw.sh @ step): the extra dims 16 and 64, then 1024 queries
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/fa
for a in "512 16 640" "512 64 640" "512 16 128"; do timeout 300 ./test_sdpa_dsv4 $a; done
