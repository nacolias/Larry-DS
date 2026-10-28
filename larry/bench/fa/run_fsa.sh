#!/bin/bash
# run_fsa.sh [ARGS..] - an abw.sh probe: test_fsa on card 1 at a few positions (the span's depth sets how many compressed
# rows its queries see; early spans mask part of the window and of the 512)
source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/fa
if [ $# -gt 0 ]; then timeout 300 ./test_fsa "$@"; exit; fi
for a in "512 30000 16000" "512 0 64" "512 600 300" "512 130000 65000"; do echo "== test_fsa $a"; timeout 300 ./test_fsa $a; done
