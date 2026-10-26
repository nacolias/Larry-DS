#!/bin/bash
# tneedle.sh - the long-prompt checks against the service with the answers shown: needles 8K x2 / 16K / 31K, the tool call
cd /opt/ai/dsv41
for n in 8000 8000 16000 31000; do
  python3 needle.py $n 1 2>&1 | python3 -c 'import sys; L = sys.stdin.read().splitlines(); print(L[-1], "|", L[0][:110], "|", (L[1].split("prompt_per_second")[1][:12] if len(L) > 1 and "prompt_per_second" in L[1] else ""))'
done
python3 tooltest.py 2>&1 | tail -2 | head -1 | cut -c1-120
