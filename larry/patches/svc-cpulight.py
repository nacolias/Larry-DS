#!/usr/bin/env python3
# svc-cpulight.py - ds41-xe-engine.sh: STRATA_XE_CPU_LIGHT=4 in the service (after the prompt-chunk combine knob).
import shutil, sys
P = '/opt/ai/dsv41/ds41-xe-engine.sh'
s = open(P).read()
if 'STRATA_XE_CPU_LIGHT' in s:
    print('already there'); sys.exit(0)
anchor = 'export STRATA_XE_COMBINE="${STRATA_XE_COMBINE:-1}"\n'
assert s.count(anchor) == 1
add = '''# prompt calls without the prefetch (< 2048 tokens): an arena expert the call would stream that serves <= 4 tokens runs
# on the CPU pool beside the cards (a pageable one streamed is read + written by the bounce copy and read by the DMA;
# the pool reads it once).  Service A/B (DSpark, clocks pinned, 5 runs): pp512 166 -> 231 tok/s, at 8K 159 -> 222, at
# 32K 154 -> 219, pp160 91 -> 125, 2047 tokens ~390 -> ~490; long prompts unchanged; prompt-path KL at the run-to-run
# floor (bench/speed/tab_all.sh, tlppl.sh).  0 = every arena expert streamed
export STRATA_XE_CPU_LIGHT="${STRATA_XE_CPU_LIGHT-4}"
'''
shutil.copy2(P, P + '.pre-cpulight-20261024')
open(P, 'w').write(s.replace(anchor, anchor + add))
print('patched', P)
