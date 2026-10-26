#!/usr/bin/env python3
# svc-pfmin.py - ds41-xe-engine.sh: the prompt prefetch from 4096 tokens a batch (an enc-skip batch's upper layers from
# 2048), after STRATA_XE_CPU_LIGHT.
import shutil, sys
P = '/opt/ai/dsv41/ds41-xe-engine.sh'
s = open(P).read()
if 'STRATA_XE_PREFETCH_MIN' in s:
    print('already there'); sys.exit(0)
anchor = 'export STRATA_XE_CPU_LIGHT="${STRATA_XE_CPU_LIGHT-4}"\n'
assert s.count(anchor) == 1
add = '''# the prompt prefetch (every arena expert of a layer staged ahead) from 4096 tokens a batch, not 2048: a whole prompt of
# 2-3K tokens runs faster on demand with the CPU pool (2,047 tokens 436 -> 502 tok/s, 2,900 455 -> 588); a long
# prompt's upper layers (enc-skip: its last ~2,560 tokens over the whole context) keep it from 2048 - without that a
# 31K prompt took 5 s longer (23.0 / 23.4 s with it)
export STRATA_XE_PREFETCH_MIN="${STRATA_XE_PREFETCH_MIN-4096}"
export STRATA_XE_PREFETCH_MIN_B="${STRATA_XE_PREFETCH_MIN_B-2048}"
'''
shutil.copy2(P, P + '.pre-pfmin-20261024')
open(P, 'w').write(s.replace(anchor, anchor + add))
print('patched', P)
