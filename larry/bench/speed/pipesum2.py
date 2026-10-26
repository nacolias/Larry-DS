#!/usr/bin/env python3
"""pipesum2.py ENGINE.log - STRATA_PIPE_TRACE lines per prompt batch: per layer the host's D (dense spans submitted;
blocks on the pass ring), J (waiting for the tier call), T (pass_sync + the call's launch), F (finish spans) ms, and
the batch's totals for layers 0-19 / 20-39."""
import re
import sys

rx = re.compile(r'pipe trace:\s+([\d.]+)\s+L(\d+)\s+(\w)\s+\[\s*(\d+),\s*(\d+)\)\s+([\d.]+) ms')
batches, cur, last_t = [], [], -1
for line in open(sys.argv[1], errors='replace'):
    m = rx.search(line)
    if not m:
        continue
    t, l, w, a0, a1, d = float(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5)), float(m.group(6))
    if cur and t < last_t:
        batches.append(cur); cur = []
    cur.append((t, l, w, a0, a1, d)); last_t = t
if cur:
    batches.append(cur)
for bi, B in enumerate(batches):
    ncol = max(a1 for t, l, w, a0, a1, d in B)
    end = max(t + d for t, l, w, a0, a1, d in B)
    per = {}
    for t, l, w, a0, a1, d in B:
        per.setdefault(l, {}).setdefault(w, 0.0)
        per[l][w] += d
    def tot(ls, w):
        return sum(per.get(l, {}).get(w, 0.0) for l in ls)
    A, Bl = range(0, 20), range(20, 40)
    print('batch %d: %d cols, %.0f ms;  layers 0-19: D %.0f J %.0f T %.0f F %.0f | layers 20-39: D %.0f J %.0f T %.0f F %.0f' % (
        bi, ncol, end, tot(A, 'D'), tot(A, 'J'), tot(A, 'T'), tot(A, 'F'), tot(Bl, 'D'), tot(Bl, 'J'), tot(Bl, 'T'), tot(Bl, 'F')))
    if '-v' in sys.argv:
        for l in sorted(per):
            print('   L%-2d ' % l + ' '.join('%s %6.1f' % (w, per[l].get(w, 0.0)) for w in 'DJTF'))
