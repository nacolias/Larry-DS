#!/usr/bin/env python3
"""pipesum3.py ENGINE_LOG - STRATA_PIPE_TRACE lines per prompt batch (a batch starts where the trace's clock restarts):
for layers 0-19 and 20-39 the summed host time of the dense parts (D), the joins (J: waiting for a tier call) and the
finishes (F), and the batch's end - which side the layers wait on."""
import re
import sys

rx = re.compile(r'pipe trace: +([\d.]+) +L(\d+) +(\w) \[ *(\d+), *(\d+)\) +([\d.]+) ms')
batches, cur, last_t = [], [], -1.0
for ln in open(sys.argv[1], errors='replace'):
    m = rx.search(ln)
    if not m:
        continue
    t, l, step, a0, a1, d = float(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5)), float(m.group(6))
    if t < last_t and cur:
        batches.append(cur)
        cur = []
    last_t = t
    cur.append((t, l, step, a0, a1, d))
if cur:
    batches.append(cur)
for i, b in enumerate(batches):
    cols = max(x[4] for x in b)
    end = max(x[0] + x[5] for x in b)
    print('batch %d: %d columns, %.0f ms' % (i, cols, end))
    for lo, hi in ((0, 19), (20, 39)):
        s = {'D': 0.0, 'J': 0.0, 'F': 0.0}
        n = {'D': 0, 'J': 0, 'F': 0}
        span = [x for x in b if lo <= x[1] <= hi]
        for x in span:
            if x[2] in s:
                s[x[2]] += x[5]
                n[x[2]] += 1
        if not span:
            continue
        t0 = min(x[0] for x in span)
        t1 = max(x[0] + x[5] for x in span)
        print('  layers %2d-%2d: %7.0f ms wall | D %7.0f ms (%3d) J %7.0f ms (%3d) F %6.0f ms (%3d) | J max %.1f ms' % (
            lo, hi, t1 - t0, s['D'], n['D'], s['J'], n['J'], s['F'], n['F'],
            max([x[5] for x in span if x[2] == 'J'] or [0])))
