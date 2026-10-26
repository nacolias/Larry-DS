#!/usr/bin/env python3
"""ptlcall.py logs/ptl.PID.tsv BATCH - per tier call of a batch (STRATA_GPU_PTL): launch (host), the expert cards' X
read done / push done (first .. last card), the combine queued (host) and reached / done on card 0, and how long card
0's queue sat empty before the combine - is a late combine the cards' work or the host's (the CPU-light part)?"""
import collections
import sys

path, want = sys.argv[1], int(sys.argv[2])
batches, cur = [], None
for line in open(path):
    if line.startswith('# batch'):
        cur = []; batches.append(cur); continue
    if line.startswith('#'):
        continue
    p = line.split('\t')
    if len(p) >= 6 and cur is not None:
        cur.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), float(p[4]), abs(float(p[5]))))
B = batches[want]
launch = [e for e in B if e[0] == 17 and e[1] == 40]
r0 = sorted([e for e in B if e[0] == 0], key=lambda e: e[5])
cards = collections.defaultdict(list)
for e in B:
    if 1 <= e[0] <= 8 and e[1] in (10, 14):
        cards[(e[2], e[1])].append(e)
# calls in launch order: (layer, a0); card records per layer in order of their X read
seen = collections.Counter()
print('%4s %6s | %7s %7s | %7s %7s | %7s %7s %7s | %6s' % ('L', 'a0', 'Xdone', 'Xlast', 'push1', 'pushN', 'cbQ', 'cbRun', 'cbEnd', 'idle'))
cbs = [e for e in r0 if e[1] in (22, 23)]
ci = 0
for (_, _, l, a0, t, _) in launch:
    k = seen[l]; seen[l] += 1
    xs = sorted(cards[(l, 10)], key=lambda e: e[5])
    ps = sorted(cards[(l, 14)], key=lambda e: e[5])
    nc = 3
    x_this = xs[k * nc:(k + 1) * nc]
    p_this = ps[k * nc:(k + 1) * nc]
    # the combine of this call: the next 22 in ring-0 order after the launch
    q = next((i for i in range(ci, len(cbs)) if cbs[i][1] == 22 and cbs[i][4] >= t - 1), None)
    if q is None or not x_this or not p_this:
        continue
    cq = cbs[q]
    ce = cbs[q + 1] if q + 1 < len(cbs) and cbs[q + 1][1] == 23 else None
    ci = q + 1
    # card 0's queue drained before the combine: the ring-0 record just before it
    idx = r0.index(cq)
    prev = r0[idx - 1][5] if idx > 0 else cq[4]
    idle = max(0.0, cq[4] - prev)
    print('%4d %6d | %7.1f %7.1f | %7.1f %7.1f | %7.1f %7.1f %7.1f | %6.1f' % (
        l, a0, min(e[5] for e in x_this) - t, max(e[5] for e in x_this) - t, min(e[5] for e in p_this) - t,
        max(e[5] for e in p_this) - t, cq[4] - t, cq[5] - t, (ce[5] - t) if ce else -1, idle))
