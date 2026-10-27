#!/usr/bin/env python3
"""cpulog_stats.py LOG [FROM TO] - the CPU pool's experts in TP passes from a STRATA_XE_CPU_LOG file (cpulog.sh): per
pass the experts on the pool, how many tokens each served, the weights' bytes and the pool's time (its GB/s), and per
format class the experts a pass, their tokens and blob size.  FROM / TO: the passes to take (default all) - e.g. the
passes of an answer alone vs two at once.  Lines: T pass rows | E pass layer expert tokens class pin uses where bytes |
L pass layer rows experts bytes us gu_ms q_ms down_ms ..."""
import collections
import sys

passes = collections.OrderedDict()
cur = None
for ln in open(sys.argv[1]):
    f = ln.split()
    if not f:
        continue
    if f[0] == 'T':
        cur = int(f[1])
        passes[cur] = {'rows': int(f[2]), 'E': [], 'L': []}
    elif f[0] == 'E' and cur is not None:
        passes[cur]['E'].append((int(f[4]), int(f[5]), int(f[6]), int(f[9])))   # tokens, class, pin, bytes
    elif f[0] == 'L' and cur is not None:
        passes[cur]['L'].append((int(f[5]), float(f[6])))                       # bytes, us
ps = list(passes.values())
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(ps)
sel = ps[lo:hi]
n = max(1, len(sel))
ne = sum(len(p['E']) for p in sel)
one = sum(1 for p in sel for e in p['E'] if e[0] == 1)
by = sum(l[0] for p in sel for l in p['L'])
us = sum(l[1] for p in sel for l in p['L'])
print("passes %d-%d: %.2f rows, %.1f pool experts a pass (%.0f %% for one token), %.0f MB, %.2f ms a pass = %.1f GB/s"
      % (lo, hi - 1, sum(p['rows'] for p in sel) / n, ne / n, 100.0 * one / max(1, ne), by / n / 1e6, us / n / 1e3,
         by / max(us, 1e-9) / 1e3))
cls = collections.defaultdict(lambda: [0, 0, 0, 0])
for p in sel:
    for e in p['E']:
        c = cls[e[1]]
        c[0] += 1
        c[1] += e[0]
        c[2] = e[3]
        c[3] += e[2] != 0
for k, v in sorted(cls.items()):
    print("  class %d: %.2f a pass, %.2f tokens each, blob %.1f MB, pinned %.0f %%" % (k, v[0] / n, v[1] / v[0], v[2] / 1e6, 100.0 * v[3] / v[0]))
