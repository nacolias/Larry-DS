#!/usr/bin/env python3
"""dtrace.py DIR [gap_us] [top] - engine1: the DSpark draft's kernels from a unitrace chrome trace of a TP decode run.
The TP passes are replayed graphs (their kernels do not show), so a device lane's visible kernels come in bursts, one
per draft step: kernels closer than gap_us (default 1500) belong to the same burst.  Per lane: bursts, kernels a
burst, device time a burst (sum of kernel durations) vs its span (first start -> last end), and the kernels by time."""
import json, sys, os, glob, collections
d = sys.argv[1]
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 1500.0
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
evs = []
for f in glob.glob(os.path.join(d, '*.json')):
    t = json.load(open(f))
    evs += t['traceEvents'] if isinstance(t, dict) else t
names = {}
for e in evs:
    if e.get('ph') == 'M' and e.get('name') in ('process_name', 'thread_name'):
        names[(e['name'][0], e.get('pid'), e.get('tid'))] = e['args'].get('name')
lanes = collections.defaultdict(list)
for e in evs:
    if e.get('ph') == 'X' and 'dur' in e:
        lanes[(e.get('pid'), e.get('tid'))].append(e)
def lname(k):
    p = names.get(('p', k[0], None)) or names.get(('p', k[0], k[1])) or str(k[0])
    t = names.get(('t', k[0], k[1])) or str(k[1])
    return f'{p} / {t}'
def short(n):
    n = n.split('(')[0] if '(' in n else n
    return n[:100]
for k, L in sorted(lanes.items(), key=lambda kv: -len(kv[1])):
    if len(L) < 200: continue
    L.sort(key=lambda e: e['ts'])
    bursts, cur = [], [L[0]]
    for a, b in zip(L, L[1:]):
        if b['ts'] - (a['ts'] + a['dur']) > GAP:
            bursts.append(cur); cur = []
        cur.append(b)
    bursts.append(cur)
    # drop the first / last (cut by the trace window) and tiny bursts
    core = [b for b in bursts[1:-1] if len(b) >= 20]
    if not core: continue
    nk = [len(b) for b in core]
    busy = [sum(e['dur'] for e in b) for b in core]
    span = [b[-1]['ts'] + b[-1]['dur'] - b[0]['ts'] for b in core]
    print(f'\n## {lname(k)}: {len(L)} kernels, {len(core)} bursts (of {len(bursts)}); a burst: {sum(nk) / len(nk):.0f} kernels, '
          f'device {sum(busy) / len(busy) / 1e3:.3f} ms, span {sum(span) / len(span) / 1e3:.3f} ms '
          f'(min {min(span) / 1e3:.3f} max {max(span) / 1e3:.3f})')
    hist = collections.Counter(nk)
    print('  kernels a burst: ' + ', '.join(f'{n} x{c}' for n, c in sorted(hist.items())[:12]))
    agg = collections.defaultdict(lambda: [0.0, 0])
    for b in core:
        for e in b:
            a = agg[short(e['name'])]; a[0] += e['dur']; a[1] += 1
    nb = len(core)
    print(f'  {"us/burst":>9s} {"n/burst":>7s} {"us each":>8s}  kernel')
    for n, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:TOP]:
        print(f'  {t / nb:9.1f} {c / nb:7.1f} {t / c:8.2f}  {n}')
    # idle inside the bursts (the host feeding the queue / dependencies)
    gaps = collections.Counter()
    for b in core:
        for a, c in zip(b, b[1:]):
            g = c['ts'] - (a['ts'] + a['dur'])
            if g > 0:
                key = '<2us' if g < 2 else '<5us' if g < 5 else '<20us' if g < 20 else '<100us' if g < 100 else '>=100us'
                gaps[key] += g
    print('  idle inside a burst (us by gap size): ' + ', '.join(f'{x} {gaps[x] / nb:.0f}' for x in ['<2us', '<5us', '<20us', '<100us', '>=100us']))
