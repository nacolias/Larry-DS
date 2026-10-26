#!/usr/bin/env python3
"""utan.py DIR [top] - summarise a unitrace chrome trace: per device lane busy time, idle gaps, top kernels."""
import json, sys, os, glob, collections
d = sys.argv[1]; TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 25
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
for k, L in sorted(lanes.items(), key=lambda kv: -len(kv[1])):
    L.sort(key=lambda e: e['ts'])
    t0, t1 = L[0]['ts'], max(e['ts'] + e['dur'] for e in L)
    busy = sum(e['dur'] for e in L)
    # union busy (overlap-safe)
    ub, cur_s, cur_e = 0.0, None, None
    for e in L:
        s, en = e['ts'], e['ts'] + e['dur']
        if cur_e is None or s > cur_e:
            if cur_e is not None: ub += cur_e - cur_s
            cur_s, cur_e = s, en
        else:
            cur_e = max(cur_e, en)
    if cur_e is not None: ub += cur_e - cur_s
    print(f'\n## {lname(k)}: {len(L)} events, span {(t1 - t0) / 1e3:.1f} ms, busy {ub / 1e3:.1f} ms ({100 * ub / max(t1 - t0, 1):.0f}%)')
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in L:
        n = e['name']
        n = n.split('(')[0] if '(' in n else n
        a = agg[n[:90]]; a[0] += e['dur']; a[1] += 1
    for n, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:TOP]:
        print(f'  {t / 1e3:8.2f} ms {c:7d} x {t / c:7.1f} us  {n}')
    if len(L) > 2:
        gaps = [L[i + 1]['ts'] - (L[i]['ts'] + L[i]['dur']) for i in range(len(L) - 1)]
        gaps = [g for g in gaps if g > 0]
        hist = collections.Counter()
        for g in gaps:
            b = '<2us' if g < 2 else '<5us' if g < 5 else '<20us' if g < 20 else '<100us' if g < 100 else '<500us' if g < 500 else '>=500us'
            hist[b] += g
        print('  idle gaps (total ms by size): ' + ', '.join(f'{b} {hist[b] / 1e3:.1f}' for b in ['<2us', '<5us', '<20us', '<100us', '<500us', '>=500us']))
