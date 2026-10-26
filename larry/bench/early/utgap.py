#!/usr/bin/env python3
"""utgap.py DIR LANE_SUBSTR [n] - on one device lane, the biggest idle gaps grouped by (kernel before -> kernel after),
plus a printed slice of the timeline (n events from the middle) to read one layer's sequence."""
import json, sys, os, glob, collections
d, sub = sys.argv[1], sys.argv[2]; NSHOW = int(sys.argv[3]) if len(sys.argv) > 3 else 120
evs = []
for f in glob.glob(os.path.join(d, '*.json')):
    t = json.load(open(f))
    evs += t['traceEvents'] if isinstance(t, dict) else t
names = {}
for e in evs:
    if e.get('ph') == 'M' and e.get('name') in ('process_name', 'thread_name'):
        names[(e['name'][0], e.get('pid'), e.get('tid'))] = e['args'].get('name')
def lname(p, t):
    return f"{names.get(('p', p, None)) or names.get(('p', p, t)) or p} / {names.get(('t', p, t)) or t}"
lanes = collections.defaultdict(list)
for e in evs:
    if e.get('ph') == 'X' and 'dur' in e:
        lanes[(e.get('pid'), e.get('tid'))].append(e)
short = lambda n: (n.split('(')[0] if '(' in n else n)[-60:]
for k, L in lanes.items():
    if sub not in lname(*k): continue
    L.sort(key=lambda e: e['ts'])
    print(f'## {lname(*k)}: {len(L)} events')
    g = collections.defaultdict(lambda: [0.0, 0])
    for i in range(len(L) - 1):
        gap = L[i + 1]['ts'] - (L[i]['ts'] + L[i]['dur'])
        if gap > 0:
            a = g[(short(L[i]['name']), short(L[i + 1]['name']))]; a[0] += gap; a[1] += 1
    for (p, n), (t, c) in sorted(g.items(), key=lambda x: -x[1][0])[:25]:
        print(f'  {t / 1e3:8.2f} ms {c:6d} x {t / c:7.1f} us   {p}  ->  {n}')
    m = len(L) // 2
    t0 = L[m]['ts']
    for e in L[m:m + NSHOW]:
        print(f"    {e['ts'] - t0:9.1f} +{e['dur']:7.1f}  {short(e['name'])}")
