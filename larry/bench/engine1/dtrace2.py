#!/usr/bin/env python3
"""dtrace2.py DIR LANE_SUBSTR [gap_us] - dtrace.py's bursts on one device lane, split by kernel count class: per class
the device time, span, the idle gaps inside, and the top kernels; plus the bursts' order (which classes follow which)."""
import json, sys, os, glob, collections
d, want = sys.argv[1], sys.argv[2]
GAP = float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0
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
for k, L in lanes.items():
    if want not in lname(k): continue
    L.sort(key=lambda e: e['ts'])
    bursts, cur = [], [L[0]]
    for a, b in zip(L, L[1:]):
        if b['ts'] - (a['ts'] + a['dur']) > GAP:
            bursts.append(cur); cur = []
        cur.append(b)
    bursts.append(cur)
    core = bursts[1:-1]
    cls = collections.defaultdict(list)
    for b in core: cls[len(b)].append(b)
    print(f'## {lname(k)}: {len(core)} bursts')
    seq = ' '.join(str(len(b)) for b in core[:40])
    print('  order:', seq)
    for n, bs in sorted(cls.items(), key=lambda x: -len(x[1]))[:4]:
        busy = sum(sum(e['dur'] for e in b) for b in bs) / len(bs)
        span = sum(b[-1]['ts'] + b[-1]['dur'] - b[0]['ts'] for b in bs) / len(bs)
        print(f'\n  class {n} kernels: {len(bs)} bursts, device {busy / 1e3:.3f} ms, span {span / 1e3:.3f} ms')
        agg = collections.defaultdict(lambda: [0.0, 0])
        for b in bs:
            for e in b:
                nm = e['name'].split('(')[0][:90]
                a = agg[nm]; a[0] += e['dur']; a[1] += 1
        for nm, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:14]:
            print(f'    {t / len(bs):8.1f} us {c / len(bs):6.1f} x {t / c:7.2f} us  {nm}')
        # the biggest gaps inside: after which kernel
        gapk = collections.defaultdict(float)
        for b in bs:
            for a, c in zip(b, b[1:]):
                g = c['ts'] - (a['ts'] + a['dur'])
                if g > 20: gapk[a['name'].split('(')[0][:60] + ' -> ' + c['name'].split('(')[0][:40]] += g
        for nm, g in sorted(gapk.items(), key=lambda x: -x[1])[:6]:
            print(f'    gap {g / len(bs):8.1f} us after  {nm}')
