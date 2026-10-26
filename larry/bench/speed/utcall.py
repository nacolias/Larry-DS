#!/usr/bin/env python3
"""utcall.py TRACE.json [--win K] [--card C] [--dump T0 T1] - an expert card's engines during a traced prompt (unitrace
--chrome-kernel-logging): per device and engine the events by name (count, total ms), and with --dump the card's GPU
events between T0 and T1 ms (relative to the window start) in start order - what a tier call's GPU time is made of."""
import collections
import json
import re
import sys

path = sys.argv[1]
win = int(sys.argv[sys.argv.index('--win') + 1]) if '--win' in sys.argv else -1
card = sys.argv[sys.argv.index('--card') + 1] if '--card' in sys.argv else None
dump = None
if '--dump' in sys.argv:
    k = sys.argv.index('--dump')
    dump = (float(sys.argv[k + 1]), float(sys.argv[k + 2]))
pname, tname, ev = {}, {}, []
with open(path) as f:
    for line in f:
        line = line.strip().rstrip(',')
        if not line.startswith('{"ph"'):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ph = o.get('ph')
        if ph == 'M':
            if o.get('name') == 'process_name':
                pname[o['pid']] = o['args']['name']
            elif o.get('name') == 'thread_name':
                tname[(o['pid'], o['tid'])] = o['args']['name']
        elif ph == 'X' and o.get('cat') == 'gpu_op':
            ev.append((o['ts'], o['dur'], o['pid'], o['tid'], o['name']))
ev.sort()
# windows at gaps > 2 s
wins, cur = [], []
for e in ev:
    if cur and e[0] - max(x[0] + x[1] for x in cur[-50:]) > 2e6:
        wins.append(cur); cur = []
    cur.append(e)
if cur:
    wins.append(cur)
print('windows:', [(len(w), round((w[-1][0] + w[-1][1] - w[0][0]) / 1e3, 1)) for w in wins])
W = wins[win]
t0 = W[0][0]
dev = {}
for p, n in pname.items():
    m = re.search(r'#(\d+)\s*$', n)
    if m:
        dev[p] = m.group(1)
def short(n):
    n = n.split('[')[0]
    if n.startswith('zeCommandList'):
        return n
    m = re.search(r'(xe_[a-z0-9_]+|zc_copy|[A-Za-z_][A-Za-z0-9_]*)(<|\(|$)', n.split('::{lambda')[0].split('(anonymous namespace)::')[-1])
    return (m.group(1) if m else n)[:60]
agg = collections.defaultdict(lambda: [0, 0.0])
for ts, du, p, t, n in W:
    agg[(dev.get(p, p), tname.get((p, t), t), short(n))][0] += 1
    agg[(dev.get(p, p), tname.get((p, t), t), short(n))][1] += du / 1e3
for k in sorted(agg, key=lambda k: (k[0], k[1], -agg[k][1])):
    if card is not None and k[0] != card:
        continue
    if agg[k][1] < 1.0:
        continue
    print('%-3s %-28s %-60s %6d %9.1f ms' % (k[0], k[1], k[2], agg[k][0], agg[k][1]))
if dump:
    for ts, du, p, t, n in W:
        r = (ts - t0) / 1e3
        if dev.get(p) == card and dump[0] <= r <= dump[1]:
            print('%10.3f %8.3f  %-26s %s' % (r, du / 1e3, tname.get((p, t), t), short(n)))
