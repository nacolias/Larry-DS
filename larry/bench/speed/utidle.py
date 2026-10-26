#!/usr/bin/env python3
"""utidle.py TRACE.json [--from MS] [--to MS] [--gap MS] - per card: kernel-busy union, copy-busy union, the idle gaps
(no kernel) longer than --gap ms (default 2) with their start time, and a histogram of idle by gap length."""
import json
import re
import sys

path = sys.argv[1]
t_from = float(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else 0
t_to = float(sys.argv[sys.argv.index('--to') + 1]) if '--to' in sys.argv else 1e12
gmin = float(sys.argv[sys.argv.index('--gap') + 1]) if '--gap' in sys.argv else 2.0
show = '--show' in sys.argv
pname, ev = {}, []
with open(path) as f:
    for line in f:
        line = line.strip().rstrip(',')
        if not line.startswith('{"ph"'):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get('ph') == 'M' and o.get('name') == 'process_name':
            pname[o['pid']] = o['args']['name']
        elif o.get('ph') == 'X' and o.get('cat') == 'gpu_op':
            ev.append((o['ts'], o['dur'], o['pid'], o['name']))
dev = {p: re.search(r'#(\d+)\s*$', n).group(1) for p, n in pname.items() if re.search(r'#(\d+)\s*$', n)}
ev.sort()
t0 = ev[0][0]
def union(iv):
    iv.sort(); tot = 0.0; out = []; cur = None
    for s, e in iv:
        if cur is None or s > cur[1]:
            if cur: out.append(cur)
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur: out.append(cur)
    return out
for card in sorted(set(dev.values())):
    E = [((ts - t0) / 1e3, du / 1e3, n) for ts, du, p, n in ev if dev.get(p) == card and t_from <= (ts - t0) / 1e3 <= t_to]
    if not E: continue
    k = union([(s, s + d) for s, d, n in E if not n.startswith('zeCommandList')])
    c = union([(s, s + d) for s, d, n in E if n.startswith('zeCommandList')])
    span = max(s + d for s, d, n in E) - E[0][0]
    kb = sum(e - s for s, e in k); cb = sum(e - s for s, e in c)
    gaps = [(k[i + 1][0] - k[i][1], k[i][1]) for i in range(len(k) - 1)]
    hist = {}
    for g, s in gaps:
        b = '<0.1' if g < 0.1 else '<1' if g < 1 else '<2' if g < 2 else '<5' if g < 5 else '<10' if g < 10 else '<20' if g < 20 else '<50' if g < 50 else '>=50'
        hist.setdefault(b, [0, 0.0]); hist[b][0] += 1; hist[b][1] += g
    print('card %s: span %.1f ms, kernels busy %.1f (%.0f%%), copies busy %.1f, idle %.1f' % (card, span, kb, 100 * kb / span, cb, span - kb))
    print('   idle by gap:', ', '.join('%s: %d x %.1f ms' % (b, hist[b][0], hist[b][1]) for b in ['<0.1', '<1', '<2', '<5', '<10', '<20', '<50', '>=50'] if b in hist))
    if show:
        for g, s in gaps:
            if g >= gmin:
                print('     gap %8.1f ms at %9.1f' % (g, s))
