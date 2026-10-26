#!/usr/bin/env python3
"""utwin.py TRACE.json T0 T1 [--bin MS] - every card's activity between T0 and T1 ms (window-relative) in bins of
--bin ms (default 2): per bin and card K = kernel-busy %, H = H2D MB, D = D2D MB started in the bin."""
import json
import re
import sys

path, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
bw = float(sys.argv[sys.argv.index('--bin') + 1]) if '--bin' in sys.argv else 2.0
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
nb = int((b - a) / bw) + 1
K = {c: [0.0] * nb for c in '0123'}
H = {c: [0.0] * nb for c in '0123'}
D = {c: [0.0] * nb for c in '0123'}
def size(n):
    m = re.search(r'\[(\d+)\]', n)
    return int(m.group(1)) if m else 0
for ts, du, p, n in ev:
    c = dev.get(p)
    if c is None: continue
    s, e = (ts - t0) / 1e3, (ts + du - t0) / 1e3
    if e < a or s > b: continue
    if n.startswith('zeCommandList'):
        i = int((s - a) / bw)
        if 0 <= i < nb:
            if 'H2D' in n: H[c][i] += size(n) / 1e6
            elif 'D2D' in n: D[c][i] += size(n) / 1e6
        continue
    for i in range(max(0, int((s - a) / bw)), min(nb, int((e - a) / bw) + 1)):
        lo, hi = a + i * bw, a + (i + 1) * bw
        K[c][i] += max(0.0, min(e, hi) - max(s, lo)) / bw * 100
print('%8s ' % 'ms' + ' | '.join('c%s  K%%   H MB   D MB' % c for c in '123'))
for i in range(nb):
    print('%8.1f ' % (a + i * bw) + ' | '.join('   %4.0f %6.1f %6.1f' % (K[c][i], H[c][i], D[c][i]) for c in '123'))
