#!/usr/bin/env python3
"""utseg.py TRACE.json CARD [--from MS] [--to MS] - one card's prompt timeline cut at its big D2D copies (the tier
call's X read and its part push): per segment the D2D (size, ms), the kernel-busy ms, the H2D copy ms and MB, idle ms."""
import json
import re
import sys

path, card = sys.argv[1], sys.argv[2]
t_from = float(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else 0
t_to = float(sys.argv[sys.argv.index('--to') + 1]) if '--to' in sys.argv else 1e12
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
E = [(ts, du, n) for ts, du, p, n in ev if dev.get(p) == card and t_from <= (ts - t0) / 1e3 <= t_to]
def size(n):
    m = re.search(r'\[(\d+)\]', n)
    return int(m.group(1)) if m and n.startswith('zeCommandList') else 0
cuts = [i for i, (ts, du, n) in enumerate(E) if 'D2D' in n and size(n) >= 8 << 20]
print('card', card, 'events', len(E), 'big D2D', len(cuts))
print('%9s %7s %7s | %8s %8s %8s %8s | %s' % ('t ms', 'D2D MB', 'D2D ms', 'kern ms', 'H2D ms', 'H2D MB', 'idle ms', 'first kernels'))
for a, b in zip(cuts, cuts[1:] + [len(E)]):
    ts, du, n = E[a]
    seg = E[a + 1:b]
    kern = [(s, d, nm) for s, d, nm in seg if not nm.startswith('zeCommandList')]
    h2d = [(s, d, nm) for s, d, nm in seg if 'H2D' in nm]
    kb = sum(d for s, d, nm in kern) / 1e3
    hb = sum(d for s, d, nm in h2d) / 1e3
    hmb = sum(size(nm) for s, d, nm in h2d) / 1e6
    end = (E[b][0] if b < len(E) else seg[-1][0] + seg[-1][1]) if seg else ts + du
    span = (end - (ts + du)) / 1e3
    # idle = segment span minus the union of kernel intervals
    iv = sorted((s, s + d) for s, d, nm in kern)
    busy, cur = 0.0, None
    for s, e in iv:
        if cur is None or s > cur[1]:
            if cur: busy += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur: busy += cur[1] - cur[0]
    names = []
    for s, d, nm in kern[:3]:
        names.append(re.sub(r'.*?(xe_[a-z0-9_]+|zc_copy|dqgemm_t|gate_up_q8|gate_up|down_y_q8|down_y|run_big|run_small).*', r'\1', nm)[:18])
    lead = ((kern[0][0] - (ts + du)) / 1e3) if kern else 0
    print('%9.1f %7.1f %7.2f | %8.2f %8.2f %8.1f %8.2f | lead %.2f %s' % ((ts - t0) / 1e3, size(n) / 1e6, du / 1e3, kb, hb, hmb,
          span - busy / 1e3, lead, ' '.join(names)))
