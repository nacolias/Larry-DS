#!/usr/bin/env python3
"""calltl2.py ENGINE.log [BATCH] - STRATA_XE_CALL_TL lines grouped into prompt batches (layer 0 starts one): per call the
largest A / G / B / E over the expert cards, the call's end; averages per batch for the layers' first / second calls in
layers 0-19 and 20-39."""
import re
import sys

rx = re.compile(r'call tl: layer (\d+) n (\d+): prep ([\d.]+) cards ([\d.]+) end ([\d.]+)')
cx = re.compile(r'c(\d)\[s ([\d.]+) st ([\d.]+) \((\d+) e (\d+) pin ([\d.]+) MB\) A ([\d.]+) G ([\d.]+) B ([\d.]+) E ([\d.]+); g (\d+) r (\d+) j (\d+)\]')
batches, cur, prev_l = [], [], 99
for line in open(sys.argv[1], errors='replace'):
    m = rx.search(line)
    if not m:
        continue
    l, n = int(m.group(1)), int(m.group(2))
    if l < prev_l and cur and not (l == prev_l):
        if l == 0 and any(c['l'] > 0 for c in cur):
            batches.append(cur); cur = []
    prev_l = l
    cards = {}
    for c in cx.finditer(line):
        cards[int(c.group(1))] = dict(s=float(c.group(2)), A=float(c.group(7)), G=float(c.group(8)), B=float(c.group(9)),
                                      E=float(c.group(10)), g=int(c.group(11)), r=int(c.group(12)), j=int(c.group(13)), nst=int(c.group(4)))
    cur.append(dict(l=l, n=n, prep=float(m.group(3)), cards_t=float(m.group(4)), end=float(m.group(5)), c=cards))
if cur:
    batches.append(cur)
sel = [int(sys.argv[2])] if len(sys.argv) > 2 else range(len(batches))
for bi in sel:
    B = batches[bi]
    print('batch %d: %d calls, n max %d' % (bi, len(B), max(c['n'] for c in B)))
    seen = {}
    rows = {}
    for c in B:
        k = seen.get(c['l'], 0); seen[c['l']] = k + 1
        ph = 'A' if c['l'] < 20 else 'B'
        ex = [v for i, v in c['c'].items() if i > 0]
        if not ex: continue
        r = rows.setdefault((ph, k), [])
        r.append((c['end'], max(v['A'] for v in ex), max(v['G'] for v in ex), max(v['E'] for v in ex), min(v['E'] for v in ex),
                  c['c'][0]['E'] if 0 in c['c'] else 0.0, c['prep'], c['n']))
    for key in sorted(rows):
        r = rows[key]
        avg = lambda i: sum(x[i] for x in r) / len(r)
        print('  layers %s call %d: %3d calls n~%5.0f  prep %5.1f  A max %5.1f  G max %5.1f  E max %5.1f (min %5.1f)  c0 E %5.1f  end %5.1f' % (
            '0-19' if key[0] == 'A' else '20-39', key[1], len(r), avg(7), avg(6), avg(1), avg(2), avg(3), avg(4), avg(5), avg(0)))
