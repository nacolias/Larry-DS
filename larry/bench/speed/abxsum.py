#!/usr/bin/env python3
"""abxsum.py LABEL ... - abx.py's rows (out/abx-LABEL.jsonl) side by side as markdown: prompts from empty (engine s, the
mean of the reps, and the dense / experts / cache-open phases), the edit (decode tok/s, tokens and ms a pass, lookup),
the decode tasks (tok/s, tokens a pass, ms a pass, CPU-pool experts a pass)."""
import json
import os
import statistics
import sys
from collections import defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
labels = sys.argv[1:]
rows = {lab: [json.loads(ln) for ln in open(os.path.join(D, 'abx-%s.jsonl' % lab)) if ln.strip()]
        for lab in labels if os.path.exists(os.path.join(D, 'abx-%s.jsonl' % lab))}
labels = [lab for lab in labels if lab in rows]


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


print('| test | ' + ' | '.join(labels) + ' |')
print('| --- |' + ' ---: |' * len(labels))
sizes = sorted({int(r['tag'][1:].split('.')[0]) for lab in labels for r in rows[lab] if r['kind'] == 'prompt'})
for n in sizes:
    cells = []
    for lab in labels:
        rs = [r for r in rows[lab] if r['kind'] == 'prompt' and r['tag'].startswith('p%d.' % n)]
        if not rs:
            cells.append('')
            continue
        s = mean([r.get('eng_prompt_s') for r in rs])
        tk = mean([r.get('eng_prompt') for r in rs])
        cells.append('%.2f s = %.0f tok/s (d %.2f e %.2f o %.2f)' % (
            s, tk / s, mean([r.get('ph_dense') for r in rs]), mean([r.get('ph_experts') for r in rs]),
            mean([r.get('ph_open') for r in rs])))
    print('| prompt %d from empty | %s |' % (n, ' | '.join(cells)))
for kind in ('edit', 'dec'):
    tags = []
    for lab in labels:
        for r in rows[lab]:
            if r['kind'] == kind and r['tag'] not in tags:
                tags.append(r['tag'])
    for tag in tags:
        cells = []
        for lab in labels:
            r = next((x for x in rows[lab] if x['tag'] == tag), None)
            if not r or 'dec_n' not in r:
                cells.append('')
                continue
            c = '%.1f tok/s' % r['dec_tps']
            if r.get('dsp_passes'):
                c += ', %.2f tok/p, %.1f ms/p, cpu %.1f/p' % (r['dsp_tpp'], 1e3 * r['dec_s'] / r['dsp_passes'],
                                                             r['tg_cpu'] / r['dsp_passes'])
            if 'lk_passes' in r:
                c += ', lookup %d/%d in %d' % (r['lk_acc'], r['lk_dr'], r['lk_passes'])
            cells.append(c)
        print('| %s | %s |' % (tag, ' | '.join(cells)))
