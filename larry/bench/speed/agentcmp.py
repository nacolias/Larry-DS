#!/usr/bin/env python3
"""agentcmp.py LABEL ... - agentx.py's rows (out/agent-LABEL.jsonl) side by side: per request its new tokens and each
label's engine prompt seconds, then the sums over the follow-up turns (reused > 0), the side requests and all, and the
decode tok/s (tokens / seconds over every request)."""
import json
import os
import sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
labels = [lab for lab in sys.argv[1:] if os.path.exists(os.path.join(D, 'agent-%s.jsonl' % lab))]
rows = {lab: [json.loads(ln) for ln in open(os.path.join(D, 'agent-%s.jsonl' % lab)) if ln.strip()] for lab in labels}
n = min(len(r) for r in rows.values()) if rows else 0
print('%-6s %7s %7s | %s' % ('tag', 'new', 'reused', ' '.join('%8s' % lab for lab in labels)))
tot = {lab: {'follow': 0.0, 'side': 0.0, 'all': 0.0, 'dn': 0, 'ds': 0.0} for lab in labels}
for i in range(n):
    r0 = rows[labels[0]][i]
    new = (r0.get('eng_prompt') or 0) - (r0.get('eng_reused') or 0)
    cells = []
    for lab in labels:
        r = rows[lab][i]
        ps = r.get('eng_prompt_s')
        cells.append('%8.2f' % ps if ps is not None else '%8s' % '-')
        if ps is not None:
            t = tot[lab]
            t['all'] += ps
            if (r.get('eng_reused') or 0) > 0:
                t['follow'] += ps
            elif r.get('kind') == 'side' or str(r.get('tag', '')).startswith('side'):
                t['side'] += ps
        if r.get('dec_n'):
            tot[lab]['dn'] += r['dec_n']
            tot[lab]['ds'] += r['dec_s']
    print('%-6s %7d %7d | %s' % (str(r0.get('tag', i))[:6], new, r0.get('eng_reused') or 0, ' '.join(cells)))
for key, name in (('follow', 'follow-up prompts s'), ('side', 'side prompts s'), ('all', 'all prompts s')):
    print('%-22s | %s' % (name, ' '.join('%8.2f' % tot[lab][key] for lab in labels)))
print('%-22s | %s' % ('decode tok/s', ' '.join('%8.1f' % (tot[lab]['dn'] / max(tot[lab]['ds'], 1e-9)) for lab in labels)))
