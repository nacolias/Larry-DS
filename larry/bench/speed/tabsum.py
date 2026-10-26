#!/usr/bin/env python3
"""tabsum.py NAME ... - the short-prompt A/B's table: per config (out/tab-NAME-PP.json) and test (pp at depth), the
prompt rate mean +- sd, its phases (dense / experts seconds), experts by tier (VRAM hit / CPU / PCIe) and tg."""
import glob
import json
import os
import statistics
import sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
rows = []
for name in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(D, 'tab-%s-*.json' % name))):
        d = json.load(open(p))
        for dep in d['depths']:
            rs = dep['runs']
            if not rs:
                continue
            f = lambda k: statistics.mean(r.get(k, 0) or 0 for r in rs)
            pp = [r['eng_pp'] for r in rs if r.get('eng_pp')]
            rows.append((d['pp'], dep['depth'], name, statistics.mean(pp), statistics.stdev(pp) if len(pp) > 1 else 0,
                         f('ph_s'), f('ph_dense'), f('ph_experts'), f('pp_hit'), f('pp_cpu'), f('pp_pcie'), f('dec_tps')))
rows.sort()
print('%5s %6s %-6s %14s %7s %6s %7s %7s %6s %6s %6s' % ('pp', 'depth', 'cfg', 'tok/s', 'prompt', 'dense', 'experts',
                                                        'hit', 'cpu', 'pcie', 'tg'))
for r in rows:
    print('%5d %6d %-6s %7.1f +- %4.1f %6.2fs %6.2f %7.2f %7.0f %6.0f %6.0f %6.1f' % r)
