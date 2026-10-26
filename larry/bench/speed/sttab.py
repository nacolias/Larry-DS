#!/usr/bin/env python3
"""sttab.py OLD NEW - the standard speed table of two runs side by side (out/speedtab-LABEL.json, out/mtbench-LABEL.json:
run_all.sh's dspark / plain, run_all2.sh's dspark2 / plain2), as markdown: llama-bench's pp512 / tg128 rows (mean +- sd,
DSpark's tokens a pass), the first messages read from empty, and vLLM's serving numbers on MT-Bench."""
import json
import os
import sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
labels = sys.argv[1:]
st = {lab: json.load(open(os.path.join(D, 'speedtab-%s.json' % lab))) for lab in labels}
mt = {}
for lab in labels:
    p = os.path.join(D, 'mtbench-%s.json' % lab)
    if os.path.exists(p):
        mt[lab] = json.load(open(p))

print('| test | ' + ' | '.join(labels) + ' |')
print('| --- |' + ' ---: |' * len(labels))
depths = [dep['depth'] for dep in st[labels[0]]['depths']]
for kind in ('pp', 'tg'):
    for d in depths:
        cells = []
        for lab in labels:
            dep = next(x for x in st[lab]['depths'] if x['depth'] == d)
            c = '%.1f ± %.1f' % (dep[kind + '_mean'], dep[kind + '_sd'])
            if kind == 'tg' and dep.get('dsp_tpp'):
                c += ' (%.2f)' % dep['dsp_tpp']
            cells.append(c)
        n = st[labels[0]]['pp'] if kind == 'pp' else st[labels[0]]['tg']
        print('| %s%d @ d%d | %s |' % (kind, n, d, ' | '.join(cells)))
firsts = []
for lab in labels:
    fs = [dep['first'] for dep in st[lab]['depths'] if dep.get('first')]
    firsts.append(' / '.join('%.1f' % f['eng_prompt_s'] for f in fs))
toks = ' / '.join('{:,}'.format(f['prompt_n']) for f in (dep['first'] for dep in st[labels[0]]['depths'] if dep.get('first')))
print('| a first message from empty: %s tokens, s | %s |' % (toks, ' | '.join(firsts)))
if mt:
    print()
    print('| MT-Bench (vllm bench serve) | ' + ' | '.join(lab for lab in labels if lab in mt) + ' |')
    print('| --- |' + ' ---: |' * len(mt))
    rows = (('time to first token, median / p90 / p99 (s)', lambda m: '%.2f / %.2f / %.2f' % (m['median_ttft_ms'] / 1e3, m['p90_ttft_ms'] / 1e3, m['p99_ttft_ms'] / 1e3)),
            ('time per output token, median (mean), ms', lambda m: '%.1f (%.1f) = %.0f tok/s' % (m['median_tpot_ms'], m['mean_tpot_ms'], 1e3 / m['median_tpot_ms'])),
            ('inter-token latency, median / p90 / p99, ms', lambda m: '%.1f / %.0f / %.0f' % (m['median_itl_ms'], m['p90_itl_ms'], m['p99_itl_ms'])),
            ('end to end, median (s)', lambda m: '%.2f' % (m['median_e2el_ms'] / 1e3)),
            ('output tok/s over the run', lambda m: '%.1f' % m['output_throughput']),
            ('completed / failed', lambda m: '%d / %d' % (m['completed'], m['failed'])))
    for name, f in rows:
        print('| %s | %s |' % (name, ' | '.join(f(mt[lab]) for lab in labels if lab in mt)))
