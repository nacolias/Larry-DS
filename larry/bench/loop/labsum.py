#!/usr/bin/env python3
"""labsum.py OUT_DIR PREFIX RUN... - agentd.py runs side by side (bench/loop/out/agentd-PREFIX-RUN.jsonl): per run the
replies' decode tok/s, tokens and ms a pass, CPU-pool experts a pass (the engine log's lines), the prompts' seconds;
then per turn the decode ms a pass of each run, and the CPU-pool uses a pass from logs/cpulog-PREFIX-RUN.txt."""
import json
import os
import sys

out, pfx, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = {}
for r in runs:
    p = os.path.join(out, 'agentd-%s-%s.jsonl' % (pfx, r))
    rows[r] = [json.loads(ln) for ln in open(p)] if os.path.exists(p) else []
print('%-6s %6s %8s %7s %8s %8s %8s %9s' % ('run', 'turns', 'tokens', 'tok/s', 'tok/pass', 'ms/pass', 'cpu/pass', 'prompts s'))
for r in runs:
    d = [x for x in rows[r] if x.get('dsp_passes')]
    if not d:
        print('%-6s (no rows)' % r)
        continue
    P = sum(x['dsp_passes'] for x in d)
    T = sum(x['dec_n'] for x in d)
    S = sum(x['dec_s'] for x in d)
    C = sum(x['tg_cpu'] for x in d)
    ps = sum(x.get('eng_prompt_s', 0) for x in rows[r])
    print('%-6s %6d %8d %7.1f %8.2f %8.1f %8.1f %9.1f' % (r, len(d), T, T / S, T / P, 1e3 * S / P, C / P, ps))
for r in runs:
    p = '/opt/ai/dsv41/logs/cpulog-%s-%s.txt' % (pfx, r)
    if not os.path.exists(p):
        continue
    passes, e, h = set(), 0, 0
    for ln in open(p):
        f = ln.split()
        if len(f) < 4:
            continue
        if f[0] == 'T':
            passes.add(int(f[1]))
        elif f[0] == 'E':
            e += 1
        elif f[0] == 'H':
            h += 1
    n = max(1, len(passes))
    print('%-6s CPU log: %d passes, CPU-pool uses %.2f a pass, promoted-slot hits %.2f a pass' % (r, len(passes), e / n, h / n))
