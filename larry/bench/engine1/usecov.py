#!/usr/bin/env python3
"""usecov.py USE.txt ... - per STRATA_XE_USAGE_OUT profile: jobs by where the expert sat as decode began (card, -1 CPU
pool), and how concentrated the CPU pool's jobs are (the top N CPU experts' share, their GiB)."""
import sys
from collections import defaultdict
prof = {}
for path in sys.argv[1:]:
    P = 0
    ex = {}
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == 'P':
            P = int(f[1])
        elif f[0] == 'U':
            l, e, jobs, picks, card, b = map(int, f[1:7])
            ex[(l, e)] = (jobs, picks, card, b)
    prof[path] = (P, ex)
    byc = defaultdict(int)
    for (jobs, picks, card, b) in ex.values():
        byc[card] += jobs
    tot = sum(byc.values())
    cpu = sorted(((v[0], v[3], k) for k, v in ex.items() if v[2] == -1 and v[0] > 0), reverse=True)
    cj = sum(j for j, _, _ in cpu)
    print('%s: %d passes, %d jobs (%.1f a pass); by card %s; CPU %d jobs = %.1f a pass, %d experts' % (
        path, P, tot, tot / max(P, 1), {c: round(100.0 * v / tot, 1) for c, v in sorted(byc.items())}, cj, cj / max(P, 1), len(cpu)))
    acc, gib = 0, 0.0
    marks = [50, 100, 200, 400, 800, 1600]
    out = []
    for i, (j, b, k) in enumerate(cpu, 1):
        acc += j
        gib += b / 2 ** 30
        if i in marks:
            out.append('top %d: %.0f%% (%.1f GiB)' % (i, 100.0 * acc / cj, gib))
    print('   ' + '; '.join(out))
    once = sum(1 for j, _, _ in cpu if j == 1)
    print('   CPU experts used once: %d (%.0f%% of CPU jobs)' % (once, 100.0 * once / max(cj, 1)))
# cross-profile: the top-400 CPU experts of one profile, their share of another profile's CPU jobs
names = list(prof)
for a in names:
    Pa, ea = prof[a]
    top = set(k for _, _, k in sorted(((v[0], v[3], k) for k, v in ea.items() if v[2] == -1), reverse=True)[:400])
    row = []
    for b in names:
        Pb, eb = prof[b]
        cj = sum(v[0] for v in eb.values() if v[2] == -1)
        cov = sum(v[0] for k, v in eb.items() if v[2] == -1 and k in top)
        row.append('%.0f%%' % (100.0 * cov / max(cj, 1)))
    print('top-400 CPU set of %s covers the CPU jobs of %s: %s' % (a.split('/')[-1], [n.split('/')[-1] for n in names], row))
