#!/usr/bin/env python3
"""cyc.py LOG... - engine1: DSpark runs compared by cycle time (verify pass + the draft after it) per pass width, from the
A / B pass timestamps of STRATA_XE_CPU_LOG, weighted by one width mix pooled over the runs given (tokens a pass move
with the text, so tok/s alone does not compare runs); also CPU-pool experts a pass and the one-expert layer time."""
import sys
from collections import defaultdict
def load(p):
    T, A, B, E, L = {}, {}, {}, 0, []
    for line in open(p):
        f = line.split()
        if not f: continue
        if f[0] == 'T': T[int(f[1])] = int(f[2])
        elif f[0] == 'A': A[int(f[1])] = float(f[2])
        elif f[0] == 'B': B[int(f[1])] = float(f[2])
        elif f[0] == 'E': E += 1
        elif f[0] == 'L' and int(f[4]) == 1: L.append(float(f[6]))
    return T, A, B, E, L
runs = sys.argv[1:]
data = {r: load(r) for r in runs}
# common width distribution: all runs pooled, widths 1-6 (7 = the prompt's passes)
dist = defaultdict(int)
for r in runs:
    T = data[r][0]
    for p, n in T.items():
        if n <= 6: dist[n] += 1
tot = sum(dist.values())
print("width mix (all runs):", {n: round(dist[n] / tot, 3) for n in sorted(dist)})
print(f"{'run':22s} {'passes':>6s} {'cpu/pass':>8s} {'k1 us':>6s}  verify by width 1..6                        draft-after by width 1..6                   cycle@mix")
for r in runs:
    T, A, B, E, L = data[r]
    v = defaultdict(list); d = defaultdict(list)
    for p, n in T.items():
        if n > 6: continue
        if p in A and p in B: v[n].append(B[p] - A[p])
        if p in B and p + 1 in A: d[n].append(A[p + 1] - B[p])
    mv = {n: sum(v[n]) / len(v[n]) for n in v if v[n]}
    md = {n: sum(d[n]) / len(d[n]) for n in d if d[n]}
    cyc = sum(dist[n] / tot * (mv.get(n, 0) + md.get(n, 0)) for n in dist)
    npass = sum(1 for n in T.values() if n <= 6)
    print(f"{r.split('/')[-1]:22s} {npass:6d} {E / max(1, len(T)):8.2f} {sum(L) / max(1, len(L)):6.0f}  " +
          " ".join(f"{mv.get(n, 0):6.2f}" for n in range(1, 7)) + "   " + " ".join(f"{md.get(n, 0):6.2f}" for n in range(1, 7)) + f"   {cyc:6.2f}")
