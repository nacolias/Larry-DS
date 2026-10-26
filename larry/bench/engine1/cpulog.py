#!/usr/bin/env python3
"""cpulog.py LOG [LOG2] - engine1: the CPU-pool experts of TP passes from STRATA_XE_CPU_LOG (tier diagnostics).

Per pass width: passes, CPU-pool experts and layers with any a pass; per layer the experts' count, bytes, wall time
and bandwidth (cold, and hot when STRATA_XE_CPU_TWICE ran them again); where the experts live (arena pinned /
pageable / file); how often each was used before (promotion: MIN 2), the promotion events, reuse distances; with the
pass timestamps (A: a pass starts, B: it is back) the verify and draft time by pass width."""
import sys
from collections import Counter, defaultdict


def load(path):
    T, E, L, ev = {}, [], [], []
    global A, B
    A, B = {}, {}
    for line in open(path):
        f = line.split()
        if not f:
            continue
        k = f[0]
        if k == 'T':
            T[int(f[1])] = int(f[2])
        elif k == 'E':
            E.append(dict(p=int(f[1]), l=int(f[2]), e=int(f[3]), nt=int(f[4]), cls=int(f[5]), pin=int(f[6]),
                          uses=int(f[7]), where=int(f[8]), b=int(f[9])))
        elif k == 'L':
            r = dict(p=int(f[1]), l=int(f[2]), n=int(f[3]), k=int(f[4]), b=int(f[5]), us=float(f[6]), gu=float(f[7]),
                     q=float(f[8]), dn=float(f[9]))
            if len(f) > 10:
                r.update(hot=float(f[10]), hgu=float(f[11]), hq=float(f[12]), hdn=float(f[13]))
            L.append(r)
        elif k == 'A':
            A[int(f[1])] = float(f[2])
        elif k == 'B':
            B[int(f[1])] = float(f[2])
        else:
            ev.append((k, int(f[1]), int(f[2])))
    return T, E, L, ev


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def report(path):
    T, E, L, ev = load(path)
    print(f"== {path}: {len(T)} passes, {len(E)} CPU-pool experts, {len(L)} layers with any")
    byn = defaultdict(list)
    for p, n in T.items():
        byn[n].append(p)
    epp = Counter(r['p'] for r in E)
    lpp = Counter(r['p'] for r in L)
    usp = defaultdict(float)
    for r in L:
        usp[r['p']] += r['us']
    print("  width  passes  experts/pass  layers/pass  cpu ms/pass  verify ms  draft ms after")
    for n in sorted(byn):
        ps = byn[n]
        vt = [B[p] - A[p] for p in ps if p in A and p in B]
        dt = [A[p + 1] - B[p] for p in ps if p in B and p + 1 in A]
        vs = f"{sum(vt) / len(vt):9.2f}" if vt else "        -"
        ds = f"{sum(dt) / len(dt):14.2f}" if dt else "             -"
        print(f"  {n:5d}  {len(ps):6d}  {sum(epp[p] for p in ps) / len(ps):12.2f}  {sum(lpp[p] for p in ps) / len(ps):11.2f}"
              f"  {sum(usp[p] for p in ps) / len(ps) / 1e3:11.2f}  {vs}  {ds}")
    # per layer with CPU experts: k, bytes, time, GB/s (cold / hot)
    print("  k  layers  MiB/layer  us cold  GB/s cold  (gate/up, quant, down ms)  us hot  GB/s hot")
    byk = defaultdict(list)
    for r in L:
        byk[min(r['k'], 5)].append(r)
    for k in sorted(byk):
        rs = byk[k]
        b = sum(r['b'] for r in rs) / len(rs)
        us = sum(r['us'] for r in rs) / len(rs)
        gu = sum(r['gu'] for r in rs) / len(rs); q = sum(r['q'] for r in rs) / len(rs); dn = sum(r['dn'] for r in rs) / len(rs)
        line = f"  {k}{'+' if k == 5 else ' '} {len(rs):6d}  {b / 1048576:9.2f}  {us:7.0f}  {b / us / 1e3:9.1f}  ({gu:.3f}, {q:.3f}, {dn:.3f})"
        hot = sum(r.get('hot', 0) for r in rs) / len(rs)
        if hot > 0:
            line += f"  {hot:6.0f}  {b / hot / 1e3:8.1f}"
        print(line)
    # tokens an expert served in its layer call
    print("  tokens per CPU expert:", dict(sorted(Counter(r['nt'] for r in E).items())))
    print("  arena: pageable %.1f %%, pinned %.1f %%, file %.1f %%" % tuple(pct(sum(1 for r in E if r['pin'] == c), len(E)) for c in (0, 1, 2)))
    print("  classes:", dict(sorted(Counter(r['cls'] for r in E).items())),
          " MiB avg %.2f" % (sum(r['b'] for r in E) / max(1, len(E)) / 1048576))
    u = Counter(min(r['uses'], 3) for r in E)
    print("  earlier CPU uses (promotion counts): 0: %.1f %%, 1: %.1f %%, 2: %.1f %%, 3+: %.1f %%" % tuple(pct(u[i], len(E)) for i in range(4)))
    w = Counter(r['where'] for r in E)
    print("  promotion state at the use: none %.1f %%, in flight %.1f %%" % (pct(w[-1], len(E)), pct(w[-2], len(E))))
    evc = Counter(k for k, _, _ in ev)
    print("  promotion events:", dict(evc))
    # reuse: passes since this expert's previous CPU use (any layer key = (l, e))
    last = {}
    dist = Counter()
    first = 0
    for r in sorted(E, key=lambda r: r['p']):
        key = (r['l'], r['e'])
        if key in last:
            d = r['p'] - last[key]
            dist['0' if d == 0 else '1' if d == 1 else '2-4' if d <= 4 else '5-16' if d <= 16 else '17-64' if d <= 64 else '65+'] += 1
        else:
            first += 1
        last[key] = r['p']
    print("  first CPU use of the run: %.1f %%; repeats by passes since the last: %s" % (
        pct(first, len(E)), {k: round(pct(v, len(E)), 1) for k, v in sorted(dist.items(), key=lambda kv: ['0', '1', '2-4', '5-16', '17-64', '65+'].index(kv[0]))}))
    # where in the stack
    lay = Counter(r['l'] for r in E)
    print("  layers with the most CPU uses:", lay.most_common(10))
    distinct = len(set((r['l'], r['e']) for r in E))
    print(f"  distinct experts {distinct}, {len(E) / max(1, distinct):.2f} uses each")


for p in sys.argv[1:]:
    report(p)
