#!/usr/bin/env python3
"""place.py USE.txt[:STAMPS.err] ... - engine1: the expert placement's per-layer balance, measured and re-dealt.

USE.txt: STRATA_XE_USAGE_OUT (P passes; U layer expert jobs picks card bytes - card as decode began, -1 the CPU pool).
STAMPS.err (optional): the same run's engine log with STRATA_TP_STAMP_LAYER=-1 (per layer: cpu-wait, the cards' expert
step) - fits the GPU model (us = bytes / BW + per-job cost) and gives card 0's CPU wait per layer.

For every profile: the per-layer expected expert time per card under the run's placement, its static imbalance
(sum over layers of the slowest card's lead over the mean, card 0 starting after its CPU wait), and the same for a
re-deal of the SAME VRAM set: each expert (heaviest first) to the card whose layer load stays lowest, within the cards'
byte totals.  A re-deal built from one profile is scored on every profile (domain shift)."""
import re
import sys
from collections import defaultdict


def load_use(path):
    P = 0
    ex = {}   # (l, e) -> [jobs, picks, card, bytes]
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == 'P':
            P = int(f[1])
        elif f[0] == 'U':
            ex[(int(f[1]), int(f[2]))] = [int(f[3]), int(f[4]), int(f[5]), int(f[6])]
    return P, ex


def load_stamps(path):
    rows = {}
    for line in open(path, errors='replace'):
        line = line.replace('\r', '\n').split('\n')[-1] if '\r' in line else line
        m = re.search(r'tp stamps L(\d+) .*cpu-wait (\d+)', line)
        if not m:
            continue
        exs = [float(x) for x in re.findall(r'ex (\d+)', line)]
        rows[int(m.group(1))] = (float(m.group(2)), exs)   # the last block wins
    return rows


def card_loads(P, ex, assign, us_per_byte, us_per_job, NL, NC=4):
    """expected us a pass per (layer, card): jobs/P x (bytes x us_per_byte + us_per_job)"""
    L = [[0.0] * NC for _ in range(NL)]
    for k, (jobs, picks, card, b) in ex.items():
        c = assign.get(k, card)
        if c < 0:
            continue
        L[k[0]][c] += jobs / P * (b * us_per_byte + us_per_job)
    return L


def imbalance(L, cpu):
    s = 0.0
    for l, row in enumerate(L):
        w = list(row)
        w[0] += cpu.get(l, 0.0)
        s += max(w) - sum(w) / len(w)
    return s


def worst(L, cpu):
    return sum(max([row[0] + cpu.get(l, 0.0)] + row[1:]) for l, row in enumerate(L))


def redeal(P, ex, cpu, us_per_byte, us_per_job, NL, NC=4):
    """the same VRAM set, each expert (expected load, heaviest first) to the card with the lowest layer load"""
    cap = [0] * NC
    for k, (jobs, picks, card, b) in ex.items():
        if card >= 0:
            cap[card] += b
    L = [[0.0] * NC for _ in range(NL)]
    for l in range(NL):
        L[l][0] = cpu.get(l, 0.0)
    left = list(cap)
    items = sorted(((jobs / P * (b * us_per_byte + us_per_job), k, b) for k, (jobs, picks, card, b) in ex.items() if card >= 0),
                   reverse=True)
    assign = {}
    for w, k, b in items:
        l = k[0]
        best = None
        for c in range(NC):
            if left[c] < b:
                continue
            key = (L[l][c] + w, -left[c])
            if best is None or key < best[0]:
                best = (key, c)
        if best is None:   # nothing fits: the card with the most room
            c = max(range(NC), key=lambda c: left[c])
        else:
            c = best[1]
        assign[k] = c
        L[l][c] += w
        left[c] -= b
    return assign


def main():
    runs = []
    for arg in sys.argv[1:]:
        up, _, sp = arg.partition(':')
        P, ex = load_use(up)
        st = load_stamps(sp) if sp else {}
        runs.append((up.split('/')[-1], P, ex, st))
    NL = 1 + max(k[0] for _, _, ex, _ in runs for k in ex)
    # fit the GPU model on the runs with stamps: us = a * bytes + b * jobs (least squares over layers x cards 1-3 + card 0)
    import numpy as np
    A, y = [], []
    for name, P, ex, st in runs:
        if not st:
            continue
        byt = defaultdict(float); job = defaultdict(float)
        for (l, e), (jobs, picks, card, b) in ex.items():
            if card >= 0:
                byt[(l, card)] += jobs / P * b
                job[(l, card)] += jobs / P
        for l, (cpu, exs) in st.items():
            for c, v in enumerate(exs):
                A.append([byt[(l, c)], job[(l, c)]]); y.append(v)
    if A:
        sol, res, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
        upb, upj = float(sol[0]), float(sol[1])
        pred = np.array(A) @ sol
        r = np.corrcoef(pred, np.array(y))[0, 1]
        print(f"GPU model fitted on {len(y)} (layer, card) points: {1 / upb / 1e3:.0f} GB/s + {upj:.1f} us a job, r = {r:.3f}")
    else:
        upb, upj = 1 / 450e3, 3.0
        print("no stamps: GPU model 450 GB/s + 3 us a job")
    cpus = {name: {l: v[0] for l, v in st.items()} for name, P, ex, st in runs}
    any_cpu = next((c for c in cpus.values() if c), {})
    for name, P, ex, st in runs:
        cpu = cpus[name] or any_cpu
        L = card_loads(P, ex, {}, upb, upj, NL)
        tot = [sum(L[l][c] for l in range(NL)) for c in range(4)]
        print(f"\n{name}: {P} passes; expected expert us a pass by card {[round(t) for t in tot]}, CPU wait {sum(cpu.values()):.0f}; "
              f"static imbalance {imbalance(L, cpu):.0f} us a pass, the slowest cards' sum {worst(L, cpu):.0f}")
    print("\nre-deal built from (rows) scored on (columns): the slowest cards' sum a pass (us), and the saving vs the run's own placement")
    hdr = ''.join(f"{n:>22s}" for n, _, _, _ in runs)
    print(f"{'':22s}{hdr}")
    for bname, bP, bex, bst in runs:
        assign = redeal(bP, bex, cpus[bname] or any_cpu, upb, upj, NL)
        line = f"{bname:22s}"
        for name, P, ex, st in runs:
            cpu = cpus[name] or any_cpu
            cur = worst(card_loads(P, ex, {}, upb, upj, NL), cpu)
            new = worst(card_loads(P, ex, assign, upb, upj, NL), cpu)
            line += f"{new:>12.0f} ({cur - new:+6.0f})"
        print(line)


if __name__ == '__main__':
    main()
