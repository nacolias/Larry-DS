#!/usr/bin/env python3
# dram-sum.py OUT [THRESH_GBS] - dram-mon.sh's CSV as GB/s per 100 ms interval; prints each burst (above THRESH, default
# 4 GB/s, gaps of <= 0.5 s merged): its start (s after the monitor's start), length, mean and peak GB/s, and its last
# 3 s as a 100 ms profile (a prompt's phase B + regrow are its end)
import sys
from collections import defaultdict
path = sys.argv[1]
th = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
iv = defaultdict(float)
for ln in open(path):
    p = ln.strip().split(",")
    if len(p) < 4 or not p[0][:1].isdigit():
        continue
    try:
        iv[round(float(p[0]), 1)] += float(p[1]) * 64 / 0.1 / 1e9
    except ValueError:
        pass
ts = sorted(iv)
bursts, cur, gap = [], None, 0
for t in ts:
    g = iv[t]
    if g >= th:
        if cur is None:
            cur = [t, t]
        cur[1] = t
        gap = 0
    elif cur is not None:
        gap += 1
        if gap > 5:
            bursts.append(cur); cur = None; gap = 0
if cur is not None:
    bursts.append(cur)
for a, b in bursts:
    vals = [iv[t] for t in ts if a <= t <= b]
    if b - a < 1.0:
        continue
    print("burst %7.1f s, %5.1f s long: mean %5.1f GB/s, peak %5.1f" % (a, b - a + 0.1, sum(vals) / len(vals), max(vals)))
    tail = [iv[t] for t in ts if b - 3.0 < t <= b]
    print("   last 3 s:", " ".join("%.0f" % v for v in tail))
