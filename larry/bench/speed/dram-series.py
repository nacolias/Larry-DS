#!/usr/bin/env python3
# dram-series.py OUT [FROM_S] [TO_S] - dram-mon.sh's CSV as a GB/s series, a line a second (ten 100 ms values)
import sys
from collections import defaultdict
iv = defaultdict(float)
for ln in open(sys.argv[1]):
    p = ln.strip().split(",")
    if len(p) < 4 or not p[0][:1].isdigit():
        continue
    try:
        iv[round(float(p[0]), 1)] += float(p[1]) * 64 / 0.1 / 1e9
    except ValueError:
        pass
a = float(sys.argv[2]) if len(sys.argv) > 2 else 0
b = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
for s in range(int(a), int(min(b, max(iv) + 1))):
    vals = [iv.get(round(s + k / 10 + 0.1, 1), 0.0) for k in range(10)]
    print("%5d s  mean %5.1f | %s" % (s, sum(vals) / 10, " ".join("%2.0f" % v for v in vals)))
