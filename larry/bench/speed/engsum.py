#!/usr/bin/env python3
# engsum.py OUT [FROM_S] [TO_S] - engmon.sh's perf CSV: per interval, each card's bcs0 / ccs0 busy % (cards 0-3)
import sys
from collections import defaultdict
rows = defaultdict(dict)
for ln in open(sys.argv[1]):
    p = ln.strip().split(";")
    if len(p) < 4 or not p[0][:1].isdigit():
        continue
    try:
        rows[float(p[0])][p[3]] = float(p[1])
    except ValueError:
        pass
a = float(sys.argv[2]) if len(sys.argv) > 2 else 0
b = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
print("  t (s)   card: bcs0% / ccs0%")
for t in sorted(rows):
    if not a <= t <= b:
        continue
    r, out = rows[t], []
    for c in ("15", "1b", "22", "26"):
        pre = "xe_0000_%s_00.0/" % c
        act = r.get(pre + "engine-active-ticks,engine_class=1,engine_instance=0,gt=0/", 0)
        tot = r.get(pre + "engine-total-ticks,engine_class=1,engine_instance=0,gt=0/", 0)
        cc = r.get(pre + "engine-active-ticks,engine_class=4,engine_instance=0,gt=0/", 0)
        out.append("%3.0f/%3.0f" % (100 * act / tot, 100 * cc / tot) if tot else "  -/  -")
    print("%7.1f  %s" % (t, "  ".join(out)))
