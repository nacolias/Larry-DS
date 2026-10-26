#!/usr/bin/env python3
"""pcsim.py LOG - engine1: the promotion cache replayed on a run's expert uses (STRATA_XE_CPU_LOG with H lines).

Every use of an expert that is not placed for good (E: the CPU pool ran it, H: a promoted slot did) in pass order;
a policy: promote at the MIN-th use while not resident, the copy lands LAT passes after the next one, at most PAR in
flight, SLOTS slots with LRU.  Prints the CPU-pool uses each policy leaves (the run's own count first)."""
import sys
from collections import OrderedDict, defaultdict


def load(path):
    uses = defaultdict(list)   # pass -> keys used (E and H)
    cpu = 0
    for line in open(path):
        f = line.split()
        if f and f[0] in ('E', 'H'):
            uses[int(f[1])].append((int(f[2]), int(f[3])))
            cpu += f[0] == 'E'
    return uses, cpu


def sim(uses, slots, pmin, lat, par):
    res = OrderedDict()        # key -> True, LRU order (oldest first)
    cnt = defaultdict(int)
    flight = {}                # key -> pass it becomes resident
    cpu = 0
    for p in sorted(uses):
        for k, ready in list(flight.items()):   # landed before this pass
            if ready <= p:
                del flight[k]
                if len(res) >= slots:
                    res.popitem(last=False)
                res[k] = True
        want = []
        for k in uses[p]:
            if k in res:
                res.move_to_end(k)
                continue
            cpu += 1
            if k in flight:
                continue
            cnt[k] += 1
            if cnt[k] >= pmin:
                want.append(k)
        for k in want:
            if len(flight) < par and k not in flight:
                flight[k] = p + 1 + lat
    return cpu


uses, cpu = load(sys.argv[1])
n = sum(len(v) for v in uses.values())
print(f"{sys.argv[1]}: {len(uses)} passes, {n} uses outside the fixed placement, {cpu} of them on the CPU pool in the run")
print("  slots  min  lat  par   cpu uses")
for slots in (369, 600, 1000):
    for pmin in (1, 2):
        for lat in (0, 1, 3):
            for par in (24, 1000):
                print(f"  {slots:5d}  {pmin:3d}  {lat:3d}  {par:4d}  {sim(uses, slots, pmin, lat, par):8d}")
