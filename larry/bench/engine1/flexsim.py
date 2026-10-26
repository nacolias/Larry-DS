#!/usr/bin/env python3
"""flexsim.py RANK.bin USE.txt ... - what a bigger promotion cache (STRATA_XE_FLEX_GIB) would leave on the CPU pool, from
STRATA_XE_USAGE_OUT profiles (aggregate jobs, so an LFU steady state: the slots hold the profile's most-used experts
that are not resident).  The resident set as decode began = cards 1-3 + card 0's cache (card >= 0 in USE.txt); a flex
of F GiB a card takes 3 * (F - 1) GiB from the resident set's tail in rank order and gives 14.5 MiB slots."""
import struct, sys
from collections import defaultdict
rk = open(sys.argv[1], 'rb').read()
n = len(rk) // 8
rank = {}
for i in range(n):
    l, e = struct.unpack_from('<ii', rk, 8 * i)
    rank[(l, e)] = i
SLOT = 14.5 * 2 ** 20
for path in sys.argv[2:]:
    P = 0
    ex = {}
    for line in open(path):
        f = line.split()
        if f and f[0] == 'P':
            P = int(f[1])
        elif f and f[0] == 'U':
            l, e, jobs, picks, card, b = map(int, f[1:7])
            ex[(l, e)] = (jobs, card, b)
    res = sorted((k for k, v in ex.items() if v[1] >= 0), key=lambda k: rank.get(k, 1 << 30))
    tot = sum(v[0] for v in ex.values())
    print('%s: %d passes, %d resident experts (%.1f GiB), jobs %.1f a pass' % (
        path.split('/')[-1], P, len(res), sum(ex[k][2] for k in res) / 2 ** 30, tot / P))
    for flex in (1, 1.5, 2, 2.5, 3, 4):
        take = 3 * (flex - 1) * 2 ** 30
        out, acc = set(), 0
        for k in reversed(res):
            if acc >= take:
                break
            out.add(k)
            acc += ex[k][2]
        slots = 372 + int(take / SLOT)
        nonres = sorted((v[0] for k, v in ex.items() if v[1] < 0 or k in out), reverse=True)
        cpu_all = sum(nonres)
        cov = sum(nonres[:slots])
        moved = sum(ex[k][0] for k in out)
        print('   flex %.1f: %4d slots, %4d experts moved out (their jobs %.1f a pass); non-resident jobs %.1f a pass, '
              'left after the slots %.1f a pass' % (flex, slots, len(out), moved / P, cpu_all / P, (cpu_all - cov) / P))
