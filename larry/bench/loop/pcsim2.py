#!/usr/bin/env python3
"""pcsim2.py LOG [SLOTS,...] - the promotion cache's policies replayed on a run's expert uses outside the fixed placement
(STRATA_XE_CPU_LOG: E = the CPU pool ran it, H = a promoted slot did), in pass order.  A promotion lands LAT passes
after the pass that asked for it (0: before the next pass, as STRATA_TP_PC_EARLY + _PUBLISH do), at most PAR in
flight.  Policies, per slot count:
  lru-min2  promote at the second use while not resident, evict the least recently used (the engine's policy)
  lru-min1  promote at the first use
  lfu-adm   promote at the second use only when the candidate's recent use count (halved every HALF passes) beats the
            victim's - the least recently used of the 8 oldest residents with the lowest count (frequency admission)
  belady    promote at the first use, evict the resident used farthest in the future (the bound no policy beats)
Each also with the engine's drop of the whole cache when a prompt's prefetch takes the flex regions (a turn: a gap of more
than GAP ms between two passes, from the A / B timestamps) and without it (the slots kept over prompts - what refilling
them after the prompt could reach).  Prints the CPU-pool uses each leaves, against the run's own count."""
import heapq
import sys
from collections import OrderedDict, defaultdict

LAT, PAR, HALF, GAP = 0, 24, 256, 500.0


def load(path):
    uses = defaultdict(list)   # pass -> keys used outside the fixed placement (E and H)
    cpu = 0
    A, B = {}, {}
    for line in open(path):
        f = line.split()
        if f and f[0] in ('E', 'H'):
            uses[int(f[1])].append((int(f[2]), int(f[3])))
            cpu += f[0] == 'E'
        elif f and f[0] == 'A':
            A[int(f[1])] = float(f[2])
        elif f and f[0] == 'B':
            B[int(f[1])] = float(f[2])
    turns = {p for p in A if p - 1 in B and A[p] - B[p - 1] > GAP}   # the first pass after a prompt
    return uses, cpu, turns


def lru(uses, slots, pmin, drops=()):
    res, cnt, flight, cpu = OrderedDict(), defaultdict(int), {}, 0
    for p in sorted(uses):
        if any(q in drops for q in range(prev[p] + 1, p + 1)):
            res.clear()
            flight.clear()
        for k, ready in list(flight.items()):
            if ready <= p:
                del flight[k]
                if len(res) >= slots:
                    res.popitem(last=False)
                res[k] = True
        for k in uses[p]:
            if k in res:
                res.move_to_end(k)
                continue
            cpu += 1
            if k in flight:
                continue
            cnt[k] += 1
            if cnt[k] >= pmin and len(flight) < PAR:
                flight[k] = p + 1 + LAT
    return cpu


def lfu_adm(uses, slots, drops=()):
    res, freq, seen, flight, cpu = OrderedDict(), defaultdict(float), defaultdict(int), {}, 0
    last_half = 0
    for p in sorted(uses):
        if any(q in drops for q in range(prev[p] + 1, p + 1)):
            res.clear()
            flight.clear()
        if p - last_half >= HALF:
            for k in list(freq):
                freq[k] *= 0.5
            last_half = p
        for k, ready in list(flight.items()):
            if ready <= p:
                del flight[k]
                if len(res) >= slots:
                    old = list(res.keys())[:8]
                    v = min(old, key=lambda x: freq[x])
                    if freq[v] >= freq[k]:
                        continue   # not admitted
                    del res[v]
                res[k] = True
        for k in uses[p]:
            freq[k] += 1
            if k in res:
                res.move_to_end(k)
                continue
            cpu += 1
            if k in flight:
                continue
            seen[k] += 1
            if seen[k] >= 2 and len(flight) < PAR:
                flight[k] = p + 1 + LAT
    return cpu


def belady(uses, slots, drops=()):
    order = sorted(uses)
    nxt, later, seq = {}, {}, []   # nxt[(i, k)]: the next pass index after i that uses k
    for i in range(len(order) - 1, -1, -1):
        ks = set(uses[order[i]])
        for k in ks:
            nxt[(i, k)] = later.get(k, 1 << 40)
            later[k] = i
        seq.append(ks)
    seq.reverse()
    res, cpu = {}, 0   # resident key -> its next use (pass index)
    for i, ks in enumerate(seq):
        if any(q in drops for q in range(prev[order[i]] + 1, order[i] + 1)):
            res.clear()
        cpu += sum(1 for k in uses[order[i]] if k not in res)
        for k in ks:   # after the pass everything it used may stay; the farthest next uses go (a bypass included)
            res[k] = nxt[(i, k)]
        if len(res) > slots:
            for k, _ in heapq.nlargest(len(res) - slots, res.items(), key=lambda kv: kv[1]):
                del res[k]
    return cpu


uses, cpu, turns = load(sys.argv[1])
slots = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [415, 600, 800, 1000, 1500]
order = sorted(uses)
prev = {p: q for q, p in zip([order[0] - 1] + order[:-1], order)}   # the pass with uses before each (drops between)
n = sum(len(v) for v in uses.values())
keys = len(set(k for v in uses.values() for k in v))
P = max(1, len(uses))
print('%s: %d passes with uses, %d turns (gaps over %.0f ms), %d uses outside the fixed placement by %d experts, %d of '
      'them on the CPU pool in the run (%.1f a pass)' % (sys.argv[1], len(uses), len(turns) + 1, GAP, n, keys, cpu, cpu / P))
print('  CPU-pool uses a pass   lru-min2           lru-min1           lfu-adm            belady')
print('  slots              drop    keep       drop    keep       drop    keep       drop    keep')
for s in slots:
    r = []
    for f in (lambda d: lru(uses, s, 2, d), lambda d: lru(uses, s, 1, d), lambda d: lfu_adm(uses, s, d),
              lambda d: belady(uses, s, d)):
        r += [f(turns), f(())]
    print('  %5d         ' % s + '   '.join('%7.2f %7.2f' % (r[i] / P, r[i + 1] / P) for i in range(0, 8, 2)))
