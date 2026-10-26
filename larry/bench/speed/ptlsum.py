#!/usr/bin/env python3
"""ptlsum.py logs/ptl.PID.tsv [BATCH] [-v] - STRATA_GPU_PTL timelines (patch-ptl.py): per prompt batch, card 0's queue
(ring 0) as a sequence of steps - dense spans (D: attention + router, F: finish), the tier's card-0 part (c0), the
combine (cb) - with the time the queue sat EMPTY before each step (the host had not queued it yet: idle) and the
time the step itself took on the card; the idle split by what came next and by layer range; each expert card's calls
(X read, prefetch barrier, GEMM / job rounds, push) and the prefetch DMAs' finish relative to the layer's first call.

Rings: 0 the dense queue, 1 + k card k's tier queue, 9 + k card k's prefetch queue, 17 host steps.  Tags: 1 D span,
2 F span, 20 / 21 card 0's part start / end, 22 / 23 the combine queued / done, 10 X read done, 11 prefetch barrier
passed, 12 GEMM round done, 13 job round done, 14 push done, 30 a layer's prefetch DMAs done, 40 tier launch, 41 / 42
join start / end, 43 / 44 pass_sync start / end.  Times are host-clock ms (a record's done time is when the monitor
saw its flag, ~20 us late)."""
import collections
import sys

path = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].lstrip('-').isdigit() else None
verbose = '-v' in sys.argv
batches, cur = [], None
for line in open(path):
    if line.startswith('# batch'):
        cur = []
        batches.append(cur)
        continue
    if line.startswith('#'):   # '# ring N: K records not seen'
        if verbose: print(line.rstrip())
        continue
    p = line.split('\t')
    if len(p) < 6 or cur is None:
        continue
    cur.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), float(p[4]), abs(float(p[5]))))   # td < 0: inferred

NAMES = {1: 'D', 2: 'F', 20: 'c0', 21: 'c0e', 22: 'cb', 23: 'cbe', 11: 'pf', 12: 'G', 13: 'J'}


def card0(B):
    r0 = [e for e in B if e[0] == 0]
    r0.sort(key=lambda e: e[5])
    host = sorted([e for e in B if e[0] == 17], key=lambda e: e[4])
    t0 = min(e[4] for e in B)
    # merge start / end pairs into steps with (kind, layer, start host time, GPU done time)
    idle = collections.Counter()
    busy = collections.Counter()
    idle_l = collections.Counter()
    gaps = []
    prev_done = None
    for e in r0:
        ring, tag, l, a, th, td = e
        kind = NAMES.get(tag, str(tag))
        if prev_done is not None:
            # the queue was empty from prev_done until the host began this step (th), if th is later
            gap = max(0.0, th - prev_done) if tag in (1, 2, 20, 22) else 0.0
            work = td - max(prev_done, th) if tag in (1, 2, 20, 22) else td - prev_done
            idle[kind] += gap
            busy[kind] += max(0.0, work)
            idle_l['0-19' if l < 20 else '20-39'] += gap
            if gap > 2.0:
                gaps.append((th - t0, gap, kind, l, a))
        prev_done = td
    span = r0[-1][5] - r0[0][4] if r0 else 0
    return span, idle, busy, idle_l, gaps, t0


def cards(B, t0):
    calls = collections.defaultdict(dict)   # (card, layer, n) -> tag -> list
    for ring, tag, l, a, th, td in B:
        if 1 <= ring <= 8:
            c = ring - 1
            calls[(c, l)].setdefault(tag, []).append((th, td, a))
    return calls


for bi, B in enumerate(batches):
    if want is not None and bi != want:
        continue
    if not B:
        continue
    span, idle, busy, idle_l, gaps, t0 = card0(B)
    ncol = max([e[3] for e in B if e[0] == 0 and e[1] == 1] + [0]) + 512
    print('batch %d (%d records, ~%d columns): card 0 queue span %.0f ms, idle %.0f ms (layers 0-19 %.0f, 20-39 %.0f)' % (
        bi, len(B), ncol, span, sum(idle.values()), idle_l['0-19'], idle_l['20-39']))
    print('   idle before:  ' + '  '.join('%s %.0f' % (k, v) for k, v in idle.most_common()))
    print('   step time:    ' + '  '.join('%s %.0f' % (k, v) for k, v in busy.most_common()))
    nd = sum(1 for e in B if e[0] == 0 and e[1] == 1)
    nf = sum(1 for e in B if e[0] == 0 and e[1] == 2)
    if nd:
        print('   per span: D %.2f ms (%d), F %.2f ms (%d)' % (busy['D'] / nd, nd, busy['F'] / max(nf, 1), nf))
    # host steps: joins and pass_syncs
    host = sorted([e for e in B if e[0] == 17], key=lambda e: e[4])
    js, ps = 0.0, 0.0
    for i, e in enumerate(host):
        if e[1] == 41:
            nxt = next((h for h in host[i + 1:] if h[1] == 42), None)
            if nxt: js += nxt[4] - e[4]
        if e[1] == 43:
            nxt = next((h for h in host[i + 1:] if h[1] == 44), None)
            if nxt: ps += nxt[4] - e[4]
    print('   host: joins %.0f ms, pass_syncs %.0f ms' % (js, ps))
    # the combine: queued (host) vs the queue reaching it; its own time
    cb = sorted([e for e in B if e[0] == 0 and e[1] in (22, 23)], key=lambda e: e[5])
    late, cbt, ncb = 0.0, 0.0, 0
    for e in cb:
        if e[1] == 22:
            ncb += 1
    # expert cards per call
    calls = cards(B, t0)
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    for (c, l), tg in calls.items():
        if 10 not in tg or 14 not in tg:
            continue
        for i, (th10, td10, n) in enumerate(tg[10]):
            if i >= len(tg[14]):
                break
            td14 = tg[14][i][1]
            td11 = tg[11][i][1] if 11 in tg and i < len(tg[11]) else td10
            ph = 'A' if l < 20 else 'B'
            k = (ph, i)
            agg[k][0] += 1
            agg[k][1] += td10 - th10          # X read: queued -> done
            agg[k][2] += max(0.0, td11 - td10)  # prefetch barrier wait after the X read
            agg[k][3] += td14 - td11          # compute + push
            agg[k][4] += td14 - th10          # the whole part
    for k in sorted(agg):
        n, x, pf, cp, tot = agg[k]
        print('   expert cards, layers %s call %d: %d card-calls: X read %.1f ms, then prefetch wait %.1f, compute + push %.1f; part %.1f ms' % (
            '0-19' if k[0] == 'A' else '20-39', k[1], n, x / n, pf / n, cp / n, tot / n))
    # prefetch DMAs done vs the layer's first X read on that card
    pfd = {}
    for ring, tag, l, a, th, td in B:
        if tag == 30:
            pfd[(a, l)] = (th, td)
    lead = collections.defaultdict(list)
    for (c, l), tg in calls.items():
        if 10 in tg and (c, l) in pfd:
            first_x = tg[10][0][1]
            lead['0-19' if l < 20 else '20-39'].append(pfd[(c, l)][1] - first_x)
    for k in sorted(lead):
        v = lead[k]
        print('   prefetch DMAs done relative to the layer\'s first X read (layers %s): avg %+.1f ms, max %+.1f, %d of %d late' % (
            k, sum(v) / len(v), max(v), sum(1 for x in v if x > 0), len(v)))
    if verbose:
        for g in sorted(gaps, key=lambda g: -g[1])[:40]:
            print('     idle %6.1f ms at %8.1f before %s L%d col %d' % (g[1], g[0], g[2], g[3], g[4]))
