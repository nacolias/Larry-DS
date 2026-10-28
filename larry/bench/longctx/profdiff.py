#!/usr/bin/env python3
"""profdiff.py ENGINE_LOG [TOP] - GGML_SYCL_OP_PROF's cumulative dumps ("larry op prof (periodic|exit): T ms over N
labels", then "  LABEL  MS ms  PCT %  COUNT x  US us" lines) turned into one table an interval: the card time between
two dumps by op (ms, share, calls, us a call), with the batches STRATA_BATCH_TRACE logged meanwhile ("batch trace: MS
batch start B") - so the positions the interval's batches read (the prompt's tokens from its start, B a batch)."""
import re
import sys

HDR = re.compile(r"larry op prof \((\w+)\): ([\d.]+) ms over (\d+) labels, (\d+) marks dropped")
ROW = re.compile(r"^  (.{44}) +([\d.]+) ms +[\d.]+ % +(\d+) x")
BAT = re.compile(r"batch trace: +([\d.]+)  batch start (\d+)")
PROMPT = re.compile(r"serve: prompt (\d+) tokens \((\d+) reused")
top = int(sys.argv[2]) if len(sys.argv) > 2 else 14
dumps = []   # (kind, total ms, {label: (ms, n)}, batches so far, tokens so far)
cur = None
nb = 0
tok = 0
for ln in open(sys.argv[1], errors="replace"):
    m = HDR.search(ln)
    if m:
        cur = {}
        dumps.append((m.group(1), float(m.group(2)), cur, nb, tok, int(m.group(4))))
        continue
    if cur is not None:
        r = ROW.match(ln)
        if r:
            cur[r.group(1).strip()] = (float(r.group(2)), int(r.group(3)))
            continue
        cur = None
    b = BAT.search(ln)
    if b:
        nb += 1
        tok += int(b.group(2))
        continue
    p = PROMPT.search(ln)
    if p:
        print("# %s" % ln.strip()[:150])
prev = ("start", 0.0, {}, 0, 0, 0)
for d in dumps:
    kind, tot, ops, b1, t1, dropped = d
    _, ptot, pops, b0, t0, _ = prev
    dt = tot - ptot
    if dt <= 0:
        prev = d
        continue
    print("\n== %s dump: %.1f s of card time since the last (%.1f s in all, %d marks dropped); batches %d..%d started meanwhile "
          "(tokens %d..%d of the prompts)" % (kind, dt / 1e3, tot / 1e3, dropped, b0 + 1, b1, t0, t1))
    rows = []
    for k, (ms, n) in ops.items():
        pms, pn = pops.get(k, (0.0, 0))
        if ms - pms > 0:
            rows.append((ms - pms, n - pn, k))
    rows.sort(reverse=True)
    for ms, n, k in rows[:top]:
        print("   %-44s %9.1f ms %5.1f %%  %8d x %9.1f us" % (k, ms, 100 * ms / dt, n, 1e3 * ms / max(n, 1)))
    prev = d
