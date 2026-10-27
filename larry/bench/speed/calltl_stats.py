#!/usr/bin/env python3
"""calltl_stats.py ENGINE_LOG - the tier's prompt calls from STRATA_XE_CALL_TL lines ("call tl: layer L n N: prep P cards C
end E light K T | c<k>[s S st ST (NS e NP pin MB MB) A a G g B b E e; g NG r ROWS j NJ] ..."), by call size: per call
the wall (end), the CPU-light part, and per card the phases - start, streamed experts done, GEMMs, the small-expert jobs,
its end - averaged; which side ends last (the CPU-light part or a card) and how often each card is the last."""
import collections
import re
import sys

LINE = re.compile(r"call tl: layer (\d+) n (\d+): prep ([\d.]+) cards ([\d.]+) end ([\d.]+) light (\d+) ([\d.]+) \|(.*)")
CARD = re.compile(r"c(\d+)\[s ([\d.]+) st ([\d.]+) \((\d+) e (\d+) pin ([\d.]+) MB\) A ([\d.]+) G ([\d.]+) B ([\d.]+) E ([\d.]+); g (\d+) r (\d+) j (\d+)\]")
bins = [(1, 128), (129, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 1 << 30)]
agg = collections.defaultdict(lambda: collections.defaultdict(float))
cnt = collections.Counter()
last = collections.defaultdict(collections.Counter)
for ln in open(sys.argv[1], errors="replace"):
    m = LINE.search(ln)
    if not m:
        continue
    n = int(m.group(2))
    b = next(i for i, (lo, hi) in enumerate(bins) if lo <= n <= hi)
    end, light_n, light_t = float(m.group(5)), int(m.group(6)), float(m.group(7))
    a = agg[b]
    cnt[b] += 1
    a["n"] += n
    a["end"] += end
    a["prep"] += float(m.group(3))
    a["light_n"] += light_n
    a["light_t"] += light_t
    worst, wk = light_t, "light"
    for c in CARD.finditer(m.group(8)):
        k = int(c.group(1))
        s, st, ns, npin, mb, A, G, B, E = (float(c.group(i)) for i in range(2, 11))
        a["c%d_E" % k] += E
        a["c%d_st" % k] += st
        a["c%d_mb" % k] += mb
        a["c%d_ns" % k] += ns
        a["c%d_gemm" % k] += max(0.0, G - A) if G > 0 and A > 0 else 0.0
        a["c%d_jobs" % k] += max(0.0, E - B) if B > 0 else 0.0
        a["c%d_n" % k] += 1
        a["c%d_g" % k] += float(c.group(11))
        a["c%d_r" % k] += float(c.group(12))
        a["c%d_j" % k] += float(c.group(13))
        a["c%d_A" % k] += A
        a["c%d_B" % k] += B
        if E > worst:
            worst, wk = E, "c%d" % k
    last[b][wk] += 1
for b, (lo, hi) in enumerate(bins):
    if not cnt[b]:
        continue
    k_ = cnt[b]
    a = agg[b]
    print("calls of %d-%s tokens: %d (mean n %.0f), wall %.1f ms, prep %.1f, CPU-light %.1f experts in %.1f ms; last to end: %s"
          % (lo, hi if hi < 1 << 30 else "", k_, a["n"] / k_, a["end"] / k_, a["prep"] / k_, a["light_n"] / k_, a["light_t"] / k_,
             ", ".join("%s %d %%" % (w, 100 * c // k_) for w, c in last[b].most_common())))
    for k in range(4):
        nk = a["c%d_n" % k]
        if not nk:
            continue
        print("   card %d (%d calls): streamed %.1f experts (%.0f MB) done at %.1f ms, GEMMs %.1f ms (from %.1f; %.0f experts, %.0f rows), jobs %.1f ms (from %.1f; %.0f), ends at %.1f ms"
              % (k, nk, a["c%d_ns" % k] / nk, a["c%d_mb" % k] / nk, a["c%d_st" % k] / nk, a["c%d_gemm" % k] / nk, a["c%d_A" % k] / nk,
                 a["c%d_g" % k] / nk, a["c%d_r" % k] / nk, a["c%d_jobs" % k] / nk, a["c%d_B" % k] / nk, a["c%d_j" % k] / nk, a["c%d_E" % k] / nk))
