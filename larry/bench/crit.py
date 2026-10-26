# crit.py TRACE.json - for every idle gap of card 0 (>= 1 ms): how much of it the expert cards spend in kernels, in
# copies only, or idle (all of them) - is the dense card waiting for the tier's compute, its streaming, or the host?
import json, sys, collections, bisect
d = json.load(open(sys.argv[1]))
ev = d["traceEvents"] if isinstance(d, dict) else d
names = {e["pid"]: e["args"].get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
dev = collections.defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and "dur" in e and "DEVICE" in str(names.get(e.get("pid"))).upper():
        dev[names[e["pid"]]].append((e["ts"], e["ts"] + e["dur"], "[SIMD" in e.get("name", ""), e.get("name", "")))
keys = sorted(dev, key=lambda n: n[-2:])
c0 = [k for k in keys if k.endswith("#0")][0]
others = [k for k in keys if k != c0]
def merge(iv):
    iv = sorted(iv); out = []
    for s, f in iv:
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], f)
        else: out.append([s, f])
    return out
kern = merge([(s, f) for k in others for s, f, isk, _ in dev[k] if isk])
copy = merge([(s, f) for k in others for s, f, isk, _ in dev[k] if not isk])
anyo = merge([(s, f) for k in others for s, f, _, _ in dev[k]])
def overlap(ivs, a, b):
    i = bisect.bisect_left([x[1] for x in ivs], a) if ivs else 0
    tot = 0.0
    while i < len(ivs) and ivs[i][0] < b:
        tot += max(0.0, min(b, ivs[i][1]) - max(a, ivs[i][0])); i += 1
    return tot
# precompute end arrays for bisect speed
kern_e = [x[1] for x in kern]; anyo_e = [x[1] for x in anyo]
def ov(ivs, ends, a, b):
    i = bisect.bisect_left(ends, a); tot = 0.0
    while i < len(ivs) and ivs[i][0] < b:
        tot += max(0.0, min(b, ivs[i][1]) - max(a, ivs[i][0])); i += 1
    return tot
e0 = merge([(s, f) for s, f, _, _ in dev[c0]])
gaps = [(e0[i][1], e0[i + 1][0]) for i in range(len(e0) - 1) if e0[i + 1][0] - e0[i][1] >= 1000]
tk = tc = ti = 0.0
big = []
for a, b in gaps:
    k = ov(kern, kern_e, a, b); x = ov(anyo, anyo_e, a, b)
    tk += k; tc += x - k; ti += (b - a) - x
    big.append((b - a, a, k, x - k))
T0 = e0[0][0]
print(f"card 0: busy (merged, kernels+copies) {sum(f - s for s, f in e0)/1e6:.2f} s of span {(e0[-1][1]-e0[0][0])/1e6:.2f} s; "
      f"gaps >= 1 ms: {len(gaps)}, {sum(b-a for a,b in gaps)/1e6:.2f} s")
print(f"  during those gaps the expert cards: some kernel running {tk/1e6:.2f} s, only copies {tc/1e6:.2f} s, all idle {ti/1e6:.2f} s")
for g, a, k, c in sorted(big, reverse=True)[:12]:
    print(f"    gap {g/1e3:7.1f} ms at {(a-T0)/1e6:6.2f} s: experts kernels {k/1e3:6.1f} ms, copies-only {c/1e3:6.1f} ms, idle {(g-k-c)/1e3:6.1f} ms")
# per-second profile of card 0 gaps by class
bins = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
for g, a, k, c in big:
    bi = int((a - T0) / 1e6)
    bins[bi][0] += k; bins[bi][1] += c; bins[bi][2] += g - k - c
print("  per second: card-0 gap ms (experts kernel / copies-only / all idle)")
for bi in sorted(bins):
    k, c, i = bins[bi]
    print(f"    {bi:3d} s: {k/1e3:6.0f} {c/1e3:6.0f} {i/1e3:6.0f}")
