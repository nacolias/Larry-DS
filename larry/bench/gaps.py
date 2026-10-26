# gaps.py TRACE.json - per device: idle-gap histogram, and a coarse timeline (bins) of busy fraction per device
import json, sys, collections
d = json.load(open(sys.argv[1]))
ev = d["traceEvents"] if isinstance(d, dict) else d
names = {e["pid"]: e["args"].get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
dev = collections.defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and "dur" in e and "DEVICE" in str(names.get(e.get("pid"))).upper():
        dev[names[e["pid"]]].append(e)
T0 = min(es[0]["ts"] for es in dev.values() for es in [sorted(es, key=lambda e: e["ts"])])
binw = float(sys.argv[2]) if len(sys.argv) > 2 else 20000.0
tl = {}
for n, es in sorted(dev.items()):
    es.sort(key=lambda e: e["ts"])
    if len(es) < 50: continue
    gaps = [(es[i + 1]["ts"] - (es[i]["ts"] + es[i]["dur"]), es[i]["ts"] + es[i]["dur"], es[i]["name"][:50], es[i + 1]["name"][:50]) for i in range(len(es) - 1)]
    h = collections.Counter()
    hs = collections.Counter()
    for g, *_ in gaps:
        k = "<10us" if g < 10 else "10-100us" if g < 100 else "0.1-1ms" if g < 1000 else "1-10ms" if g < 10000 else ">10ms"
        h[k] += 1; hs[k] += g
    print(f"== {n[-6:]}: " + ", ".join(f"{k}: {h[k]}x {hs[k]/1e3:.1f} ms" for k in ["<10us", "10-100us", "0.1-1ms", "1-10ms", ">10ms"]))
    big = sorted(gaps, key=lambda x: -x[0])[:6]
    for g, t, a, b in big:
        print(f"     gap {g/1e3:7.2f} ms at {(t-T0)/1e3:8.1f} ms  after {a!r} before {b!r}")
    nb = int((es[-1]["ts"] - T0) / binw) + 1
    busy = [0.0] * nb
    for e in es:
        s, f = e["ts"] - T0, e["ts"] - T0 + e["dur"]
        while s < f:
            bi = int(s / binw); be = min(f, (bi + 1) * binw)
            if bi < nb: busy[bi] += be - s
            s = be
    tl[n[-6:]] = busy
keys = sorted(tl)
print("timeline (busy pct per %.0f ms bin): " % (binw / 1e3) + " ".join(keys))
nb = max(len(v) for v in tl.values())
for i in range(min(nb, int(sys.argv[3]) if len(sys.argv) > 3 else 60)):
    print(f"{i*binw/1e3:7.0f} " + " ".join(f"{100*tl[k][i]/binw:5.0f}" if i < len(tl[k]) else "    -" for k in keys))
