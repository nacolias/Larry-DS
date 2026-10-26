# sumlag.py TRACE.json - GPU-timestamp view of the decode / verify sum: for each tier sum kernel (run_small's lambda)
# on card 0, the end of the last part it can depend on: the expert cards' part copies (D2D of n_out*4 bytes) and
# card 0's own expert kernels, plus card 0's previous kernel (the dense queue's shexp end, roughly)
import json, sys, collections, re, bisect, statistics
d = json.load(open(sys.argv[1]))
ev = d["traceEvents"] if isinstance(d, dict) else d
names = {e["pid"]: e["args"].get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
dev = collections.defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and "dur" in e and "DEVICE" in str(names.get(e.get("pid"))).upper():
        dev[str(names[e["pid"]])[-2:]].append(e)
for k in dev: dev[k].sort(key=lambda e: e["ts"])
c0 = dev["#0"]
sums = [e for e in c0 if "run_small" in e["name"]]
copies = {}
for k in ["#1", "#2", "#3"]:
    copies[k] = [e for e in dev[k] if "MemoryCopy(D2D)" in e["name"]]
expk0 = [e for e in c0 if re.search(r"gate_up_q8|down_q8|xe_quant_rows|strata::v41::xe::", e["name"])]
ends0 = [e["ts"] + e["dur"] for e in expk0]
rows = []
for s in sums:
    t = s["ts"]
    lastp = []
    for k, cl in copies.items():
        # the latest copy on card k ending before the sum starts, within 1.5 ms
        best = None
        for e in reversed(cl):
            if e["ts"] + e["dur"] <= t:
                if t - (e["ts"] + e["dur"]) < 1500: best = e["ts"] + e["dur"]
                break
        if best: lastp.append(("card" + k[1], best))
    i = bisect.bisect_right(ends0, t) - 1
    if i >= 0 and t - ends0[i] < 1500: lastp.append(("card0", ends0[i]))
    # card 0's kernel just before the sum (any queue)
    j = bisect.bisect_left([e["ts"] for e in c0], t) - 1
    prev_end = c0[j]["ts"] + c0[j]["dur"] if j >= 0 else t
    if not lastp: continue
    who, last = max(lastp, key=lambda p: p[1])
    rows.append((t - last, who, t - prev_end, s["dur"], len(lastp)))
lags = [r[0] for r in rows]
print(f"{len(rows)} sums: lag after the last part: median {statistics.median(lags):.1f} us, mean {statistics.mean(lags):.1f}, p90 {sorted(lags)[int(len(lags)*0.9)]:.1f}")
print(f"  gap after card 0's previous kernel: median {statistics.median([r[2] for r in rows]):.1f} us")
c = collections.Counter(r[1] for r in rows)
print("  last part from:", dict(c))
for who in c:
    l = [r[0] for r in rows if r[1] == who]
    print(f"   {who}: median lag {statistics.median(l):.1f} us (n={len(l)})")
hist = collections.Counter(min(int(r[0] // 10) * 10, 200) for r in rows)
print("  lag histogram (10 us bins):", sorted(hist.items()))
