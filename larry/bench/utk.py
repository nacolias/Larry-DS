# utk.py TRACE.json - per-device kernel summary of a unitrace chrome trace: busy, gaps, top kernels by time
import json, sys, collections, re
d = json.load(open(sys.argv[1]))
ev = d["traceEvents"] if isinstance(d, dict) else d
names = {}
for e in ev:
    if e.get("ph") == "M" and e.get("name") == "process_name":
        names[e["pid"]] = e["args"].get("name")
dev = collections.defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and "dur" in e and e.get("pid") in names and "evice" in str(names.get(e["pid"])) or (e.get("ph") == "X" and "[SIMD" in e.get("name", "")):
        dev[e["pid"]].append(e)
def short(n):
    n = re.sub(r"\(anonymous namespace\)::", "", n)
    m = re.match(r"([^(<]*)(<[^(]*>)?\(", n)
    base = m.group(1) if m else n[:60]
    simd = re.search(r"\[SIMD.*\]$", n)
    if "fastmv" in n or "run_r8m" in n:
        t = re.search(r"run(_r8m)?<([^>]*)>", n)
        base = ("fastmv::run" + (t.group(1) or "") + "<" + t.group(2) + ">") if t else n[:70]
    if "lambda" in base or base.strip() == "":
        base = n[:80]
    return base.strip() + " " + (simd.group(0) if simd else "")
for pid, es in dev.items():
    es.sort(key=lambda e: e["ts"])
    if len(es) < 50: continue
    t0, t1 = es[0]["ts"], es[-1]["ts"] + es[-1]["dur"]
    busy = sum(e["dur"] for e in es)
    gaps = [es[i + 1]["ts"] - (es[i]["ts"] + es[i]["dur"]) for i in range(len(es) - 1)]
    print(f"== {names.get(pid, pid)}: {len(es)} ops, span {(t1 - t0) / 1e3:.1f} ms, busy {busy / 1e3:.1f} ms ({100 * busy / (t1 - t0):.0f}%)")
    gs = sorted(gaps)
    if gs:
        print(f"   gaps: median {gs[len(gs) // 2]:.1f} us, p90 {gs[int(len(gs) * .9)]:.1f}, sum {sum(gs) / 1e3:.1f} ms; gaps <10us sum {sum(g for g in gs if g < 10) / 1e3:.1f} ms, >=10us sum {sum(g for g in gs if g >= 10) / 1e3:.1f} ms")
    agg = collections.defaultdict(lambda: [0, 0.0])
    for e in es:
        k = short(e["name"])
        agg[k][0] += 1; agg[k][1] += e["dur"]
    for k, (c, s) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
        print(f"   {s / 1e3:8.2f} ms {c:6d}x {s / c:7.1f} us  {k[:150]}")
