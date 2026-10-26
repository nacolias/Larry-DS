# vlayer.py TRACE.json - card 0's verify layers (main model, 5-col fastmv run_r8m<5,...>): one layer's kernel list
# (durations), the median layer's per-kernel durations summed by name over many layers
import json, sys, collections, re, statistics
d = json.load(open(sys.argv[1]))
ev = d["traceEvents"] if isinstance(d, dict) else d
names = {e["pid"]: e["args"].get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
es = sorted([e for e in ev if e.get("ph") == "X" and "dur" in e and str(names.get(e.get("pid"))).endswith("#0")], key=lambda e: e["ts"])
def short(n):
    n = re.sub(r"\(anonymous namespace\)::", "", n)
    m = re.match(r"([^(<\[]*)", n)
    b = (m.group(1) if m else n).strip()
    if "fastmv" in n:
        t = re.search(r"run(_r8m)?<([^>]*)>", n); b = ("fastmv::run" + (t.group(1) or "") + "<" + t.group(2) + ">") if t else b
    sz = re.search(r"\[SIMD\d+ (\{[^}]*\}) (\{[^}]*\})\]$", n)
    return (b[:56] + " " + (sz.group(1) if sz else "")).strip()
# a verify layer starts at the hc_pre / fused hc kernel before the first 5-col mat-vec; split at the q_b-like big
# 5-col kernel: we take windows between consecutive 'run_r8m<5, 4, 2, true>' (wo_a-like, once a layer)
marks = [i for i, e in enumerate(es) if "run_r8m<5, 4, 2, true>" in e["name"]]
layers = []
for a, b in zip(marks, marks[1:]):
    if es[b]["ts"] - es[a]["ts"] > 5000: continue   # a layer is < 5 ms apart even traced
    layers.append(es[a:b])
print(f"{len(layers)} verify layers (wo_a to wo_a)")
tot = collections.defaultdict(list)
for L in layers:
    s = collections.Counter()
    for e in L: s[short(e["name"])] += e["dur"]
    for k, v in s.items(): tot[k].append(v)
n = len(layers)
rows = sorted(((sum(v) / n, k, len(v)) for k, v in tot.items()), reverse=True)
print(f"per layer kernel time {sum(r[0] for r in rows):.1f} us (mean over {n})")
for m, k, c in rows[:45]:
    print(f"  {m:7.1f} us  {c/n:4.2f}/layer  {k}")
