# draft3.py TRACE.json - card 0's DSpark drafts: a draft = the kernels from the end of a verify pass to its last
# argmax (only the draft runs argmax); verify layers marked by the main model's 5..8-column wo_a / q_b mat-vecs
import json, sys, collections, re
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
isarg = [("argmax" in e["name"]) for e in es]
# the main model's layer: its 'gate_up_q8' / 'down_q8' (card 0's routed experts) or the tier's quant - use the
# fused hc kernels with 5..8 columns? simpler: a verify layer has ggml_sycl_op_dsv41_hc_fb
islay = [("dsv41_hc_fb" in e["name"]) for e in es]
drafts = []
i = 0
n = len(es)
while i < n:
    # find the next argmax run
    while i < n and not isarg[i]: i += 1
    if i >= n: break
    j = i
    while j < n and not islay[j]: j += 1          # the draft's argmaxes end before the next verify layer
    last_arg = max(k for k in range(i, j) if isarg[k])
    # the draft's start: back from the first argmax to the previous verify layer kernel
    s = i
    while s > 0 and not islay[s - 1]: s -= 1
    drafts.append((s, last_arg))
    i = j
drafts = [(a, b) for a, b in drafts if b > a]
tot = collections.Counter(); cnt = collections.Counter(); spans = []
for a, b in drafts[1:-1]:
    seg = es[a + 30:b + 1]   # skip the verify's last layer finish + LM head (~30 kernels)
    spans.append((seg[-1]["ts"] + seg[-1]["dur"] - seg[0]["ts"], sum(e["dur"] for e in seg), len(seg)))
    for e in seg: tot[short(e["name"])] += e["dur"]; cnt[short(e["name"])] += 1
m = len(spans)
print(f"{m} drafts: span {sum(s[0] for s in spans)/m/1e3:.2f} ms, kernel busy {sum(s[1] for s in spans)/m/1e3:.2f} ms, {sum(s[2] for s in spans)/m:.0f} ops each")
for k, v in tot.most_common(40):
    print(f"  {v/m:8.1f} us/draft  {cnt[k]/m:6.1f}x  {v/cnt[k]:7.1f} us  {k}")
