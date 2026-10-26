#!/usr/bin/env python3
# memsum3.py ABW_LOG MEMTRACE3_LOG [VMWATCH2_JSONL] - per config of an abw.sh window: memtrace3.bt's counters summed
# over its load (server start .. up) and its prompts (up .. done), in MiB (pages * 4 KiB) or counts; with vmwatch2's
# file: kswapd / direct reclaim scans (MiB), the page cache's growth (MiB) and the lowest free memory (GiB) too.
import datetime, json, re, sys
def secs(h):
    a, b, c = (int(x) for x in h.split(":")); return a * 3600 + b * 60 + c
cfg, order = {}, []
for l in open(sys.argv[1], errors="replace"):
    m = re.match(r"\[(\w+)\] server pid \d+, env:.*?(\d\d:\d\d:\d\d)\s*$", l)
    if m: cfg[m.group(1)] = {"start": secs(m.group(2))}; order.append(m.group(1)); continue
    m = re.match(r"\[(\w+)\] (up|done) (\d\d:\d\d:\d\d)", l)
    if m and m.group(1) in cfg: cfg[m.group(1)][m.group(2)] = secs(m.group(3))
K = ["fe", "fo", "ae", "ao", "se", "so", "bk", "rs"]
rows = []
for l in open(sys.argv[2], errors="replace"):
    m = re.match(r"(\d\d:\d\d:\d\d) fadd (\d+) (\d+) anon (\d+) (\d+) swin (\d+) (\d+) bk (\d+) rs (\d+)", l)
    if m: rows.append((secs(m.group(1)), [int(x) for x in m.groups()[1:]]))
mib = lambda p: p * 4096 / 2**20
print("%-5s %-7s %10s %10s %10s %10s %8s %8s %9s %9s" % ("cfg", "phase", "fadd_eng", "fadd_oth", "anon_eng", "anon_oth",
      "swin_e", "swin_o", "bk_MiB", "rs_MiB"))
for n in order:
    c = cfg[n]
    for ph, a, b in (("load", c.get("start"), c.get("up")), ("prompts", c.get("up"), c.get("done"))):
        if a is None or b is None: continue
        s = [0] * len(K)
        for t, v in rows:
            if a < t <= b + 2: s = [x + y for x, y in zip(s, v)]
        print("%-5s %-7s %8.0fMB %8.0fMB %10d %10d %8d %8d %9.0f %9.0f" % (n, ph, mib(s[0]), mib(s[1]), s[2], s[3], s[4], s[5], mib(s[6]), mib(s[7])))

if len(sys.argv) > 3:
    vw = []
    for l in open(sys.argv[3]):
        r = json.loads(l)
        t = datetime.datetime.fromtimestamp(r["t"], datetime.timezone.utc)
        vw.append((t.hour * 3600 + t.minute * 60 + t.second, r))
    print()
    print("%-5s %-7s %10s %10s %10s %12s %8s" % ("cfg", "phase", "kswScan", "dirScan", "allocstall", "file_grow", "freeMin"))
    for n in order:
        c = cfg[n]
        for ph, a, b in (("load", c.get("start"), c.get("up")), ("prompts", c.get("up"), c.get("done"))):
            if a is None or b is None: continue
            ks = ds = st = 0; f0 = f1 = None; fm = None
            for t, r in vw:
                if a < t <= b + 2:
                    d = r["d"]; g = r["g"]
                    ks += d.get("pgscan_kswapd", 0); ds += d.get("pgscan_direct", 0)
                    st += d.get("allocstall_normal", 0) + d.get("allocstall_movable", 0)
                    f = g.get("nr_file_pages"); fr = g.get("nr_free_pages")
                    if f is not None:
                        f0 = f if f0 is None else f0; f1 = f
                    if fr is not None: fm = fr if fm is None else min(fm, fr)
            print("%-5s %-7s %8.0fMB %8.0fMB %10d %10.0fMB %7.1fG" % (n, ph, mib(ks), mib(ds), st,
                  mib((f1 or 0) - (f0 or 0)), (fm or 0) * 4096 / 2**30))
