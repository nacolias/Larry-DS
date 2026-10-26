#!/usr/bin/env python3
# memsum.py ABW_LOG MEMTRACE_LOG VMWATCH2_JSONL - the first-prompt stall A/B (2026-10-26): per config of an abw.sh
# window, its arm (the last "memset" probe before it), load time, prompt times (pN.0 = the first after the start), and
# what memtrace.bt / vmwatch2.py saw while it ran its prompts ("up" .. "done"); then the means per arm.
import collections
import datetime
import json
import re
import sys

abw, mt, vw = sys.argv[1:4]


def secs(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


cfgs = collections.OrderedDict()
arm = "?"
for line in open(abw, errors="replace"):
    m = re.match(r"\[probe\] memset (\w+): (.*)", line)
    if m:
        arm = m.group(1)
        if m.group(1) == "tuned":
            arm = "tuned " + " ".join(kv.split("=")[1] for kv in m.group(2).split()[:3])
        continue
    m = re.match(r"\[(\w+)\] server pid \d+, env:.*?(\d\d:\d\d:\d\d)\s*$", line)
    if m:
        cfgs[m.group(1)] = {"arm": arm, "start": secs(m.group(2)), "p": {}}
        continue
    m = re.match(r"\[(\w+)\] up (\d\d:\d\d:\d\d)", line)
    if m and m.group(1) in cfgs:
        cfgs[m.group(1)]["up"] = secs(m.group(2))
        continue
    m = re.match(r"\[(\w+)\] (p\d+\.\d)\s+prompt\s+\d+ \(\d+ reused\)\s+([\d.]+) s", line)
    if m and m.group(1) in cfgs:
        cfgs[m.group(1)]["p"][m.group(2)] = float(m.group(3))
        continue
    m = re.match(r"\[(\w+)\] done (\d\d:\d\d:\d\d)", line)
    if m and m.group(1) in cfgs:
        cfgs[m.group(1)]["done"] = secs(m.group(2))

MTF = ["edr_n", "edr_ms", "edr_max", "odr_n", "odr_ms", "odr_max", "edc_n", "edc_ms", "edc_max", "odc_n", "odc_ms",
       "odc_max", "mig", "migc", "upinv", "uprbx", "uprbw", "x_n", "x_slow", "x_slow_ms", "x_max", "thr",
       "bk_n", "bk_pages", "bk_ms", "rs_n", "rs_pages", "rs_ms", "rs_max", "pr_n", "pr_ms", "pr_max", "sh_n", "sh_freed",
       "sh_ms"]   # (the last 13: memtrace2.bt only)
mts = []
for line in open(mt, errors="replace"):
    m = re.match(r"(\d\d:\d\d:\d\d) edr (\d+) (\d+) (\d+) odr (\d+) (\d+) (\d+) edc (\d+) (\d+) (\d+) odc (\d+) (\d+) (\d+) "
                 r"mig (\d+) migc (\d+) up (\d+) (\d+) (\d+) x (\d+) (\d+) (\d+) (\d+) thr (\d+)"
                 r"(?: bk (\d+) (\d+) (\d+) rs (\d+) (\d+) (\d+) (\d+) pr (\d+) (\d+) (\d+) sh (\d+) (\d+) (\d+))?", line)
    if m:
        mts.append((secs(m.group(1)), dict(zip(MTF, (int(x) for x in m.groups()[1:] if x is not None)))))
vws = []
for line in open(vw):
    r = json.loads(line)
    t = datetime.datetime.fromtimestamp(r["t"], datetime.timezone.utc)
    vws.append((t.hour * 3600 + t.minute * 60 + t.second, r))


def agg(a, b):
    o = collections.Counter()
    mx = collections.Counter()
    for t, d in mts:
        if a < t <= b + 2:
            for k, v in d.items():
                if k.endswith("max"):
                    mx[k] = max(mx[k], v)
                else:
                    o[k] += v
    o.update(mx)
    free_min = None
    for t, r in vws:
        if a < t <= b + 2:
            for k, v in r["d"].items():
                o["vm_" + k] += v
            f = r["g"].get("nr_free_pages")
            if f is not None:
                free_min = f if free_min is None else min(free_min, f)
    o["free_min_gib"] = round((free_min or 0) * 4096 / 2**30, 1)
    return o


COLS = [("p8000.0", "8K.0"), ("p8000.1", "8K.1"), ("p16000.0", "16K.0"), ("p16000.1", "16K.1"), ("p31000.0", "31K.0"),
        ("p31000.1", "31K.1")]
MCOLS = [("edr_n", "eDR"), ("edr_ms", "eDRms"), ("edr_max", "eDRmax"), ("edc_n", "eDC"), ("edc_ms", "eDCms"),
         ("mig", "migr"), ("upinv", "upInv"), ("uprbx", "upRbX"), ("uprbw", "upRbW"), ("x_slow", "x>2ms"),
         ("x_slow_ms", "xSlowMs"), ("x_max", "xMax"), ("vm_pgscan_kswapd", "kswScan"), ("vm_pgscan_direct", "dirScan"),
         ("vm_compact_stall", "cStall"), ("vm_workingset_refault_file", "refault"), ("vm_pswpin", "swpIn"),
         ("vm_pgmajfault", "majflt"), ("free_min_gib", "freeMin")]
if any("bk_n" in d for _, d in mts):   # memtrace2.bt: the xe shrinker's BO backups, their restores, the rebind worker
    MCOLS += [("bk_pages", "bkPages"), ("bk_ms", "bkMs"), ("rs_n", "rsN"), ("rs_pages", "rsPages"), ("rs_ms", "rsMs"),
              ("rs_max", "rsMax"), ("pr_n", "prN"), ("pr_ms", "prMs"), ("pr_max", "prMax"), ("sh_freed", "shFreed")]
print("%-5s %-26s %5s " % ("cfg", "arm", "load") + " ".join("%6s" % c[1] for c in COLS))
arms = collections.OrderedDict()
for n, c in cfgs.items():
    ld = c["up"] - c["start"] if "up" in c else float("nan")
    print("%-5s %-26s %5.0f " % (n, c["arm"], ld) + " ".join("%6.2f" % c["p"].get(k, float("nan")) for k, _ in COLS))
    arms.setdefault(c["arm"], []).append(n)
print()
print("while the prompts ran (up .. done):")
print("%-5s " % "cfg" + " ".join("%8s" % c[1] for c in MCOLS))
for n, c in cfgs.items():
    if "up" not in c or "done" not in c:
        continue
    c["m"] = agg(c["up"], c["done"])
    c["ml"] = agg(c["start"], c["up"])
    print("%-5s " % n + " ".join("%8s" % c["m"].get(k, 0) for k, _ in MCOLS))
print()
print("while the engine loaded (start .. up):")
print("%-5s " % "cfg" + " ".join("%8s" % c[1] for c in MCOLS))
for n, c in cfgs.items():
    if "ml" in c:
        print("%-5s " % n + " ".join("%8s" % c["ml"].get(k, 0) for k, _ in MCOLS))
print()
print("means per arm:")
for a, ns in arms.items():
    ps = [cfgs[n] for n in ns]
    row = []
    for k, lab in COLS:
        v = [p["p"][k] for p in ps if k in p["p"]]
        row.append("%s %.2f" % (lab, sum(v) / len(v)) if v else "%s -" % lab)
    print("  %-26s n=%d  %s" % (a, len(ps), "  ".join(row)))
    ms = [p["m"] for p in ps if "m" in p]
    if ms:
        print("  %-26s       %s" % ("", "  ".join("%s %.0f" % (lab, sum(m.get(k, 0) for m in ms) / len(ms))
                                            for k, lab in MCOLS)))
