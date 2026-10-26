#!/usr/bin/env python3
# vmwatch2.py OUT [SECONDS] - /proc/vmstat every 2 s as JSON lines (larry, 2026-10-26: the first-prompt stall A/B):
# deltas of the reclaim / compaction / migration / swap / refault counters, and the free / file / anon page gauges.
# (vmwatch.sh's successor: more counters, machine-readable for memsum.py.)
import json
import sys
import time

OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
D = ("allocstall_normal allocstall_movable pgscan_direct pgsteal_direct pgscan_kswapd pgsteal_kswapd compact_stall "
     "compact_fail compact_success compact_daemon_wake pgmigrate_success pswpin pswpout workingset_refault_file "
     "workingset_refault_anon pgmajfault kswapd_low_wmark_hit_quickly kswapd_high_wmark_hit_quickly thp_fault_alloc "
     "thp_fault_fallback pgpgin pgpgout").split()
G = "nr_free_pages nr_file_pages nr_anon_pages nr_inactive_file nr_active_file".split()


def rd():
    v = {}
    with open("/proc/vmstat") as f:
        for line in f:
            k, x = line.split()
            v[k] = int(x)
    return v


prev = rd()
t_end = time.time() + N
with open(OUT, "a") as o:
    while time.time() < t_end:
        time.sleep(2)
        cur = rd()
        rec = {"t": round(time.time(), 1),
               "d": {k: cur[k] - prev[k] for k in D if k in cur and cur[k] != prev.get(k)},
               "g": {k: cur[k] for k in G if k in cur}}
        o.write(json.dumps(rec) + "\n")
        o.flush()
        prev = cur
