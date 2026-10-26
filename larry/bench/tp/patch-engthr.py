#!/usr/bin/env python3
# patch-engthr.py - STRATA_ENGRAM_THREADS: the Engram row reader's pool (its queue depth across both NVMe drives;
# default the config's 32).  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_ENGRAM_THREADS" in s:
    print("already"); sys.exit(0)
old = "    if (!cfg.engram_dirs.empty() && !im.engram.open(G, cfg.engram_dirs, cfg.engram_threads, err)) return false;"
new = """    // larry, STRATA_ENGRAM_THREADS: the row reader's pool = its queue depth over both NVMe drives (a cold 16K prompt
    // batch is ~1.4M random reads: at 32 a cold 31.6K prompt waited ~4 s for its first batch's rows)
    const int eng_threads = std::getenv("STRATA_ENGRAM_THREADS") ? std::atoi(std::getenv("STRATA_ENGRAM_THREADS")) : cfg.engram_threads;
    if (!cfg.engram_dirs.empty() && !im.engram.open(G, cfg.engram_dirs, eng_threads, err)) return false;"""
if s.count(old) != 1: sys.exit("anchor")
s = s.replace(old, new)
open(P, "w").write(s)
print("patched")
