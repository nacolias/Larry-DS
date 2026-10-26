#!/usr/bin/env python3
# patch-dskip.py - DEBUG STRATA_XE_DENSE_SKIP_DEBUG=1 (WRONG OUTPUT, timing only): the dense card's share of the
# routed experts is not computed (an empty kernel stands in), to bound what it costs the verify / decode path.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_DENSE_SKIP_DEBUG" in s: print("already"); sys.exit(0)
old = """    const XeJob* jobs = (const XeJob*) c.stage_dev;
    const float* X = xsrc ? xsrc : (const float*) (c.stage_dev + xoff);   // xsrc: pinned host rows, read over PCIe"""
new = """    const XeJob* jobs = (const XeJob*) c.stage_dev;
    static const bool dskip = env_i("STRATA_XE_DENSE_SKIP_DEBUG", 0) != 0;   // DEBUG: wrong output, timing only
    if (dskip && c.dense) return c.q.single_task([] {});
    const float* X = xsrc ? xsrc : (const float*) (c.stage_dev + xoff);   // xsrc: pinned host rows, read over PCIe"""
if s.count(old) != 1: sys.exit("anchor")
open(P, "w").write(s.replace(old, new))
print("patched")
