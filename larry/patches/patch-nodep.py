#!/usr/bin/env python3
# patch-nodep.py - DEBUG STRATA_XE_NODEP_DEBUG=1 (WRONG OUTPUT, timing only): the decode / verify sum does not wait
# for the armed expert cards' parts.  Memory-safe (their queues stay in order; the parts land in the slots regardless).
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_NODEP_DEBUG" in s:
    print("already patched"); sys.exit(0)
old = """            deps.push_back(c.arm_ev);
"""
new = """            static const bool nodep_dbg = env_i("STRATA_XE_NODEP_DEBUG", 0) != 0;   // DEBUG: wrong output, timing only
            if (!nodep_dbg) deps.push_back(c.arm_ev);
"""
if s.count(old) != 1: sys.exit("anchor")
s = s.replace(old, new)
open(P, "w").write(s)
print("patched")
