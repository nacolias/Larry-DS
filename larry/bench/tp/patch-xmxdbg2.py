#!/usr/bin/env python3
# patch-xmxdbg2.py - the one-token-job debug also prints H (xmx ht, slots 0..2 of rows 0..3)
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/test_ds41_xe.cpp"
s = open(P).read()
if "ht slots" in s:
    print("already"); sys.exit(0)
old = """                        std::vector<float> g1((size_t) NT * E);"""
new = """                        std::vector<sycl::half> hh((size_t) F * 8);
                        q.memcpy(hh.data(), pht, hh.size() * 2).wait();
                        std::printf("    ht slots (row: s0 s1 s2):");
                        for (int r = 0; r < 4; ++r) std::printf("  %d: %.3f %.3f %.3f", r, (float) hh[r * 8], (float) hh[r * 8 + 1], (float) hh[r * 8 + 2]);
                        std::printf("\\n");
                        std::vector<float> g1((size_t) NT * E);"""
if s.count(old) != 1: sys.exit("anchor")
s = s.replace(old, new)
open(P, "w").write(s)
print("patched")
