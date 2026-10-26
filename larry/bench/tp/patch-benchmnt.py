#!/usr/bin/env python3
# patch-benchmnt.py - test_ds41_xe --bench-q8: the kernel template (max_nt) as the armed decode / verify sequences pick
# it - from the CALL's width, not the jobs' tokens: configs with mostly 1-token jobs run at max_nt 2 / 4 / 8
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/test_ds41_xe.cpp"
s = open(P).read()
if "int mnt;" in s:
    print("already"); sys.exit(0)
def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"anchor x{s.count(old)}: {old[:80]!r}")
    s = s.replace(old, new)
rep("        struct Cfg { int nj, nt, ntok; };", "        struct Cfg { int nj, nt, ntok, mnt; };")
rep("""        for (const Cfg cf : {Cfg{2, 1, 1}, Cfg{3, 1, 1}, Cfg{2, 1, 3}, Cfg{3, 1, 3}, Cfg{3, 2, 3}, Cfg{3, 3, 3},
                             Cfg{3, 2, 5}, Cfg{4, 2, 5}, Cfg{6, 2, 5}, Cfg{3, 5, 5}, Cfg{6, 1, 1}}) {""",
    """        for (const Cfg cf : {Cfg{3, 1, 1, 1}, Cfg{3, 1, 3, 1}, Cfg{3, 1, 3, 2}, Cfg{3, 1, 3, 4}, Cfg{3, 1, 5, 8},
                             Cfg{3, 2, 3, 2}, Cfg{3, 2, 3, 4}, Cfg{3, 3, 3, 4}, Cfg{3, 2, 5, 8}}) {""")
rep("xe::xe_experts_q8(q, djs + (size_t) c * cf.nj, cf.nj, cf.ntok, Xq, Hq, dO, (int) E, (int) F, L, false, cf.nt);",
    "xe::xe_experts_q8(q, djs + (size_t) c * cf.nj, cf.nj, cf.ntok, Xq, Hq, dO, (int) E, (int) F, L, false, cf.mnt);")
rep("""            std::printf("  %d experts x %d tokens (call of %d): dp4a %6.1f us = %4.0f GB/s | xmx %6.1f us = %4.0f GB/s | max|diff| %.2e of %.2e\\n",
                        cf.nj, cf.nt, cf.ntok, us, bytes / us / 1e3, ux, bytes / ux / 1e3, md, mx);""",
    """            std::printf("  %d experts x %d tokens (call of %d, template %d): dp4a %6.1f us = %4.0f GB/s | xmx %6.1f us = %4.0f GB/s | max|diff| %.2e of %.2e\\n",
                        cf.nj, cf.nt, cf.ntok, cf.mnt, us, bytes / us / 1e3, ux, bytes / ux / 1e3, md, mx);""")
open(P, "w").write(s)
print("patched")
