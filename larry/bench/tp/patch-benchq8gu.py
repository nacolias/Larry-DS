#!/usr/bin/env python3
# patch-benchq8gu.py - test_ds41_xe --bench-q8-gu T: --bench-q8 over experts whose gate/up type is T only (11 Q3_K,
# 42 Q2_0, 143 PTQ1_0).  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/test_ds41_xe.cpp"
s = open(P).read()
if "bench_q8_gu" in s:
    print("already"); sys.exit(0)
def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"anchor x{s.count(old)}: {old[:80]!r}")
    s = s.replace(old, new)
rep("    int bench_q8 = 0;\n", "    int bench_q8 = 0, bench_q8_gu = -1;\n")
rep("        else if (k == \"--bench-q8\" && i + 1 < argc) bench_q8 = std::atoi(argv[++i]);\n",
    "        else if (k == \"--bench-q8\" && i + 1 < argc) bench_q8 = std::atoi(argv[++i]);\n"
    "        else if (k == \"--bench-q8-gu\" && i + 1 < argc) bench_q8_gu = std::atoi(argv[++i]);\n")
rep("            const cpu::NativeFmt& f = fl[(size_t) g.cls(l, e)];\n            std::vector<uint8_t> blob(f.bytes);\n            src.read(l, e, blob.data());\n            uint8_t* fb",
    "            const cpu::NativeFmt& f = fl[(size_t) g.cls(l, e)];\n            if (bench_q8_gu >= 0 && f.gu_type != bench_q8_gu) continue;\n            std::vector<uint8_t> blob(f.bytes);\n            src.read(l, e, blob.data());\n            uint8_t* fb")
rep("rotated\\n\", l, (int) pl.size(),", "rotated (gate/up type %d)\\n\", l, (int) pl.size(), bench_q8_gu,")
open(P, "w").write(s)
print("patched")
