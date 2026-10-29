#!/usr/bin/env python3
# patch-pclend2.py [XOLOTL_DIR] - larry, 2026-10-29 (after patch-pclend.py): tp_room_ahead gives the lent promotion slots
# back only when it has KV rows to make room for.  Every request's start calls it with nothing (patch-roomahead.py drops
# an earlier read-ahead there), so the lent slots went back at every turn - also before the turns read as TP passes,
# which allocate nothing.  An empty call sizes nothing from the free VRAM.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
T = os.path.join(X, "tools/v41/engine/xe/ds41_xe_tier.cpp")
s = open(T).read()
if "patch-pclend2.py" in s:
    print("already patched")
    sys.exit(0)
old = """void Ds41XeTier::tp_room_ahead(const std::vector<std::pair<int, size_t>>& reqs) {
    Ds41XeImpl& im = *im_;
    pc_unlend(im);   // larry (patch-pclend.py)
"""
new = """void Ds41XeTier::tp_room_ahead(const std::vector<std::pair<int, size_t>>& reqs) {
    Ds41XeImpl& im = *im_;
    // larry (patch-pclend.py; patch-pclend2.py: only with rows to make room for - every request's start calls it empty)
    if (!reqs.empty()) pc_unlend(im);
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(T, "w").write(s.replace(old, new))
print("patched", T)
