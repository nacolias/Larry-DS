#!/usr/bin/env python3
# patch-fn16min.py [TIER_CPP] - larry, STRATA_XE_FN16_MIN=N (default 0): FN16 only for calls of at least N columns.  The
# fp16 conversion on the dense card plus the cross-card event before the experts' X lands cost a phase-B call (1.2-1.5K
# columns) ~4 ms (STRATA_XE_CALL_TL "A" 5 -> 9 ms) and phase B 1.68 -> 1.82 s, while phase A's big calls' X lands 3-8 ms
# earlier with FN16=2 (abwT44).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_FN16_MIN" in s:
    print("already patched"); sys.exit(0)
old = """    bool fn16 = fn16_env && x_dev && cd16 && n > kMaxJobTok && ((size_t) n * H) % 4 == 0;"""
new = """    static const int fn16_min = env_i("STRATA_XE_FN16_MIN", 0);   // larry, patch-fn16min.py: the big calls only
    bool fn16 = fn16_env && x_dev && cd16 && n > kMaxJobTok && n >= fn16_min && ((size_t) n * H) % 4 == 0;"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
