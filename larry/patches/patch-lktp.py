#!/usr/bin/env python3
# patch-lktp.py [GENERATE_CPP] - larry, engine1: STRATA_LOOKUP_TP (default 0 = off): a serve lookup draft stays within
# the TP core's pass width (7 rows: the token, a re-fed one and the drafts).  A wider lookup pass takes the old dense
# path, which pages card 0's copies of the TP cards' weights in for every layer: in an agentic session (2026-10-24
# 14:13, 19 turns) such passes cost ~0.1-0.2 s each against ~0.04 s for a TP pass, ~2x a TP pass's time a token.
# Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "STRATA_LOOKUP_TP" in s:
    print("already patched"); sys.exit(0)
old = """                        int cap = std::min(lk_max, kVerify - 1 - (refeed ? 1 : 0));
"""
new = """                        int cap = std::min(lk_max, kVerify - 1 - (refeed ? 1 : 0));
                        // larry, engine1 (STRATA_LOOKUP_TP=1): the pass stays on the TP core (<= 7 rows) - a wider one
                        // takes the old path, which pages card 0's dense weights in every layer (~0.1-0.2 s a pass)
                        static const bool lk_tp = std::getenv("STRATA_LOOKUP_TP") && std::getenv("STRATA_LOOKUP_TP")[0] == '1';
                        if (lk_tp && tp_on) cap = std::min(cap, 7 - 1 - (refeed ? 1 : 0));
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
