#!/usr/bin/env python3
# patch-pipesplit.py [GENERATE_CPP] - larry, STRATA_PIPE_MIN_PART (default 0 = as before): the pipelined batch splits a
# layer's columns into two parts only when both have at least this many.  The split point is rounded to a 512-column
# sub-batch, so a 513-767-token batch split 512 + 1..255: abwT89's 530-token prompt ran an 18-token tier call after
# every 512-token one - 12-16 ms of fixed cost (the CPU-light part's read back and pool start, the cards' setup and
# joins) in each of the 40 layers, ~0.5 s of a 1.96 s prompt.  No layer splitting = the non-pipelined path: one tier
# call a layer for all the batch's columns.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "STRATA_PIPE_MIN_PART" in s:
    print("already patched"); sys.exit(0)
old = """                auto split_of = [&](int l, int & lo, int & mid) {
                    lo = (cut > 0 && l >= CL) ? cut : 0;
                    mid = lo + (int) std::lround((double) (B - lo) / 2.0 / SB) * SB;
                    if (mid <= lo || mid >= B) mid = -1;
                };
"""
new = """                // larry, STRATA_PIPE_MIN_PART (patch-pipesplit.py, default 0): a layer splits only when both parts have at
                // least this many columns (a 530-token batch went 512 + 18: an 18-token tier call a layer, ~12-16 ms each)
                static const int pipe_min_part = std::getenv("STRATA_PIPE_MIN_PART") ? std::atoi(std::getenv("STRATA_PIPE_MIN_PART")) : 0;
                auto split_of = [&](int l, int & lo, int & mid) {
                    lo = (cut > 0 && l >= CL) ? cut : 0;
                    mid = lo + (int) std::lround((double) (B - lo) / 2.0 / SB) * SB;
                    if (mid <= lo || mid >= B) mid = -1;
                    if (mid > 0 && (mid - lo < pipe_min_part || B - mid < pipe_min_part)) mid = -1;
                };
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
