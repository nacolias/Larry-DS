#!/usr/bin/env python3
# patch-pipesplit2.py [GENERATE_CPP] - larry, after patch-pipesplit.py: a batch whose layers STRATA_PIPE_MIN_PART kept
# whole stays on the pipelined path (one part a layer: the FFN input and routed sums on the cards) - abwT90: dropping
# to the non-pipelined path sent each layer's FFN input through the host (530 tokens: dense 0.18 -> 0.78 s, the
# prompt 1.97 -> 2.15-2.21 s).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "patch-pipesplit2" in s:
    print("already patched"); sys.exit(0)
if "STRATA_PIPE_MIN_PART" not in s:
    sys.exit("patch-pipesplit.py first")


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""                static const int pipe_min_part = std::getenv("STRATA_PIPE_MIN_PART") ? std::atoi(std::getenv("STRATA_PIPE_MIN_PART")) : 0;
                auto split_of = [&](int l, int & lo, int & mid) {
                    lo = (cut > 0 && l >= CL) ? cut : 0;
                    mid = lo + (int) std::lround((double) (B - lo) / 2.0 / SB) * SB;
                    if (mid <= lo || mid >= B) mid = -1;
                    if (mid > 0 && (mid - lo < pipe_min_part || B - mid < pipe_min_part)) mid = -1;
                };
""", """                static const int pipe_min_part = std::getenv("STRATA_PIPE_MIN_PART") ? std::atoi(std::getenv("STRATA_PIPE_MIN_PART")) : 0;
                bool kept_whole = false;   // larry, patch-pipesplit2: a layer STRATA_PIPE_MIN_PART kept in one part
                auto split_of = [&](int l, int & lo, int & mid) {
                    lo = (cut > 0 && l >= CL) ? cut : 0;
                    mid = lo + (int) std::lround((double) (B - lo) / 2.0 / SB) * SB;
                    if (mid <= lo || mid >= B) mid = -1;
                    if (mid > 0 && (mid - lo < pipe_min_part || B - mid < pipe_min_part)) { mid = -1; kept_whole = true; }
                };
""")
rep("""                pipe = pipe && any2 && part_max <= piece;
""", """                pipe = pipe && (any2 || kept_whole) && part_max <= piece;   // larry, patch-pipesplit2: kept_whole too
""")
open(P, "w").write(s)
print("patched", P)
