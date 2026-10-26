#!/usr/bin/env python3
# patch-pfrange.py [GENERATE_CPP] - larry: an enc-skip batch whose lower layers' calls (all B columns) are below
# STRATA_XE_PREFETCH_MIN but whose upper layers' (the phase-B columns) reach STRATA_XE_PREFETCH_MIN_B prefetches layers
# CL..39 only (it prefetched them all): the lower layers then run on demand with STRATA_XE_CPU_LIGHT.  Prompts of 3-4K
# tokens with the service's 4096 / 2048.  Identical to before whenever MIN_B = MIN (the defaults).  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "pf_first" in s:
    print("already patched"); sys.exit(0)
old = """                    const int nb = cut > 0 ? B - cut : B;
                    const int pf_last = (cut > 0 ? nb >= pf_min_b : nb >= pf_min) ? n_layer - 1 : (cut > 0 && B >= pf_min ? CL - 1 : -1);
                    if (pf_last >= 0 && !mc.cpu_only) tier.prefetch_begin(0, pf_last, pipe ? part_max : B);
"""
new = """                    const int nb = cut > 0 ? B - cut : B;
                    // the lower layers' calls have B columns, an enc-skip batch's upper layers' (CL on) nb: each part
                    // its own threshold; only the upper part reaching its own = layers CL..39 (a 3-4K-token prompt)
                    const bool lo_pf = B >= pf_min, hi_pf = cut > 0 ? nb >= pf_min_b : nb >= pf_min;
                    const int pf_first = hi_pf && !lo_pf && cut > 0 ? CL : 0;
                    const int pf_last = hi_pf ? n_layer - 1 : (cut > 0 && lo_pf ? CL - 1 : -1);
                    if (pf_last >= 0 && !mc.cpu_only) tier.prefetch_begin(pf_first, pf_last, pipe ? part_max : B);
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
