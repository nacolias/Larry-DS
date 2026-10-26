#!/usr/bin/env python3
# patch-pfdepth.py [GENERATE_CPP] - larry, STRATA_XE_PREFETCH_DEPTH (default 0 = off): a prompt batch that ends before
# this context position takes no prefetch at all - every call streams its routed arena experts on demand and the CPU
# pool runs the light ones (STRATA_XE_CPU_LIGHT).  At depth 0 that beats any prefetch (a whole 3.5K prompt: 510 tok/s
# all prefetched, 559 the upper layers only, 642 none); a deep batch's pipelined calls are bound by card 0's attention
# over the long context, which the pool's host work slows - those keep the thresholds.  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "STRATA_XE_PREFETCH_DEPTH" in s:
    print("already patched"); sys.exit(0)
old = """                    const bool lo_pf = B >= pf_min, hi_pf = cut > 0 ? nb >= pf_min_b : nb >= pf_min;
                    const int pf_first = hi_pf && !lo_pf && cut > 0 ? CL : 0;
                    const int pf_last = hi_pf ? n_layer - 1 : (cut > 0 && lo_pf ? CL - 1 : -1);
                    if (pf_last >= 0 && !mc.cpu_only) tier.prefetch_begin(pf_first, pf_last, pipe ? part_max : B);
"""
new = """                    const bool lo_pf = B >= pf_min, hi_pf = cut > 0 ? nb >= pf_min_b : nb >= pf_min;
                    const int pf_first = hi_pf && !lo_pf && cut > 0 ? CL : 0;
                    const int pf_last = hi_pf ? n_layer - 1 : (cut > 0 && lo_pf ? CL - 1 : -1);
                    // STRATA_XE_PREFETCH_DEPTH (0 = off): a batch ending before this position takes none (a shallow
                    // batch is bound by its experts: on demand with the CPU pool's light experts it runs faster)
                    static const int64_t pf_depth = std::getenv("STRATA_XE_PREFETCH_DEPTH") ? std::atoll(std::getenv("STRATA_XE_PREFETCH_DEPTH")) : 0;
                    const bool pf_shallow = (int64_t) pos0 + c0 + B < pf_depth;
                    if (pf_last >= 0 && !pf_shallow && !mc.cpu_only) tier.prefetch_begin(pf_first, pf_last, pipe ? part_max : B);
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
