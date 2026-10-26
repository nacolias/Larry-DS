#!/usr/bin/env python3
# patch-pfminb.py [GENERATE_CPP] - larry, STRATA_XE_PREFETCH_MIN_B (default: STRATA_XE_PREFETCH_MIN): the prefetch
# threshold for an enc-skip batch's upper layers (layer CL on: only its phase-B columns, at most --enc-replay of them),
# apart from the whole batch's.  With STRATA_XE_CPU_LIGHT a whole prompt of 2-3K tokens runs faster without the
# prefetch (2,048 tokens ~345 -> ~500 tok/s), but a long prompt's upper layers (~2,560 columns over a long context)
# want it: STRATA_XE_PREFETCH_MIN=4096 alone cost a 31K prompt 5 s.  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "STRATA_XE_PREFETCH_MIN_B" in s:
    print("already patched"); sys.exit(0)
old = """                // larry: the tier streams each layer's arena experts onto the expert cards ahead of its run_chunk_dev
                // (layers whose tier call has >= STRATA_XE_PREFETCH_MIN tokens; with enc-skip the layers from CL on
                // run only the phase-B columns)
                {
                    static const int pf_min = std::getenv("STRATA_XE_PREFETCH_MIN") ? std::atoi(std::getenv("STRATA_XE_PREFETCH_MIN")) : 2048;
                    const int nb = cut > 0 ? B - cut : B;
                    const int pf_last = nb >= pf_min ? n_layer - 1 : (cut > 0 && B >= pf_min ? CL - 1 : -1);
"""
new = """                // larry: the tier streams each layer's arena experts onto the expert cards ahead of its run_chunk_dev
                // (layers whose tier call has >= STRATA_XE_PREFETCH_MIN tokens; with enc-skip the layers from CL on
                // run only the phase-B columns: STRATA_XE_PREFETCH_MIN_B, default the same).  Below its threshold a
                // layer's routed arena experts stream on demand and STRATA_XE_CPU_LIGHT runs the light ones on the CPU
                // pool: a whole prompt of 2-3K tokens is faster so (2,048 tokens ~345 -> ~500 tok/s), a long prompt's
                // upper layers (~2,560 columns over a long context, card 0's attention the bottleneck) are not (31K +5 s)
                {
                    static const int pf_min = std::getenv("STRATA_XE_PREFETCH_MIN") ? std::atoi(std::getenv("STRATA_XE_PREFETCH_MIN")) : 2048;
                    static const int pf_min_b = std::getenv("STRATA_XE_PREFETCH_MIN_B") ? std::atoi(std::getenv("STRATA_XE_PREFETCH_MIN_B")) : pf_min;
                    const int nb = cut > 0 ? B - cut : B;
                    const int pf_last = (cut > 0 ? nb >= pf_min_b : nb >= pf_min) ? n_layer - 1 : (cut > 0 && B >= pf_min ? CL - 1 : -1);
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
