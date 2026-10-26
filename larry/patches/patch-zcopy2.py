#!/usr/bin/env python3
# patch-zcopy2.py [TIER_CPP] - larry, STRATA_XE_ZCOPY=2: the per-call pinned staging (patch-zcopy.py) on the expert cards
# for EVERY call, not only single-round ones.  Phase-A calls mostly carry streamed / kept experts (the layer's overflow,
# kept for its second call), so ZCOPY=1 left them on the old path - a host wait after the row-list upload and two per
# job batch, while the queue drained the card-to-card X copy and the streamed experts: under unitrace an expert card's
# kernels were busy 38-44 ms of a 104-114 ms span a 16K call.  The staging is sized for the worst case: every round
# takes a slot map (n x K) and its last job batch may be partial (rounds <= 2 + the call's streamed experts).  The
# overrun guard stays.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_ZCOPY=2" in s:
    print("already patched"); sys.exit(0)


def rep(old, new):
    global s
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:100]))
    s = s.replace(old, new)


rep("""        const bool zc = zcopy_on() && !c.dense && streamed[(size_t) c.idx].empty() && kept[(size_t) c.idx].empty();""",
    """        // larry, STRATA_XE_ZCOPY=2 (patch-zcopy2.py): every call (rounds included), the staging sized for them
        static const int zcopy_mode = env_i("STRATA_XE_ZCOPY", 0);
        const bool zc = zcopy_on() && !c.dense &&
                        (zcopy_mode >= 2 || (streamed[(size_t) c.idx].empty() && kept[(size_t) c.idx].empty()));
        const size_t zc_rounds = zc ? 2 + std::min<size_t>(streamed[(size_t) c.idx].size(), 64) : 1;""")
rep("""            const size_t need = 4 * 256 + (size_t) rg * 4 + (size_t) n * K * 4 +
                                (size_t) nbat * ((size_t) BJ * sizeof(XeJob) + (size_t) n * K * 4 + 512);""",
    """            const size_t need = 4 * 256 + (size_t) rg * 4 + zc_rounds * ((size_t) n * K * 4 + 512) +
                                ((size_t) nbat + zc_rounds) * ((size_t) BJ * sizeof(XeJob) + (size_t) n * K * 4 + 512);""")
open(P, "w").write(s)
print("patched", P)
