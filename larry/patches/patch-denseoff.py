#!/usr/bin/env python3
# patch-denseoff.py [TIER_CPP] - larry, STRATA_XE_DENSE_OFFLOAD=N (default 0 = off; with STRATA_XE_DENSE_LITE): in a
# pipelined prompt call of >= N tokens the experts cached on the dense card are computed by the expert cards instead,
# from their pinned arena copies (every dense-card cache expert is a pinned arena expert: direct DMAs, no CPU copy),
# streamed like the call's other overflow experts (and kept in the stream buffer for the layer's second call).  The
# dense card then runs nothing of the call but the combine: its tier kernels competed with the dense half (DENSE_LITE
# alone: the dense half +0.6 s at 16K).  The cost is PCIe: ~27 experts (~0.29 GB) a layer at 16K, once a layer - the
# lower layers' calls (8K tokens) have the uplink to spare; the upper layers' (~1.5K, staging-bound) stay as they are
# with N above their size.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_DENSE_OFFLOAD" in s:
    print("already patched"); sys.exit(0)
if "STRATA_XE_DENSE_LITE" not in s:
    sys.exit("needs patch-denselite.py first")


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    const bool lite = dense_lite && x_dev && out_dev && im.dense >= 0 && NC > 1;
""", """    const bool lite = dense_lite && x_dev && out_dev && im.dense >= 0 && NC > 1;
    // larry, STRATA_XE_DENSE_OFFLOAD: the dense card's cached experts to the expert cards (see patch-denseoff.py)
    static const int dense_offload = env_i("STRATA_XE_DENSE_OFFLOAD", 0);
    const bool offl_on = lite && dense_offload > 0 && n >= dense_offload && !im.apin.empty();
    auto offl = [&](int64_t e) -> bool {
        if (!offl_on) return false;
        const size_t key = (size_t) im.key(l, e);
        return im.card_of[key] == (int8_t) im.dense && im.aptr[key] && im.apin[key];
    };
""")
rep("""        const int c = im.card_of[(size_t) im.key(l, e)];
        if (c >= 0) cl[(size_t) c].push_back((int32_t) e);
    }
    for (int64_t e = 0; e < E; ++e) {
        if (by_e[(size_t) e].empty() || im.card_of[(size_t) im.key(l, e)] >= 0) continue;
""", """        const int c = im.card_of[(size_t) im.key(l, e)];
        if (c >= 0 && !offl(e)) cl[(size_t) c].push_back((int32_t) e);   // larry, STRATA_XE_DENSE_OFFLOAD: offloaded ones below
    }
    for (int64_t e = 0; e < E; ++e) {
        if (by_e[(size_t) e].empty() || (im.card_of[(size_t) im.key(l, e)] >= 0 && !offl(e))) continue;
""")
rep("""        if (light_on && (int) by_e[(size_t) e].size() <= light_t && im.aptr[(size_t) im.key(l, e)] &&
""", """        if (light_on && !offl(e) && (int) by_e[(size_t) e].size() <= light_t && im.aptr[(size_t) im.key(l, e)] &&
""")
rep("""                if (im.card_of[key] >= 0) return im.dptr[key];
                if (ps && ps->card[(size_t) e] >= 0) return c.pbuf[pslot] + ps->off[(size_t) e];
                return c.stream + soff[(size_t) e];
""", """                if (im.card_of[key] >= 0 && !offl(e)) return im.dptr[key];
                if (ps && ps->card[(size_t) e] >= 0) return c.pbuf[pslot] + ps->off[(size_t) e];
                return c.stream + soff[(size_t) e];
""")
rep("""                return im.card_of[(size_t) im.key(l, e)] >= 0 || (ps && ps->card[(size_t) e] >= 0) ? r == (overlap ? -1 : 0)
                                                                                               : round_of[(size_t) e] == r;
""", """                return (im.card_of[(size_t) im.key(l, e)] >= 0 && !offl(e)) || (ps && ps->card[(size_t) e] >= 0)
                           ? r == (overlap ? -1 : 0) : round_of[(size_t) e] == r;
""")
rep("""            auto is_planar = [&](int32_t e) { return im.planar && im.card_of[(size_t) im.key(l, e)] >= 0; };
""", """            auto is_planar = [&](int32_t e) { return im.planar && im.card_of[(size_t) im.key(l, e)] >= 0 && !offl(e); };
""")
open(P, "w").write(s)
print("patched", P)
