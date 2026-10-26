#!/usr/bin/env python3
# patch-keepdepth.py [XOLOTL_DIR] - larry, 2026-10-26: STRATA_SERVE_KEEP_LEARN_DEPTH_GIB=<g> (engine, default 0): a
# kept-cache prompt batch gets g * min(1, conversation tokens / 1M) GiB more room on the dense card - in the cache
# shrink (the calibrated need and the learned one) and in the batch width.  At 1M a turn's batch needs more than
# either estimate: abwT142's 11.7K-token turn at 973K context took ALL 10.27 GiB the shrink left (0.00 GiB free at its
# low point) and passed; abwT141's same turn had 9.85 GiB and died with UR_RESULT_ERROR_OUT_OF_RESOURCES (DSV4_HC_PRE).
# A 30K conversation gets +0.04 GiB with g 1.5, a 1M one all of it.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
s = open(P).read()
if "STRATA_SERVE_KEEP_LEARN_DEPTH_GIB" in s:
    print("already patched", P)
    sys.exit(0)
pairs = [
("""            static const double kl_margin = std::getenv("STRATA_SERVE_KEEP_LEARN_MARGIN_GIB") ? std::atof(std::getenv("STRATA_SERVE_KEEP_LEARN_MARGIN_GIB")) : 1.5;
""",
"""            static const double kl_margin = std::getenv("STRATA_SERVE_KEEP_LEARN_MARGIN_GIB") ? std::atof(std::getenv("STRATA_SERVE_KEEP_LEARN_MARGIN_GIB")) : 1.5;
            // larry, STRATA_SERVE_KEEP_LEARN_DEPTH_GIB (patch-keepdepth.py): more room for a kept batch the deeper the
            // conversation (abwT141 / abwT142: an 11.7K batch at 973K needed all of 10.27 GiB; with 9.85 it died)
            static const double kl_depth = std::getenv("STRATA_SERVE_KEEP_LEARN_DEPTH_GIB") ? std::atof(std::getenv("STRATA_SERVE_KEEP_LEARN_DEPTH_GIB")) : 0.0;
            const double kl_dadd = kl_depth * std::min(1.0, (double) total / 1048576.0) * 1073741824.0;   // bytes
"""),
("""                    double need = (kfix * 1073741824.0 + kcol * 1048576.0 * (double) std::min(dh, PB)) +
                                  a.vram_margin_gib * 1073741824.0;
""",
"""                    double need = (kfix * 1073741824.0 + kcol * 1048576.0 * (double) std::min(dh, PB)) +
                                  a.vram_margin_gib * 1073741824.0 + kl_dadd;
"""),
("""                        const double nl = est + kl_margin * 1073741824.0;
""",
"""                        const double nl = est + kl_margin * 1073741824.0 + kl_dadd;
"""),
("""                        const double nl = kl_used + kl_col * 1048576.0 * (double) std::max(0, std::min(dh, PB) - kl_B) +
                                          kl_margin * 1073741824.0;
""",
"""                        const double nl = kl_used + kl_col * 1048576.0 * (double) std::max(0, std::min(dh, PB) - kl_B) +
                                          kl_margin * 1073741824.0 + kl_dadd;
"""),
("""                const double room = (double) frb - a.vram_margin_gib * 1073741824.0 - fit_fixed * 1073741824.0;
""",
"""                const double room = (double) frb - a.vram_margin_gib * 1073741824.0 - fit_fixed * 1073741824.0 -
                                    (kept ? kl_dadd : 0.0);   // larry, patch-keepdepth.py
"""),
("""                    const double lr = (double) frb - kl_margin * 1073741824.0 - kl_sel_used;   // patch-keeplearn2 / keephist
""",
"""                    const double lr = (double) frb - kl_margin * 1073741824.0 - kl_sel_used - kl_dadd;   // patch-keeplearn2 / keephist / keepdepth
"""),
]
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:80]))
    s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
