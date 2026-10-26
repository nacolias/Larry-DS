#!/usr/bin/env python3
# patch-dsparkcum.py [XOLOTL_DIR] - larry, 2026-10-26: STRATA_DSPARK_CUM=<c> (engine, default 0 = off).  Besides the floor
# on each draft's own confidence (STRATA_DSPARK_CONF), the verify stops at the first draft where the product of the
# confidences so far drops below c: a window whose later drafts are likely lost is not verified whole.  Lossless (the
# emitted tokens are the trunk's; only the verify width changes).  From Project Maya's MAYA_DS4_DSPARK_CUM (0.5 there:
# +6.5 % decode on a Radeon 8065S; DeepInfer's cumulative rule).  Both loops: the server's and the one-shot one.
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")

s = open(P).read()
if "STRATA_DSPARK_CUM" in s:
    print("already patched", P)
    sys.exit(0)
pairs = [(
"""                    static const float conf_tau_s = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
                    if (conf_tau_s >= 0) {
                        int kk = 0;
                        while (kk < k && 1.0f / (1.0f + std::exp(-sp_conf[kk])) >= conf_tau_s) ++kk;
                        k = kk;
                    }
""",
"""                    static const float conf_tau_s = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
                    // larry, STRATA_DSPARK_CUM=<c> (patch-dsparkcum.py; Project Maya's MAYA_DS4_DSPARK_CUM): also stop at the
                    // first draft where the product of the confidences so far drops below c
                    static const float cum_tau_s = std::getenv("STRATA_DSPARK_CUM") ? (float) std::atof(std::getenv("STRATA_DSPARK_CUM")) : 0.0f;
                    if (conf_tau_s >= 0 || cum_tau_s > 0) {
                        int kk = 0;
                        float cum = 1.0f;
                        while (kk < k) {
                            const float p = 1.0f / (1.0f + std::exp(-sp_conf[kk]));
                            if (p < conf_tau_s || (cum *= p) < cum_tau_s) break;
                            ++kk;
                        }
                        k = kk;
                    }
"""), (
"""            if (conf_tau >= 0) {
                int kk = 0;
                while (kk < k && sig(sp_conf[kk]) >= conf_tau) ++kk;
                k = kk;
            }
""",
"""            // larry, STRATA_DSPARK_CUM=<c> (patch-dsparkcum.py): also stop where the confidences' product drops below c
            static const float cum_tau = std::getenv("STRATA_DSPARK_CUM") ? (float) std::atof(std::getenv("STRATA_DSPARK_CUM")) : 0.0f;
            if (conf_tau >= 0 || cum_tau > 0) {
                int kk = 0;
                float cum = 1.0f;
                while (kk < k && sig(sp_conf[kk]) >= conf_tau && (cum *= sig(sp_conf[kk])) >= cum_tau) ++kk;
                k = kk;
            }
""")]
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:80]))
    s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
