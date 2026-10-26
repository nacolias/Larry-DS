#!/usr/bin/env python3
# patch-keepneed.py [GENERATE_CPP] - larry: the keep-cache shrink (a batch of <= STRATA_SERVE_KEEP_MAX new tokens gives
# back only the VRAM it needs from card 0's expert cache) sized the batch with the CUDA engine's constants (4.3 GiB +
# 0.95 MiB a column: 20.5 GiB for 16,384 columns - the whole 19.9 GiB cache went for any batch over ~8K tokens) while
# the batch width right below it uses the xe calibration from the environment (STRATA_SERVE_BATCH_FIXED_GIB 3.5 /
# STRATA_SERVE_BATCH_COL_MIB 0.45).  Both from the same constants now.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "larry, patch-keepneed" in s:
    print("already patched"); sys.exit(0)
old = """                if (Ds41Tier::keep_cache_mode() && dh <= keep_max && !mc.cpu_only) {
                    const double need = (4.3 * 1073741824.0 + 0.95 * 1048576.0 * (double) std::min(dh, PB)) +
                                        a.vram_margin_gib * 1073741824.0;
"""
new = """                if (Ds41Tier::keep_cache_mode() && dh <= keep_max && !mc.cpu_only) {
                    // larry, patch-keepneed: the batch's VRAM from the calibration the batch width below uses
                    // (STRATA_SERVE_BATCH_FIXED_GIB / _COL_MIB; the CUDA engine's 4.3 / 0.95 released the whole cache)
                    static const double kfix = std::getenv("STRATA_SERVE_BATCH_FIXED_GIB") ? std::atof(std::getenv("STRATA_SERVE_BATCH_FIXED_GIB")) : 4.3;
                    static const double kcol = std::getenv("STRATA_SERVE_BATCH_COL_MIB") ? std::atof(std::getenv("STRATA_SERVE_BATCH_COL_MIB")) : 0.95;
                    const double need = (kfix * 1073741824.0 + kcol * 1048576.0 * (double) std::min(dh, PB)) +
                                        a.vram_margin_gib * 1073741824.0;
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
