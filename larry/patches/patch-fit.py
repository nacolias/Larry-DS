# patch-fit.py - serve: the batch-width fit's constants from the environment (STRATA_SERVE_BATCH_FIXED_GIB 4.3,
# STRATA_SERVE_BATCH_COL_MIB 0.95: the CUDA engine's calibration; larry's Xe tier keeps its chunk buffers off the
# dense card, ~0.32 MiB a column there), and the pipelined loop's VRAM trace points
p = '/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp'
s = open(p).read()
pairs = [
    ('''                const double room = (double) frb - a.vram_margin_gib * 1073741824.0 - 4.3 * 1073741824.0;
                const int fit = (int) std::max<int64_t>(2, (int64_t) (room / (0.95 * 1048576.0)));''',
     '''                // larry: the two constants from the environment (the Xe tier's batch keeps its chunk buffers on the
                // expert cards: ~0.32 MiB of state a column on the dense card, measured with STRATA_VRAM_TRACE)
                static const double fit_fixed = std::getenv("STRATA_SERVE_BATCH_FIXED_GIB") ? std::atof(std::getenv("STRATA_SERVE_BATCH_FIXED_GIB")) : 4.3;
                static const double fit_col = std::getenv("STRATA_SERVE_BATCH_COL_MIB") ? std::atof(std::getenv("STRATA_SERVE_BATCH_COL_MIB")) : 0.95;
                const double room = (double) frb - a.vram_margin_gib * 1073741824.0 - fit_fixed * 1073741824.0;
                const int fit = (int) std::max<int64_t>(2, (int64_t) (room / (fit_col * 1048576.0)));'''),
    ('''                    for (int l = 0; okp && l < n_layer; ++l) {
                        int lo, mid;
                        split_of(l, lo, mid);
                        if (mid < 0 || lo != plo || mid != pmid || (cut > 0 && l == CL)) okp = PJ();''',
     '''                    for (int l = 0; okp && l < n_layer; ++l) {
                        if (l < 4 || l == n_layer - 1) vram_log("pipelined batch layer", l);
                        int lo, mid;
                        split_of(l, lo, mid);
                        if (mid < 0 || lo != plo || mid != pmid || (cut > 0 && l == CL)) okp = PJ();'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(p, 'w').write(s)
print('ok')
