#!/usr/bin/env python3
# patch-keeplearn.py [GENERATE_CPP] - larry, STRATA_SERVE_KEEP_LEARN=1 (default 0 until measured): the keep-cache shrink
# sized from what the last kept batch really took on the dense card.  The calibrated need (STRATA_SERVE_BATCH_FIXED_GIB
# + _COL_MIB a column + --vram-margin) is right for a start's first big batch (its pools grow: one 16K batch ended with
# 0.00 GiB free), but later batches find those pools allocated already - STRATA_VRAM_TRACE (abwT61): an 8K batch took
# 3.32 GiB of the 8.04 freed for it, a 12K one 4.53 of 10.04 - so every fresh prompt evicted ~4.7-5.9 GiB of card 0's
# expert cache for nothing, streamed those experts during the batch and re-uploaded them after it (the regrow:
# 0.29 s at 8K, 0.47 s at 16K, before the first token).  Now: after a kept batch, used = free after its shrink - free
# at its end, and the next batch of B columns needs used + _COL_MIB * (B - B_last, if more) + the margin (never more
# than the calibrated need); the batch width follows the same model.  A batch that ends with < 0.5 GiB free (or a
# start's first one) goes back to the calibrated need.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "STRATA_SERVE_KEEP_LEARN" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""                    const double need = (kfix * 1073741824.0 + kcol * 1048576.0 * (double) std::min(dh, PB)) +
                                        a.vram_margin_gib * 1073741824.0;
                    std::string kerr;
""", """                    double need = (kfix * 1073741824.0 + kcol * 1048576.0 * (double) std::min(dh, PB)) +
                                  a.vram_margin_gib * 1073741824.0;
                    // larry, STRATA_SERVE_KEEP_LEARN: the last kept batch's own VRAM, plus the columns it did not have
                    kl_now = false;
                    if (keep_learn && kl_B > 0) {
                        const double nl = kl_used + kcol * 1048576.0 * (double) std::max(0, std::min(dh, PB) - kl_B) +
                                          a.vram_margin_gib * 1073741824.0;
                        if (nl < need) {
                            std::fprintf(stderr, "serve: keep-cache learned: the last batch (%d tokens) took %.2f GiB -> %.2f GiB "
                                         "for this one (calibrated %.2f)\\n", kl_B, kl_used / 1073741824.0, nl / 1073741824.0,
                                         need / 1073741824.0);
                            need = nl;
                            kl_now = true;
                        }
                    }
                    std::string kerr;
""")
rep("""                const double room = (double) frb - a.vram_margin_gib * 1073741824.0 - fit_fixed * 1073741824.0;
                const int fit = (int) std::max<int64_t>(2, (int64_t) (room / (fit_col * 1048576.0)));
                pb_cap = std::min(PB, std::max(SB, fit));
""", """                const double room = (double) frb - a.vram_margin_gib * 1073741824.0 - fit_fixed * 1073741824.0;
                int fit = (int) std::max<int64_t>(2, (int64_t) (room / (fit_col * 1048576.0)));
                if (kept && kl_now) {   // larry, STRATA_SERVE_KEEP_LEARN: the width from the same model as the shrink
                    const double lr = (double) frb - a.vram_margin_gib * 1073741824.0 - kl_used;
                    fit = (int) std::max<int64_t>(2, (int64_t) kl_B + (int64_t) (lr / (fit_col * 1048576.0)));
                }
                pb_cap = std::min(PB, std::max(SB, fit));
                kl_free0 = (double) frb;
""")
rep("""            if (kept) {   // the batch is done: the tail slots back (exactly the evicted experts), decode biases on
                std::string kerr;
                dense.release_pass();   // the batch's pass buffers first (open_cache does the same before a seed)
""", """            if (kept && keep_learn && !mc.cpu_only && kl_free0 > 0) {   // larry, STRATA_SERVE_KEEP_LEARN: what it took
                size_t fre = 0, tote = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &fre, &tote);
                const double used = std::max(0.0, kl_free0 - (double) fre);
                const bool tight = (double) fre < 0.5 * 1073741824.0;
                std::fprintf(stderr, "serve: keep-cache: the %d-token batch took %.2f GiB (%.2f GiB free at its end%s)\\n",
                             std::min(dh, pb_cap), used / 1073741824.0, fre / 1073741824.0, tight ? ": tight - calibrated next" : "");
                kl_used = used;
                kl_B = tight ? 0 : std::min(dh, pb_cap);
            }
            kl_free0 = 0;
            if (kept) {   // the batch is done: the tail slots back (exactly the evicted experts), decode biases on
                std::string kerr;
                dense.release_pass();   // the batch's pass buffers first (open_cache does the same before a seed)
""")
# the state (once, before the serve loop's first use): find the keep-cache block's opening and put statics above it
rep("""            if (!by_step && defer_vram && cache_open && dh >= 2) {
                static const int keep_max = std::getenv("STRATA_SERVE_KEEP_MAX") ? std::atoi(std::getenv("STRATA_SERVE_KEEP_MAX")) : 4096;
""", """            // larry, STRATA_SERVE_KEEP_LEARN (patch-keeplearn.py): the last kept batch's dense-card VRAM (bytes) and width
            static const bool keep_learn = std::getenv("STRATA_SERVE_KEEP_LEARN") && std::atoi(std::getenv("STRATA_SERVE_KEEP_LEARN")) != 0;
            static double kl_used = 0, kl_free0 = 0;
            static int kl_B = 0;
            static bool kl_now = false;
            kl_now = false;
            if (!by_step && defer_vram && cache_open && dh >= 2) {
                static const int keep_max = std::getenv("STRATA_SERVE_KEEP_MAX") ? std::atoi(std::getenv("STRATA_SERVE_KEEP_MAX")) : 4096;
""")
open(P, "w").write(s)
print("patched", P)
