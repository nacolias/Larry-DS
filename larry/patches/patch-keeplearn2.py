#!/usr/bin/env python3
# patch-keeplearn2.py [GENERATE_CPP] - larry, STRATA_SERVE_KEEP_LEARN fix (after patch-keeplearn.py): the batch's VRAM
# was read after feed() returned - by then ~3.3 GiB of the batch's own buffers were given back (abwT66: an 8K batch
# "took 1.91 GiB" with 7.05 GiB free at its end while STRATA_VRAM_TRACE had 3.74 GiB free at its last layer), so the
# next batch was sized from too little.  Now the dense card's free VRAM is sampled at every layer of a kept batch (and
# after its layers), and its LOW point is what the batch took.  Guards (one bad estimate OOMed an A/B engine): the
# learned size only for a turn ending no deeper than max(the learned one's end, 16384) - a deeper context's upper
# layers have more rows; STRATA_SERVE_KEEP_LEARN_COL_MIB (0.6, measured 0.30) a column more than the learned batch and
# STRATA_SERVE_KEEP_LEARN_MARGIN_GIB (1.5) on top; a low point under 1 GiB sends the next batch back to the
# calibrated size.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "patch-keeplearn2" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""// larry, STRATA_BATCH_TRACE=1: a served prompt's steps, ms since its keep-cache step (btr_reset)
""", """// larry, STRATA_SERVE_KEEP_LEARN (patch-keeplearn2): a kept batch's lowest free VRAM on the dense card (bytes; < 0 = not
// tracking), sampled at every layer by the batch loops
static double g_kl_min_free = -1;
// larry, STRATA_BATCH_TRACE=1: a served prompt's steps, ms since its keep-cache step (btr_reset)
""")
rep("""    auto vram_log = [&](const char * where, int a) {
        if (!vram_trace) return;
""", """    auto kl_sample = [&]() {   // larry, patch-keeplearn2: the kept batch's low point
        if (g_kl_min_free < 0) return;
        size_t tf = 0, tt = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
        if ((double) tf < g_kl_min_free) g_kl_min_free = (double) tf;
    };
    auto vram_log = [&](const char * where, int a) {
        if (!vram_trace) return;
""")
rep("""                        if (l < 4 || l == n_layer - 1) vram_log("pipelined batch layer", l);
""", """                        if (l < 4 || l == n_layer - 1) vram_log("pipelined batch layer", l);
                        if ((l & 3) == 3 || l == n_layer - 1) kl_sample();   // larry, patch-keeplearn2 (every 4th layer)
""")
rep("""                    if (l < 4) vram_log("batch layer", l);
""", """                    if (l < 4) vram_log("batch layer", l);
                    if ((l & 3) == 3) kl_sample();   // larry, patch-keeplearn2 (every 4th layer)
""")
rep("""                tier.prefetch_end();
                fd_dense += now_ms() - t0;
                btr("layers / prefetch end");
""", """                kl_sample();   // larry, patch-keeplearn2: after the last layer (before the batch's buffers go)
                tier.prefetch_end();
                fd_dense += now_ms() - t0;
                btr("layers / prefetch end");
""")
# start tracking before feed (a kept batch with the learning on), use the low point after it
rep("""            if (!ensure_kv(total + 64)) { std::printf("ERR %s\\n", err.c_str()); std::fflush(stdout); continue; }
            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
""", """            if (!ensure_kv(total + 64)) { std::printf("ERR %s\\n", err.c_str()); std::fflush(stdout); continue; }
            g_kl_min_free = kept && keep_learn && kl_free0 > 0 ? 1e30 : -1;   // larry, patch-keeplearn2
            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
""")
rep("""                size_t fre = 0, tote = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &fre, &tote);
                const double used = std::max(0.0, kl_free0 - (double) fre);
                const bool tight = (double) fre < 0.5 * 1073741824.0;
                std::fprintf(stderr, "serve: keep-cache: the %d-token batch took %.2f GiB (%.2f GiB free at its end%s)\\n",
                             std::min(dh, pb_cap), used / 1073741824.0, fre / 1073741824.0, tight ? ": tight - calibrated next" : "");
""", """                size_t fre = 0, tote = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &fre, &tote);
                const double lo = std::min((double) fre, g_kl_min_free > 0 ? g_kl_min_free : (double) fre);   // patch-keeplearn2
                const double used = std::max(0.0, kl_free0 - lo);
                const bool tight = lo < 1.0 * 1073741824.0;
                std::fprintf(stderr, "serve: keep-cache: the %d-token batch took %.2f GiB (%.2f GiB free at its low point, %.2f at "
                             "its end%s)\\n", std::min(dh, pb_cap), used / 1073741824.0, lo / 1073741824.0, fre / 1073741824.0,
                             tight ? ": tight - calibrated next" : "");
""")
rep("""            kl_free0 = 0;
            if (kept) {   // the batch is done""", """            kl_free0 = 0;
            g_kl_min_free = -1;   // larry, patch-keeplearn2
            if (kept) {   // the batch is done""")
rep("""                    if (keep_learn && kl_B > 0) {
                        const double nl = kl_used + kcol * 1048576.0 * (double) std::max(0, std::min(dh, PB) - kl_B) +
                                          a.vram_margin_gib * 1073741824.0;
""", """                    if (keep_learn && kl_B > 0 && total <= std::max(kl_end, 16384)) {   // patch-keeplearn2: no deeper
                        const double nl = kl_used + kl_col * 1048576.0 * (double) std::max(0, std::min(dh, PB) - kl_B) +
                                          kl_margin * 1073741824.0;
""")
rep("""                    const double lr = (double) frb - a.vram_margin_gib * 1073741824.0 - kl_used;
                    fit = (int) std::max<int64_t>(2, (int64_t) kl_B + (int64_t) (lr / (fit_col * 1048576.0)));
""", """                    const double lr = (double) frb - kl_margin * 1073741824.0 - kl_used;   // patch-keeplearn2
                    fit = (int) std::max<int64_t>(2, (int64_t) kl_B + (int64_t) (lr / (kl_col * 1048576.0)));
""")
rep("""            static double kl_used = 0, kl_free0 = 0;
            static int kl_B = 0;
""", """            static double kl_used = 0, kl_free0 = 0;
            static int kl_B = 0;
            static int kl_end = 0;   // patch-keeplearn2: the learned turn's end position
            static const double kl_col = std::getenv("STRATA_SERVE_KEEP_LEARN_COL_MIB") ? std::atof(std::getenv("STRATA_SERVE_KEEP_LEARN_COL_MIB")) : 0.6;
            static const double kl_margin = std::getenv("STRATA_SERVE_KEEP_LEARN_MARGIN_GIB") ? std::atof(std::getenv("STRATA_SERVE_KEEP_LEARN_MARGIN_GIB")) : 1.5;
""")
rep("""                kl_used = used;
                kl_B = tight ? 0 : std::min(dh, pb_cap);
""", """                kl_used = used;
                kl_B = tight ? 0 : std::min(dh, pb_cap);
                kl_end = total;   // patch-keeplearn2
""")
open(P, "w").write(s)
print("patched", P)
