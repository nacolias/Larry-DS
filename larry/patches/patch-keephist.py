#!/usr/bin/env python3
# patch-keephist.py [GENERATE_CPP] - larry, STRATA_SERVE_KEEP_LEARN_HIST=1 (default 0; needs STRATA_SERVE_KEEP_LEARN=1):
# the keep-cache shrink sized from the last 8 kept batches instead of the last one alone.  A chat's follow-up turns
# (131-1303 new tokens at 25-36K, the user's session 2026-10-26) never used the learned size: each ended deeper than the
# learned one (patch-keeplearn2's guard), so they shrank by the calibration (3.5 GiB + 0.45 MiB a column + 1 GiB) while
# taking 0.15-1.4 GiB - 250-330 of card 0's experts out and back (the regrow, 0.15-0.19 s a turn).  Now: of the kept
# batches at least as wide as this one, the smallest VRAM it took, times total / max(its end, 16384) when this turn
# ends deeper (a deeper context's rows grow its attention buffers at most linearly); none as wide: the widest one's,
# STRATA_SERVE_KEEP_LEARN_COL_MIB a column more (as before).  STRATA_SERVE_KEEP_LEARN_MARGIN_GIB (1.5) on top, the
# batch width from the same pair, a batch that ends under 1 GiB free forgets them all (the next goes by the
# calibration).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "STRATA_SERVE_KEEP_LEARN_HIST" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""            static bool kl_now = false;
            kl_now = false;
""", """            static bool kl_now = false;
            kl_now = false;
            // larry, STRATA_SERVE_KEEP_LEARN_HIST (patch-keephist.py): the last 8 kept batches - width, VRAM taken, end
            struct KlEnt { int B; double used; int end; };
            static std::vector<KlEnt> kl_hist;
            static const bool kl_hist_on = keep_learn && std::getenv("STRATA_SERVE_KEEP_LEARN_HIST") &&
                                           std::atoi(std::getenv("STRATA_SERVE_KEEP_LEARN_HIST")) != 0;
            static int kl_sel_B = 0;          // the pair the shrink used (the batch width follows it)
            static double kl_sel_used = 0;
""")
rep("""                    kl_now = false;
                    if (keep_learn && kl_B > 0 && total <= std::max(kl_end, 16384)) {   // patch-keeplearn2: no deeper
                        const double nl = kl_used + kl_col * 1048576.0 * (double) std::max(0, std::min(dh, PB) - kl_B) +
                                          kl_margin * 1073741824.0;
                        if (nl < need) {
                            std::fprintf(stderr, "serve: keep-cache learned: the last batch (%d tokens) took %.2f GiB -> %.2f GiB "
                                         "for this one (calibrated %.2f)\\n", kl_B, kl_used / 1073741824.0, nl / 1073741824.0,
                                         need / 1073741824.0);
                            need = nl;
                            kl_now = true;
                        }
                    }""", """                    kl_now = false;
                    if (kl_hist_on && !kl_hist.empty()) {   // larry, patch-keephist: the closest batch as wide, depth-scaled
                        const int w = std::min(dh, PB);
                        auto depth = [&](const KlEnt & e) {
                            const int lim = std::max(e.end, 16384);
                            return total > lim ? (double) total / lim : 1.0;
                        };
                        const KlEnt * be = nullptr;
                        double est = 0;
                        for (const KlEnt & e : kl_hist)   // a batch at least as wide bounds this one
                            if (e.B >= w && (!be || e.used * depth(e) < est)) { be = &e; est = e.used * depth(e); }
                        if (!be) {   // none as wide: the widest one's, kl_col a column more
                            for (const KlEnt & e : kl_hist) if (!be || e.B > be->B) be = &e;
                            est = be->used * depth(*be) + kl_col * 1048576.0 * (double) (w - be->B);
                        }
                        const double nl = est + kl_margin * 1073741824.0;
                        if (nl < need) {
                            std::fprintf(stderr, "serve: keep-cache learned: a %d-token batch ending at %d took %.2f GiB -> %.2f GiB for "
                                         "this one (%d tokens ending at %d; calibrated %.2f)\\n", be->B, be->end, be->used / 1073741824.0,
                                         nl / 1073741824.0, w, total, need / 1073741824.0);
                            need = nl;
                            kl_now = true;
                            kl_sel_B = be->B;
                            kl_sel_used = be->used * depth(*be);
                        }
                    } else if (keep_learn && kl_B > 0 && total <= std::max(kl_end, 16384)) {   // patch-keeplearn2: no deeper
                        const double nl = kl_used + kl_col * 1048576.0 * (double) std::max(0, std::min(dh, PB) - kl_B) +
                                          kl_margin * 1073741824.0;
                        if (nl < need) {
                            std::fprintf(stderr, "serve: keep-cache learned: the last batch (%d tokens) took %.2f GiB -> %.2f GiB "
                                         "for this one (calibrated %.2f)\\n", kl_B, kl_used / 1073741824.0, nl / 1073741824.0,
                                         need / 1073741824.0);
                            need = nl;
                            kl_now = true;
                            kl_sel_B = kl_B;
                            kl_sel_used = kl_used;
                        }
                    }""")
rep("""                    const double lr = (double) frb - kl_margin * 1073741824.0 - kl_used;   // patch-keeplearn2
                    fit = (int) std::max<int64_t>(2, (int64_t) kl_B + (int64_t) (lr / (kl_col * 1048576.0)));""",
    """                    const double lr = (double) frb - kl_margin * 1073741824.0 - kl_sel_used;   // patch-keeplearn2 / keephist
                    fit = (int) std::max<int64_t>(2, (int64_t) kl_sel_B + (int64_t) (lr / (kl_col * 1048576.0)));""")
rep("""                kl_used = used;
                kl_B = tight ? 0 : std::min(dh, pb_cap);
                kl_end = total;   // patch-keeplearn2
""", """                kl_used = used;
                kl_B = tight ? 0 : std::min(dh, pb_cap);
                kl_end = total;   // patch-keeplearn2
                if (kl_hist_on) {   // larry, patch-keephist: a tight batch forgets them all
                    if (tight) kl_hist.clear();
                    else {
                        kl_hist.push_back({kl_B, used, total});
                        if (kl_hist.size() > 8) kl_hist.erase(kl_hist.begin());
                    }
                }
""")
open(P, "w").write(s)
print("patched", P)
