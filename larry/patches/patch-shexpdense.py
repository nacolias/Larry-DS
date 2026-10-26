#!/usr/bin/env python3
# patch-shexpdense.py [ENGINE_DIR] - larry, STRATA_SHEXP_TIER (with patch-shexptier.py / patch-shexpgen.py): the dense
# half's prompt spans can leave the shared expert to the expert tier.  Ds41Dense::set_shexp_skip(true): the span
# graphs write zeros into the shared expert's columns (the finish adds shexp + routed; the tier puts the shared
# expert into routed) instead of running it.  Decode / verify graphs are untouched.  Idempotent.
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine"
P, H = D + "/ds41_dense.cpp", D + "/ds41_dense.hpp"
s, h = open(P).read(), open(H).read()
if "set_shexp_skip" in s:
    print("already patched"); sys.exit(0)


def rep(text, old, new, n=1, what="cpp"):
    if text.count(old) != n:
        sys.exit("%s anchor x%d (want %d): %s" % (what, text.count(old), n, old[:100]))
    return text.replace(old, new)


h = rep(h, """    void set_pass_ring(bool on);
""", """    void set_pass_ring(bool on);
    /// larry, STRATA_SHEXP_TIER (patch-shexpdense.py): the following prompt spans leave the shared expert to the expert
    /// tier (their graphs write zeros into its columns; the tier adds it into the routed sum)
    void set_shexp_skip(bool on);
""", what="hpp")
s = rep(s, """    std::mutex * sub_mu = nullptr;            // larry: set_submit_lock
""", """    std::mutex * sub_mu = nullptr;            // larry: set_submit_lock
    bool shexp_skip = false;                  // larry, STRATA_SHEXP_TIER: set_shexp_skip
""")
s = rep(s, """    // ---- shared expert
    if (with_shexp) {
        ggml_tensor * sx = b.cols2(im.shexp, D, n, col0);
        outs.push_back(fn8 ? shexp_ffn(im, gc, il, fn, fn8, sx) : ggml_cpy(gc, shexp_ffn(im, gc, il, fn), sx));
    }
""", """    // ---- shared expert
    if (with_shexp) {
        ggml_tensor * sx = b.cols2(im.shexp, D, n, col0);
        if (im.shexp_skip && !dec)   // larry, STRATA_SHEXP_TIER: the expert cards add it into the routed sum - zeros here
            outs.push_back(ggml_cpy(gc, ggml_scale(gc, fn, 0.0f), sx));
        else
            outs.push_back(fn8 ? shexp_ffn(im, gc, il, fn, fn8, sx) : ggml_cpy(gc, shexp_ffn(im, gc, il, fn), sx));
    }
""")
s = rep(s, """void Ds41Dense::set_submit_lock(std::mutex * m, std::atomic<int> * prio) { p_->sub_mu = m; p_->sub_prio = m ? prio : nullptr; }
""", """void Ds41Dense::set_submit_lock(std::mutex * m, std::atomic<int> * prio) { p_->sub_mu = m; p_->sub_prio = m ? prio : nullptr; }
void Ds41Dense::set_shexp_skip(bool on) { p_->shexp_skip = on; }   // larry, STRATA_SHEXP_TIER
""")
open(P, "w").write(s)
open(H, "w").write(h)
print("patched", P, H)
