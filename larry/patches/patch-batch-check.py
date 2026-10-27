#!/usr/bin/env python3
# patch-batch-check.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, stage 2: batched decode; diagnostic, default
# off): STRATA_SERVE_JOINT_CHECK=1 - before each joint pass every run's rows also go through a pass of their own (the
# conversation bound; its ratio-2 compressor states kept aside and put back by Ds41Tp::st_keep - every other state row
# the joint pass writes again), then the joint pass's logits rows are compared with those: the largest difference, the
# argmax agreement and KL(own || joint), every 32 joint passes in the engine log ("serve: joint check").  The baselines,
# no joint rows compared: =2 the own pass against the same rows with one more row after them (a width's rounding alone:
# the kernels' forms go by width), =3 against the same pass again (the run-to-run spread).  Run it with
# STRATA_TP_PROMOTE_CHUNKS=0 (no expert promotions between the passes: the same experts on the same cards).  Needs
# patch-batch-tp.py and patch-batch-serve.py.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
H = os.path.join(X, "tools/v41/engine/tp/tp_core.hpp")
C = os.path.join(X, "tools/v41/engine/tp/tp_core.cpp")
G = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
h = open(H).read()
c = open(C).read()
g = open(G).read()
if "st_keep" in h and "STRATA_SERVE_JOINT_CHECK" in g:
    print("already patched", H, C, G)
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


h = rep(h, """    int seq() const;
""", """    int seq() const;
    /// larry (batched decode, patch-batch-check.py; diagnostic): the bound conversation's ratio-2 compressor states (every
    /// card) copied aside (restore false) or back (true) - a check pass at the same positions then leaves them as they were
    bool st_keep(bool restore);
""", "hpp")
c = rep(c, """    uint64_t * stamps = nullptr;                               // STRATA_TP_STAMP_LAYER: the clock at each step of that layer
""", """    uint64_t * stamps = nullptr;                               // STRATA_TP_STAMP_LAYER: the clock at each step of that layer
    float * st_keep = nullptr;                                 // larry (patch-batch-check.py): st_keep's copies
""", "Card st_keep")
c = rep(c, """            c->graphs_m.clear();
            if (c->peer_blk) sycl::free(c->peer_blk, c->q);
""", """            c->graphs_m.clear();
            if (c->st_keep) sycl::free(c->st_keep, c->q);
            if (c->peer_blk) sycl::free(c->peer_blk, c->q);
""", "dtor")
c = rep(c, """int Ds41Tp::seq() const { return p_->seq; }
""", """int Ds41Tp::seq() const { return p_->seq; }

bool Ds41Tp::st_keep(bool restore) {   // larry (patch-batch-check.py, diagnostic)
    Impl & im = *p_;
    size_t floats = 0;
    for (int l = 0; l < im.NL; ++l)
        if (im.li[(size_t) l].kv_src && im.li[(size_t) l].ratio == 2) floats += (size_t) (4 * im.DH);
    for (auto & cp : im.cards) {
        Card & c = *cp;
        if (!c.st_keep) c.st_keep = sycl::malloc_device<float>(std::max<size_t>(floats, 1), c.q);
        if (!c.st_keep) { im.err = "tp: st_keep: no memory"; return false; }
        float * k = c.st_keep;
        const size_t b = (size_t) (2 * im.DH) * 4;
        for (int l = 0; l < im.NL; ++l) {
            const Impl::LInfo & L = im.li[(size_t) l];
            if (!L.kv_src || L.ratio != 2) continue;
            LS & s = c.ls[(size_t) l];
            if (restore) { c.q.memcpy(s.st_kv, k, b); c.q.memcpy(s.st_sc, k + 2 * im.DH, b); }
            else { c.q.memcpy(k, s.st_kv, b); c.q.memcpy(k + 2 * im.DH, s.st_sc, b); }
            k += 4 * im.DH;
        }
    }
    try {
        for (auto & cp : im.cards) cp->q.wait_and_throw();
    } catch (const std::exception & e) {
        im.err = std::string("tp: st_keep: ") + e.what();
        return false;
    }
    return true;
}
""", "st_keep def")
g = rep(g, """    auto tp_run_multi = [&](const int * toks, const int * seqs, const int * ns, const int * p0s, int nrun, float * vh_out,
                            float * lg_out) -> bool {
""", """    auto tp_st_keep = [&](bool restore) -> bool {   // (patch-batch-check.py, diagnostic)
        if (!tp->st_keep(restore)) { ferr = tp->last_error(); return false; }
        return true;
    };
    auto tp_run_multi = [&](const int * toks, const int * seqs, const int * ns, const int * p0s, int nrun, float * vh_out,
                            float * lg_out) -> bool {
""", "tp_st_keep")
g = rep(g, """    auto tp_sync_bound = [&]() -> bool { return true; };
""", """    auto tp_sync_bound = [&]() -> bool { return true; };
    auto tp_st_keep = [&](bool) -> bool { return false; };
""", "tp_st_keep stub")
g = rep(g, """            bool ok;
            if (live.size() == 1) {   // the others ended in their plans: a plain pass in its conversation
""", """            // larry, STRATA_SERVE_JOINT_CHECK=1 (diagnostic, patch-batch-check.py): each run's rows through a pass of its
            // own first (its ratio-2 compressor states kept aside and put back: the joint pass sees the state as it was)
            // (=2 / =3: baselines - the own pass against the same rows + one more row / the same pass again; no joint rows)
            static const int jcheck = std::getenv("STRATA_SERVE_JOINT_CHECK") ? std::atoi(std::getenv("STRATA_SERVE_JOINT_CHECK")) : 0;
            static std::vector<float> jc_lg, jc_lg2;
            static std::vector<int> jc_vt;
            static double jc_max = 0, jc_kl = 0;
            static long jc_rows = 0, jc_agree = 0, jc_n = 0;
            bool jc_ok = jcheck > 0 && live.size() >= 2;
            if (jc_ok) {
                jc_lg.clear();
                jc_lg2.clear();
                for (int i : live) {
                    const VPlan & v = plans[(size_t) i];
                    if (!seq_switch(slots[(size_t) js[(size_t) i]].phys) || !tp_st_keep(false) ||
                        !pass_v(v.vt.data(), (int) v.vt.size(), v.p0) || !tp_st_keep(true)) { jc_ok = false; break; }
                    jc_lg.insert(jc_lg.end(), vlog.begin(), vlog.begin() + (int64_t) v.vt.size() * nv);
                    if (jcheck >= 2) {
                        jc_vt = v.vt;
                        if (jcheck == 2 && jc_vt.size() < 7) jc_vt.push_back(jc_vt.back());
                        if (!tp_st_keep(false) || !pass_v(jc_vt.data(), (int) jc_vt.size(), v.p0) || !tp_st_keep(true)) { jc_ok = false; break; }
                        jc_lg2.insert(jc_lg2.end(), vlog.begin(), vlog.begin() + (int64_t) v.vt.size() * nv);
                    }
                }
                if (!jc_ok) std::fprintf(stderr, "serve: joint check: a run's own pass failed: %s\\n", ferr.c_str());
            }
            bool ok;
            if (live.size() == 1) {   // the others ended in their plans: a plain pass in its conversation
""", "check before")
g = rep(g, """            const std::string perr = ok ? std::string() : (ferr.empty() ? err : ferr);
""", """            const std::string perr = ok ? std::string() : (ferr.empty() ? err : ferr);
            if (jc_ok && ok && live.size() >= 2) {   // (patch-batch-check.py) the joint rows against their own passes' rows
                size_t row = 0;
                for (int i : live) {
                    const size_t rows = plans[(size_t) i].vt.size();
                    for (size_t r = 0; r < rows; ++r, ++row) {
                        const float * pa = jc_lg.data() + row * (size_t) nv;
                        const float * pb = (jcheck >= 2 ? jc_lg2.data() : vlog.data()) + row * (size_t) nv;
                        int64_t ia = 0, ib = 0;
                        double mx = 0;
                        for (int64_t t = 0; t < (int64_t) nv; ++t) {
                            mx = std::max(mx, (double) std::fabs(pa[t] - pb[t]));
                            if (pa[t] > pa[ia]) ia = t;
                            if (pb[t] > pb[ib]) ib = t;
                        }
                        double za = 0, zb = 0;
                        for (int64_t t = 0; t < (int64_t) nv; ++t) { za += std::exp((double) pa[t] - pa[ia]); zb += std::exp((double) pb[t] - pb[ib]); }
                        const double la = std::log(za), lb = std::log(zb);
                        double kl = 0;
                        for (int64_t t = 0; t < (int64_t) nv; ++t) {
                            const double xa = (double) pa[t] - pa[ia] - la, xb = (double) pb[t] - pb[ib] - lb;
                            if (xa > -30) kl += std::exp(xa) * (xa - xb);
                        }
                        jc_max = std::max(jc_max, mx);
                        jc_kl += kl;
                        jc_agree += ia == ib;
                        ++jc_rows;
                    }
                }
                if (++jc_n % 32 == 0) {
                    std::fprintf(stderr, "serve: joint check %s (%ld joint passes, %ld rows): max |logit diff| %.4f, argmax agree %.2f %%, mean KL %.3g\\n",
                                 jcheck == 1 ? "own vs joint" : jcheck == 2 ? "own vs own + 1 row" : "own vs own again", jc_n, jc_rows, jc_max,
                                 100.0 * (double) jc_agree / (double) std::max(1L, jc_rows), jc_kl / (double) std::max(1L, jc_rows));
                    jc_max = jc_kl = 0;
                    jc_rows = jc_agree = 0;
                }
            }
""", "check after")
open(H, "w").write(h)
open(C, "w").write(c)
open(G, "w").write(g)
print("patched", H, C, G)
