#!/usr/bin/env python3
# patch-batch-serve.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, stage 2: batched decode): the batch slots'
# DSpark rounds share ONE TP pass.  A round is split in two halves - plan (emit t, choose the drafts, the verify rows)
# and accept (against the pass's rows, DSpark's step on the taps, emit the accepted drafts); a request alone runs plan,
# its own pass, accept, as before.  When two or more slots decode with DSpark on the TP cards (STRATA_SERVE_JOINT,
# default 1), each plans in its own conversation (its drafts capped so all rows fit the pass's 7), one
# Ds41Tp::pass_multi runs them all (each run in its conversation's state: patch-batch-tp.py), then each accepts.  Needs
# patch-batch-tp.py, patch-batch-dense.py (the Engram rows of a run's own conversation) and patch-widthbench.py (its
# timing hooks sit in the round).  Idempotent.
import hashlib
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
s = open(P).read()
if "joint_round" in s:
    print("already patched", P)
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


# the TP core's per-conversation Engram rows
s = rep(s, """        th.engram_slot = [&dense](const int * t, int n, int p0, int s, float * o) { return dense.engram_rows_slot(t, n, p0, s, o); };
""", """        th.engram_slot = [&dense](const int * t, int n, int p0, int s, float * o) { return dense.engram_rows_slot(t, n, p0, s, o); };
        // larry (batched decode, patch-batch-serve.py): a pass of several conversations - each run's rows from its own history
        th.engram_slot_seq = [&dense](int q, const int * t, int n, int p0, int s, float * o) { return dense.engram_rows_slot_seq(q, t, n, p0, s, o); };
""", "engram hook")
# the bound conversation's sync, a pass of several conversations
s = rep(s, """    const bool tp_on = tp != nullptr;
#else
    auto tp_run = [&](const int *, int, int, float *, float * = nullptr) -> bool { return false; };
    const bool tp_on = false;
#endif
""", """    const bool tp_on = tp != nullptr;
    // larry (batched decode, patch-batch-serve.py): the bound conversation's card-0 state to the other cards when the
    // prompt path / a restore wrote it (tp_run's first step - a pass of several conversations syncs each one first)
    auto tp_sync_bound = [&]() -> bool {
        int64_t lo = 0, hi = 0;
        bool changed = false;
        dense.take_written(&lo, &hi, &changed);
        if (changed) {
            dense.pass_sync();
            if (!tp->sync_state(lo, hi, false)) { ferr = tp->last_error(); return false; }
        }
        return true;
    };
    // a pass of several conversations' runs (Ds41Tp::pass_multi): their logits rows into lg_out, DSpark's taps into vh_out
    auto tp_run_multi = [&](const int * toks, const int * seqs, const int * ns, const int * p0s, int nrun, float * vh_out,
                            float * lg_out) -> bool {
#if defined(DS41_XE)
        tier.regrow_poll(false);
#endif
        if (!tp->pass_multi(toks, seqs, ns, p0s, nrun, lg_out, vh_out)) { ferr = tp->last_error(); return false; }
        last_tp = true;
        tp_lg = lg_out;
        static const int pchunks = std::getenv("STRATA_TP_PROMOTE_CHUNKS") ? std::atoi(std::getenv("STRATA_TP_PROMOTE_CHUNKS")) : 64;
        tier.tp_between_passes(pchunks);
        return true;
    };
#else
    auto tp_run = [&](const int *, int, int, float *, float * = nullptr) -> bool { return false; };
    auto tp_sync_bound = [&]() -> bool { return true; };
    auto tp_run_multi = [&](const int *, const int *, const int *, const int *, int, float *, float *) -> bool { return false; };
    const bool tp_on = false;
#endif
""", "tp_run")
s = rep(s, """        vlog.resize((size_t) m * nv);
        if (!dense.logits(0, m, vlog.data())) { ferr = dense.last_error(); return false; }
        lap(3);
        return true;
    };
""", """        vlog.resize((size_t) m * nv);
        if (!dense.logits(0, m, vlog.data())) { ferr = dense.last_error(); return false; }
        lap(3);
        return true;
    };
    // larry (batched decode, patch-batch-serve.py): pass_v for a pass of several conversations' runs (TP only) - vlog /
    // vh hold the rows of every run, in run order
    auto pass_vm = [&](const int * toks, const int * seqs, const int * ns, const int * p0s, int nrun) -> bool {
#if defined(DS41_XE)
        tier.tl_flush();
#endif
        int m = 0;
        for (int u = 0; u < nrun; ++u) m += ns[u];
        ++pv_n; pv_tok += m;
        const double tq = now_ms();
        vh.resize((size_t) m * 3 * n_embd);
        vlog.resize((size_t) m * nv);
        if (!tp_run_multi(toks, seqs, ns, p0s, nrun, vh.data(), vlog.data())) return false;
        pv_tp = true;
        pv_ms[0] += now_ms() - tq;
        return true;
    };
""", "pass_vm")

# the DSpark round: plan / accept halves
a = s.index("        auto dspark_cycle = [&](DecState & d, const std::function<bool()> & stopped) -> bool {\n")
b = s.index("#endif\n        // ---- one round of the plain path (no DSpark)", a)
if hashlib.md5(s[a:b].encode()).hexdigest() != "9671c79b02745d1bcc0b5bd21b4f10b0":
    sys.exit("dspark_cycle is not the one this patch was written against")
s = s[:a] + r"""        // larry (batched decode, patch-batch-serve.py): a round in two halves - plan (emit t, the drafts: at most kcap,
        // the verify rows) and accept (against the pass's rows) - so the slots' rounds can share one TP pass
        struct VPlan {
            int k = 0, r0 = 0, p0 = 0;
            bool is_lk = false;
            int dr[kVerify] = { 0 };
            std::vector<int> vt;
        };
        auto dspark_plan = [&](DecState & d, const std::function<bool()> & stopped, int kcap, VPlan & v) -> bool {   // false = ended
            static const int lk_max = std::getenv("STRATA_SERVE_LOOKUP") ? std::max(0, std::min(kVerify - 2, std::atoi(std::getenv("STRATA_SERVE_LOOKUP")))) : 0;
            static const int lk_g = std::getenv("STRATA_LOOKUP_NGRAM") ? std::max(1, std::atoi(std::getenv("STRATA_LOOKUP_NGRAM"))) : 3;
            static const bool lk_agree = !(std::getenv("STRATA_LOOKUP_AGREE") && std::getenv("STRATA_LOOKUP_AGREE")[0] == '0');
            // STRATA_DSPARK_CONF=<tau>, as the one-shot loop: verify only the leading drafts whose confidence-head
            // probability is >= tau (the emitted tokens are still the trunk's - only the verify width changes)
            static const float conf_tau_s = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
            // larry, STRATA_DSPARK_CUM=<c> (patch-dsparkcum.py; Project Maya's MAYA_DS4_DSPARK_CUM): also stop at the
            // first draft where the product of the confidences so far drops below c
            static const float cum_tau_s = std::getenv("STRATA_DSPARK_CUM") ? (float) std::atof(std::getenv("STRATA_DSPARK_CUM")) : 0.0f;
            static const bool lk_tp = std::getenv("STRATA_LOOKUP_TP") && std::getenv("STRATA_LOOKUP_TP")[0] == '1';
            int * dr = v.dr;
            for (int i = 0; i < kVerify; ++i) dr[i] = 0;
            if (!d.skip_emit && !emit_tok(d, d.t)) return false;
            d.skip_emit = false;
            if (stopped()) { d.finish = "cancel"; return false; }
            int k = sp_depth_serve;
            if (d.max_new > 0) k = std::min(k, d.max_new - d.gen);
            k = std::min(k, ctxv - (d.P + 3));   // the pass writes P+1 .. P+1+k
            if (k < 0) return false;
            k = std::min(k, kcap);   // (batched decode: this slot's share of the pass)
            if (conf_tau_s >= 0 || cum_tau_s > 0) {
                int kk = 0;
                float cum = 1.0f;
                while (kk < k) {
                    const float p = 1.0f / (1.0f + std::exp(-d.sp_conf[kk]));
                    if (p < conf_tau_s || (cum *= p) < cum_tau_s) break;
                    ++kk;
                }
                k = kk;
            }
            const bool refeed = d.dirty && ((d.P + 1) & 1);
            for (int i = 0; i < k; ++i) dr[i] = d.sp_ids[i];
            bool is_lk = false;
            if (lk_max > 0 && d.lk_cool > 0) --d.lk_cool;
            else if (lk_max > 0) {
                // the sequence is hist + t (t = the token for P+1, not in hist yet); find the latest earlier q with
                // seq[q-G .. q] == seq[n-G .. n] (G+1 tokens, the last being t), then copy seq[q+1 ..]
                const int n = (int) hist.size();
                auto sq = [&](int i) { return i == n ? d.t : hist[(size_t) i]; };
                int cap = std::min(lk_max, kVerify - 1 - (refeed ? 1 : 0));
                // larry, engine1 (STRATA_LOOKUP_TP=1): the pass stays on the TP core (<= 7 rows) - a wider one
                // takes the old path, which pages card 0's dense weights in every layer (~0.1-0.2 s a pass)
                if (lk_tp && tp_on) cap = std::min(cap, 7 - 1 - (refeed ? 1 : 0));
                if (d.max_new > 0) cap = std::min(cap, d.max_new - d.gen);
                cap = std::min(cap, ctxv - (d.P + 3));
                cap = std::min(cap, kcap);
                if (cap >= 2 && n > lk_g + 1) {
                    for (int q = n - 1; q >= lk_g; --q) {
                        if (hist[(size_t) q] != d.t) continue;
                        int m = 1;
                        while (m < 64 && q - m >= 0 && hist[(size_t) (q - m)] == sq(n - m)) ++m;
                        if (m < lk_g + 1) continue;
                        const int L = std::min({cap, m - lk_g + 1, n - q});
                        if (L < 2) break;   // the latest match copies too little: not worth a pass
                        if (lk_agree && sq(q + 1) != d.sp_ids[0]) break;
                        for (int i = 0; i < L; ++i) dr[i] = sq(q + 1 + i);
                        k = L;
                        is_lk = true;
                        break;
                    }
                }
            }
            v.vt.clear();
            if (refeed) v.vt.push_back(d.prev);
            v.vt.push_back(d.t);
            for (int i = 0; i < k; ++i) v.vt.push_back(dr[i]);
            v.k = k;
            v.is_lk = is_lk;
            v.r0 = refeed ? 1 : 0;
            v.p0 = refeed ? d.P : d.P + 1;
            return true;
        };
        // lg: the round's logits rows (v.vt.size()), hv: its taps' host rows, tap_row: their first row on the cards (-1:
        // none).  false = the request ended (d.finish says how)
        auto dspark_accept = [&](DecState & d, const VPlan & v, const float * lg, const float * hv, int tap_row) -> bool {
            static thread_local std::vector<std::pair<int, double>> dist;
            const int k = v.k, r0 = v.r0;
            const int * dr = v.dr;
            int acc = 0, tn = -1;
            for (; acc < k; ++acc) {
                sample_dist(lg + (size_t) (r0 + acc) * nv, nv, d.temp, d.tk, d.top_p, d.min_p, dist);
                if (!spec_accept(dist, dr[acc], *d.rng, tn)) break;
            }
            if (acc == k) tn = sample_p(lg + (size_t) (r0 + k) * nv, nv, d.temp, d.tk, d.top_p, d.min_p, *d.rng);
            ++d.rq_pass; d.rq_acc += acc;
            if (v.is_lk) {
                ++d.lk_pass; d.lk_dr += k; d.lk_acc += acc;
                if (2 * acc < k) { d.lk_cool = d.lk_back; d.lk_back = std::min(64, d.lk_back * 2); }
                else d.lk_back = 1;
            }
            int spos[kVerify + 1];
            for (int i = 0; i <= acc; ++i) spos[i] = d.P + 1 + i;
            int drafts[kVerify];
            for (int i = 0; i < k; ++i) drafts[i] = dr[i];
            const double wb_s0 = wb_prof ? now_ms() : 0.0;
            if (!dsp->step(hv + (size_t) r0 * 3 * n_embd, spos, acc + 1, tn, d.sp_ids, d.sp_conf, tap_row >= 0 ? tap_row + r0 : -1)) {
                d.ferr = "DSpark step: " + dsp->last_error(); d.finish = "error"; return false;
            }
            if (wb_prof) wb_cyc[3] += now_ms() - wb_s0;
            hist.push_back(d.t);
            bool more = true;
            for (int i = 0; i < acc && more; ++i) { hist.push_back(drafts[i]); more = emit_tok(d, drafts[i]); }
            d.prev = acc == 0 ? d.t : drafts[acc - 1];
            d.dirty = acc < k;
            d.P += 1 + acc;
            d.t = tn;
            return more;
        };
        auto dspark_cycle = [&](DecState & d, const std::function<bool()> & stopped) -> bool {
            static thread_local VPlan v;
            if (!dspark_plan(d, stopped, kVerify, v)) return false;
            if (!ensure_kv(v.p0 + (int64_t) v.vt.size() + 64)) { d.ferr = err; d.finish = "error"; return false; }
            const double wb_v0 = wb_prof ? now_ms() : 0.0;
            if (!pass_v(v.vt.data(), (int) v.vt.size(), v.p0)) { d.ferr = ferr; d.finish = "error"; return false; }
            if (wb_prof) { wb_cyc[0] += 1; wb_cyc[1] += (double) v.vt.size(); wb_cyc[2] += now_ms() - wb_v0; }
            const bool more = dspark_accept(d, v, vlog.data(), vh.data(), pv_tp ? 0 : -1);
            if (std::strcmp(d.finish, "error") != 0 && !in_seam) tier.rebalance_tick();
            return more;
        };
""" + s[b:]

# the joint passes' log
s = rep(s, """        auto finish_slot = [&](int k) {   // (its conversation slot is bound)
""", """        long jr_pass = 0, jr_rows = 0;   // larry (batched decode): joint passes and their rows since the last slot ended
        auto finish_slot = [&](int k) {   // (its conversation slot is bound)
""", "finish_slot")
s = rep(s, """            std::printf("BDONE %d %d %s %.1f\\n", k, d.gen, d.finish, ms);
""", """            if (jr_pass > 0) {   // (larry, batched decode)
                std::fprintf(stderr, "serve: batched decode: %ld joint passes, %.2f rows a pass\\n", jr_pass, (double) jr_rows / jr_pass);
                jr_pass = jr_rows = 0;
            }
            std::printf("BDONE %d %d %s %.1f\\n", k, d.gen, d.finish, ms);
""", "finish_slot log")
# the window: the DSpark slots' rounds in one pass
s = rep(s, """        int win_rr = 0;
        auto run_window = [&]() {   // one round of every decoding slot, each in its own conversation slot
            if (!bias_now) { bias_now = true; apply_bias(); }   // (the decode biases)
            for (int i = 0; i < n_slots; ++i) {
                const int k = (win_rr + i) % n_slots;
                SlotReq & sr = slots[(size_t) k];
                if (!sr.decoding) continue;
""", """        int win_rr = 0;
#if defined(DS41_HAVE_DSPARK)
        // larry (batched decode, patch-batch-serve.py): the DSpark slots' rounds in ONE TP pass (STRATA_SERVE_JOINT, default
        // 1): each plans in its own conversation - the drafts capped so every row fits the pass's 7 - then one pass of
        // them all (each run in its conversation's state: Ds41Tp::pass_multi), then each accepts against its rows and
        // steps DSpark on its taps.  Measured 2026-10-27 (patch-widthbench.py): two conversations' rows in one pass cost
        // what one conversation's rows of that width cost.
        static const bool batch_dec = !(std::getenv("STRATA_SERVE_JOINT") && std::getenv("STRATA_SERVE_JOINT")[0] == '0');
        auto joint_round = [&](const std::vector<int> & js) {
            const int nj = (int) js.size();
            int base = 0;
            for (int k : js) { const DecState & d = slots[(size_t) k].d; base += 1 + ((d.dirty && ((d.P + 1) & 1)) ? 1 : 0); }
            const int extra = std::max(0, 7 - base);
            static thread_local std::vector<VPlan> plans;
            static thread_local std::vector<int> live, jt, j_seq, j_n, j_p0;
            plans.resize((size_t) nj);
            live.clear();
            for (int i = 0; i < nj; ++i) {
                const int k = js[(size_t) i];
                SlotReq & sr = slots[(size_t) k];
                if (!seq_switch(sr.phys)) { sr.d.ferr = err; sr.d.finish = "error"; finish_slot(k); continue; }
                auto stopped = [&]() { poll_cmds(false); return bstop_req[(size_t) k] != 0; };
                const int kcap = extra / nj + ((i + win_rr) % nj < extra % nj ? 1 : 0);
                VPlan & v = plans[(size_t) i];
                if (!dspark_plan(sr.d, stopped, kcap, v)) { finish_slot(k); continue; }
                const bool kv_ok = ensure_kv(v.p0 + (int64_t) v.vt.size() + 64);
                if (!kv_ok || !tp_sync_bound()) { sr.d.ferr = kv_ok ? ferr : err; sr.d.finish = "error"; finish_slot(k); continue; }
                live.push_back(i);
            }
            if (live.empty()) return;
            jt.clear(); j_seq.clear(); j_n.clear(); j_p0.clear();
            for (int i : live) {
                const VPlan & v = plans[(size_t) i];
                jt.insert(jt.end(), v.vt.begin(), v.vt.end());
                j_seq.push_back(slots[(size_t) js[(size_t) i]].phys);
                j_n.push_back((int) v.vt.size());
                j_p0.push_back(v.p0);
            }
            bool ok;
            if (live.size() == 1) {   // the others ended in their plans: a plain pass in its conversation
                ok = seq_switch(j_seq[0]) && pass_v(jt.data(), (int) jt.size(), j_p0[0]);
            } else {
                ok = pass_vm(jt.data(), j_seq.data(), j_n.data(), j_p0.data(), (int) live.size());
                ++jr_pass;
                jr_rows += (long) jt.size();
            }
            const std::string perr = ok ? std::string() : (ferr.empty() ? err : ferr);
            int roff = 0;
            for (int i : live) {
                const int k = js[(size_t) i];
                SlotReq & sr = slots[(size_t) k];
                const VPlan & v = plans[(size_t) i];
                const int ro = roff;
                roff += (int) v.vt.size();
                if (!seq_switch(sr.phys)) { sr.d.ferr = err; sr.d.finish = "error"; finish_slot(k); continue; }
                if (!ok) { sr.d.ferr = perr; sr.d.finish = "error"; finish_slot(k); continue; }
                if (!dspark_accept(sr.d, v, vlog.data() + (size_t) ro * nv, vh.data() + (size_t) ro * 3 * n_embd, pv_tp ? ro : -1))
                    finish_slot(k);
            }
            if (!in_seam) tier.rebalance_tick();
        };
#endif
        auto run_window = [&]() {   // one round of every decoding slot, each in its own conversation slot
            if (!bias_now) { bias_now = true; apply_bias(); }   // (the decode biases)
            // larry (batched decode): the DSpark slots' rounds share one pass (in conversation-slot order - the pass's
            // run order: fewer graph shapes); the rest take theirs one by one below
            static thread_local std::vector<int> js;
            js.clear();
#if defined(DS41_HAVE_DSPARK)
            if (batch_dec && tp_on) {
                int base = 0;
                for (int q = 0; q < n_seq; ++q)
                    for (int k = 0; k < n_slots; ++k) {
                        const SlotReq & sr = slots[(size_t) k];
                        if (!sr.decoding || !sr.d.spec || sr.phys != q) continue;
                        const int b = 1 + ((sr.d.dirty && ((sr.d.P + 1) & 1)) ? 1 : 0);
                        if (base + b > 7) continue;
                        base += b;
                        js.push_back(k);
                    }
                if (js.size() >= 2) joint_round(js);
                else js.clear();
            }
#endif
            for (int i = 0; i < n_slots; ++i) {
                const int k = (win_rr + i) % n_slots;
                SlotReq & sr = slots[(size_t) k];
                if (!sr.decoding || std::find(js.begin(), js.end(), k) != js.end()) continue;
""", "run_window")
open(P, "w").write(s)
print("patched", P)
