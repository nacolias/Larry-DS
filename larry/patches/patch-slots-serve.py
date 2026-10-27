#!/usr/bin/env python3
# patch-slots-serve.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, stage 1 step B): Strata's batch-slot
# protocol on the ds41 serve loop (serve/server.py generate_batched; upstream Xolotl 9ae1def6's lines), with the
# conversation slots of patch-seqs-*.py.  --batch-slots N (the server passes it for "parallel": N; default 1 = off):
#   in:  GEN / GENI (a request alone: as before, DSpark included) | BGEN <slot> <max_new> [k=v ..] <ids> |
#        BGENI <slot> .. | BSTOP <slot> | BYIELD <slot> (accepted, not acted on: the read finishes) | STOP | QUIT
#   out: a BGEN reads its prompt as a GEN 1 (RESUME / PP / T / DONE), then BADM <slot> <1 = decodes on | 0 = done>;
#        the slot's tokens are BT <slot> <id>, its end BDONE <slot> <produced> <stop|length|cancel> <ms>;
#        INFO .. batch_slots=N at the start.
# The busy slots take turns: a window = one DSpark round (draft, verify, accept) of each decoding slot, each in its own
# conversation slot (bound by pointer, no copy) - every pass stays a single-sequence pass, so the kernels are unchanged
# and each slot decodes exactly as it would alone.  Windows run whenever no control line is waiting, and between the
# layer-major batches of a prompt being read (and every 4 passes of a short one): a long prompt pauses the others for
# one batch at most.  The decode loop is now a per-request state (DecState) with one round a call (dspark_cycle /
# plain_cycle): the request alone runs them in a loop as before.  A BGEN's conversation slot is the idle one whose
# conversation its prompt continues best (the logical <slot> is mapped onto it); its own RNG.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
s = open(P).read()
if "patch-slots-serve.py" in s:
    print("already patched", P)
    sys.exit(0)


def rep(old, new):
    global s
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:90]))
    s = s.replace(old, new)


# ---- the option
rep(r'''    int threads = 0, prefill_chunk = 0, prompt_batch = 16384;''',
    r'''    int batch_slots = 1;   // larry (patch-slots-serve.py): --batch-slots N (Strata's --batch N): the slot protocol
    int threads = 0, prefill_chunk = 0, prompt_batch = 16384;''')
rep(r'''        else if (k == "--prompt-batch") a.prompt_batch = std::atoi(next().c_str());''',
    r'''        else if (k == "--prompt-batch") a.prompt_batch = std::atoi(next().c_str());
        else if (k == "--batch-slots" || k == "--batch") a.batch_slots = std::max(1, std::min(8, std::atoi(next().c_str())));''')
rep(r'''    const int serve_seqs = a.serve && std::getenv("STRATA_SERVE_SEQS") ? std::max(1, std::min(8, std::atoi(std::getenv("STRATA_SERVE_SEQS")))) : 1;''',
    r'''    const int serve_seqs = std::max(a.serve ? a.batch_slots : 1,
        a.serve && std::getenv("STRATA_SERVE_SEQS") ? std::max(1, std::min(8, std::atoi(std::getenv("STRATA_SERVE_SEQS")))) : 1);''')

# ---- the prompt reader's seam: windows of the decoding slots between batches
rep(r'''    auto feed = [&](const int * ids, int n, int pos0, const std::function<void(int)> & progress,''',
    r'''    // larry (patch-slots-serve.py): set while a BGEN reads its prompt with other slots decoding - their windows run
    // between the layer-major batches (and between the short path's passes)
    std::function<void()> feed_seam;
    auto feed = [&](const int * ids, int n, int pos0, const std::function<void(int)> & progress,''')
rep(r'''                if (Ds41XeTier::ptl_on()) tier.ptl_dump("batch");   // larry, STRATA_GPU_PTL
#endif
                c0 += B;
''',
    r'''                if (Ds41XeTier::ptl_on()) tier.ptl_dump("batch");   // larry, STRATA_GPU_PTL
#endif
                c0 += B;
                if (feed_seam && c0 < n) feed_seam();   // larry (batch slots): the other slots' windows
''')

# ---- the serve loop: the command intake, the slots, the decode rounds, the windows
rep(r'''        auto stop_pending = [&]() -> bool {
            std::lock_guard<std::mutex> lk(qm);
            for (auto it = q.begin(); it != q.end(); ++it)
                if (*it == "STOP") { q.erase(it); return true; }
            return false;
        };''',
    r'''        // larry (patch-slots-serve.py): the command intake - control lines (GEN / GENI / BGEN / BGENI) queue in ctl_q, STOP /
        // BSTOP / BYIELD / QUIT become flags; poll_cmds(true) blocks until a line comes
        std::deque<std::string> ctl_q;
        bool ctl_stop = false, quit_seen = false;
        std::vector<char> bstop_req;   // BSTOP <slot> (sized with the slots below)
        auto poll_cmds = [&](bool block) {
            std::unique_lock<std::mutex> lk(qm);
            if (block) qc.wait(lk, [&] { return !q.empty() || eof; });
            while (!q.empty()) {
                std::string ln = std::move(q.front());
                q.pop_front();
                if (ln == "QUIT") quit_seen = true;
                else if (ln == "STOP") ctl_stop = true;
                else if (ln.rfind("BSTOP ", 0) == 0) {
                    const int k = std::atoi(ln.c_str() + 6);
                    if (k >= 0 && (size_t) k < bstop_req.size()) bstop_req[(size_t) k] = 1;
                } else if (ln.rfind("BYIELD ", 0) == 0) {
                    std::fprintf(stderr, "serve: %s - not supported, the read goes on\n", ln.c_str());
                } else if (!ln.empty()) {
                    ctl_q.push_back(std::move(ln));
                }
            }
            if (eof && q.empty()) quit_seen = true;
        };
        auto stop_pending = [&]() -> bool {   // the control-line request's STOP (the decode loop checks it every round)
            poll_cmds(false);
            const bool st = ctl_stop;
            ctl_stop = false;
            return st;
        };''')

# the slots (after the conversation slots of patch-seqs-gen.py) and the decode rounds
rep(r'''        if (n_seq > 1)
            std::fprintf(stderr, "serve: %d conversation slots resident on the cards (STRATA_SERVE_SEQS)\n", n_seq);''',
    r'''        if (n_seq > 1)
            std::fprintf(stderr, "serve: %d conversation slots resident on the cards (STRATA_SERVE_SEQS)\n", n_seq);
        // larry (patch-slots-serve.py): a request's decode state - one round a call (the request alone loops them, the
        // batch slots' windows take one each).  slot -1: the control lines' request (T lines), else BT <slot> lines.
        struct DecState {
            int slot = -1;
            int max_new = 0, tk = 0, gen = 0;
            float temp = 1.0f, top_p = 1.0f, min_p = 0.0f;
            std::mt19937_64 * rng = nullptr;
            const char * finish = "length";
            std::string ferr;
            bool spec = false, skip_emit = false;   // DSpark (else the plain path); skip_emit: t was emitted already
            int P = 0, t = -1, prev = 0;
            bool dirty = false;
            int sp_ids[5] = { 0, 0, 0, 0, 0 };
            float sp_conf[5] = { 0, 0, 0, 0, 0 };
            int lk_cool = 0, lk_back = 1;
            long lk_pass = 0, lk_dr = 0, lk_acc = 0, rq_pass = 0, rq_acc = 0;
            std::vector<float> lg;   // the plain path: the next token's logits
            int pos = 0;
        };
        bool in_seam = false;   // windows inside a prompt read: no rebalance ticks (the read owns the flex regions)
        auto emit_tok = [&](DecState & d, int tok) -> bool {   // false = the request ends here
            if (d.slot < 0) std::printf("T %d\n", tok);
            else std::printf("BT %d %d\n", d.slot, tok);
            std::fflush(stdout);
            ++d.gen;
            if (is_stop(tok)) { d.finish = "stop"; return false; }
            if (d.max_new > 0 && d.gen >= d.max_new) return false;
            return true;
        };
#if defined(DS41_HAVE_DSPARK)
        // ---- one DSpark round (as the one-shot loop: P = last fed position, t = the token for P+1, sp_ids = drafts for
        // P+2..; each verify row's distribution is sample_p's, spec_accept keeps it exact): emit t, verify the drafts after
        // it, emit the accepted ones.  false = the request ended (d.finish says how)
        // STRATA_SERVE_LOOKUP=L (s6, opt-in until measured; mab776/Strata-DS4 M11's design): when the context's last G+1
        // tokens (STRATA_LOOKUP_NGRAM=G, default 3) occurred earlier in this request (prompt + output), the tokens that
        // followed them there are the verify pass's drafts instead of DSpark's - an agent's output copies its context
        // (edits, rewrites, quoted tool results).  spec_accept treats every draft as a deterministic token (accept with p,
        // else resample without it), so the output distribution is unchanged.  A match of M tokens drafts M - G + 1 (<= L,
        // <= the pass width).  STRATA_LOOKUP_AGREE (default 1): only when the copy's first token is DSpark's own first
        // draft.  A lookup pass with less than half accepted sits the lookup out 1, 2, 4 .. 64 passes.  DSpark steps on
        // the verified rows either way.  kVerify (s6): rows one verify pass may take - 16, as the prompt passes (pass_m
        // splits the tier calls at kMaxMulti = 8 and the pass buffers hold 17).
        constexpr int kVerify = 16;
        auto dspark_cycle = [&](DecState & d, const std::function<bool()> & stopped) -> bool {
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
            static thread_local std::vector<int> vt;
            static thread_local std::vector<std::pair<int, double>> dist;
            int dr[kVerify] = { 0 };
            if (!d.skip_emit && !emit_tok(d, d.t)) return false;
            d.skip_emit = false;
            if (stopped()) { d.finish = "cancel"; return false; }
            int k = sp_depth_serve;
            if (d.max_new > 0) k = std::min(k, d.max_new - d.gen);
            k = std::min(k, ctxv - (d.P + 3));   // the pass writes P+1 .. P+1+k
            if (k < 0) return false;
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
            vt.clear();
            if (refeed) vt.push_back(d.prev);
            vt.push_back(d.t);
            for (int i = 0; i < k; ++i) vt.push_back(dr[i]);
            const int r0 = refeed ? 1 : 0, p0 = refeed ? d.P : d.P + 1;
            if (!ensure_kv(p0 + (int64_t) vt.size() + 64)) { d.ferr = err; d.finish = "error"; return false; }
            if (!pass_v(vt.data(), (int) vt.size(), p0)) { d.ferr = ferr; d.finish = "error"; return false; }
            int acc = 0, tn = -1;
            for (; acc < k; ++acc) {
                sample_dist(vlog.data() + (size_t) (r0 + acc) * nv, nv, d.temp, d.tk, d.top_p, d.min_p, dist);
                if (!spec_accept(dist, dr[acc], *d.rng, tn)) break;
            }
            if (acc == k) tn = sample_p(vlog.data() + (size_t) (r0 + k) * nv, nv, d.temp, d.tk, d.top_p, d.min_p, *d.rng);
            ++d.rq_pass; d.rq_acc += acc;
            if (is_lk) {
                ++d.lk_pass; d.lk_dr += k; d.lk_acc += acc;
                if (2 * acc < k) { d.lk_cool = d.lk_back; d.lk_back = std::min(64, d.lk_back * 2); }
                else d.lk_back = 1;
            }
            int spos[kVerify + 1];
            for (int i = 0; i <= acc; ++i) spos[i] = d.P + 1 + i;
            int drafts[kVerify];
            for (int i = 0; i < k; ++i) drafts[i] = dr[i];
            if (!dsp->step(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, tn, d.sp_ids, d.sp_conf, pv_tp ? r0 : -1)) {
                d.ferr = "DSpark step: " + dsp->last_error(); d.finish = "error"; return false;
            }
            hist.push_back(d.t);
            bool more = true;
            for (int i = 0; i < acc && more; ++i) { hist.push_back(drafts[i]); more = emit_tok(d, drafts[i]); }
            d.prev = acc == 0 ? d.t : drafts[acc - 1];
            d.dirty = acc < k;
            d.P += 1 + acc;
            d.t = tn;
            if (!in_seam) tier.rebalance_tick();
            return more;
        };
#endif
        // ---- one round of the plain path (no DSpark): sample from d.lg, emit, feed it (its next logits)
        auto plain_cycle = [&](DecState & d, const std::function<bool()> & stopped) -> bool {
            if (d.max_new > 0 && d.gen >= d.max_new) return false;
            if (stopped()) { d.finish = "cancel"; return false; }
            const int tok = sample_p(d.lg.data(), nv, d.temp, d.tk, d.top_p, d.min_p, *d.rng);
            if (!emit_tok(d, tok)) return false;
            if (d.pos + 1 >= ctxv) return false;
            if (!ensure_kv(d.pos + 64)) { d.ferr = err; d.finish = "error"; return false; }
            if (!step(tok, d.pos) || !logits_rows(0, 1, d.lg.data())) { d.ferr = ferr; d.finish = "error"; return false; }
            hist.push_back(tok);
            ++d.pos;
            if (!in_seam) tier.rebalance_tick();
            return true;
        };
        // ---- the batch slots (--batch-slots N: Strata's protocol) - logical slot k decodes in conversation slot phys
        struct SlotReq {
            bool decoding = false;
            int phys = 0;
            double t0 = 0;
            std::mt19937_64 rng;
            DecState d;
        };
        const int n_slots = a.batch_slots > 1 ? n_seq : 0;
        std::vector<SlotReq> slots((size_t) n_slots);
        for (int k = 0; k < n_slots; ++k) slots[(size_t) k].phys = k;
        bstop_req.assign((size_t) n_slots, 0);
        auto any_decoding = [&]() {
            for (const auto & sr : slots) if (sr.decoding) return true;
            return false;
        };
        auto phys_busy = [&](int q) {   // conversation slot q holds a decoding slot's conversation
            for (const auto & sr : slots) if (sr.decoding && sr.phys == q) return true;
            return false;
        };
        auto finish_slot = [&](int k) {   // (its conversation slot is bound)
            SlotReq & sr = slots[(size_t) k];
            DecState & d = sr.d;
            const double ms = now_ms() - sr.t0;
            const bool bad = std::strcmp(d.finish, "error") == 0;
#if defined(DS41_HAVE_DSPARK)
            if (d.spec) {
                dsp_next = bad ? -1 : d.P + 1;   // the ring holds every position <= P (the last emitted token is not fed)
                sv_pass += d.rq_pass; sv_acc += d.rq_acc;
            }
#endif
            if (bad) {
                std::fprintf(stderr, "serve: slot %d: decode failed: %s\n", k, d.ferr.c_str());
                hist.clear();
                conv_img.clear();
                snap_len = -1;
                d.finish = "length";
            }
            std::fprintf(stderr, "serve: slot %d (conversation slot %d): %d tokens in %.2f s = %.2f tok/s (%s)%s\n", k, sr.phys,
                         d.gen, ms / 1000.0, 1000.0 * std::max(0, d.gen - 1) / std::max(ms, 1e-9), d.finish,
                         d.rq_pass > 0 ? (", DSpark " + std::to_string(1.0 + (double) d.rq_acc / d.rq_pass).substr(0, 4) + " tokens/pass").c_str() : "");
            std::printf("BDONE %d %d %s %.1f\n", k, d.gen, d.finish, ms);
            std::fflush(stdout);
            sr.decoding = false;
            bstop_req[(size_t) k] = 0;
        };
        int win_rr = 0;
        auto run_window = [&]() {   // one round of every decoding slot, each in its own conversation slot
            if (!bias_now) { bias_now = true; apply_bias(); }   // (the decode biases)
            for (int i = 0; i < n_slots; ++i) {
                const int k = (win_rr + i) % n_slots;
                SlotReq & sr = slots[(size_t) k];
                if (!sr.decoding) continue;
                if (!seq_switch(sr.phys)) { sr.d.ferr = err; sr.d.finish = "error"; finish_slot(k); continue; }
                auto stopped = [&]() { poll_cmds(false); return bstop_req[(size_t) k] != 0; };
#if defined(DS41_HAVE_DSPARK)
                const bool more = sr.d.spec ? dspark_cycle(sr.d, stopped) : plain_cycle(sr.d, stopped);
#else
                const bool more = plain_cycle(sr.d, stopped);
#endif
                if (!more) finish_slot(k);
            }
            win_rr = n_slots ? (win_rr + 1) % n_slots : 0;
        };
        // the next control line; the decoding slots' windows run while none is waiting.  false: QUIT / end of input
        auto next_request = [&](std::string & out) -> bool {
            for (;;) {
                poll_cmds(false);
                if (!ctl_q.empty()) {
                    out = std::move(ctl_q.front());
                    ctl_q.pop_front();
                    return true;
                }
                if (quit_seen) return false;
                if (any_decoding()) { run_window(); continue; }
                poll_cmds(true);
            }
        };''')

# the INFO line: the slots
rep(r'''        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld conversation_cache_slots=%d conversation_cache_mib=%d\n",
                    ctxv, (long long) tier.resident(), (long long) tier.arena_experts(), park_on ? park_slots : 0,
                    park_on ? (int) (park_gib * 1024) : 0);   // (0 MiB: no budget - the Monitor card then says "off")''',
    r'''        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld conversation_cache_slots=%d conversation_cache_mib=%d%s\n",
                    ctxv, (long long) tier.resident(), (long long) tier.arena_experts(), park_on ? park_slots : 0,
                    park_on ? (int) (park_gib * 1024) : 0,   // (0 MiB: no budget - the Monitor card then says "off")
                    n_slots > 1 ? (" batch_slots=" + std::to_string(n_slots)).c_str() : "");   // (patch-slots-serve.py)''')

# the loop: requests from the scheduler; BGEN parsed
rep(r'''        std::string line;
        while (next_line(line)) {
            if (line == "QUIT") break;
            if (line == "STOP" || line.empty()) continue;
            // larry (vision): GENI <max_new> [keys] [img_keys=k,..] <embeddings file> <ids csv> - a request with pictures
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
            std::istringstream ss(line);
            std::string w, csv, img_path;
            std::vector<std::string> img_keys;
            int max_new = 0, tk = 0;
            float temp = a.temp, top_p = 1.0f, min_p = 0.0f;
            ss >> w >> max_new;''',
    r'''        std::string line;
        (void) next_line;
        while (next_request(line)) {   // larry (patch-slots-serve.py): the decoding slots' windows run in between
            ctl_stop = false;          // (a STOP before this request is not for it)
            // larry (vision): GENI <max_new> [keys] [img_keys=k,..] <embeddings file> <ids csv> - a request with pictures
            // larry (batch slots): BGEN / BGENI <slot> ... - read into batch slot <slot>, which then decodes in the windows
            const bool bgen = line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0;
            const bool geni = line.rfind("GENI ", 0) == 0 || line.rfind("BGENI ", 0) == 0;
            if (!geni && !bgen && line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
            std::istringstream ss(line);
            std::string w, csv, img_path;
            std::vector<std::string> img_keys;
            int max_new = 0, tk = 0, admit = -1;
            float temp = a.temp, top_p = 1.0f, min_p = 0.0f;
            ss >> w;
            if (bgen) ss >> admit;
            ss >> max_new;
            if (bgen && (n_slots < 2 || admit < 0 || admit >= n_slots || slots[(size_t) admit].decoding)) {
                std::printf("ERR BGEN: slot %d is not free\n", admit);
                std::fflush(stdout);
                continue;
            }''')

# the conversation slot: an idle one (BGEN: mapped onto the logical slot)
rep(r'''            if (n_seq > 1) {   // larry (batch slots): the conversation slot this request continues best
                int best = -1;
                size_t bl = 0;
                for (int q = 0; q < n_seq; ++q) {
                    const std::vector<int> & h = seq_hist(q);''',
    r'''            if (n_seq > 1) {   // larry (batch slots): the conversation slot this request continues best
                int best = -1;
                size_t bl = 0;
                for (int q = 0; q < n_seq; ++q) {
                    if (phys_busy(q)) continue;   // (patch-slots-serve.py: a decoding slot's)
                    const std::vector<int> & h = seq_hist(q);''')
rep(r'''                if (best < 0 || bl < (size_t) park_min) {   // continues none: an empty slot, else the one used longest ago
                    best = -1;
                    for (int q = 0; q < n_seq && best < 0; ++q) if (seq_hist(q).empty()) best = q;
                    if (best < 0) {
                        best = 0;
                        for (int q = 1; q < n_seq; ++q) if (seq_used[(size_t) q] < seq_used[(size_t) best]) best = q;
                    }
                }
                if (!seq_switch(best)) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); continue; }''',
    r'''                if (best < 0 || bl < (size_t) park_min) {   // continues none: an empty slot, else the one used longest ago
                    best = -1;
                    for (int q = 0; q < n_seq && best < 0; ++q) if (!phys_busy(q) && seq_hist(q).empty()) best = q;
                    if (best < 0)
                        for (int q = 0; q < n_seq; ++q)
                            if (!phys_busy(q) && (best < 0 || seq_used[(size_t) q] < seq_used[(size_t) best])) best = q;
                }
                if (best < 0) { std::printf("ERR %sno idle conversation slot\n", bgen ? "BGEN: " : ""); std::fflush(stdout); continue; }
                if (admit >= 0 && slots[(size_t) admit].phys != best)   // (patch-slots-serve.py: logical slot -> this one)
                    for (auto & sr : slots)
                        if (sr.phys == best) { std::swap(sr.phys, slots[(size_t) admit].phys); break; }
                if (!seq_switch(best)) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); continue; }''')

# a BGEN's read: the windows between its batches / passes
rep(r'''            const double tph0 = now_ms();
            btr("feed start (batch width)", pb_cap);''',
    r'''            // larry (patch-slots-serve.py): a BGEN read with other slots decoding: their windows between its batches
            const int feed_seq = cur_seq;
            if (admit >= 0 && any_decoding())
                feed_seam = [&]() {
                    const bool b0 = bias_now;
                    in_seam = true;
                    run_window();
                    in_seam = false;
                    if (!seq_switch(feed_seq)) std::fprintf(stderr, "serve: back to the read's conversation slot: %s\n", err.c_str());
                    if (bias_now != b0) { bias_now = b0; apply_bias(); }
                };
            const double tph0 = now_ms();
            btr("feed start (batch width)", pb_cap);''')
rep(r'''                    const int before = i;
                    i += m;
                    last_rows = m;
                    if (i / 64 != before / 64 && i < dh) pp(i);''',
    r'''                    const int before = i;
                    i += m;
                    last_rows = m;
                    if (i / 64 != before / 64 && i < dh) pp(i);
                    if (feed_seam && i < dh && (i / m) % 4 == 0) feed_seam();   // (larry, batch slots)''')
rep(r'''            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));''',
    r'''            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
            feed_seam = nullptr;''')

# ---- the decode: the request's state, its rounds; a BGEN's own token
a = s.index(r'''#if defined(DS41_HAVE_DSPARK)
            // ---- DSpark speculative sampling (as the one-shot loop''')
b = s.index(r'''            const double dec_ms = now_ms() - td0;''')
s = s[:a] + r'''#if defined(DS41_HAVE_DSPARK)
            long rq_pass = 0, rq_acc = 0;
#endif
            // larry (patch-slots-serve.py): the decode as a state and its rounds - this request alone loops them here; a
            // BGEN emits its own token (a GEN 1) and its slot takes the rounds in the windows from there
            DecState dd;
            dd.max_new = max_new; dd.tk = tk; dd.temp = temp; dd.top_p = top_p; dd.min_p = min_p;
            dd.rng = &rng;
            dd.spec = sp_t >= 0;
            dd.P = total - 1; dd.t = sp_t; dd.prev = ids.back();
            std::copy(sp_ids, sp_ids + 5, dd.sp_ids);
            std::copy(sp_conf, sp_conf + 5, dd.sp_conf);
            dd.lg = lg; dd.pos = pos;
            bool slot_cont = false;
            if (admit >= 0) {
                SlotReq & sr = slots[(size_t) admit];
                sr.rng.seed(rng());   // the slot's own sampler (from the request's seed= when it had one)
                dd.rng = &sr.rng;
                if (stop_pending()) dd.finish = "cancel";
                else if (dd.spec) slot_cont = emit_tok(dd, dd.t);
                else slot_cont = plain_cycle(dd, [] { return false; });
                dd.skip_emit = dd.spec;   // the slot's first round verifies t's drafts without emitting t again
#if defined(DS41_HAVE_DSPARK)
                if (!slot_cont && dd.spec) dsp_next = dd.P + 1;
#endif
            }
#if defined(DS41_HAVE_DSPARK)
            else if (dd.spec) {
                while (dspark_cycle(dd, stop_pending)) {}
                if (dd.lk_pass > 0)
                    std::fprintf(stderr, "serve: lookup %ld/%ld drafts accepted in %ld passes\n", dd.lk_acc, dd.lk_dr, dd.lk_pass);
                dsp_next = dd.P + 1;   // the ring holds every position <= P (the last emitted token is not fed)
                if (std::strcmp(dd.finish, "error") == 0) dsp_next = -1;
                sv_pass += dd.rq_pass; sv_acc += dd.rq_acc;
            }
#endif
            else {
                while (plain_cycle(dd, stop_pending)) {}
            }
            gen = dd.gen;
            finish = dd.finish;
            if (std::strcmp(finish, "error") == 0) ferr = dd.ferr;
            pos = dd.pos;
#if defined(DS41_HAVE_DSPARK)
            rq_pass = dd.rq_pass; rq_acc = dd.rq_acc;
#endif
''' + s[b:]

# after DONE: a BGEN's BADM, its slot decoding on
rep(r'''            std::printf("DONE %d %d %.1f %.1f %s 0 0 %d %lld %lld 0 %lld 0 %d\n", gen, total, prompt_ms, dec_ms, finish, reused,
                        (long long) st.hits, (long long) st.lookups(), (long long) st.file_tier, total - reused);
            std::fflush(stdout);''',
    r'''            std::printf("DONE %d %d %.1f %.1f %s 0 0 %d %lld %lld 0 %lld 0 %d\n", gen, total, prompt_ms, dec_ms, finish, reused,
                        (long long) st.hits, (long long) st.lookups(), (long long) st.file_tier, total - reused);
            std::fflush(stdout);
            if (admit >= 0) {   // larry (patch-slots-serve.py): BADM, and the slot decodes on in the windows
                std::printf("BADM %d %d\n", admit, slot_cont ? 1 : 0);
                std::fflush(stdout);
                if (slot_cont) {
                    SlotReq & sr = slots[(size_t) admit];
                    sr.d = std::move(dd);
                    sr.d.slot = admit;
                    sr.d.rng = &sr.rng;
                    sr.decoding = true;
                    sr.phys = cur_seq;
                    sr.t0 = now_ms();
                    bstop_req[(size_t) admit] = 0;
                }
            }''')

open(P, "w").write(s)
print("patched", P)
