#!/usr/bin/env python3
# patch-seqs-gen.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, stage 1 step A): STRATA_SERVE_SEQS=N (default
# 1 = as before) resident conversations - the dense half's and the TP core's per-conversation state (patch-seqs-dense.py,
# patch-seqs-tp.py) wired up, and the serve loop serving each request in the conversation slot whose conversation it
# continues best, else in the slot used longest ago (an empty one first; its conversation is parked first, as before).
# Requests still run one at a time - this is the state, the interleaving comes next.  The serve loop's live
# conversation variables are the bound slot's; the others wait in seq_stash (swapped, no copy); the lazy KV map's mapped
# extent is per slot too.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
s = open(P).read()
if "STRATA_SERVE_SEQS" in s:
    print("already patched", P)
    sys.exit(0)
pairs = [
# the dense half's conversation count
("""    Ds41DenseConfig dc;
    dc.backend = be;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.ctx_max = ctx;
""",
"""    Ds41DenseConfig dc;
    dc.backend = be;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.ctx_max = ctx;
    // larry (batch slots, patch-seqs-gen.py): STRATA_SERVE_SEQS conversations resident side by side (--serve only)
    const int serve_seqs = a.serve && std::getenv("STRATA_SERVE_SEQS") ? std::max(1, std::min(8, std::atoi(std::getenv("STRATA_SERVE_SEQS")))) : 1;
    dc.n_seqs = serve_seqs;
"""),
# the lazy map's extent: per conversation
("""    int64_t kv_pos = 0;   // positions the map covers
""",
"""    int64_t kv_pos = 0;   // positions the map covers (larry, batch slots: the bound conversation's; the others' in kv_pos_seq)
    std::vector<int64_t> kv_pos_seq;
"""),
("""        const int64_t init = std::min<int64_t>(ctx, std::getenv("STRATA_DENSE_KV_INIT") ? std::atoll(std::getenv("STRATA_DENSE_KV_INIT")) : 262144);
        std::vector<size_t> kb;
        dense.kv_bytes_for(init, kb);
        if (!kvmap->grow(kb, err)) { std::fprintf(stderr, "ds41_generate: KV map: %s\\n", err.c_str()); return 1; }
        kv_pos = init;
""",
"""        const int64_t init = std::min<int64_t>(ctx, std::getenv("STRATA_DENSE_KV_INIT") ? std::atoll(std::getenv("STRATA_DENSE_KV_INIT")) : 262144);
        std::vector<size_t> kb;
        for (int q = dense.n_seqs() - 1; q >= 0; --q) {   // (larry, batch slots: every conversation's first positions)
            if (!dense.bind_seq(q)) { std::fprintf(stderr, "ds41_generate: KV map: no conversation slot %d\\n", q); return 1; }
            dense.kv_bytes_for(init, kb);
            if (!kvmap->grow(kb, err)) { std::fprintf(stderr, "ds41_generate: KV map: %s\\n", err.c_str()); return 1; }
        }
        kv_pos = init;
        kv_pos_seq.assign((size_t) dense.n_seqs(), init);
"""),
# the TP core: a state set a conversation
("""        tc.ctx_max = ctx;
""",
"""        tc.ctx_max = ctx;
        tc.n_seqs = dense.n_seqs();   // (larry, batch slots)
"""),
("""            th.state.push_back({ (float *) p[0], (uint16_t *) p[1], (uint16_t *) p[2], (float *) p[3], (float *) p[4] });
        }
""",
"""            th.state.push_back({ (float *) p[0], (uint16_t *) p[1], (uint16_t *) p[2], (float *) p[3], (float *) p[4] });
        }
        for (int q = 1; q < dense.n_seqs(); ++q) {   // larry (batch slots): the other conversations' rank-0 state
            th.seq_state.emplace_back();
            if (!dense.bind_seq(q)) { std::fprintf(stderr, "ds41_generate: no conversation slot %d\\n", q); return 1; }
            for (int l = 0; l < n_layer; ++l) {
                void * p[5] = {};
                if (!dense.state_ptrs(l, &p[0], &p[1], &p[2], &p[3], &p[4])) {
                    std::fprintf(stderr, "ds41_generate: STRATA_TP: conversation %d, layer %d: no state\\n", q, l);
                    return 1;
                }
                th.seq_state.back().push_back({ (float *) p[0], (uint16_t *) p[1], (uint16_t *) p[2], (float *) p[3], (float *) p[4] });
            }
        }
        if (!dense.bind_seq(0)) return 1;
"""),
# the serve loop: the other conversations' variables, the switch
("""        int dsp_next = -1;
        const int sp_depth_serve = std::getenv("STRATA_DSPARK_DEPTH") ? std::max(1, std::min(5, std::atoi(std::getenv("STRATA_DSPARK_DEPTH")))) : 4;
        long sv_pass = 0, sv_acc = 0;
#endif
""",
"""        int dsp_next = -1;
        const int sp_depth_serve = std::getenv("STRATA_DSPARK_DEPTH") ? std::max(1, std::min(5, std::atoi(std::getenv("STRATA_DSPARK_DEPTH")))) : 4;
        long sv_pass = 0, sv_acc = 0;
#endif
        // larry (batch slots, patch-seqs-gen.py): STRATA_SERVE_SEQS conversation slots, each holding a conversation on
        // the cards (the dense half's and the TP core's per-conversation state).  The live variables above are the
        // bound slot's; the others' wait in seq_stash (swapped in and out, no copies).
        struct SeqStash {
            std::vector<int> hist;
            std::vector<ImgId> img;
            int snap_len = -1;
            std::vector<float> snap_logits, snap_seed;
            int snap_seed_n = 0, dsp_next = -1;
        };
        const int n_seq = dense.n_seqs();
        std::vector<SeqStash> seq_stash((size_t) n_seq);
        std::vector<long> seq_used((size_t) n_seq, 0);
        int cur_seq = 0;
        long seq_tick = 0;
        auto seq_swap = [&](SeqStash & o) {
            std::swap(o.hist, hist);
            std::swap(o.img, conv_img);
            std::swap(o.snap_len, snap_len);
            std::swap(o.snap_logits, snap_logits);
            std::swap(o.snap_seed, snap_seed);
            std::swap(o.snap_seed_n, snap_seed_n);
#if defined(DS41_HAVE_DSPARK)
            std::swap(o.dsp_next, dsp_next);
#endif
        };
        auto seq_switch = [&](int q) -> bool {   // conversation slot q becomes the live one (between requests / passes)
            if (q == cur_seq) return true;
            if (!dense.bind_seq(q)) { err = "no conversation slot " + std::to_string(q); return false; }
#if defined(DS41_TP)
            if (tp && !tp->bind_seq(q)) { err = tp->last_error(); return false; }
#endif
            seq_swap(seq_stash[(size_t) cur_seq]);   // the live conversation into its own entry ...
            seq_swap(seq_stash[(size_t) q]);         // ... and q's out of its entry
            if (!kv_pos_seq.empty()) {
                kv_pos_seq[(size_t) cur_seq] = kv_pos;
                kv_pos = kv_pos_seq[(size_t) q];
            }
            cur_seq = q;
            return true;
        };
        auto seq_hist = [&](int q) -> const std::vector<int> & { return q == cur_seq ? hist : seq_stash[(size_t) q].hist; };
        auto seq_img = [&](int q) -> const std::vector<ImgId> & { return q == cur_seq ? conv_img : seq_stash[(size_t) q].img; };
        if (n_seq > 1)
            std::fprintf(stderr, "serve: %d conversation slots resident on the cards (STRATA_SERVE_SEQS)\\n", n_seq);
"""),
# a request: the conversation slot it continues best, else the one used longest ago
("""            size_t lcp = 0;
            while (lcp < hist.size() && lcp < ids.size() && hist[lcp] == ids[lcp]) ++lcp;
            lcp = img_cap(lcp, conv_img, req_img);   // (larry, vision)
""",
"""            if (n_seq > 1) {   // larry (batch slots): the conversation slot this request continues best
                int best = -1;
                size_t bl = 0;
                for (int q = 0; q < n_seq; ++q) {
                    const std::vector<int> & h = seq_hist(q);
                    size_t l = 0;
                    while (l < h.size() && l < ids.size() && h[l] == ids[l]) ++l;
                    l = img_cap(l, seq_img(q), req_img);
                    if (l > bl) { bl = l; best = q; }
                }
                if (best < 0 || bl < (size_t) park_min) {   // continues none: an empty slot, else the one used longest ago
                    best = -1;
                    for (int q = 0; q < n_seq && best < 0; ++q) if (seq_hist(q).empty()) best = q;
                    if (best < 0) {
                        best = 0;
                        for (int q = 1; q < n_seq; ++q) if (seq_used[(size_t) q] < seq_used[(size_t) best]) best = q;
                    }
                }
                if (!seq_switch(best)) { std::printf("ERR %s\\n", err.c_str()); std::fflush(stdout); continue; }
                seq_used[(size_t) cur_seq] = ++seq_tick;
            }
            size_t lcp = 0;
            while (lcp < hist.size() && lcp < ids.size() && hist[lcp] == ids[lcp]) ++lcp;
            lcp = img_cap(lcp, conv_img, req_img);   // (larry, vision)
"""),
("""            std::fprintf(stderr, "serve: reuse lcp=%zu park=%zu/%zu lp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d\\n", lcp,
                         dense.parked(0) ? pk[0].hist.size() : (size_t) 0, dense.parked(1) ? pk[1].hist.size() : (size_t) 0, lp_show,
                         hist.size(), snap_len, ids.size(), reused);
""",
"""            std::fprintf(stderr, "serve: reuse lcp=%zu park=%zu/%zu lp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d seq=%d\\n", lcp,
                         dense.parked(0) ? pk[0].hist.size() : (size_t) 0, dense.parked(1) ? pk[1].hist.size() : (size_t) 0, lp_show,
                         hist.size(), snap_len, ids.size(), reused, cur_seq);
"""),
]
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:90]))
    s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
