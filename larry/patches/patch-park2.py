#!/usr/bin/env python3
# patch-park2.py [ENGINE_DIR] - larry: two parked conversations (STRATA_SERVE_PARK_SLOTS, default 2; 1 = the old way).
# One slot meant a parked conversation nobody continued (an agent session ended right after a side request; a test)
# kept the slot until a request wrote past its rows, and meanwhile the next side request could not park the live
# conversation: its next turn read everything again (2026-10-24 replays: an 18K and a 34K main conversation, +7 / +16 s).
# Now a side request parks the live conversation into a free slot, else the least recently parked one, and a request
# unparks the slot it continues best.  A park saves the whole conversation when it fits STRATA_SERVE_PARK_MAX (then no
# later write can spoil it - it used to save only the side request's reach); a bigger one is saved partially as before
# and dropped once anything writes past its saved rows (a request, or the unpark of a slot that saved more).
# Idempotent per file.
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine"


def patch(path, edits, marker):
    s = open(path).read()
    if marker in s:
        print("already patched", path)
        return
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit("%s: anchor x%d: %s" % (path, s.count(old), old[:100]))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


patch(os.path.join(D, "ds41_dense.hpp"), [("""    bool park(int64_t positions);
""", """    /// larry: `slot` (0 / 1): two conversations can be parked (STRATA_SERVE_PARK_SLOTS)
    bool park(int64_t positions, int slot = 0);
"""), ("""    bool unpark();
    void drop_park();
    bool parked() const;
""", """    bool unpark(int slot = 0);
    void drop_park(int slot = 0);
    bool parked(int slot = 0) const;
""")], "int slot = 0")

patch(os.path.join(D, "ds41_dense.cpp"), [("""    int64_t parked_n = 0;
""", ""), ("""    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; };
    std::vector<ParkedRows> parked;     // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites
    int parked_pos = -1;
""", """    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; };
    // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites; larry: two slots
    struct ParkSlot { std::vector<ParkedRows> rows; int pos = -1; int64_t n = 0; };
    ParkSlot pk[2];
"""), ("""bool Ds41Dense::park(int64_t positions) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1) return false;
    ggml_backend_synchronize(im.backend);
    im.parked.clear();
    auto put = [&](ggml_tensor * t, size_t off, size_t n) {
        Impl::ParkedRows r{ t, off, std::vector<uint8_t>(n) };
        ggml_backend_tensor_get(t, r.b.data(), off, n);
        im.parked.push_back(std::move(r));
    };
""", """bool Ds41Dense::park(int64_t positions, int slot) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1 || slot < 0 || slot > 1) return false;
    ggml_backend_synchronize(im.backend);
    Impl::ParkSlot & S = im.pk[slot];
    S.rows.clear();
    auto put = [&](ggml_tensor * t, size_t off, size_t n) {
        Impl::ParkedRows r{ t, off, std::vector<uint8_t>(n) };
        ggml_backend_tensor_get(t, r.b.data(), off, n);
        S.rows.push_back(std::move(r));
    };
"""), ("""    im.parked_pos = im.snap_pos;
    im.parked_n = positions;
    return true;
}

bool Ds41Dense::unpark() {
    Impl & im = *p_;
    if (im.parked_pos < 0) return false;
    ggml_backend_synchronize(im.backend);
    for (auto & r : im.parked) ggml_backend_tensor_set(r.t, r.b.data(), r.off, r.b.size());
    ggml_backend_synchronize(im.backend);
    im.wrote(0, 2 * (im.parked_n + 1));   // the parked rows (rows [0, positions + 1] of either ratio)
    im.snap_pos = im.parked_pos;
    drop_park();
    return true;
}

void Ds41Dense::drop_park() {
    p_->parked.clear();
    p_->parked.shrink_to_fit();
    p_->parked_pos = -1;
}

bool Ds41Dense::parked() const { return p_->parked_pos >= 0; }
""", """    S.pos = im.snap_pos;
    S.n = positions;
    return true;
}

bool Ds41Dense::unpark(int slot) {
    Impl & im = *p_;
    if (slot < 0 || slot > 1 || im.pk[slot].pos < 0) return false;
    Impl::ParkSlot & S = im.pk[slot];
    ggml_backend_synchronize(im.backend);
    for (auto & r : S.rows) ggml_backend_tensor_set(r.t, r.b.data(), r.off, r.b.size());
    ggml_backend_synchronize(im.backend);
    im.wrote(0, 2 * (S.n + 1));   // the parked rows (rows [0, positions + 1] of either ratio)
    im.snap_pos = S.pos;
    drop_park(slot);
    return true;
}

void Ds41Dense::drop_park(int slot) {
    if (slot < 0 || slot > 1) return;
    p_->pk[slot].rows.clear();
    p_->pk[slot].rows.shrink_to_fit();
    p_->pk[slot].pos = -1;
}

bool Ds41Dense::parked(int slot) const { return slot >= 0 && slot <= 1 && p_->pk[slot].pos >= 0; }
""")], "ParkSlot pk[2]")

patch(os.path.join(D, "ds41_generate.cpp"), [("""        std::vector<int> park_hist;       // the parked conversation's tokens (its snapshot length)
        std::vector<float> park_logits;
        int64_t park_rows = 0;            // positions saved: a request writing past this invalidates the park
""", """        // larry: STRATA_SERVE_PARK_SLOTS (default 2, 1 = the old way) conversations kept beside the live one - with one, a
        // parked conversation nobody continued kept the slot and the next side request could not park the live one (its
        // next turn read everything again).  A park saves the whole conversation when it fits park_max (no later write
        // can spoil it); a bigger one only the side request's reach, and is dropped once anything writes past that.
        static const int park_slots = std::max(1, std::min(2, std::getenv("STRATA_SERVE_PARK_SLOTS") ? std::atoi(std::getenv("STRATA_SERVE_PARK_SLOTS")) : 2));
        struct ParkMeta {
            std::vector<int> hist;        // the parked conversation's tokens (its snapshot length)
            std::vector<float> logits;
            int64_t saved = 0;            // positions saved
            int64_t valid = 0;            // a write past this spoils it (INT64_MAX: saved whole)
            long stamp = 0;               // when it was parked (the oldest goes first)
        };
        ParkMeta pk[2];
        long park_tick = 0;
"""), ("""                const int64_t save = std::min<int64_t>(rows, (int64_t) snap_len + 1);
                const double tpk = now_ms();
                if (park_on && !dense.parked() && snap_len >= park_min && hist.size() >= (size_t) snap_len && save <= park_max
                    && dense.park(save)) {
                    park_hist.assign(hist.begin(), hist.begin() + snap_len);
                    park_logits = snap_logits;
                    park_rows = rows;
                    std::fprintf(stderr, "serve: parked the %d-token conversation (%lld positions saved, valid to %lld; %.2f s)\\n",
                                 snap_len, (long long) save, (long long) rows, (now_ms() - tpk) / 1000.0);
                }
""", """                // (larry, two slots: the whole conversation when it fits park_max - then no later write spoils it)
                const bool whole = park_slots > 1 && (int64_t) snap_len + 1 <= park_max;
                const int64_t save = whole ? (int64_t) snap_len + 1 : std::min<int64_t>(rows, (int64_t) snap_len + 1);
                const double tpk = now_ms();
                int ps = -1;   // a free slot, else the one parked longest ago
                for (int s = 0; s < park_slots && ps < 0; ++s) if (!dense.parked(s)) ps = s;
                if (ps < 0) {
                    ps = 0;
                    for (int s = 1; s < park_slots; ++s) if (pk[s].stamp < pk[ps].stamp) ps = s;
                }
                if (park_on && snap_len >= park_min && hist.size() >= (size_t) snap_len && save <= park_max) {
                    if (dense.parked(ps)) {
                        dense.drop_park(ps);
                        std::fprintf(stderr, "serve: park %d replaced (its %zu-token conversation was parked longest ago)\\n", ps, pk[ps].hist.size());
                    }
                    if (dense.park(save, ps)) {
                        pk[ps].hist.assign(hist.begin(), hist.begin() + snap_len);
                        pk[ps].logits = snap_logits;
                        pk[ps].saved = save;
                        pk[ps].valid = save == (int64_t) snap_len + 1 ? INT64_MAX : rows;
                        pk[ps].stamp = ++park_tick;
                        std::fprintf(stderr, "serve: parked the %d-token conversation in slot %d (%lld positions saved, %s; %.2f s)\\n",
                                     snap_len, ps, (long long) save, save == (int64_t) snap_len + 1 ? "whole" : "partial",
                                     (now_ms() - tpk) / 1000.0);
                    }
                }
"""), ("""            std::fprintf(stderr, "serve: reuse lcp=%zu park=%zu lp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d\\n", lcp,
                         park_hist.size(), lp_show, hist.size(), snap_len, ids.size(), reused);
""", """            std::fprintf(stderr, "serve: reuse lcp=%zu park=%zu/%zu lp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d\\n", lcp,
                         dense.parked(0) ? pk[0].hist.size() : (size_t) 0, dense.parked(1) ? pk[1].hist.size() : (size_t) 0, lp_show,
                         hist.size(), snap_len, ids.size(), reused);
""")], "STRATA_SERVE_PARK_SLOTS")


# the unpark section: rewritten whole (the slot it continues best, the spoiled partial parks, then the unpark)
G = os.path.join(D, "ds41_generate.cpp")
g = open(G).read()
if "int un = 0;   // 1: unpark ub whole" not in g:
    a = g.index("            size_t lp_show = 0;        // how far this request actually continues the PARKED conversation (logged)\n")
    b = g.index("            int reused = 0;\n", a)
    blk = """            size_t lp_show = 0;        // how far this request actually continues the PARKED conversation (logged)
            int ub = -1;               // larry: the parked slot it continues best (STRATA_SERVE_PARK_SLOTS)
            for (int s = 0; s < park_slots; ++s) {
                if (!dense.parked(s)) continue;
                size_t l = 0;
                while (l < pk[s].hist.size() && l < ids.size() && pk[s].hist[l] == ids[l]) ++l;
                if (ub < 0 || l > lp_show) { ub = s; lp_show = l; }
            }
            // STRATA_SERVE_PARK_PARTIAL (default on): a request that continues MOST of the parked conversation is
            // worth restoring, not only one that continues ALL of it.  A chat template that re-renders its tail -
            // or an EMPTY assistant turn, as the s13 harness produces - leaves lp just under park_hist.size(), and
            // requiring an exact prefix threw away a 21,349-token cache over a 27-token difference: run8/run15 both
            // logged "reuse lcp=27 hist=0 snap=-1 prompt=23531 -> reused=0" and re-read the whole prompt.
            // Positional caches make a partial resume safe: unpark() restores positions [0, park_size) and the
            // forward pass then re-feeds from lp, overwriting the stale tail (that is the documented fix: "resume
            // at the longest common prefix even when it is below the snapshot").
            static const bool park_partial = !(std::getenv("STRATA_SERVE_PARK_PARTIAL") && std::getenv("STRATA_SERVE_PARK_PARTIAL")[0] == '0');
            int un = 0;   // 1: unpark ub whole, 2: partially (the request re-feeds from its lcp)
            if (ub >= 0) {
                const size_t n_ub = pk[ub].hist.size();
                const size_t lp_min = (size_t) std::max<int64_t>(park_min, (int64_t) n_ub * 3 / 4);
                if (lp_show > lcp && lp_show == n_ub) un = 1;
                else if (lp_show > lcp && park_partial && lp_show < n_ub && lp_show >= lp_min) un = 2;
            }
            {   // a partially saved park is spoiled by a write past its saved rows: this request's, or the unpark of a slot that saved more
                const int64_t reach = (int64_t) ids.size() + std::max(0, max_new);
                for (int s = 0; s < park_slots; ++s)
                    if (dense.parked(s) && !(un && s == ub) && (reach > pk[s].valid || (un && pk[ub].saved > pk[s].valid))) {
                        dense.drop_park(s);
                        std::fprintf(stderr, "serve: park %d dropped (a write past its %lld saved positions)\\n", s, (long long) pk[s].valid);
                    }
            }
            if (un) {   // this request continues a parked conversation (better than the live one)
                const double tup = now_ms();
                const std::vector<int> & park_hist = pk[ub].hist;
                const size_t lp = lp_show;
                if (un == 2 && dense.unpark(ub)) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation from slot %d at lcp %zu (partial: re-feeds from there; %.2f s)\\n",
                                 park_hist.size(), ub, lp, (now_ms() - tup) / 1000.0);
                    hist = park_hist;
                    snap_len = -1;        // the tail under the park's end is re-fed, so no snapshot survives this
                    reuse_override = (int) lp;
                    lcp = lp;
                    unparked = true;
#if defined(DS41_HAVE_DSPARK)
                    dsp_next = -1;        // DSpark's window holds the side request's states
#endif
                } else if (un == 1 && dense.unpark(ub)) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation from slot %d (%.2f s)\\n", park_hist.size(), ub,
                                 (now_ms() - tup) / 1000.0);
                    hist = park_hist;
                    snap_len = (int) park_hist.size();
                    snap_logits = pk[ub].logits;
                    snap_seed_n = 0;
                    lcp = lp;
                    unparked = true;   // the live rings are the side request's: resume through restore()
#if defined(DS41_HAVE_DSPARK)
                    dsp_next = -1;     // DSpark's window holds the side request's states
#endif
                }
            }
"""
    g = g[:a] + blk + g[b:]
    open(G, "w").write(g)
    print("rewrote the unpark section", G)

