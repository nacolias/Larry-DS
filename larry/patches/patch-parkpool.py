#!/usr/bin/env python3
# patch-parkpool.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrency stage 0): the parked conversations become a pool and
# two resume bugs are fixed (both as upstream Xolotl's dffa55a3 found them):
#  1. A partial unpark (STRATA_SERVE_PARK_PARTIAL) resumed at any lp >= 3/4 of the parked conversation with the SIDE
#     request's live window ring and ratio-2 compressor state.  The ring at the park's snapshot P holds positions
#     [P - RAW, P) (RAW = 128 + 16) and the compressor state P - 2, P - 1, so a resume at s < P is exact only for an
#     even s with P - s <= 16, and only after restore().  Now exactly that (otherwise the prompt is read fresh).
#  2. The Engram history is kept by position and is not parked: a side request longer than the parked conversation
#     left its own tokens before the resume point, which the first new positions' hashes read.  Every resume now writes
#     the 8 tokens before `reused` back (the vision path did it for picture requests only).
#  3. STRATA_SERVE_PARK_SLOTS up to 64 (was capped at 2) and STRATA_SERVE_PARK_GIB = the host memory the parked
#     conversations may hold together (0 = no budget, the old way): the conversations parked longest ago go first.
#     A park copies a ratio-2 source's positions / 2 + 2 rows (was positions + 1): 5,120 -> ~3,200 bytes a position.
#  4. Taking a parked conversation back discarded the LIVE one (only an unrelated request parked it): rotating three
#     conversations re-read one every time.  Now the live conversation goes to the pool first (park_live), and a parked
#     earlier state of the same conversation is dropped as superseded.
#  5. "conversation cache:" lines + INFO conversation_cache_mib / _slots, the server's Monitor card format.
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
E = os.path.join(X, "tools/v41/engine")


def patch(path, pairs, mark):
    s = open(path).read()
    if mark in s:
        print("already patched", path)
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit("%s: anchor x%d: %s" % (path, s.count(old), old[:90]))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


patch(os.path.join(E, "ds41_dense.hpp"), [
("""    /// larry: `slot` (0 / 1): two conversations can be parked (STRATA_SERVE_PARK_SLOTS)
    bool park(int64_t positions, int slot = 0);
""",
"""    /// larry: `slot` 0 .. 63: a pool of parked conversations (STRATA_SERVE_PARK_SLOTS; patch-parkpool.py)
    bool park(int64_t positions, int slot = 0);
"""),
("""    bool parked(int slot = 0) const;
""",
"""    bool parked(int slot = 0) const;
    /// larry (patch-parkpool.py): the host bytes parked slot `slot` holds / a park of `positions` would take
    size_t park_bytes(int slot) const;
    size_t park_bytes_for(int64_t positions) const;
"""),
], "park_bytes_for")

patch(os.path.join(E, "ds41_dense.cpp"), [
("""    // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites; larry: two slots
    struct ParkSlot { std::vector<ParkedRows> rows; int pos = -1; int64_t n = 0; };
    ParkSlot pk[2];
""",
"""    // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites; larry: a pool of
    // slots, grown on demand (patch-parkpool.py)
    struct ParkSlot { std::vector<ParkedRows> rows; int pos = -1; int64_t n = 0; size_t bytes = 0; };
    std::vector<ParkSlot> pk;
"""),
("""bool Ds41Dense::park(int64_t positions, int slot) {
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
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) put(t, 0, ggml_nbytes(t));
        for (ggml_tensor * t : { L.comp, L.idxk }) {   // row r holds a position >= r (ratio 1 or 2): rows [0, n] cover n positions
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            put(t, 0, (size_t) n * t->nb[1]);
            put(t, (size_t) (rows - 1) * t->nb[1], t->nb[1]);   // the spare row (open compressor groups write it)
        }
    }
    S.pos = im.snap_pos;
    S.n = positions;
    return true;
}
""",
"""// larry (patch-parkpool.py): the rows of a ratio-r source that hold positions [0, positions): row k holds positions
// [k r, k r + r), so positions / r + 1 rows; + 1 more as a margin (kv_bytes_for's rule).  Was positions + 1 for both.
static int64_t park_rows(const ggml_tensor * t, int64_t ratio, int64_t positions) {
    const int64_t r = ratio == 2 ? 2 : 1;
    return std::min<int64_t>(t->ne[1] - 1, positions / r + 2);
}

bool Ds41Dense::park(int64_t positions, int slot) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1 || slot < 0 || slot >= 64) return false;
    if ((size_t) slot >= im.pk.size()) im.pk.resize((size_t) slot + 1);
    ggml_backend_synchronize(im.backend);
    Impl::ParkSlot & S = im.pk[slot];
    S.rows.clear();
    S.bytes = 0;
    auto put = [&](ggml_tensor * t, size_t off, size_t n) {
        Impl::ParkedRows r{ t, off, std::vector<uint8_t>(n) };
        ggml_backend_tensor_get(t, r.b.data(), off, n);
        S.rows.push_back(std::move(r));
        S.bytes += n;
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) put(t, 0, ggml_nbytes(t));
        for (ggml_tensor * t : { L.comp, L.idxk }) {
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = park_rows(t, L.ratio, positions);
            put(t, 0, (size_t) n * t->nb[1]);
            put(t, (size_t) (rows - 1) * t->nb[1], t->nb[1]);   // the spare row (open compressor groups write it)
        }
    }
    S.pos = im.snap_pos;
    S.n = positions;
    return true;
}

size_t Ds41Dense::park_bytes(int slot) const {
    return slot >= 0 && (size_t) slot < p_->pk.size() && p_->pk[slot].pos >= 0 ? p_->pk[slot].bytes : 0;
}

size_t Ds41Dense::park_bytes_for(int64_t positions) const {
    const Impl & im = *p_;
    size_t b = 0;
    for (const auto & L : im.ly) {
        for (const ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) b += ggml_nbytes(t);
        for (const ggml_tensor * t : { L.comp, L.idxk })
            if (t) b += (size_t) (park_rows(t, L.ratio, positions) + 1) * t->nb[1];
    }
    return b;
}
"""),
("""    if (slot < 0 || slot > 1 || im.pk[slot].pos < 0) return false;
    Impl::ParkSlot & S = im.pk[slot];
""",
"""    if (slot < 0 || (size_t) slot >= im.pk.size() || im.pk[slot].pos < 0) return false;
    Impl::ParkSlot & S = im.pk[slot];
"""),
("""void Ds41Dense::drop_park(int slot) {
    if (slot < 0 || slot > 1) return;
    p_->pk[slot].rows.clear();
    p_->pk[slot].rows.shrink_to_fit();
    p_->pk[slot].pos = -1;
}

bool Ds41Dense::parked(int slot) const { return slot >= 0 && slot <= 1 && p_->pk[slot].pos >= 0; }
""",
"""void Ds41Dense::drop_park(int slot) {
    if (slot < 0 || (size_t) slot >= p_->pk.size()) return;
    p_->pk[slot].rows.clear();
    p_->pk[slot].rows.shrink_to_fit();
    p_->pk[slot].pos = -1;
    p_->pk[slot].bytes = 0;
}

bool Ds41Dense::parked(int slot) const { return slot >= 0 && (size_t) slot < p_->pk.size() && p_->pk[slot].pos >= 0; }
"""),
], "park_bytes_for")

patch(os.path.join(E, "ds41_generate.cpp"), [
("""        static const int park_slots = std::max(1, std::min(2, std::getenv("STRATA_SERVE_PARK_SLOTS") ? std::atoi(std::getenv("STRATA_SERVE_PARK_SLOTS")) : 2));
""",
"""        // larry, patch-parkpool.py: up to 64 slots (a pool), and STRATA_SERVE_PARK_GIB = the host memory the parked
        // conversations may hold together (0 = no budget): a park that would pass it drops the ones parked longest ago
        static const int park_slots = std::max(1, std::min(64, std::getenv("STRATA_SERVE_PARK_SLOTS") ? std::atoi(std::getenv("STRATA_SERVE_PARK_SLOTS")) : 2));
        static const double park_gib = std::getenv("STRATA_SERVE_PARK_GIB") ? std::max(0.0, std::atof(std::getenv("STRATA_SERVE_PARK_GIB"))) : 0.0;
"""),
("""        ParkMeta pk[2];
        long park_tick = 0;
""",
"""        std::vector<ParkMeta> pk((size_t) park_slots);
        long park_tick = 0, park_evictions = 0;
        auto park_pool = [&](size_t & bytes) {   // parked conversations now, and the host memory they hold
            int n = 0;
            bytes = 0;
            for (int s = 0; s < park_slots; ++s)
                if (dense.parked(s)) { ++n; bytes += dense.park_bytes(s); }
            return n;
        };
        // larry, patch-parkpool.py: the live conversation into the pool - before an unrelated request overwrites it,
        // and now also before a parked conversation is taken back (that discarded the live one: rotating three
        // conversations re-read one every time).  A free slot, else the one parked longest ago - never `avoid` (the
        // slot about to be unparked; then only a whole conversation: a partial one relies on rows the unpark rewrites).
        // `rows`: the positions the next request may write.  The slot, or -1.
        auto park_live = [&](int64_t rows, int avoid) -> int {
            if (!park_on || snap_len < park_min || hist.size() < (size_t) snap_len) return -1;
            // the conversation's own rows only - it re-writes everything past its snapshot when it goes on (an agent's
            // side call asks for 32000 tokens: a 12.6K conversation was saved as 47K positions); the whole conversation
            // when it fits park_max (then no later write spoils it)
            const bool whole = park_slots > 1 && (int64_t) snap_len + 1 <= park_max;
            const int64_t save = whole ? (int64_t) snap_len + 1 : std::min<int64_t>(rows, (int64_t) snap_len + 1);
            if (save > park_max || (avoid >= 0 && !whole)) return -1;
            size_t pb = 0;
            int pn = 0;
            for (int s = 0; s < park_slots; ++s)   // an earlier state of this conversation is superseded
                if (s != avoid && dense.parked(s) && pk[s].hist.size() <= (size_t) snap_len &&
                    std::equal(pk[s].hist.begin(), pk[s].hist.end(), hist.begin())) {
                    dense.drop_park(s);
                    pn = park_pool(pb);
                    std::fprintf(stderr, "serve: conversation cache: dropped 1 superseded (slot %d, %zu tokens); parked=%d\\n", s,
                                 pk[s].hist.size(), pn);
                }
            const size_t need = dense.park_bytes_for(save);
            const size_t budget = park_gib > 0 ? (size_t) (park_gib * 1073741824.0) : SIZE_MAX;
            if (need > budget) {
                pn = park_pool(pb);
                std::fprintf(stderr, "serve: conversation cache: skipped %d tokens (%.2f GiB > the %.2f GiB budget); parked=%d bytes=%zu\\n",
                             snap_len, need / 1073741824.0, park_gib, pn, pb);
                return -1;
            }
            const double tpk = now_ms();
            int ps = -1;
            for (int s = 0; s < park_slots && ps < 0; ++s) if (s != avoid && !dense.parked(s)) ps = s;
            if (ps < 0)
                for (int s = 0; s < park_slots; ++s) if (s != avoid && (ps < 0 || pk[s].stamp < pk[ps].stamp)) ps = s;
            if (ps < 0) return -1;   // one slot, and it is about to be unparked
            if (dense.parked(ps)) {
                dense.drop_park(ps);
                ++park_evictions;
                std::fprintf(stderr, "serve: park %d replaced (its %zu-token conversation was parked longest ago)\\n", ps, pk[ps].hist.size());
            }
            for (;;) {   // the budget (the slot about to be unparked is leaving): the oldest others go first
                park_pool(pb);
                if (avoid >= 0) pb -= std::min(pb, dense.park_bytes(avoid));
                if (pb + need <= budget) break;
                int old = -1;
                for (int s = 0; s < park_slots; ++s)
                    if (s != ps && s != avoid && dense.parked(s) && (old < 0 || pk[s].stamp < pk[old].stamp)) old = s;
                if (old < 0) break;
                dense.drop_park(old);
                ++park_evictions;
                std::fprintf(stderr, "serve: park %d dropped (its %zu-token conversation, for the %.2f GiB budget)\\n", old,
                             pk[old].hist.size(), park_gib);
            }
            if (!dense.park(save, ps)) return -1;
            pk[ps].hist.assign(hist.begin(), hist.begin() + snap_len);
            pk[ps].img = img_trim(conv_img, snap_len);   // (larry, vision)
            pk[ps].logits = snap_logits;
            pk[ps].saved = save;
            pk[ps].valid = save == (int64_t) snap_len + 1 ? INT64_MAX : rows;
            pk[ps].stamp = ++park_tick;
            std::fprintf(stderr, "serve: parked the %d-token conversation in slot %d (%lld positions saved, %s; %.2f s)\\n",
                         snap_len, ps, (long long) save, save == (int64_t) snap_len + 1 ? "whole" : "partial",
                         (now_ms() - tpk) / 1000.0);
            pn = park_pool(pb);
            std::fprintf(stderr, "serve: conversation cache: parked %d tokens (slot %d, %.2f GiB); parked=%d bytes=%zu evictions=%ld\\n",
                         snap_len, ps, dense.park_bytes(ps) / 1073741824.0, pn, pb, park_evictions);
            return ps;
        };
"""),
("""        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld\\n", ctxv, (long long) tier.resident(),
                    (long long) tier.arena_experts());
""",
"""        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld conversation_cache_slots=%d conversation_cache_mib=%d\\n",
                    ctxv, (long long) tier.resident(), (long long) tier.arena_experts(), park_on ? park_slots : 0,
                    park_on ? (int) (park_gib * 1024) : 0);   // (0 MiB: no budget - the Monitor card then says "off")
"""),
("""            // Positional caches make a partial resume safe: unpark() restores positions [0, park_size) and the
            // forward pass then re-feeds from lp, overwriting the stale tail (that is the documented fix: "resume
            // at the longest common prefix even when it is below the snapshot").
""",
"""            // larry, patch-parkpool.py (upstream dffa55a3): NOT every cache is positional.  The window ring at the
            // park's snapshot P holds positions [P - RAW, P) (RAW = 128 + 16) and the ratio-2 compressor state P - 2,
            // P - 1, so a resume at s < P reads the window [s - 127, s) from the ring and, at an odd s, the state of
            // s - 1: exact only for an even s with P - s <= 16, and only after restore() (the live rings are the side
            // request's).  The old rule (any lp >= 3/4 of P) resumed with the side request's rings: upstream's CPU
            // check measured KL 2.1 against a fresh read 20 tokens back, top-1 wrong.
"""),
("""                if (lp_show > lcp && lp_show == n_ub) un = 1;
                else if (lp_show > lcp && park_partial && lp_show < n_ub && lp_show >= lp_min) un = 2;
""",
"""                const size_t ps2 = lp_show & ~(size_t) 1;   // the even resume point at or under lp (see above)
                if (lp_show > lcp && lp_show == n_ub) un = 1;
                else if (park_partial && lp_show < n_ub && lp_show >= lp_min && ps2 > lcp && n_ub - ps2 <= 16) un = 2;
"""),
("""                const size_t lp = lp_show;
                if (un == 2 && dense.unpark(ub)) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation from slot %d at lcp %zu (partial: re-feeds from there; %.2f s)\\n",
                                 park_hist.size(), ub, lp, (now_ms() - tup) / 1000.0);
                    hist = park_hist;
                    conv_img = pk[ub].img;   // (larry, vision)
                    snap_len = -1;        // the tail under the park's end is re-fed, so no snapshot survives this
                    reuse_override = (int) lp;
                    lcp = lp;
""",
"""                const size_t lp = lp_show;
                const size_t ps2 = lp & ~(size_t) 1;   // patch-parkpool.py: resume at an even position, after restore()
                if (un == 2 && dense.unpark(ub) && dense.restore()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation from slot %d at %zu (lcp %zu; partial: re-feeds from there; %.2f s)\\n",
                                 park_hist.size(), ub, ps2, lp, (now_ms() - tup) / 1000.0);
                    hist = park_hist;
                    conv_img = pk[ub].img;   // (larry, vision)
                    snap_len = -1;        // the tail under the park's end is re-fed, so no snapshot survives this
                    reuse_override = (int) ps2;
                    lcp = ps2;
"""),
("""            } else {
                // an unrelated request: park the conversation it would overwrite (see STRATA_SERVE_PARK)
                const int64_t rows = (int64_t) ids.size() + (max_new > 0 ? max_new : park_max) + 64;
                // larry: the conversation's own rows only - it re-writes everything past its snapshot when it goes on
                // (an agent's side call asks for 32000 tokens: a 12.6K conversation was saved as 47K positions)
                // (larry, two slots: the whole conversation when it fits park_max - then no later write spoils it)
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
                        pk[ps].img = img_trim(conv_img, snap_len);   // (larry, vision)
                        pk[ps].logits = snap_logits;
                        pk[ps].saved = save;
                        pk[ps].valid = save == (int64_t) snap_len + 1 ? INT64_MAX : rows;
                        pk[ps].stamp = ++park_tick;
                        std::fprintf(stderr, "serve: parked the %d-token conversation in slot %d (%lld positions saved, %s; %.2f s)\\n",
                                     snap_len, ps, (long long) save, save == (int64_t) snap_len + 1 ? "whole" : "partial",
                                     (now_ms() - tpk) / 1000.0);
                    }
                }
                hist.clear();
""",
"""            } else {
                // an unrelated request: park the conversation it would overwrite (see STRATA_SERVE_PARK; park_live)
                park_live((int64_t) ids.size() + (max_new > 0 ? max_new : park_max) + 64, -1);
                hist.clear();
"""),
("""            if (un) {   // this request continues a parked conversation (better than the live one)
                const double tup = now_ms();
""",
"""            if (un) {   // this request continues a parked conversation (better than the live one)
                // patch-parkpool.py: the live conversation goes to the pool first (it was discarded)
                park_live((int64_t) ids.size() + (max_new > 0 ? max_new : park_max) + 64, ub);
                const double tup = now_ms();
"""),
("""            if (reused > 0) { std::printf("RESUME %d\\n", reused); std::fflush(stdout); }
""",
"""            if (unparked) {   // patch-parkpool.py: the server's Monitor card
                size_t pb = 0;
                const int pn = park_pool(pb);
                std::fprintf(stderr, "serve: conversation cache: restored %d tokens (slot %d); parked=%d bytes=%zu\\n", reused, ub, pn, pb);
            }
            if (reused > 0) { std::printf("RESUME %d\\n", reused); std::fflush(stdout); }
"""),
("""            if (!img_spans.empty()) {   // larry (vision): the pictures' rows for the feed; the Engram's look-back history again
                dense.set_images(img_spans);
                const int kb = std::min(reused, 8);
                if (kb > 0) dense.engram_tokens(reused - kb, ids.data() + reused - kb, kb);
            }
""",
"""            if (!img_spans.empty()) dense.set_images(img_spans);   // larry (vision): the pictures' rows for the feed
            if (reused > 0) {
                // the Engram's look-back history again - on EVERY resume (patch-parkpool.py; it was the vision path's
                // only): the history is kept by position and not parked, so a side request since wrote its own tokens
                // before `reused`, and the first new positions' hashes read them.  The reused prefix is this prompt's.
                const int kb = std::min(reused, 8);
                dense.engram_tokens(reused - kb, ids.data() + reused - kb, kb);
            }
"""),
], "STRATA_SERVE_PARK_GIB")
