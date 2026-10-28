#!/usr/bin/env python3
# patch-roomahead.py [XOLOTL_DIR] - larry, 2026-10-28 (optimization #5, long contexts: the wait before the first reply).
# After a long prompt the first pass grows the bound conversation's KV replicas on the expert cards to the prompt's cap
# bucket, and each card first gives VRAM back: its evictable last slabs go back to the arena (STRATA_XE_TP_EVICT), the
# experts' host copies read from the model file then.  r1m-stage.sh's 967K prompt: 1152 / 4972 / 994 ms on cards 1-3,
# one card after another - 7.1 of the 7.9 s its sync took (131K: 3 x ~0.63 s).  Now:
#  - the tier reads those copies AHEAD: near a long prompt's end (STRATA_TP_AHEAD_S, default 30: once the batches left
#    look shorter than that) the engine tells the TP core where the prompt ends, the core tells the tier each card's
#    coming growth (Ds41Tp::room_ahead -> tier.tp_room_ahead), and the tier works out the slabs evict_slabs would free
#    and reads their experts on a helper thread (STRATA_XE_TP_AHEAD, default 1; _READERS 2; _CPUS: where they run,
#    default the process's CPUs - pinned to the helper CPUs, where the prompt's staging runs, 8 cost a 131K prompt 5 s;
#    it needs STRATA_XE_TP_AHEAD_MEM_GIB 8 of MemAvailable left over); re-checked every batch; evict_slabs adopts a
#    slab's copy when the slab's experts are the ones read (else it reads them as before); what the growth did not
#    take is dropped when the conversation's replicas reach the target (or at the next request);
#  - the cards evict side by side (tier.tp_make_room_multi, one thread a card; STRATA_XE_TP_EVICT_PAR=0: in turn), the
#    shared bookkeeping under a lock, the dense card's replicas dropped once before;
#  - evict_slabs frees as many slabs as the shortfall takes, then waits for the driver to show them once a round
#    (STRATA_XE_TP_EVICT_ROUND=0: once a slab, as before); its reads go straight into the copies' buffer (no pinned
#    bounce); its line gives the slabs, how many were read ahead and where the time went.
#  - sync_state copies the written rows only below where its own growth started (ensure_rows copied the rest from card
#    0 already: the 967K prompt's rows went twice, 0.34 s of its 0.77 s sync).
# The TP core's "KV replicas now" line gives the room's and the maps + rows' time.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
TH = os.path.join(X, "tools/v41/engine/tp/tp_core.hpp")
TC = os.path.join(X, "tools/v41/engine/tp/tp_core.cpp")
XH = os.path.join(X, "tools/v41/engine/xe/ds41_xe_tier.hpp")
XC = os.path.join(X, "tools/v41/engine/xe/ds41_xe_tier.cpp")
G = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
th, tc, xh, xc, g = (open(p).read() for p in (TH, TC, XH, XC, G))
if "tp_room_ahead" in xh:
    print("already patched")
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


def rep_span(s, start, end, new, what):
    """the text from `start` up to (not including) `end` -> new"""
    if s.count(start) != 1:
        sys.exit("%s: start anchor x%d" % (what, s.count(start)))
    i = s.index(start)
    j = s.find(end, i)
    if j < 0:
        sys.exit("%s: no end anchor" % what)
    return s[:i] + new + s[j:]


# ---------------------------------------------------------------- the TP core
th = rep(th, """    void set_vram_hook(std::function<bool(int gpu, size_t bytes)> make_room);
""", """    void set_vram_hook(std::function<bool(int gpu, size_t bytes)> make_room);
    /// larry (patch-roomahead.py): the tier's room on every expert card in one call (it evicts on all of them at once;
    /// ensure_rows then takes it instead of set_vram_hook's), and its read-ahead of a coming growth's evictions (an
    /// empty list: none coming - it drops what it read ahead)
    void set_room_hooks(std::function<bool(const std::vector<std::pair<int, size_t>> & reqs)> make_room_all,
                        std::function<void(const std::vector<std::pair<int, size_t>> & reqs)> ahead);
    /// larry (patch-roomahead.py): the bound conversation's replicas will grow to position hi's cap bucket (a prompt
    /// that ends there) - each card's extra bytes to the read-ahead hook; hi < 0: nothing coming.  Between passes.
    void room_ahead(int64_t hi);
""", "hpp api")

tc = rep(tc, """    std::function<bool(int, size_t)> make_room;   // the tier: room for that many more bytes on that GPU
""", """    std::function<bool(int, size_t)> make_room;   // the tier: room for that many more bytes on that GPU
    // larry (patch-roomahead.py): the tier's room on every card in one call; its read-ahead of a coming growth's
    // evictions (room_ahead: for conversation ahead_seq up to ahead_want rows - dropped once its replicas get there)
    std::function<bool(const std::vector<std::pair<int, size_t>> &)> make_room_all;
    std::function<void(const std::vector<std::pair<int, size_t>> &)> room_ahead;
    int ahead_seq = -1;
    int64_t ahead_want[3] = { 0, 0, 0 };
""", "impl fields")

tc = rep(tc, """    // STRATA_TP_LAZY_KV: ranks >= 1 map the pages of rows [0, want[r]) of every ratio-r source (up to cmax[r]) before
""", """    // larry (patch-roomahead.py): the bytes card c's replicas map to hold want[r] rows of every ratio-r source
    size_t extra_of(const Card & c, const int64_t want[3]) const {
        size_t extra = 0;
        for (int l = 0; l < NL; ++l) {
            const LInfo & L = li[(size_t) l];
            if (!L.kv_src) continue;
            const LS & s = c.ls[(size_t) l];
            extra += s.vcomp.grow_bytes((size_t) (want[L.ratio] * DH) * 2) + s.vidxk.grow_bytes((size_t) (want[L.ratio] * ID) * 2);
        }
        return extra;
    }
    // STRATA_TP_LAZY_KV: ranks >= 1 map the pages of rows [0, want[r]) of every ratio-r source (up to cmax[r]) before
""", "extra_of")

tc = rep(tc, """        const Card & c0 = *cards[0];
        size_t total = 0;
        for (int k = 1; k < N; ++k) {
            Card & c = *cards[(size_t) k];
            ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c.dev);
            size_t extra = 0;
            for (int l = 0; l < NL; ++l) {
                const LInfo & L = li[(size_t) l];
                if (!L.kv_src) continue;
                const LS & s = c.ls[(size_t) l];
                extra += s.vcomp.grow_bytes((size_t) (want[L.ratio] * DH) * 2) + s.vidxk.grow_bytes((size_t) (want[L.ratio] * ID) * 2);
            }
            if (extra && make_room && !make_room(c.gpu, extra)) { err = "tp: no VRAM on GPU " + std::to_string(c.gpu) + " for the KV rows"; return false; }
            total += extra;
""", """        const Card & c0 = *cards[0];
        size_t total = 0;
        // larry (patch-roomahead.py): every card's extra bytes first - the tier makes room on all of them in one call
        std::vector<size_t> extra_k((size_t) N, 0);
        std::vector<std::pair<int, size_t>> reqs;
        for (int k = 1; k < N; ++k) {
            extra_k[(size_t) k] = extra_of(*cards[(size_t) k], want);
            if (extra_k[(size_t) k]) reqs.emplace_back(cards[(size_t) k]->gpu, extra_k[(size_t) k]);
        }
        const double tr0 = now_us();
        if (!reqs.empty() && make_room_all && !make_room_all(reqs)) { err = "tp: no VRAM on the expert cards for the KV rows"; return false; }
        const double tr1 = now_us();
        for (int k = 1; k < N; ++k) {
            Card & c = *cards[(size_t) k];
            ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c.dev);
            const size_t extra = extra_k[(size_t) k];
            if (extra && !make_room_all && make_room && !make_room(c.gpu, extra)) { err = "tp: no VRAM on GPU " + std::to_string(c.gpu) + " for the KV rows"; return false; }
            total += extra;
""", "ensure_rows room")

tc = rep(tc, """        std::fprintf(stderr, "tp: KV replicas now %lld / %lld rows (ratio 1 / 2), +%.0f MiB on the other cards\\n",
                     (long long) want[1], (long long) want[2], total / 1048576.0);
        for (int r = 1; r <= 2; ++r) rows_done[r] = want[r];
        return true;
""", """        std::fprintf(stderr, "tp: KV replicas now %lld / %lld rows (ratio 1 / 2), +%.0f MiB on the other cards (room %.0f ms, "
                     "maps + rows %.0f ms)\\n", (long long) want[1], (long long) want[2], total / 1048576.0, (tr1 - tr0) / 1e3,
                     (now_us() - tr1) / 1e3);
        for (int r = 1; r <= 2; ++r) rows_done[r] = want[r];
        if (ahead_seq == seq && rows_done[1] >= ahead_want[1] && rows_done[2] >= ahead_want[2]) {   // (patch-roomahead.py)
            ahead_seq = -1;
            if (room_ahead) room_ahead({});   // the tier's read-ahead this growth did not take goes
        }
        return true;
""", "ensure_rows end")

tc = rep(tc, """void Ds41Tp::set_vram_hook(std::function<bool(int, size_t)> make_room) { p_->make_room = std::move(make_room); }
""", """void Ds41Tp::set_vram_hook(std::function<bool(int, size_t)> make_room) { p_->make_room = std::move(make_room); }

void Ds41Tp::set_room_hooks(std::function<bool(const std::vector<std::pair<int, size_t>> &)> make_room_all,
                            std::function<void(const std::vector<std::pair<int, size_t>> &)> ahead) {
    p_->make_room_all = std::move(make_room_all);
    p_->room_ahead = std::move(ahead);
}

// larry (patch-roomahead.py)
void Ds41Tp::room_ahead(int64_t hi) {
    Impl & im = *p_;
    if (!im.room_ahead || !im.lazy || im.N < 2) return;
    std::vector<std::pair<int, size_t>> reqs;
    if (hi >= 0) {   // the bucket sync_state / the first pass will map (as ensure_rows: never below what is mapped)
        int64_t want[3] = { 0, 0, 0 };
        bool any = false;
        for (int r = 1; r <= 2; ++r) {
            want[r] = std::min<int64_t>(im.cmax[r], std::max<int64_t>(im.rows_done[r], next_pow2(std::max<int64_t>(1024, (hi + r - 1) / r + 1))));
            any = any || want[r] > im.rows_done[r];
        }
        if (any) {
            for (int k = 1; k < im.N; ++k) {
                const size_t e = im.extra_of(*im.cards[(size_t) k], want);
                if (e) reqs.emplace_back(im.cards[(size_t) k]->gpu, e);
            }
            for (int r = 1; r <= 2; ++r) im.ahead_want[r] = want[r];
        }
    }
    im.ahead_seq = reqs.empty() ? -1 : im.seq;
    im.room_ahead(reqs);
}
""", "api defs")

# sync_state: the rows ensure_rows just copied are not copied again
tc = rep(tc, """    if (im.lazy && !every) {   // the replicas' pages for the rows up to hi first (as a pass there would: a cap bucket)
        int64_t want[3] = { 0, 0, 0 };
        for (int r = 1; r <= 2; ++r) want[r] = std::min<int64_t>(im.cmax[r], next_pow2(std::max<int64_t>(1024, (hi + r - 1) / r + 1)));
        if (!im.ensure_rows(want)) return false;
    }
""", """    // larry (patch-roomahead.py): the rows ensure_rows maps here it copies from card 0 (the prompt has written them) -
    // the copies below stop where they start (a fresh 967K prompt's rows went twice: 0.34 s more before its first reply)
    int64_t fresh0[3] = { INT64_MAX, INT64_MAX, INT64_MAX };
    if (im.lazy && !every) {   // the replicas' pages for the rows up to hi first (as a pass there would: a cap bucket)
        int64_t want[3] = { 0, 0, 0 };
        for (int r = 1; r <= 2; ++r) want[r] = std::min<int64_t>(im.cmax[r], next_pow2(std::max<int64_t>(1024, (hi + r - 1) / r + 1)));
        for (int r = 1; r <= 2; ++r) fresh0[r] = im.rows_done[r];
        if (!im.ensure_rows(want)) return false;
    }
""", "sync fresh")
tc = rep(tc, """            int64_t r0 = every ? 0 : std::max<int64_t>(0, lo / r), r1 = every ? top : std::min<int64_t>(top, (hi + r - 1) / r + 1);
""", """            int64_t r0 = every ? 0 : std::max<int64_t>(0, lo / r), r1 = every ? top : std::min<int64_t>({ top, (hi + r - 1) / r + 1, fresh0[r] });
""", "sync range")
tc = rep(tc, """            const int64_t r0 = every ? 0 : std::max<int64_t>(0, lo / r), r1 = every ? top : std::min<int64_t>(top, (hi + r - 1) / r + 1);
            if (!im.pull_rows_staged(l, r0, r1) || !im.pull_rows_staged(l, rows - 1, rows)) return false;
""", """            const int64_t r0 = every ? 0 : std::max<int64_t>(0, lo / r), r1 = every ? top : std::min<int64_t>({ top, (hi + r - 1) / r + 1, fresh0[r] });
            if (!im.pull_rows_staged(l, r0, r1) || !im.pull_rows_staged(l, rows - 1, rows)) return false;
""", "sync staged range")

# ---------------------------------------------------------------- the tier
xh = rep(xh, """    bool tp_make_room(int gpu, size_t bytes);
""", """    bool tp_make_room(int gpu, size_t bytes);
    /// larry (patch-roomahead.py): tp_make_room for several GPUs in one call - the cards' evictions side by side
    /// (STRATA_XE_TP_EVICT_PAR, default 1), from what tp_room_ahead read where it did.  Between passes.  False: a card
    /// has no room.
    bool tp_make_room_multi(const std::vector<std::pair<int, size_t>>& reqs);
    /// larry (patch-roomahead.py): the TP core's next growth will want these bytes on these GPUs (a prompt still runs) -
    /// the host copies of the experts its evictions would free are read meanwhile, on a helper thread
    /// (STRATA_XE_TP_AHEAD, default 1); an empty list drops what was read.  At a quiet point (between batches / passes).
    void tp_room_ahead(const std::vector<std::pair<int, size_t>>& reqs);
""", "tier hpp")

xc = rep(xc, """    std::vector<uint8_t*> evict_bufs;  ///< the evicted experts' host copies (read from the model file at eviction)
""", """    std::vector<uint8_t*> evict_bufs;  ///< the evicted experts' host copies (read from the model file at eviction)
    /// larry (patch-roomahead.py): an eviction's host copies read ahead - an entry a slab the TP core's coming rows will
    /// free (tp_room_ahead), read by pre_thr while a prompt runs; evict_slabs adopts a slab's copy when it frees it
    struct PreRead {
        int card = -1;
        const uint8_t* slab = nullptr;
        std::vector<std::pair<int32_t, int32_t>> list;   // the slab's experts, in the order evict_slabs lists them
        std::vector<size_t> off;                         // their offsets in hb (4 KiB rounded, as evict_slabs')
        size_t tot = 0;
        uint8_t* hb = nullptr;
        int state = 0;                                   // 0 queued, 1 being read, 2 read, -1 dropped / failed
        double ms = 0;
    };
    std::mutex pre_mu;
    std::condition_variable pre_cv;
    std::vector<std::shared_ptr<PreRead>> pre;           // the current set
    std::deque<std::shared_ptr<PreRead>> pre_q;          // pre_thr's queue
    std::thread pre_thr;
    bool pre_stop = false;
    std::mutex ev_mu;                  ///< larry (patch-roomahead.py): the cards' evictions side by side - their shared bookkeeping
""", "impl fields")

EVICT_NEW = r'''/// larry (patch-roomahead.py): the experts of `list` into hb + off[i] - nth threads, each through its own O_DIRECT
/// scratch and bounce; on `cpus` when given (a read ahead while a prompt runs: STRATA_XE_TP_AHEAD_CPUS), else on the
/// process's CPUs (not pinned to the helper CPUs: the prompt's staging runs there - 8 readers pinned there cost a 131K
/// prompt's expert phase 3 s)
void read_into(Ds41XeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& list, const std::vector<size_t>& off,
               uint8_t* hb, int nth, const cpu_set_t* cpus) {
    static const bool direct = env_i("STRATA_XE_LOAD_DIRECT", 1) != 0;
    const size_t sc = direct ? (im.blobs->read_staged_scratch() + 4095) & ~(size_t) 4095 : 0;
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < std::max(1, nth); ++i)
        th.emplace_back([&]() {
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), cpus ? cpus : &xe::process_cpus());
            uint8_t* buf = (uint8_t*) std::aligned_alloc(4096, ((size_t) im.blob_max + 4095) & ~(size_t) 4095);
            uint8_t* scr = sc ? (uint8_t*) std::aligned_alloc(4096, sc) : nullptr;
            for (;;) {
                const size_t k = next.fetch_add(1);
                if (k >= list.size() || !buf) break;
                if (scr) im.blobs->read_staged(list[k].first, list[k].second, buf, scr, sc);
                else im.blobs->read(list[k].first, list[k].second, buf);
                std::memcpy(hb + off[k], buf, im.bbytes(list[k].first, list[k].second));
            }
            std::free(scr);
            std::free(buf);
        });
    for (auto& t : th) t.join();
}

/// larry (patch-roomahead.py): the read-ahead helper - queued entries read one after another
void pre_worker(Ds41XeImpl& im) {
    xe::place_helper(pthread_self());
    static const int readers = std::max(1, env_i("STRATA_XE_TP_AHEAD_READERS", 2));
    static cpu_set_t ahead_cpus;
    static const bool ahead_set = xe::parse_cpus(std::getenv("STRATA_XE_TP_AHEAD_CPUS"), ahead_cpus);
    for (;;) {
        std::shared_ptr<Ds41XeImpl::PreRead> e;
        {
            std::unique_lock<std::mutex> lk(im.pre_mu);
            im.pre_cv.wait(lk, [&] { return im.pre_stop || !im.pre_q.empty(); });
            if (im.pre_stop) return;
            e = im.pre_q.front();
            im.pre_q.pop_front();
            if (e->state != 0) continue;   // dropped while queued
            e->state = 1;
        }
        const double t0 = now_ms();
        uint8_t* hb = (uint8_t*) std::aligned_alloc(4096, e->tot);
        if (hb) read_into(im, e->list, e->off, hb, readers, ahead_set ? &ahead_cpus : nullptr);
        {
            std::lock_guard<std::mutex> lk(im.pre_mu);
            e->ms = now_ms() - t0;
            if (e->state == 1 && hb) { e->hb = hb; e->state = 2; }
            else { std::free(hb); e->state = -1; }   // dropped while it was read (or no memory)
        }
        im.pre_cv.notify_all();
    }
}

/// an entry out of the read-ahead (pre_mu held): its copy freed now, or by the helper when it is being read
void pre_drop_locked(Ds41XeImpl::PreRead& e) {
    if (e.state == 2) { std::free(e.hb); e.hb = nullptr; }
    e.state = -1;
}

/// larry (patch-roomahead.py): the copy read ahead of slab `last`'s experts when it was read for exactly `list` (a read
/// still running is waited for, at most STRATA_XE_TP_AHEAD_WAIT_S - 60 s - then it is dropped); null: none - the caller
/// reads them
uint8_t* pre_take(Ds41XeImpl& im, const XeCard& c, const uint8_t* last, const std::vector<std::pair<int32_t, int32_t>>& list,
                  double& waited) {
    std::unique_lock<std::mutex> lk(im.pre_mu);
    auto it = std::find_if(im.pre.begin(), im.pre.end(), [&](const std::shared_ptr<Ds41XeImpl::PreRead>& e) {
        return e->card == c.idx && e->slab == last;
    });
    if (it == im.pre.end()) return nullptr;
    std::shared_ptr<Ds41XeImpl::PreRead> e = *it;
    im.pre.erase(it);
    if (e->list != list) { pre_drop_locked(*e); return nullptr; }
    static const double wait_s = env_d("STRATA_XE_TP_AHEAD_WAIT_S", 60.0);
    const double t0 = now_ms();
    const bool got = im.pre_cv.wait_for(lk, std::chrono::milliseconds((int64_t) (wait_s * 1000.0)),
                                        [&] { return e->state == 2 || e->state < 0; });
    waited += now_ms() - t0;
    if (!got) {
        std::fprintf(stderr, "xe tier: card %d: the read-ahead of a slab's experts did not finish in %.0f s (state %d) - read now\n",
                     c.idx, wait_s, e->state);
        pre_drop_locked(*e);
        return nullptr;
    }
    if (e->state != 2) return nullptr;
    uint8_t* hb = e->hb;
    e->hb = nullptr;
    e->state = -1;
    return hb;
}

/// the card's evictable last slabs back to the arena until `need` bytes are free; their experts become arena experts.
/// Between TP passes only: no kernel reads them (the card's queues are idle; promotions write flex slots).
/// larry (patch-roomahead.py): the cards may run this side by side (tp_make_room_multi drops the dense card's replicas
/// first: drop_rep false; the shared bookkeeping under ev_mu), a slab's host copies come from tp_room_ahead's read when
/// it read that slab's experts, and the driver is waited for once a round (the slabs the shortfall takes), not a slab
int64_t evict_slabs(Ds41XeImpl& im, XeCard& c, int64_t need, bool drop_rep = true) {
    const size_t ci = (size_t) c.idx;
    if (ci >= im.card_set.size()) return 0;
    auto& set = im.card_set[ci];
    const size_t from = im.evict_from[ci];
    if (set.size() <= from || c.slabs.size() < 2) return 0;
    const double t0 = now_ms();
    if (drop_rep) rep_drop(im);   // the dense card's copies of expert-card experts first (route tables: those entries change too)
    c.q.wait();
    static const bool round_wait = env_i("STRATA_XE_TP_EVICT_ROUND", 1) != 0;
    int64_t n = 0, bytes = 0;
    int slabs = 0, ahead = 0;
    double t_read = 0, t_ahead = 0, t_drv = 0;
    bool stop = false;
    while (!stop && c.free_vram() < need && c.slabs.size() > 1 && set.size() > from) {
        const int64_t fr0 = c.free_vram();
        const int64_t round_n = round_wait ? std::max<int64_t>(1, (need - fr0 + (int64_t) kSlab - 1) / (int64_t) kSlab) : 1;
        int64_t freed = 0;
        for (int64_t j = 0; j < round_n && c.slabs.size() > 1 && set.size() > from; ++j) {
            uint8_t* last = c.slabs.back();
            // larry, patch-pcrace: the slab's experts wherever they are in the set (not only its suffix) - moved to its end
            auto in_slab = [&](const std::pair<int32_t, int32_t>& le) {
                const size_t key = (size_t) im.key(le.first, le.second);
                const uint8_t* p = im.dptr[key];
                return im.card_of[key] == (int8_t) c.idx && p >= last && p < last + kSlab;
            };
            bool fixed_in = false;
            for (size_t i = 0; i < from && i < set.size(); ++i) fixed_in = fixed_in || in_slab(set[i]);
            if (fixed_in) { stop = true; break; }   // the slab holds experts that are not evictable
            const size_t k = (size_t) (std::stable_partition(set.begin() + (std::ptrdiff_t) from, set.end(),
                                                             [&](const std::pair<int32_t, int32_t>& le) { return !in_slab(le); }) -
                                       set.begin());
            // their host copies first: read ahead, else from the model file now (pageable; the arena's own blobs stay)
            std::vector<std::pair<int32_t, int32_t>> list(set.begin() + (std::ptrdiff_t) k, set.end());
            std::vector<size_t> off(list.size());
            size_t tot = 0;
            for (size_t i = 0; i < list.size(); ++i) { off[i] = tot; tot += (im.bbytes(list[i].first, list[i].second) + 4095) & ~(size_t) 4095; }
            uint8_t* hb = list.empty() ? nullptr : pre_take(im, c, last, list, t_ahead);
            if (hb) ++ahead;
            else if (!list.empty()) {
                const int64_t mem_keep = (int64_t) (env_d("STRATA_XE_TP_EVICT_MEM_GIB", 4.0) * 1073741824.0);
                const int64_t avail = mem_available_bytes();
                if (avail > 0 && avail - (int64_t) tot < mem_keep) {
                    std::fprintf(stderr, "xe tier: card %d: no eviction - %.2f GiB of host memory available, %.2f GiB needed\n", c.idx,
                                 avail / 1073741824.0, tot / 1073741824.0);
                    stop = true;
                    break;
                }
                hb = (uint8_t*) std::aligned_alloc(4096, tot);
                if (!hb) { stop = true; break; }
                const double tr = now_ms();
                read_into(im, list, off, hb, 8, nullptr);
                t_read += now_ms() - tr;
            }
            {
                std::lock_guard<std::mutex> lk(im.ev_mu);
                if (hb) im.evict_bufs.push_back(hb);
                for (size_t i = 0; i < list.size(); ++i) {
                    const size_t key = (size_t) im.key(list[i].first, list[i].second);
                    im.aptr[key] = hb + off[i];
                    im.apin[key] = 0;
                }
                im.arena_bytes += (int64_t) tot;
                im.arena_n += (int64_t) list.size();
                for (size_t i = k; i < set.size(); ++i) {
                    const auto& le = set[i];
                    const size_t key = (size_t) im.key(le.first, le.second);
                    im.card_of[key] = -1;
                    im.dptr[key] = nullptr;
                    rt_put(im, key);
                    c.used -= (int64_t) im.bbytes(le.first, le.second);
                    bytes += (int64_t) im.bbytes(le.first, le.second);
                    --c.n_experts;
                    ++n;
                }
            }
            set.resize(k);
            local_free(c.q, last);
            c.slabs.pop_back();
            c.slab_used = kSlab;   // the slabs left are full: the next expert opens a new one
            freed += (int64_t) kSlab;
            ++slabs;
        }
        if (!freed) break;
        // the xe driver shows freed VRAM a little later: wait until the driver shows this round's slabs (or the reading
        // stops rising) before the next test - else every evictable slab went at once
        const double tw = now_ms();
        for (int i = 0, same = 0; i < 50 && c.free_vram() < fr0 + freed * 9 / 10 && same < 5; ++i) {
            const int64_t f0 = c.free_vram();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            same = c.free_vram() == f0 ? same + 1 : 0;
        }
        t_drv += now_ms() - tw;
    }
    if (!n) return 0;
    {
        std::lock_guard<std::mutex> lk(im.ev_mu);
        im.resident_n = 0;
        for (auto& cp : im.cards) im.resident_n += cp->n_experts;
        im.evicted_n += n;
    }
    std::fprintf(stderr, "xe tier: card %d gave %lld experts (%.2f GiB, %d slabs, %d read ahead) back to the arena for the TP "
                 "core's KV rows (%.0f ms: reads %.0f, read-ahead waits %.0f, driver %.0f; %.2f GiB free)\n", c.idx, (long long) n,
                 bytes / 1073741824.0, slabs, ahead, now_ms() - t0, t_read, t_ahead, t_drv, c.free_vram() / 1073741824.0);
    return n;
}'''

xc = rep_span(xc, "/// the card's evictable last slabs back to the arena until `need` bytes are free; their experts become arena experts.\n",
              "\n}  // namespace\n\nbool Ds41XeTier::build_arena_ranked", EVICT_NEW, "evict_slabs")

ROOM_NEW = r'''bool Ds41XeTier::tp_make_room(int gpu, size_t bytes) { return tp_make_room_multi({{gpu, bytes}}); }

// larry (patch-roomahead.py): tp_make_room's steps for every card at once - the evictions side by side
bool Ds41XeTier::tp_make_room_multi(const std::vector<std::pair<int, size_t>>& reqs) {
    Ds41XeImpl& im = *im_;
    // the rows fit beside what is free (STRATA_TP_ROOM_KEEP_GIB, 0.25, left over); else the flex region gives the
    // shortfall back - after a long prompt its chunk buffers hold the card's slack, as they did with eager replicas
    const int64_t keep = (int64_t) (env_d("STRATA_TP_ROOM_KEEP_GIB", 0.25) * kGiB);
    // larry, STRATA_XE_TP_EVICT: the free VRAM down to STRATA_XE_TP_EVICT_KEEP_GIB (as the flex logic: 0.25 - the
    // run_big buffers kept after a prompt already hold its reserve; 1.5 evicted 570 experts for 36 MiB of rows), then
    // the card's evictable slabs, the flex region last
    const int64_t ekeep = (int64_t) (env_d("STRATA_XE_TP_EVICT_KEEP_GIB", 0.25) * kGiB);
    struct Rq { XeCard* c; size_t bytes; int64_t fr; bool evict; int64_t got; };
    std::vector<Rq> rq;
    for (const auto& r : reqs) {
        XeCard* cp = nullptr;
        for (auto& c : im.cards) if (c->gpu == r.first) cp = c.get();
        if (!cp || cp->dense) continue;   // not an expert card of the tier
        const int64_t fr = cp->free_vram();
        auto tdl = im.tp_defer.find(r.first);
        if (tdl != im.tp_defer.end()) tdl->second = tdl->second > r.second ? tdl->second - r.second : 0;
        if (fr < 0) continue;
        rq.push_back({cp, r.second, fr, tp_evict_on() && !cp->pbuf_cap && (int64_t) r.second + ekeep > fr, 0});
    }
    std::vector<Rq*> ev;
    for (auto& r : rq) if (r.evict) ev.push_back(&r);
    if (!ev.empty()) {
        rep_drop(im);   // once, before the cards' evictions (evict_slabs would on each)
        static const bool par = env_i("STRATA_XE_TP_EVICT_PAR", 1) != 0;
        if (par && ev.size() > 1) {
            std::vector<std::thread> th;
            for (Rq* r : ev) th.emplace_back([&im, r, ekeep] { r->got = evict_slabs(im, *r->c, (int64_t) r->bytes + ekeep, false); });
            for (auto& t : th) t.join();
        } else {
            for (Rq* r : ev) r->got = evict_slabs(im, *r->c, (int64_t) r->bytes + ekeep, false);
        }
    }
    bool ok = true;
    for (auto& r : rq) {
        XeCard& c = *r.c;
        const int64_t give0 = (int64_t) r.bytes + keep - (r.got > 0 ? c.free_vram() : r.fr);
        if (give0 <= 0) continue;
        if (!c.flex || c.pbuf_cap) {   // no flex region to give back (a prompt holds it)
            if (c.free_vram() <= (int64_t) r.bytes) {
                std::fprintf(stderr, "xe tier: card %d: no room for %.2f GiB of the TP core's KV rows\n", c.idx, r.bytes / kGiB);
                ok = false;
            }
            continue;
        }
        const int64_t give = std::min<int64_t>(give0, (int64_t) c.flex_cap);
        pc_drop(im);   // the promoted experts out (every card; the slots come back at the next promotion)
        c.q.wait();
        const size_t was = c.flex_cap;
        local_free(c.q, c.flex);
        const int64_t left = ((int64_t) was - give) & ~(int64_t) 255;
        c.flex = left >= 8 * (int64_t) im.blob_max ? card_alloc(c.q, (size_t) left) : nullptr;   // larry, patch-barstage
        c.flex_cap = c.flex ? (size_t) left : 0;
        std::fprintf(stderr, "xe tier: card %d's flex region %.2f -> %.2f GiB (the TP core's KV rows)\n", c.idx, was / kGiB,
                     c.flex_cap / kGiB);
        if (c.free_vram() <= (int64_t) r.bytes) {
            std::fprintf(stderr, "xe tier: card %d: no room for %.2f GiB of the TP core's KV rows\n", c.idx, r.bytes / kGiB);
            ok = false;
        }
    }
    return ok;
}

// larry (patch-roomahead.py)
void Ds41XeTier::tp_room_ahead(const std::vector<std::pair<int, size_t>>& reqs) {
    Ds41XeImpl& im = *im_;
    static const bool on = env_i("STRATA_XE_TP_AHEAD", 1) != 0;
    if (!on || !tp_evict_on()) return;
    const int64_t ekeep = (int64_t) (env_d("STRATA_XE_TP_EVICT_KEEP_GIB", 0.25) * kGiB);
    // the slabs evict_slabs would free on each card for its bytes now, and their experts in its order
    std::vector<std::shared_ptr<Ds41XeImpl::PreRead>> want;
    for (const auto& r : reqs) {
        XeCard* cp = nullptr;
        for (auto& c : im.cards) if (c->gpu == r.first) cp = c.get();
        if (!cp || cp->dense || cp->pbuf_cap) continue;
        XeCard& c = *cp;
        const size_t ci = (size_t) c.idx;
        if (ci >= im.card_set.size()) continue;
        const auto& set = im.card_set[ci];
        const size_t from = im.evict_from[ci];
        const int64_t need = (int64_t) r.second + ekeep;
        int64_t fr = c.free_vram();
        if (fr < 0) continue;
        size_t evictable = set.size() > from ? set.size() - from : 0;
        for (size_t si = c.slabs.size(); si > 1 && fr < need && evictable > 0; --si) {
            const uint8_t* last = c.slabs[si - 1];
            auto in_slab = [&](const std::pair<int32_t, int32_t>& le) {
                const size_t key = (size_t) im.key(le.first, le.second);
                const uint8_t* p = im.dptr[key];
                return im.card_of[key] == (int8_t) c.idx && p >= last && p < last + kSlab;
            };
            bool fixed_in = false;
            for (size_t i = 0; i < from && i < set.size(); ++i) fixed_in = fixed_in || in_slab(set[i]);
            if (fixed_in) break;
            auto e = std::make_shared<Ds41XeImpl::PreRead>();
            e->card = c.idx;
            e->slab = last;
            for (size_t i = from; i < set.size(); ++i) if (in_slab(set[i])) e->list.push_back(set[i]);
            e->off.resize(e->list.size());
            for (size_t i = 0; i < e->list.size(); ++i) {
                e->off[i] = e->tot;
                e->tot += (im.bbytes(e->list[i].first, e->list[i].second) + 4095) & ~(size_t) 4095;
            }
            evictable -= std::min(evictable, e->list.size());
            fr += (int64_t) kSlab;
            if (!e->list.empty()) want.push_back(e);
        }
    }
    std::lock_guard<std::mutex> lk(im.pre_mu);
    std::vector<std::shared_ptr<Ds41XeImpl::PreRead>> keep;
    size_t newb = 0, newn = 0;
    static const double mem_gib = env_d("STRATA_XE_TP_AHEAD_MEM_GIB", 8.0);
    int64_t avail = -2;
    for (auto& w : want) {
        auto it = std::find_if(im.pre.begin(), im.pre.end(), [&](const std::shared_ptr<Ds41XeImpl::PreRead>& e) {
            return e->card == w->card && e->slab == w->slab && e->list == w->list && e->state >= 0;
        });
        if (it != im.pre.end()) { keep.push_back(*it); continue; }
        if (avail == -2) avail = mem_available_bytes();
        if (avail > 0 && avail - (int64_t) (newb + w->tot) < (int64_t) (mem_gib * kGiB)) continue;   // left to the eviction
        newb += w->tot;
        ++newn;
        keep.push_back(w);
        im.pre_q.push_back(w);
    }
    size_t dropped = 0;
    for (auto& e : im.pre)
        if (std::find(keep.begin(), keep.end(), e) == keep.end()) { pre_drop_locked(*e); ++dropped; }
    im.pre = keep;
    if (newn || dropped || want.size() > keep.size()) {
        size_t tb = 0;
        for (auto& e : im.pre) tb += e->tot;
        if (avail == -2) avail = mem_available_bytes();
        std::fprintf(stderr, "xe tier: read-ahead for the TP core's KV rows: %zu slabs' experts queued (%.2f GiB), %zu dropped; "
                     "%zu held (%.2f GiB), MemAvailable %.1f GiB%s\n", newn, newb / kGiB, dropped, im.pre.size(), tb / kGiB,
                     avail / kGiB, want.size() > keep.size() ? " - too little for the rest" : "");
    }
    if (newn && !im.pre_thr.joinable()) im.pre_thr = std::thread([&im] { pre_worker(im); });
    if (newn) im.pre_cv.notify_all();
}

'''

xc = rep_span(xc, "bool Ds41XeTier::tp_make_room(int gpu, size_t bytes) {\n", "void Ds41XeTier::tp_set_promote_queues(",
              ROOM_NEW, "tp_make_room")

xc = rep(xc, """void Ds41XeTier::close() {
    if (!im_) return;
    Ds41XeImpl& im = *im_;
""", """void Ds41XeTier::close() {
    if (!im_) return;
    Ds41XeImpl& im = *im_;
    {   // larry (patch-roomahead.py): the read-ahead helper stopped, what it read freed - and ready again (init closes first)
        { std::lock_guard<std::mutex> lk(im.pre_mu); im.pre_stop = true; }
        im.pre_cv.notify_all();
        if (im.pre_thr.joinable()) im.pre_thr.join();
        for (auto& e : im.pre) { std::free(e->hb); e->hb = nullptr; }
        im.pre.clear();
        im.pre_q.clear();
        im.pre_stop = false;
    }
""", "close")

# ---------------------------------------------------------------- the engine
g = rep(g, """        tp->set_vram_hook([&tier](int gpu, size_t bytes) { return tier.tp_make_room(gpu, bytes); });
""", """        tp->set_vram_hook([&tier](int gpu, size_t bytes) { return tier.tp_make_room(gpu, bytes); });
        // larry (patch-roomahead.py): every expert card's room in one call (the evictions side by side); a long
        // prompt's evictions read ahead while it runs
        tp->set_room_hooks([&tier](const std::vector<std::pair<int, size_t>> & r) { return tier.tp_make_room_multi(r); },
                           [&tier](const std::vector<std::pair<int, size_t>> & r) { tier.tp_room_ahead(r); });
""", "engine hooks")

g = rep(g, """                int B = std::min(pb_cap, n - c0);
                btr("batch start", B);
""", """                int B = std::min(pb_cap, n - c0);
                btr("batch start", B);
                const double tb0 = now_ms();   // larry (patch-roomahead.py)
""", "batch start")

g = rep(g, """                c0 += B;
                if (feed_seam && c0 < n) feed_seam();   // larry (batch slots): the other slots' windows
""", """                c0 += B;
#if defined(DS41_TP)
                // larry (patch-roomahead.py): once the batches left look shorter than STRATA_TP_AHEAD_S (30 s; 0 = off),
                // the expert cards' evictions for this prompt's KV replicas are read ahead (re-checked every batch)
                if (tp && c0 < n) {
                    static const double ahead_s = std::getenv("STRATA_TP_AHEAD_S") ? std::atof(std::getenv("STRATA_TP_AHEAD_S")) : 30.0;
                    const double left_ms = (now_ms() - tb0) * (double) (n - c0) / (double) B;
                    if (ahead_s > 0 && left_ms <= ahead_s * 1000.0) tp->room_ahead((int64_t) pos0 + n + 64);
                }
#endif
                if (feed_seam && c0 < n) feed_seam();   // larry (batch slots): the other slots' windows
""", "batch seam")

g = rep(g, """            if (!ensure_kv(total + 64)) { std::printf("ERR %s\\n", err.c_str()); std::fflush(stdout); continue; }
""", """            if (!ensure_kv(total + 64)) { std::printf("ERR %s\\n", err.c_str()); std::fflush(stdout); continue; }
#if defined(DS41_TP)
            if (tp) tp->room_ahead(-1);   // larry (patch-roomahead.py): an earlier prompt's read-ahead left over goes
#endif
""", "request start")

for p, s in ((TH, th), (TC, tc), (XH, xh), (XC, xc), (G, g)):
    open(p, "w").write(s)
print("patched", TH, TC, XH, XC, G)
