#!/usr/bin/env python3
# patch-regrowasync.py [XE_DIR] - larry (with patch-regrowasync-gen.py; the engine's STRATA_SERVE_REGROW_ASYNC=1):
# after a kept prompt batch, the dense card's cache regrow (the tail experts the batch's VRAM evicted, back from their
# pinned arena copies: 0.15-0.47 s at 131-16K tokens, all inside the prompt's time before the first token) is queued
# and left running.  regrow_cache_async: the slabs are taken and the uploads queued as before, but the experts are
# published (card_of / dptr / the route table) only when their copies are done - regrow_poll(wait), on the engine's
# thread before a TP pass.  Until then every reader sees them in the arena (a TP pass computes them on the CPU pool),
# except resident(): the route bias counts them already, so the decode routes exactly as after a synchronous regrow.
# The copies go on the tier's own queue of the dense card (the TP passes and DSpark run on others).  While one is
# pending no promotion starts (an expert of it must not get a second home) and every cache change (a shrink, a release,
# a seed, a sync regrow, close) waits for it first.  Pageable experts (bounced) are copied before it returns, so the
# bounce is free for the next prompt's streams.  Idempotent.
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe"
P, H = D + "/ds41_xe_tier.cpp", D + "/ds41_xe_tier.hpp"
s, h = open(P).read(), open(H).read()
if "regrow_cache_async" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1, which="cpp"):
    global s, h
    t = s if which == "cpp" else h
    if t.count(old) != n:
        sys.exit("%s anchor x%d (want %d): %s" % (which, t.count(old), n, old[:100]))
    t = t.replace(old, new)
    if which == "cpp": s = t
    else: h = t


rep("""    int64_t regrow_cache(std::string& err);""", """    int64_t regrow_cache(std::string& err);
    /// larry, patch-regrowasync.py: regrow_cache's uploads queued and left running - published by regrow_poll once
    /// they are done (until then the experts are arena experts to every reader but resident()).  -1: error.
    int64_t regrow_cache_async(std::string& err);
    /// a pending async regrow published if its copies are done (`wait`: after waiting for them); true when this call
    /// published it.  The engine's thread, between passes.
    bool regrow_poll(bool wait);""", which="hpp")
rep("""    std::map<const void*, std::vector<GivenSlice>> head_given;
""", """    std::map<const void*, std::vector<GivenSlice>> head_given;
    // larry, patch-regrowasync: a queued regrow of the dense card's cache - its experts, published by regrow_poll
    struct RgEnt { size_t key; uint8_t* p; std::pair<int32_t, int32_t> le; };
    std::vector<RgEnt> rg_list;
    std::vector<uint8_t> rg_mark;   ///< by key: in rg_list (resident() counts them for the route bias)
    sycl::event rg_ev;              ///< the dense card's queue behind the last copy / repack of rg_list
    bool rg_pending = false;
    double rg_t0 = 0;
""")
# seed_dense: the deferred form
rep("""int64_t seed_dense(Ds41XeImpl& im, XeCard& c, const std::vector<std::pair<int32_t, int32_t>>& ranked, const char* what, std::string& err) {
    const double t0 = now_ms();""", """int64_t seed_dense(Ds41XeImpl& im, XeCard& c, const std::vector<std::pair<int32_t, int32_t>>& ranked, const char* what, std::string& err,
                   bool defer = false) {   // larry, patch-regrowasync: defer = the uploads queued, published by regrow_poll
    const double t0 = now_ms();""")
rep("""        im.card_of[key] = (int8_t) c.idx;
        im.dptr[key] = p;
        rt_put(im, key);
        im.dense_set.push_back(le);
    }
    c.q.wait();
    im.resident_n = 0;""", """        if (defer) {   // larry, patch-regrowasync
            im.rg_list.push_back({key, p, le});
            if (im.rg_mark.size() != im.card_of.size()) im.rg_mark.assign(im.card_of.size(), 0);
            im.rg_mark[key] = 1;
            continue;
        }
        im.card_of[key] = (int8_t) c.idx;
        im.dptr[key] = p;
        rt_put(im, key);
        im.dense_set.push_back(le);
    }
    if (defer) {   // larry, patch-regrowasync: the bounce's copies done (the next prompt's streams use it), the rest queued
        hev[0].wait();
        hev[1].wait();
        im.rg_ev = c.q.ext_oneapi_submit_barrier();
        im.rg_pending = !im.rg_list.empty();
        im.rg_t0 = t0;
        std::fprintf(stderr, "xe tier: dense card %d %s: %lld experts (%lld pinned / %lld pageable) queued in %.2f s (slabs %.0f ms)\\n",
                     c.idx, what, (long long) (n_pin + n_page), (long long) n_pin, (long long) n_page, (now_ms() - t0) / 1000.0, t_alloc);
        return n_pin + n_page;
    }
    c.q.wait();
    im.resident_n = 0;""")
# the API: async regrow + poll; the sync regrow waits for a pending one first
rep("""/// larry (serve keep-cache): after the batch, the tail back - the seed continued within what is free now.
int64_t Ds41XeTier::regrow_cache(std::string& err) {
    Ds41XeImpl& im = *im_;
    if (im.dense < 0 || !im.cache_enabled || im.seed_ranked.empty()) return 0;""", """/// larry, patch-regrowasync: regrow_cache's copies queued, published by regrow_poll (see the header)
int64_t Ds41XeTier::regrow_cache_async(std::string& err) {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);
    if (im.dense < 0 || !im.cache_enabled || im.seed_ranked.empty()) return 0;
    XeCard& c = *im.cards[(size_t) im.dense];
    c.budget = dense_budget(im, c);
    return c.used >= c.budget ? 0 : seed_dense(im, c, im.seed_ranked, "regrow", err, true);   // (the replicas: at publish)
}

bool Ds41XeTier::regrow_poll(bool wait) {
    Ds41XeImpl& im = *im_;
    if (!im.rg_pending) return false;
    if (!wait && im.rg_ev.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete)
        return false;
    const double tw = now_ms();
    im.rg_ev.wait();
    const double tp = now_ms();
    for (const auto& r : im.rg_list) {
        im.card_of[r.key] = (int8_t) im.dense;
        im.dptr[r.key] = r.p;
        rt_put(im, r.key);
        im.dense_set.push_back(r.le);
        im.rg_mark[r.key] = 0;
    }
    const size_t n = im.rg_list.size();
    im.rg_list.clear();
    im.rg_pending = false;
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
    rep_upload(im);   // larry, engine1: the replicas after the cache
    std::fprintf(stderr, "xe tier: dense card regrow published: %zu experts, %.2f s after it was queued (%s %.0f ms); %zu experts "
                 "(%.2f GiB) on it\\n", n, (tp - im.rg_t0) / 1000.0, wait ? "waited" : "done,", tp - tw, im.dense_set.size(),
                 im.cards[(size_t) im.dense]->used / kGiB);
    return true;
}

/// larry (serve keep-cache): after the batch, the tail back - the seed continued within what is free now.
int64_t Ds41XeTier::regrow_cache(std::string& err) {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync
    if (im.dense < 0 || !im.cache_enabled || im.seed_ranked.empty()) return 0;""")
# every other cache change waits for a pending regrow first
rep("""int64_t Ds41XeTier::shrink_cache_for(int64_t need_bytes, std::string& err) {
    Ds41XeImpl& im = *im_;""", """int64_t Ds41XeTier::shrink_cache_for(int64_t need_bytes, std::string& err) {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync""")
rep("""void Ds41XeTier::release_cache() {
    Ds41XeImpl& im = *im_;""", """void Ds41XeTier::release_cache() {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync""")
rep("""bool Ds41XeTier::seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    Ds41XeImpl& im = *im_;""", """bool Ds41XeTier::seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync""")
rep("""void Ds41XeTier::close() {
    if (!im_) return;
    Ds41XeImpl& im = *im_;""", """void Ds41XeTier::close() {
    if (!im_) return;
    Ds41XeImpl& im = *im_;
    regrow_poll(true);   // larry, patch-regrowasync""")
# no promotion while a regrow is pending
rep("""    for (const auto& lk : im.tp_want) if (im.pc_ready) pc_want(im, lk.first, lk.second);
    im.tp_want.clear();""", """    // larry, patch-regrowasync: no new promotion while a regrow is pending (its experts are arena experts until published)
    for (const auto& lk : im.tp_want) if (im.pc_ready && !im.rg_pending) pc_want(im, lk.first, lk.second);
    im.tp_want.clear();""")
# resident(): a pending regrow's experts count (the route bias as after a synchronous regrow)
rep("""bool Ds41XeTier::resident(int64_t l, int64_t e) const {
    if (l < 0 || l >= im_->g.n_layers || e < 0 || e >= im_->g.n_experts) return false;
    return im_->card_of[(size_t) im_->key(l, e)] >= 0;
}""", """bool Ds41XeTier::resident(int64_t l, int64_t e) const {
    if (l < 0 || l >= im_->g.n_layers || e < 0 || e >= im_->g.n_experts) return false;
    const size_t key = (size_t) im_->key(l, e);
    // larry, patch-regrowasync: a pending regrow's experts are resident for the route bias
    return im_->card_of[key] >= 0 || (im_->rg_pending && key < im_->rg_mark.size() && im_->rg_mark[key]);
}""")
open(P, "w").write(s)
open(H, "w").write(h)
print("patched", P, H)
