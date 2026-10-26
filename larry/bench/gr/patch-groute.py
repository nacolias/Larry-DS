#!/usr/bin/env python3
# patch-groute.py - larry, STRATA_XE_GROUTE (default 0 for the A/B): the dense card dispatches a decode / verify call's
# routed experts itself.  Today the host waits for the router readback, groups the routing, writes every card's jobs
# and releases the armed expert cards (~42-60 us readback -> card start), then queues the dense card's own experts.
# With GROUTE, right behind the fn readback the dense queue runs a route kernel (the host's group() + card assignment
# on the GPU, reading a pinned table of where every expert is) that writes every card's job list into dense-card
# memory and the CPU pool's entries into pinned memory; the release queue then releases the armed expert cards, which
# pull their jobs card-to-card; the dense card's own share is queued behind the shared expert (pre_wait).  The host,
# after its readback wait, only runs the CPU pool's entries and queues the sum - same parts in the same order, so the
# same bits as the host's dispatch.  STRATA_XE_GROUTE_CHECK=1 (debug): every routed call's job lists are read back and
# compared with the host's grouping.
# Usage: patch-groute.py [engine dir]   (default /opt/ai/dsv41/xolotl/tools/v41/engine)   Idempotent.
import sys, shutil
D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine"
FILES = {"tier": D + "/xe/ds41_xe_tier.cpp", "hpp": D + "/xe/ds41_xe_tier.hpp", "dense": D + "/ds41_dense.cpp",
         "dhpp": D + "/ds41_dense.hpp", "gen": D + "/ds41_generate.cpp"}
src = {k: open(p).read() for k, p in FILES.items()}
if "STRATA_XE_GROUTE" in src["tier"]:
    print("already"); sys.exit(0)


def rep(k, old, new, cnt=1):
    s = src[k]
    if s.count(old) != cnt: sys.exit(f"{k}: anchor x{s.count(old)}: {old[:100]!r}")
    src[k] = s.replace(old, new)


# ================================ ds41_xe_tier.cpp ================================
# ---- the route table / output types
rep("tier", """constexpr int kPcBounce = 3;                     ///< STRATA_XE_PROMOTE: pinned bounces (promotions in flight)
""", """constexpr int kPcBounce = 3;                     ///< STRATA_XE_PROMOTE: pinned bounces (promotions in flight)

/// larry, STRATA_XE_GROUTE: where a (layer, expert) is, for the dense card's route kernel (a pinned table kept with
/// card_of / dptr: rt_put), and what the kernel leaves the host (pinned): its card entries' count and the CPU pool's
/// entries (Ent's fields)
struct RtEnt {
    const uint8_t* blob;                ///< the device address on `card`
    uint32_t up_off, down_off;
    uint16_t gu_type, d_type, gu_row, d_row;
    int32_t card;                       ///< -1: on no card
    int32_t pad;
};
struct RtCpu { int32_t e, nt, tok[kMaxJobTok], k[kMaxJobTok]; float w[kMaxJobTok]; };
struct RtHdr { int32_t ncard, ncpu, pad[2]; RtCpu cpu[kMaxDecodeJobs]; };
""")
# ---- per card: an armed sequence that pulls its jobs from the route kernel's list
rep("tier", """    bool armed = false;             ///< an armed sequence waits for its release
""", """    bool armed = false;             ///< an armed sequence waits for its release
    bool arm_routed = false;        ///< STRATA_XE_GROUTE: it pulls its jobs from the dense card's route list (released there)
    /// STRATA_XE_ARM_GRAPH: a routed armed sequence per shape (jobs, tokens, kernel width), recorded once and replayed
    std::map<uint32_t, std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>>> agraphs;
    std::map<uint32_t, const float*> agraph_x;   ///< per recording: the X its quant kernel reads (baked in)
""")
# ---- the impl's route state
rep("tier", """    int64_t arm_used = 0, arm_disarmed = 0;
""", """    int64_t arm_used = 0, arm_disarmed = 0;
    // larry, STRATA_XE_GROUTE (default 0): the dense card's route kernel dispatches decode / verify calls (route)
    bool groute_on = false;
    bool groute_want = false;          ///< groute_next(): the coming attn_router's tier call is one route() may take
    RtEnt* rt = nullptr;               ///< pinned [layers * experts]
    XeJob* rt_jobs = nullptr;          ///< dense card: [cards][kMaxDecodeJobs] - the routed call's jobs, per card
    XeJob* rt_null = nullptr;          ///< dense card: [cards] - each card's null job (the lists' padding)
    RtHdr* rt_hdr = nullptr;           ///< pinned: the route kernel's counts and CPU entries
    int64_t routed_l = -1;             ///< the routed call: layer, width, job bucket, kernel width, the cards it fed
    int routed_nt = 0, routed_ng = 0, routed_mv = 0;
    uint32_t routed_mask = 0;
    const float* rt_x = nullptr;       ///< its X on the dense card (the router's fn)
    bool rt_dense_armed = false;       ///< the dense card's own share is queued (arm: behind the shared expert)
    sycl::event rt_dense_ev;
    int64_t groute_used = 0, groute_miss = 0, groute_bad = 0;
""")
# ---- rt_put: after make_job
rep("tier", """/// Copies the file-layout blob of (l, e) at `src` (host USM, or device-readable) into its planar slot `dst` on card
""", """/// STRATA_XE_GROUTE: the route table's entry of `key` from card_of / dptr - after every change of them.  The host never
/// changes them while a route kernel is queued (it waits for each call's readbacks, which are behind the kernel).
void rt_put(Ds41XeImpl& im, size_t key) {
    if (!im.rt) return;
    RtEnt& r = im.rt[key];
    const int c = im.card_of[key];
    if (c < 0 || !im.dptr[key]) { r.card = -1; r.blob = nullptr; return; }
    const XeJob j = make_job(im, (int64_t) key / im.g.n_experts, (int64_t) key % im.g.n_experts, im.dptr[key]);
    r.blob = j.blob;
    r.up_off = j.up_off;
    r.down_off = j.down_off;
    r.gu_type = j.gu_type;
    r.d_type = j.d_type;
    r.gu_row = j.gu_row;
    r.d_row = j.d_row;
    r.card = c;
}

/// Copies the file-layout blob of (l, e) at `src` (host USM, or device-readable) into its planar slot `dst` on card
""")
# ---- every change of card_of / dptr keeps the table
rep("tier", """            im.card_of[(size_t) im.key(le.first, le.second)] = (int8_t) c.idx;
            im.dptr[(size_t) im.key(le.first, le.second)] = p;
""", """            im.card_of[(size_t) im.key(le.first, le.second)] = (int8_t) c.idx;
            im.dptr[(size_t) im.key(le.first, le.second)] = p;
            rt_put(im, (size_t) im.key(le.first, le.second));
""")
rep("tier", """        im.card_of[key] = (int8_t) c.idx;
        im.dptr[key] = p;
        im.dense_set.push_back(le);
""", """        im.card_of[key] = (int8_t) c.idx;
        im.dptr[key] = p;
        rt_put(im, key);
        im.dense_set.push_back(le);
""")
rep("tier", """            im.card_of[key] = -1;
            im.dptr[key] = nullptr;
            c.used -= (int64_t) im.bbytes(le.first, le.second);
""", """            im.card_of[key] = -1;
            im.dptr[key] = nullptr;
            rt_put(im, key);
            c.used -= (int64_t) im.bbytes(le.first, le.second);
""")
rep("tier", """        const size_t key = (size_t) im.key(le.first, le.second);
        im.card_of[key] = -1;
        im.dptr[key] = nullptr;
    }
    im.dense_set.clear();
""", """        const size_t key = (size_t) im.key(le.first, le.second);
        im.card_of[key] = -1;
        im.dptr[key] = nullptr;
        rt_put(im, key);
    }
    im.dense_set.clear();
""")
rep("tier", """                im.card_of[(size_t) j.key] = (int8_t) c.idx;
                im.dptr[(size_t) j.key] = slot;
""", """                im.card_of[(size_t) j.key] = (int8_t) c.idx;
                im.dptr[(size_t) j.key] = slot;
                rt_put(im, (size_t) j.key);
""")
rep("tier", """        im.card_of[(size_t) old] = -1;
        im.dptr[(size_t) old] = nullptr;
""", """        im.card_of[(size_t) old] = -1;
        im.dptr[(size_t) old] = nullptr;
        rt_put(im, (size_t) old);
""")
rep("tier", """            if (k >= 0) { im.card_of[(size_t) k] = -1; im.dptr[(size_t) k] = nullptr; }
""", """            if (k >= 0) { im.card_of[(size_t) k] = -1; im.dptr[(size_t) k] = nullptr; rt_put(im, (size_t) k); }
""")
# ---- the buffers, once placed (after the arm switches)
rep("tier", """    im.arm_on = env_i("STRATA_XE_ARM", 1) != 0 && im.planar && im.x_direct;
    im.arm_multi = im.arm_on && env_i("STRATA_XE_ARM_MULTI", 1) != 0;
""", """    im.arm_on = env_i("STRATA_XE_ARM", 1) != 0 && im.planar && im.x_direct;
    im.arm_multi = im.arm_on && env_i("STRATA_XE_ARM_MULTI", 1) != 0;
    // larry, STRATA_XE_GROUTE (default 0): the route table (pinned, every key) and the route kernel's lists / output
    im.groute_on = env_i("STRATA_XE_GROUTE", 0) != 0 && im.arm_on && im.dense >= 0 && !im.host_slots && im.cards.size() <= 8;
    if (im.groute_on && !im.rt) {
        XeCard& dcd = *im.cards[(size_t) im.dense];
        const size_t NK = im.card_of.size();
        im.rt = sycl::malloc_host<RtEnt>(NK, im.ctx);
        im.rt_hdr = sycl::malloc_host<RtHdr>(1, im.ctx);
        im.rt_jobs = sycl::malloc_device<XeJob>(im.cards.size() * kMaxDecodeJobs, dcd.q);
        im.rt_null = sycl::malloc_device<XeJob>(im.cards.size(), dcd.q);
        if (!im.rt || !im.rt_hdr || !im.rt_jobs || !im.rt_null) {
            std::fprintf(stderr, "xe tier: route buffers: allocation failed - STRATA_XE_GROUTE off\\n");
            im.groute_on = false;
        } else {
            for (size_t k = 0; k < NK; ++k) { im.rt[k] = RtEnt{}; im.rt[k].card = -1; rt_put(im, k); }
            std::vector<XeJob> nulls(im.cards.size());   // an expert-less card's: blob null (a null job's blob is never read)
            for (auto& cp : im.cards) { nulls[(size_t) cp->idx] = cp->arm_null; nulls[(size_t) cp->idx].nt = 0; }
            dcd.q.memcpy(im.rt_null, nulls.data(), nulls.size() * sizeof(XeJob)).wait();
            std::memset((void*) im.rt_hdr, 0, sizeof(RtHdr));
            std::fprintf(stderr, "xe tier: STRATA_XE_GROUTE: the dense card routes decode / verify calls\\n");
        }
    }
""")
# ---- card_experts: the jobs / queue / X explicit (the dense card's routed share runs on the dense queue)
rep("tier", """/// A card's share of a decode / verify call: its n jobs (staged in c.stage_dev, X at xoff) into dst [nt][n_embd].
sycl::event card_experts(Ds41XeImpl& im, XeCard& c, int n, int nt, int max_nt, size_t xoff, float* dst, const float* xsrc = nullptr) {
    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
    const XeJob* jobs = (const XeJob*) c.stage_dev;
    const float* X = xsrc ? xsrc : (const float*) (c.stage_dev + xoff);   // xsrc: pinned host rows, read over PCIe
    if (!im.planar) return xe::xe_experts(c.q, jobs, n, nt, X, c.H, dst, H, F, im.L, false);
    static const bool xmx = env_i("STRATA_XE_XMX", 0) != 0;   // larry: the XMX kernels (xe_xmx.cpp)
    if (xmx) {
        xe::xe_xt(c.q, X, nt, H, c.xt);
        return xe::xe_experts_x(c.q, jobs, n, nt, c.xt, c.ht, dst, H, F, im.L, false);
    }
    xe::xe_quant_rows(c.q, X, nt, H, c.xq);
    return xe::xe_experts_q8(c.q, jobs, n, nt, c.xq, c.hq, dst, H, F, im.L, false, max_nt);
}
""", """/// The planar kernels of a card's share on queue q: n jobs (device-readable) on X [nt][n_embd] into dst, c's scratch.
sycl::event card_experts_at(Ds41XeImpl& im, XeCard& c, sycl::queue& q, const XeJob* jobs, int n, int nt, int max_nt,
                            const float* X, float* dst) {
    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
    static const bool xmx = env_i("STRATA_XE_XMX", 0) != 0;   // larry: the XMX kernels (xe_xmx.cpp)
    if (xmx) {
        xe::xe_xt(q, X, nt, H, c.xt);
        return xe::xe_experts_x(q, jobs, n, nt, c.xt, c.ht, dst, H, F, im.L, false);
    }
    xe::xe_quant_rows(q, X, nt, H, c.xq);
    return xe::xe_experts_q8(q, jobs, n, nt, c.xq, c.hq, dst, H, F, im.L, false, max_nt);
}

/// A card's share of a decode / verify call: its n jobs (staged in c.stage_dev, X at xoff) into dst [nt][n_embd].
sycl::event card_experts(Ds41XeImpl& im, XeCard& c, int n, int nt, int max_nt, size_t xoff, float* dst, const float* xsrc = nullptr) {
    const int H = (int) im.g.n_embd, F = (int) im.g.n_ff;
    const XeJob* jobs = (const XeJob*) c.stage_dev;
    const float* X = xsrc ? xsrc : (const float*) (c.stage_dev + xoff);   // xsrc: pinned host rows, read over PCIe
    if (!im.planar) return xe::xe_experts(c.q, jobs, n, nt, X, c.H, dst, H, F, im.L, false);
    return card_experts_at(im, c, c.q, jobs, n, nt, max_nt, X, dst);
}
""")
# ---- disarm: a routed sequence has its release already (route) - only waited for; the dense card's routed share too
rep("tier", """    bool any = false;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.armed) continue;
        for (int k = 0; k < c.arm_ng; ++k) std::memcpy(c.stage_host + (size_t) k * sizeof(XeJob), &c.arm_null, sizeof(XeJob));
        arm_release(im, c);
        c.armed = false;
        any = true;
    }
    if (!any) return;
    for (auto& cp : im.cards) if (cp->arm_flag) cp->arm_ev.wait();
    im.armed_l = -1;
    ++im.arm_disarmed;
}""", """    bool any = false;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.armed) continue;
        if (!c.arm_routed) {   // STRATA_XE_GROUTE: a routed one was released by the dense card (route) with its jobs
            for (int k = 0; k < c.arm_ng; ++k) std::memcpy(c.stage_host + (size_t) k * sizeof(XeJob), &c.arm_null, sizeof(XeJob));
            arm_release(im, c);
        }
        c.armed = false;
        c.arm_routed = false;
        any = true;
    }
    if (im.rt_dense_armed) { im.rt_dense_ev.wait(); im.rt_dense_armed = false; any = true; }
    im.routed_l = -1;
    if (!any) return;
    for (auto& cp : im.cards) if (cp->arm_flag) cp->arm_ev.wait();
    im.armed_l = -1;
    ++im.arm_disarmed;
}""")
# ---- run_small: a routed call takes the routed path; anything else waits for the routed sequences first
rep("tier", """    const double tl_h0 = Ds41XeTier::tl_on() && nt == 1 ? tl_now_us() : 0;
    const int64_t H = im.g.n_embd;
    const int K = (int) im.g.top_k;
    const double t0 = now_ms();
    static thread_local std::vector<Ent> ents;
    static thread_local std::vector<int32_t> first;
    group(ids, w, nt, K, im.g.n_experts, ents, first);
""", """    const double tl_h0 = Ds41XeTier::tl_on() && nt == 1 ? tl_now_us() : 0;
    const int64_t H = im.g.n_embd;
    const int K = (int) im.g.top_k;
    im.groute_want = false;   // STRATA_XE_GROUTE: a groute_next() whose attn_router did not route ends with this call
    if (im.routed_l >= 0) {   // the dense card dispatched this call (route)
        if (im.routed_l == l && im.routed_nt == nt && out_dev && im.dense_q && x == im.arm_x) return run_routed(im, l, nt, ids, w, x, out_dev);
        ++im.groute_miss;     // another call: the routed sequences run their jobs, this one goes the host's way behind them
        disarm(im);
    }
    const double t0 = now_ms();
    static thread_local std::vector<Ent> ents;
    static thread_local std::vector<int32_t> first;
    group(ids, w, nt, K, im.g.n_experts, ents, first);
""")
# ---- run_routed: before run_small
rep("tier", """/// One decode / verify call: nt <= 8 tokens.  out_host (nt x n_embd) or out_dev (the dense card) gets the sum.
bool run_small(""", """/// STRATA_XE_GROUTE_CHECK=1 (debug): the route kernel's job lists and CPU entries of this call against the host's own
/// grouping and card assignment (run_small's) - counts mismatches into groute_bad, prints the first few
void groute_check(Ds41XeImpl& im, int64_t l, int nt, const int32_t* ids, const float* w) {
    const int K = (int) im.g.top_k;
    const size_t NC = im.cards.size();
    std::vector<Ent> ents;
    std::vector<int32_t> first;
    group(ids, w, nt, K, im.g.n_experts, ents, first);
    std::vector<XeJob> got(NC * kMaxDecodeJobs);
    im.dense_q->memcpy(got.data(), im.rt_jobs, got.size() * sizeof(XeJob)).wait();
    std::vector<int> nj(NC, 0);
    int ncpu = 0;
    bool bad = false;
    for (const Ent& en : ents) {
        const size_t key = (size_t) im.key(l, en.e);
        const int c = im.card_of[key];
        if (c < 0 || !((im.routed_mask >> c) & 1u)) {
            const RtCpu& rc = im.rt_hdr->cpu[ncpu++];
            bad = bad || rc.e != en.e || rc.nt != en.nt;
            for (int s = 0; s < en.nt && !bad; ++s) bad = rc.tok[s] != en.tok[s] || rc.k[s] != en.k[s] || rc.w[s] != en.w[s];
            continue;
        }
        const XeJob& j = got[(size_t) c * kMaxDecodeJobs + (size_t) nj[(size_t) c]++];
        XeJob r = make_job(im, l, en.e, im.dptr[key]);
        r.nt = (uint8_t) en.nt;
        for (int s = 0; s < en.nt; ++s) { r.tok[s] = en.tok[s]; r.w[s] = en.w[s]; }
        bad = bad || j.blob != r.blob || j.up_off != r.up_off || j.down_off != r.down_off || j.gu_type != r.gu_type ||
              j.d_type != r.d_type || j.gu_row != r.gu_row || j.d_row != r.d_row || j.nt != r.nt;
        for (int s = 0; s < kMaxJobTok && !bad; ++s) bad = j.tok[s] != r.tok[s] || j.w[s] != r.w[s];
    }
    bad = bad || ncpu != im.rt_hdr->ncpu;
    for (size_t c = 0; c < NC && !bad; ++c)
        if ((im.routed_mask >> c) & 1u)
            for (int k = nj[c]; k < im.routed_ng && !bad; ++k) bad = got[c * kMaxDecodeJobs + (size_t) k].nt != 0;
    if (bad && im.groute_bad++ < 8)
        std::fprintf(stderr, "xe tier: STRATA_XE_GROUTE_CHECK: layer %lld (%d tokens): the route kernel's lists differ from the host's\\n",
                     (long long) l, nt);
}

/// STRATA_XE_GROUTE: a call route() dispatched - every card has its jobs (the dense card's route kernel); here the CPU
/// pool's entries (the kernel's list) and the sum, its parts in run_small's order (armed cards, the dense card, the
/// CPU pool; an armed card without jobs adds its zeros)
bool run_routed(Ds41XeImpl& im, int64_t l, int nt, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    const double tl_h0 = Ds41XeTier::tl_on() && nt == 1 ? tl_now_us() : 0;
    const double t0 = now_ms();
    const int64_t H = im.g.n_embd;
    const int K = (int) im.g.top_k;
    const size_t NC = im.cards.size();
    XeCard* dc = im.cards[(size_t) im.dense].get();
    sycl::queue& qd = *im.dense_q;
    const size_t n_out = (size_t) nt * H, stride = (size_t) kMaxJobTok * H;
    float* slots = dc->dslots;
    static const bool check = env_i("STRATA_XE_GROUTE_CHECK", 0) != 0;
    if (check) groute_check(im, l, nt, ids, w);
    if (im.pc_mode < 0) im.pc_mode = env_i("STRATA_XE_PROMOTE", 1) != 0 && im.planar && im.dense >= 0 ? 1 : 0;
    if (im.pc_mode > 0) {   // as run_small (the route kernel ran on the table as it was before this)
        ++im.pc_clock;
        if (!im.pc_ready && !im.pc_tried && !im.pf_on && ++im.pc_calls >= env_i("STRATA_XE_PROMOTE_AFTER", 1)) {
            im.pc_tried = true;
            pc_setup(im);
        }
        if (!im.pc_jobs.empty()) {
            const double tq = now_ms();
            pc_poll(im, false);
            im.pc_poll_us += (now_ms() - tq) * 1000.0;
        }
        if (im.pc_ready)   // the promoted experts this call ran: their slots used now (the LRU)
            for (int i = 0; i < nt * K; ++i) {
                const int32_t e = ids[i];
                if (e < 0 || e >= im.g.n_experts) continue;
                const int32_t pw = im.pc_where[(size_t) im.key(l, e)];
                if (pw >= 0) { im.cards[(size_t) (pw >> 16)]->pc_tick[(size_t) (pw & 0xFFFF)] = im.pc_clock; ++im.pc_hit; }
            }
    }
    // the CPU pool's entries (pinned, complete: the host waited for the readbacks queued behind the route kernel)
    static thread_local std::vector<Ent> cents;
    static thread_local std::vector<const Ent*> cpu_list;
    const int ncpu = std::max(0, std::min(im.rt_hdr->ncpu, kMaxDecodeJobs));
    cents.resize((size_t) ncpu);
    cpu_list.clear();
    for (int i = 0; i < ncpu; ++i) {
        const RtCpu& rc = im.rt_hdr->cpu[i];
        Ent& en = cents[(size_t) i];
        en.e = rc.e;
        en.nt = std::min(rc.nt, kMaxJobTok);
        for (int s = 0; s < kMaxJobTok; ++s) { en.tok[s] = rc.tok[s]; en.k[s] = rc.k[s]; en.w[s] = rc.w[s]; }
        cpu_list.push_back(&en);
    }
    for (const Ent* en : cpu_list) if (im.pc_ready) pc_want(im, l, (size_t) im.key(l, en->e));
    im.st.hits += im.rt_hdr->ncard;
    if (!im.rt_dense_armed) {   // arm() did not queue the dense card's share: now (behind what the dense queue has)
        im.rt_dense_ev = card_experts_at(im, *dc, qd, im.rt_jobs + (size_t) dc->idx * kMaxDecodeJobs, im.routed_ng, nt,
                                         im.routed_mv, im.rt_x, slots + (size_t) dc->idx * stride);
        im.rt_dense_armed = true;
    }
    std::vector<sycl::event> deps;
    struct Src { const float* p[10]; } src{};   // <= 7 expert cards, the dense card, the CPU pool
    int ns = 0;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.armed || !c.arm_routed) continue;
        deps.push_back(c.arm_ev);
        c.stage_ev = c.arm_ev;
        c.armed = false;
        c.arm_routed = false;
        ++im.arm_used;
        src.p[ns++] = slots + (size_t) c.idx * stride;
    }
    deps.push_back(im.rt_dense_ev);
    im.rt_dense_armed = false;
    src.p[ns++] = slots + (size_t) dc->idx * stride;
    if (!cpu_list.empty()) {
        im.hsum_ev.wait();   // the previous call's copy of hsum has been made
        std::memset(im.hsum, 0, n_out * 4);
        cpu_ents(im, l, cpu_list, x, im.hsum);
        im.hsum_ev = qd.memcpy(slots + NC * stride, im.hsum, n_out * 4);
        deps.push_back(im.hsum_ev);
        src.p[ns++] = slots + NC * stride;
    }
    float* o = (float*) out_dev;
    if (Ds41XeTier::tl_on() && nt == 1) tl_stamp(im, 0, qd, 15);
    im.last_sum = qd.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n_out), [=](sycl::id<1> i) {
            float sum = 0.0f;
            for (int k = 0; k < ns; ++k) sum += src.p[k][i];
            o[i] = sum;
        });
    });
    if (Ds41XeTier::tl_on()) tl_stamp(im, 0, qd, nt == 1 ? 3 : 11);
    if (tl_h0 > 0) im.tl_hmark.emplace_back(tl_h0, tl_now_us());
    im.routed_l = -1;
    im.armed_l = -1;
    ++im.groute_used;
    if (im.pc_mode > 0 && !im.pc_jobs.empty()) {   // promotions: behind this call's card work
        const double tq = now_ms();
        pc_poll(im, true);
        im.pc_push_us += (now_ms() - tq) * 1000.0;
    }
    im.st.wall_ms += now_ms() - t0;
    return true;
}

/// One decode / verify call: nt <= 8 tokens.  out_host (nt x n_embd) or out_dev (the dense card) gets the sum.
bool run_small(""")
# ---- arm(): the per-card sequence factored out (arm_card), the routed dense share; route(), groute_next()
s = src["tier"]
a0 = s.find("void Ds41XeTier::arm(int64_t l, int nt) {\n")
a1 = s.find("\nbool Ds41XeTier::run_chunk(int64_t l, int n,")
if a0 < 0 or a1 < 0 or "h.single_task([=] {   // until the host's release count passes the consumed one" not in s[a0:a1]:
    sys.exit("tier: arm() not found as expected")
ARM = r'''namespace {
/// STRATA_XE_ARM: card c's share of a coming call, queued ahead of its routing - a spin until its release (bounded),
/// its ng jobs (from the host's stage, or STRATA_XE_GROUTE: pulled from the dense card's route list), the experts on
/// the call's X (arm_x: pinned, read over PCIe), the part's copy into its slot on the dense card.  Under g_submit.
void arm_card(Ds41XeImpl& im, XeCard& c, int ng, int nt, int mv, const sycl::event& prev, const XeJob* jobs_src) {
    namespace sx = sycl::ext::oneapi::experimental;
    const int64_t H = im.g.n_embd;
    const size_t xoff = (kMaxDecodeJobs * sizeof(XeJob) + kStageAlign - 1) / kStageAlign * kStageAlign;
    float* slots = im.cards[(size_t) im.dense]->dslots;
    int* errp = im.arm_err;
    int* flag = c.arm_flag;
    int* seen = c.arm_seen;
    const bool tl = Ds41XeTier::tl_on();
    // STRATA_XE_GROUTE (jobs_src): the release comes from the dense card behind its route kernel, which its queue runs
    // after the previous call's sum - the slot this writes is free by then, no wait on `prev` (a cross-card dependency
    // costs the host ~10 us a submission)
    auto seq = [&](bool dep) {
        c.q.submit([&](sycl::handler& h) {
            if (dep) h.depends_on(prev);
            h.single_task([=] {   // until the host's release count passes the consumed one (bounded: a few seconds)
                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space> f(*flag);
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> sn(*seen);
                const int want = sn.load() + 1;
                int k = 0;
                for (; k < 0x7fffffff && f.load(sycl::memory_order::acquire) < want; ++k) {}
                if (k == 0x7fffffff)
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                     sycl::access::address_space::global_space>(*errp).store(1, sycl::memory_order::release);
                sn.store(want);
            });
        });
        if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 10);   // released: the card starts
        if (jobs_src) c.q.memcpy(c.stage_dev, jobs_src, (size_t) ng * sizeof(XeJob));   // card-to-card (the dense card's list)
        else c.q.memcpy(c.stage_dev, c.stage_host, (size_t) ng * sizeof(XeJob));
        card_experts(im, c, ng, nt, mv, xoff, c.OUT, im.arm_x);
        if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 11);
        sycl::event ev = c.q.memcpy(slots + (size_t) c.idx * kMaxJobTok * H, c.OUT, (size_t) nt * H * 4);
        if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 14);   // its copy is done
        return ev;
    };
    // STRATA_XE_ARM_GRAPH=1 (routed, no timeline): the sequence recorded once per shape and replayed - one submission
    static const bool agraph = env_i("STRATA_XE_ARM_GRAPH", 0) != 0;
    if (jobs_src && agraph && !tl && im.arm_x) {
        const uint32_t key = (uint32_t) ng | ((uint32_t) nt << 8) | ((uint32_t) mv << 12);
        auto it = c.agraphs.find(key);
        // X is baked into the recording (the job source and the slot are fixed): a different X - the dense half's
        // readback mirror reallocated by set_columns - records afresh
        if (it != c.agraphs.end() && c.agraph_x[key] != im.arm_x) { c.agraphs.erase(it); it = c.agraphs.end(); }
        if (it == c.agraphs.end()) {
            c.agraph_x[key] = im.arm_x;
            sx::command_graph<sx::graph_state::modifiable> g(c.q.get_context(), c.q.get_device());
            g.begin_recording(c.q);
            seq(false);
            g.end_recording();
            it = c.agraphs.emplace(key, std::make_unique<sx::command_graph<sx::graph_state::executable>>(g.finalize())).first;
        }
        sx::command_graph<sx::graph_state::executable>& eg = *it->second;
        c.arm_ev = c.q.submit([&](sycl::handler& h) { h.ext_oneapi_graph(eg); });
    } else {
        c.arm_ev = seq(jobs_src == nullptr);
    }
    c.armed = true;
    c.arm_routed = jobs_src != nullptr;
    c.arm_ng = ng;
}

/// The dense card's route kernel for a call of nt tokens (ids / wts: its router's [nt][K] on the device): run_small's
/// group() + card assignment.  One work-group, a work-item per routing slot (token-major): an expert's first slot builds
/// its entry (its tokens = its slots in order, one a token) and writes it as a job of its card - index = the earlier
/// entries of that card, the host's order - or (on no card / a card not taking jobs) as a CPU-pool entry (pinned);
/// then every taking card's list is null-padded to ng, and the counts written.
sycl::event route_kernel(sycl::queue& q, const int32_t* ids, const float* wts, int nt, int K, int64_t E, int64_t l,
                         const RtEnt* rt, XeJob* jobs, const XeJob* nulls, RtHdr* hdr, int ng, uint32_t mask, int NC) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> se(sycl::range<1>(64), h);   // slot -> expert (-1 none)
        sycl::local_accessor<int32_t, 1> sc(sycl::range<1>(64), h);   // slot -> entry's card (-1 CPU pool, -2 no entry)
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(64), sycl::range<1>(64)), [=](sycl::nd_item<1> it) {
            const int i = (int) it.get_local_id(0);
            const int n = nt * K;
            int e = -1;
            if (i < n) {
                e = ids[i];
                if (e < 0 || e >= E) e = -1;
            }
            se[i] = e;
            sycl::group_barrier(it.get_group());
            bool first = e >= 0;
            for (int j = 0; j < i && first; ++j) if (se[j] == e) first = false;
            RtEnt re{};
            int card = -2;
            if (first) {
                re = rt[l * E + e];
                card = re.card >= 0 && re.card < NC && ((mask >> re.card) & 1u) ? re.card : -1;
            }
            sc[i] = card;
            sycl::group_barrier(it.get_group());
            if (first) {
                int idx = 0;
                for (int j = 0; j < i; ++j) if (sc[j] == card) ++idx;
                int ntk = 0;
                int32_t tok[kMaxJobTok], kk[kMaxJobTok];
                float w[kMaxJobTok];
                for (int s = 0; s < kMaxJobTok; ++s) { tok[s] = 0; kk[s] = 0; w[s] = 0.f; }
                for (int j = i; j < n && ntk < kMaxJobTok; ++j)
                    if (se[j] == e) { tok[ntk] = j / K; kk[ntk] = j; w[ntk] = wts[j]; ++ntk; }
                if (card >= 0) {
                    XeJob jb;
                    jb.blob = re.blob;
                    jb.up_off = re.up_off;
                    jb.down_off = re.down_off;
                    jb.gu_type = re.gu_type;
                    jb.d_type = re.d_type;
                    jb.gu_row = re.gu_row;
                    jb.d_row = re.d_row;
                    jb.nt = (uint8_t) ntk;
                    for (int s = 0; s < kMaxJobTok; ++s) { jb.tok[s] = tok[s]; jb.w[s] = w[s]; }
                    jobs[(size_t) card * kMaxDecodeJobs + (size_t) idx] = jb;
                } else {
                    RtCpu& c = hdr->cpu[idx];
                    c.e = e;
                    c.nt = ntk;
                    for (int s = 0; s < kMaxJobTok; ++s) { c.tok[s] = tok[s]; c.k[s] = kk[s]; c.w[s] = w[s]; }
                }
            }
            if (i < NC && ((mask >> i) & 1u)) {   // card i's list, null-padded
                int cnt = 0;
                for (int j = 0; j < n; ++j) cnt += sc[j] == i;
                for (int k = cnt; k < ng; ++k) jobs[(size_t) i * kMaxDecodeJobs + (size_t) k] = nulls[i];
            }
            if (i == 0) {
                int ncard = 0, ncpu = 0;
                for (int j = 0; j < n; ++j) { ncard += sc[j] >= 0; ncpu += sc[j] == -1; }
                hdr->ncard = ncard;
                hdr->ncpu = ncpu;
            }
        });
    });
}
}  // namespace

void Ds41XeTier::arm(int64_t l, int nt) {
    Ds41XeImpl& im = *im_;
    if (im.routed_l >= 0 && im.routed_l == l && im.routed_nt == nt) {
        // STRATA_XE_GROUTE: route() armed and released the expert cards; the dense card's own share now, on the dense
        // queue behind the shared expert (its jobs: the route kernel's list)
        XeCard& dc = *im.cards[(size_t) im.dense];
        std::lock_guard<PrioMutex> lk(g_submit);
        im.rt_dense_ev = card_experts_at(im, dc, *im.dense_q, im.rt_jobs + (size_t) dc.idx * kMaxDecodeJobs, im.routed_ng, nt,
                                         im.routed_mv, im.rt_x, dc.dslots + (size_t) dc.idx * kMaxJobTok * im.g.n_embd);
        im.rt_dense_armed = true;
        return;
    }
    disarm(im);   // an earlier arm nobody took
    if (!im.arm_on || !im.relq || !im.dense_q || !im.arm_x || nt < 1 || nt > kMaxJobTok || (nt > 1 && !im.arm_multi)) return;
    if (im.arm_err && *(volatile int*) im.arm_err) {   // a spin kernel timed out once: arming stays off
        std::fprintf(stderr, "xe tier: an armed card waited out its release - STRATA_XE_ARM off\n");
        im.arm_on = im.arm_multi = false;
        return;
    }
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    if (!dc || !dc->dslots) return;
    const int K = (int) im.g.top_k;
    static const int kb[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, kMaxDecodeJobs};
    int ng = kMaxDecodeJobs;
    for (int b : kb) if (b >= K * nt) { ng = b; break; }
    const int mv = nt <= 1 ? 1 : nt <= 2 ? 2 : nt <= 4 ? 4 : 8;
    const sycl::event prev = im.last_sum;   // the previous call's sum reads the slots these sequences write
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.arm_flag || !c.arm_null_ok) continue;
        std::lock_guard<PrioMutex> lk(g_submit);
        arm_card(im, c, ng, nt, mv, prev, nullptr);
    }
    im.armed_l = l;
    im.armed_nt = nt;
}

void Ds41XeTier::groute_next() { im_->groute_want = im_->groute_on; }

void Ds41XeTier::route(int64_t l, int nt, const void* ids_dev, const void* wts_dev, const void* fn_dev, const float* fn_host) {
    Ds41XeImpl& im = *im_;
    const bool want = im.groute_want;
    im.groute_want = false;
    if (!want || !im.groute_on) return;
    if (!im.arm_on || !im.relq || !im.dense_q || !fn_host || !ids_dev || !wts_dev || !fn_dev || nt < 1 || nt > kMaxJobTok ||
        (nt > 1 && !im.arm_multi))
        return;   // the host's dispatch (arm(), run_small)
    {   // the expert cards read this layer's X straight from the fn readback's mirror: pinned host memory only
        static const float* chk_p = nullptr;
        static bool chk_ok = false;
        if (fn_host != chk_p) { chk_p = fn_host; chk_ok = sycl::get_pointer_type(fn_host, im.ctx) == sycl::usm::alloc::host; }
        if (!chk_ok) return;
    }
    if (im.arm_err && *(volatile int*) im.arm_err) return;   // arm() turns arming off (and says so)
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    const int K = (int) im.g.top_k;
    if (!dc || !dc->dslots || nt * K > 64) return;   // (the route kernel: a work-item a routing slot, one work-group)
    disarm(im);   // an earlier arm nobody took
    im.arm_x = fn_host;   // what the armed sequences read, and what run_small's take test compares the call's x with
    static const int kb[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, kMaxDecodeJobs};
    int ng = kMaxDecodeJobs;
    for (int b : kb) if (b >= K * nt) { ng = b; break; }
    const int mv = nt <= 1 ? 1 : nt <= 2 ? 2 : nt <= 4 ? 4 : 8;
    const sycl::event prev = im.last_sum;
    uint32_t mask = 1u << dc->idx;
    std::lock_guard<PrioMutex> lk(g_submit);
    // the expert cards' sequences (each spins until its release below, then pulls its list card-to-card)
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.arm_flag || !c.arm_null_ok) continue;
        arm_card(im, c, ng, nt, mv, prev, im.rt_jobs + (size_t) c.idx * kMaxDecodeJobs);
        mask |= 1u << c.idx;
    }
    // the route kernel on the dense queue (behind the fn readback the expert cards read), then the releases on the
    // release queue once it is done (the dense queue goes on with the readbacks and the shared expert meanwhile)
    sycl::queue& qd = *im.dense_q;
    const sycl::event rev = route_kernel(qd, (const int32_t*) ids_dev, (const float*) wts_dev, nt, K, im.g.n_experts, l, im.rt,
                                         im.rt_jobs, im.rt_null, im.rt_hdr, ng, mask, (int) im.cards.size());
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.armed || !c.arm_routed) continue;
        int* srcp = c.arm_src + 16 * (c.arm_src_i++ & 63);
        *srcp = ++c.arm_rel;
        im.relq->submit([&](sycl::handler& h) {
            h.depends_on(rev);
            h.memcpy(c.arm_flag, srcp, sizeof(int));
        });
    }
    im.routed_l = l;
    im.routed_nt = nt;
    im.routed_ng = ng;
    im.routed_mv = mv;
    im.routed_mask = mask;
    im.rt_x = (const float*) fn_dev;
    im.rt_dense_armed = false;
    im.armed_l = l;
    im.armed_nt = nt;
}
'''
src["tier"] = s[:a0] + ARM + s[a1:]
# ---- close: the report line and the buffers
rep("tier", """    if (im.arm_used || im.arm_disarmed)
        std::fprintf(stderr, "xe tier: armed card shares used %lld, disarmed calls %lld\\n", (long long) im.arm_used,
                     (long long) im.arm_disarmed);
""", """    if (im.arm_used || im.arm_disarmed)
        std::fprintf(stderr, "xe tier: armed card shares used %lld, disarmed calls %lld\\n", (long long) im.arm_used,
                     (long long) im.arm_disarmed);
    if (im.groute_used || im.groute_miss || im.groute_bad)
        std::fprintf(stderr, "xe tier: STRATA_XE_GROUTE: %lld routed calls, %lld others after a route, %lld check mismatches\\n",
                     (long long) im.groute_used, (long long) im.groute_miss, (long long) im.groute_bad);
""")
rep("tier", """    im.achunks.clear();
    if (im.hsum) { im.hsum_ev.wait(); sycl::free(im.hsum, im.ctx); im.hsum = nullptr; }
""", """    im.achunks.clear();
    for (void* p : {(void*) im.rt, (void*) im.rt_hdr, (void*) im.rt_jobs, (void*) im.rt_null}) if (p) sycl::free(p, im.ctx);
    im.rt = nullptr; im.rt_hdr = nullptr; im.rt_jobs = nullptr; im.rt_null = nullptr;
    if (im.hsum) { im.hsum_ev.wait(); sycl::free(im.hsum, im.ctx); im.hsum = nullptr; }
""")
# ---- the dense card's slabs freed (prompts): a routed share on the dense queue may still read them (review item 6)
rep("tier", """    if (im.dense < 0 || im.dense_set.empty()) return -1;
    XeCard& c = *im.cards[(size_t) im.dense];
    c.q.wait();
""", """    if (im.dense < 0 || im.dense_set.empty()) return -1;
    XeCard& c = *im.cards[(size_t) im.dense];
    c.q.wait();
    if (im.groute_on) { im.rt_dense_ev.wait(); im.last_sum.wait(); }   // STRATA_XE_GROUTE: its share on the dense queue
""")
rep("tier", """void Ds41XeTier::release_cache() {
    Ds41XeImpl& im = *im_;
    if (im.dense < 0 || im.dense_set.empty()) return;
""", """void Ds41XeTier::release_cache() {
    Ds41XeImpl& im = *im_;
    if (im.dense < 0 || im.dense_set.empty()) return;
    if (im.groute_on) { im.rt_dense_ev.wait(); im.last_sum.wait(); }   // STRATA_XE_GROUTE: its share on the dense queue
""")
# run_routed is defined before run_small and uses pc_* / cpu_ents / tl_stamp (all defined above run_small)

# ================================ ds41_xe_tier.hpp ================================
rep("hpp", """    void arm(int64_t layer, int nt);
""", """    void arm(int64_t layer, int nt);
    /// larry, STRATA_XE_GROUTE (default 0): the dense card routes the call itself - the dense half calls this right
    /// after queueing the layer's fn readback (ids / wts / fn: the router's outputs on the dense card); a route kernel
    /// on its queue writes every card's jobs and the CPU pool's entries, the expert cards are armed and released by the
    /// dense card, and the matching run_to_dev / run_multi_to_dev only runs the CPU part and the sum.  Only for the call
    /// right after groute_next() (the generate loop's decode / verify paths: their tier call matches the pass).
    void route(int64_t layer, int nt, const void* ids_dev, const void* wts_dev, const void* fn_dev, const float* fn_host);
    void groute_next();
""")

# ================================ ds41_dense ================================
rep("dhpp", """    void set_pre_wait_hook(std::function<void(int, int)> h);
""", """    void set_pre_wait_hook(std::function<void(int, int)> h);
    /// larry, STRATA_XE_GROUTE: called right after a decode / verify layer's fn readback is queued (layer, width, the
    /// router's ids / wts and fn on the device, the readback's host mirror) - the tier queues its route kernel there
    /// (Ds41XeTier::route)
    void set_route_hook(std::function<void(int, int, const void *, const void *, const void *, const float *)> h);
""")
rep("dense", """    std::function<void(int, int)> pre_wait;   // larry: set_pre_wait_hook
""", """    std::function<void(int, int)> pre_wait;   // larry: set_pre_wait_hook
    std::function<void(int, int, const void *, const void *, const void *, const float *)> route_hook;   // larry: set_route_hook
""")
rep("dense", """void Ds41Dense::set_pre_wait_hook(std::function<void(int, int)> h) { p_->pre_wait = std::move(h); }
""", """void Ds41Dense::set_pre_wait_hook(std::function<void(int, int)> h) { p_->pre_wait = std::move(h); }
void Ds41Dense::set_route_hook(std::function<void(int, int, const void *, const void *, const void *, const float *)> h) { p_->route_hook = std::move(h); }
""")
rep("dense", """    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
""", """    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    // larry, STRATA_XE_GROUTE: the tier's route kernel behind the fn readback (the expert cards read that copy)
    if (async && im.rb_ev && im.route_hook) im.route_hook(il, (int) n, im.o_ids->data, im.o_wts->data, im.o_fn->data, im.host_fn.data());
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
""")

# ================================ ds41_generate.cpp ================================
rep("gen", """        dense.set_pre_wait_hook([&tier](int il, int n) { tier.arm(il, n); });   // larry: STRATA_XE_ARM
""", """        dense.set_pre_wait_hook([&tier](int il, int n) { tier.arm(il, n); });   // larry: STRATA_XE_ARM
        dense.set_route_hook([&tier](int il, int n, const void * i, const void * w, const void * f, const float * fh) { tier.route(il, n, i, w, f, fh); });
""")
rep("gen", """            const float * fn = nullptr;
            if (!dense.attn_router(l, ids1.data(), w1.data(), &fn)) { ferr = dense.last_error(); return false; }
""", """            const float * fn = nullptr;
#if defined(DS41_XE)
            if (rdev1) tier.groute_next();   // larry, STRATA_XE_GROUTE: this layer's tier call (run_to_dev) may take the routing
#endif
            if (!dense.attn_router(l, ids1.data(), w1.data(), &fn)) { ferr = dense.last_error(); return false; }
""")
rep("gen", """            const float * fn = nullptr;
            if (!dense.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
            lap(0);
""", """            const float * fn = nullptr;
#if defined(DS41_XE)
            if (rdev1 && !no_tier_batch && m <= Ds41Tier::kMaxMulti) tier.groute_next();   // larry, STRATA_XE_GROUTE: one run_multi_to_dev
#endif
            if (!dense.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
            lap(0);
""")

for k, p in FILES.items():
    shutil.copy(p, p + ".pre-groute")
    open(p, "w").write(src[k])
print("patched")
