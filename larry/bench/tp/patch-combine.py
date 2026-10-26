#!/usr/bin/env python3
# patch-combine.py - larry, STRATA_XE_COMBINE (default 0 for the A/B): a prompt chunk call's combine.  0: after every
# card is done, the dense card copies each expert card's part (n x n_embd f32) over its own link and adds them (31.6K
# prompt: 2.2 s of the tier's 18 s, all on its critical path).  1: each expert card pushes its part from its own queue
# the moment its work is done (a card that finishes early overlaps the others), the dense card only adds.  2: as 1,
# the parts pushed as fp16 (the call's X buffer holds the conversion: half the bytes over the dense card's link).
# Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_COMBINE" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


rep("""    float* cX2 = nullptr; size_t cX2_cap = 0;              ///< the dense card's copy of the other cards' sum
""", """    float* cX2 = nullptr; size_t cX2_cap = 0;              ///< the dense card's copy of the other cards' sum
    sycl::half* cX2h = nullptr; size_t cX2h_cap = 0;       ///< STRATA_XE_COMBINE=2: the parts as fp16 (slot = card)
""")
# the pushes at the end of each card's work
rep("""        if (!out_dev) {
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.memcpy(c.cOUT_host, c.cOUT, (size_t) n * H * 4);
        }
        c.q.wait();""", """        if (!out_dev) {
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.memcpy(c.cOUT_host, c.cOUT, (size_t) n * H * 4);
        }
        if (out_dev && !c.dense && cmb > 0 && cdc) {   // STRATA_XE_COMBINE: this card's part onto the dense card now
            const size_t ne = (size_t) n * H;
            std::lock_guard<PrioMutex> lk(g_submit);
            if (cmb == 2) {   // fp16 in the call's X buffer (no longer read), half the bytes over the dense card's link
                sycl::half* h16 = reinterpret_cast<sycl::half*>(c.cX);
                const float* src = c.cOUT;
                c.q.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> i) { h16[i] = (sycl::half) src[i]; });
                c.q.memcpy(cdc->cX2h + (size_t) c.idx * ne, h16, ne * 2);
            } else {
                c.q.memcpy(cdc->cX2 + (size_t) c.idx * ne, c.cOUT, ne * 4);
            }
        }
        c.q.wait();""")
rep("""    auto card_work = [&](XeCard& c) -> bool {
        std::vector<int32_t> S((size_t) n * K);""", """    // STRATA_XE_COMBINE (see the combine below): the cards push their parts themselves; the dense card's landing
    // buffer (a slot per card) must exist before their threads start
    static const int cmb_env = env_i("STRATA_XE_COMBINE", 0);
    XeCard* cdc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    int cmb = out_dev && cdc ? cmb_env : 0;
    if (cmb > 0) {
        std::lock_guard<PrioMutex> lk(g_submit);
        const size_t need = im.cards.size() * (size_t) n * H;
        const bool okb = cmb == 2 ? grow_dev(cdc->cX2h, cdc->cX2h_cap, need, cdc->q) : grow_dev(cdc->cX2, cdc->cX2_cap, need, cdc->q);
        if (!okb) { std::fprintf(stderr, "xe tier: dense card: combine slots allocation failed - the copy-after combine\\n"); cmb = 0; }
    }
    auto card_work = [&](XeCard& c) -> bool {
        std::vector<int32_t> S((size_t) n * K);""")
# the combine: the parts already landed
rep("""        sycl::event cev;
        {
            std::lock_guard<PrioMutex> lk(g_submit);
            if (!others.empty() && !grow_dev(dc->cX2, dc->cX2_cap, NE * others.size(), dc->q)) {""", """        sycl::event cev;
        if (cmb > 0) {   // STRATA_XE_COMBINE: every busy card's part is in its slot (its thread waited for the push)
            std::lock_guard<PrioMutex> lk(g_submit);
            int idx[8] = {};
            const int np = (int) std::min<size_t>(others.size(), 8);
            for (int k = 0; k < np; ++k) idx[k] = others[(size_t) k]->idx;
            const float* own = dc->busy ? dc->cOUT : nullptr;
            float* o = (float*) out_dev;
            if (cmb == 2) {
                const sycl::half* parts = dc->cX2h;
                cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                    float sm = own ? own[i] : 0.0f;
                    for (int k = 0; k < np; ++k) sm += (float) parts[(size_t) idx[k] * NE + i];
                    o[i] = sm;
                });
            } else {
                const float* parts = dc->cX2;
                cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                    float sm = own ? own[i] : 0.0f;
                    for (int k = 0; k < np; ++k) sm += parts[(size_t) idx[k] * NE + i];
                    o[i] = sm;
                });
            }
        } else {
            std::lock_guard<PrioMutex> lk(g_submit);
            if (!others.empty() && !grow_dev(dc->cX2, dc->cX2_cap, NE * others.size(), dc->q)) {""")
# free the fp16 slots with the rest
rep("""        fr(c.gscratch); fr(c.cX2); fr(c.bounce);""", """        fr(c.gscratch); fr(c.cX2); fr(c.cX2h); fr(c.bounce);""")
open(P, "w").write(s)
print("patched")
