#!/usr/bin/env python3
# patch-fn16.py [TIER_CPP] - larry, STRATA_XE_FN16 (default 0 until measured): a prompt chunk call's X (the FFN input,
# n x 5120 floats on the dense card) goes to the expert cards as fp16 - converted once on the dense card into a buffer
# the peers' copy engines read (USM), copied by each expert card's own queue (half the bytes over card 0's link: an
# 8192-token call's 168 MB took ~17 ms, three cards at once) and widened back to f32 there, so every kernel after it is
# as before.  The GEMM experts gather X as fp16 anyway (the same values); the small-job (q8) path quantises an
# fp16-rounded X.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_FN16" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    float* cX = nullptr; size_t cX_cap = 0;          ///< floats
""", """    float* cX = nullptr; size_t cX_cap = 0;          ///< floats
    sycl::half* cXh = nullptr; size_t cXh_cap = 0;   ///< larry, STRATA_XE_FN16: X as it arrives (halves; card-local)
""")
rep("""    PfSlot pf[2];
""", """    PfSlot pf[2];
    sycl::half* fn16 = nullptr; size_t fn16_cap = 0;   ///< larry, STRATA_XE_FN16: the call's X as fp16 on the dense card
""")
rep("""    std::unique_lock<PrioMutex> prep_lk(g_submit);
    const double t_lk = now_ms();
""", """    std::unique_lock<PrioMutex> prep_lk(g_submit);
    const double t_lk = now_ms();
    // larry, STRATA_XE_FN16: the call's X once as fp16 on the dense card (the dense half's pass_sync made it final);
    // the expert cards copy half the bytes and widen it
    static const int fn16_env = env_i("STRATA_XE_FN16", 0);
    XeCard* cd16 = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    bool fn16 = fn16_env && x_dev && cd16 && n > kMaxJobTok && ((size_t) n * H) % 4 == 0;
    sycl::event fn16_ev;
    if (fn16 && grow_dev(im.fn16, im.fn16_cap, (size_t) n * H, cd16->q)) {
        const float* xs = x_dev;
        sycl::half* hd = im.fn16;
        fn16_ev = cd16->q.parallel_for(sycl::range<1>((size_t) n * H / 4), [=](sycl::id<1> i) {
            const sycl::float4 v = *reinterpret_cast<const sycl::float4*>(xs + 4 * i[0]);
            *reinterpret_cast<sycl::vec<sycl::half, 4>*>(hd + 4 * i[0]) =
                sycl::vec<sycl::half, 4>((sycl::half) v.x(), (sycl::half) v.y(), (sycl::half) v.z(), (sycl::half) v.w());
        });
    } else {
        fn16 = false;
    }
""")
rep("""        c.q.memcpy(c.cX, x_dev ? x_dev : x, (size_t) n * H * 4);   // x_dev: card-to-card from the dense card
""", """        if (fn16 && !c.dense && grow_loc(c.cXh, c.cXh_cap, (size_t) n * H, c.q)) {   // larry, STRATA_XE_FN16
            c.q.memcpy(c.cXh, im.fn16, (size_t) n * H * 2, fn16_ev);   // card-to-card from the dense card, this card's queue
            const sycl::half* hs = c.cXh;
            float* xd = c.cX;
            c.q.parallel_for(sycl::range<1>((size_t) n * H / 4), [=](sycl::id<1> i) {
                const sycl::vec<sycl::half, 4> v = *reinterpret_cast<const sycl::vec<sycl::half, 4>*>(hs + 4 * i[0]);
                *reinterpret_cast<sycl::float4*>(xd + 4 * i[0]) = sycl::float4((float) v[0], (float) v[1], (float) v[2], (float) v[3]);
            });
        } else {
            c.q.memcpy(c.cX, x_dev ? x_dev : x, (size_t) n * H * 4);   // x_dev: card-to-card from the dense card
        }
""")
rep("""        local_free(c.q, c.cX); fr(c.cOUT); fr(c.cOUT_host);""",
    """        local_free(c.q, c.cX); local_free(c.q, c.cXh); fr(c.cOUT); fr(c.cOUT_host);""")
rep("""    im.lt_dev = nullptr;
    im.lt_dev_cap = 0;
""", """    im.lt_dev = nullptr;
    im.lt_dev_cap = 0;
    if (im.fn16 && im.dense >= 0) { im.cards[(size_t) im.dense]->q.wait(); sycl::free(im.fn16, im.cards[(size_t) im.dense]->q); }   // larry
    im.fn16 = nullptr;
""")
open(P, "w").write(s)
print("patched", P)
