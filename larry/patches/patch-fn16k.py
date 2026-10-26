#!/usr/bin/env python3
# patch-fn16k.py [TIER_CPP] - larry, STRATA_XE_FN16=2 (default 0 until measured): FN16's X (fp16, the tier's own buffer
# on the dense card) read by a KERNEL on each expert card - peer loads on the compute engine, widened to f32 as they
# land - instead of a queue memcpy.  A card has one user copy engine and it runs copies FIFO across its queues: while
# the calls' card-to-card X copies ran, the prompt's staging DMAs onto cards 1-3 fell from 28.4 to 2.7 GB/s
# (bench/dqb/dma_load); X read by kernels at the same 28.5 GB/s leaves them 14.6 (bench/dqb/p2p_kread: the bytes
# match).  The buffer is in the tier's context (peer-readable: engine1 enables peer access between the cards); ggml's
# own buffers are not (their kernels' peer writes faulted) - which is why it reads FN16's copy, not x_dev.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


if "patch-fn16k" not in s:
  rep("""        if (fn16 && !c.dense && grow_loc(c.cXh, c.cXh_cap, (size_t) n * H, c.q)) {   // larry, STRATA_XE_FN16
            c.q.memcpy(c.cXh, im.fn16, (size_t) n * H * 2, fn16_ev);   // card-to-card from the dense card, this card's queue""",
    """        if (fn16 && !c.dense && fn16_env >= 2 && (((size_t) n * H) % 8) == 0) {
            // larry, STRATA_XE_FN16=2 (patch-fn16k.py): the dense card's fp16 X by peer loads, widened here - the copy
            // engine stays with the prefetch's staging
            const sycl::half* hs = im.fn16;
            float* xd = c.cX;
            c.q.submit([&](sycl::handler& h) {
                h.depends_on(fn16_ev);
                h.parallel_for(sycl::range<1>((size_t) n * H / 8), [=](sycl::id<1> i) {
                    const sycl::vec<sycl::half, 8> v = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(hs + 8 * i[0]);
                    *reinterpret_cast<sycl::float4*>(xd + 8 * i[0]) = sycl::float4((float) v[0], (float) v[1], (float) v[2], (float) v[3]);
                    *reinterpret_cast<sycl::float4*>(xd + 8 * i[0] + 4) = sycl::float4((float) v[4], (float) v[5], (float) v[6], (float) v[7]);
                });
            });
        } else if (fn16 && !c.dense && grow_loc(c.cXh, c.cXh_cap, (size_t) n * H, c.q)) {   // larry, STRATA_XE_FN16
            c.q.memcpy(c.cXh, im.fn16, (size_t) n * H * 2, fn16_ev);   // card-to-card from the dense card, this card's queue""")
  open(P, "w").write(s)
  print("patched", P)

# part 2: peer access for the expert cards' reads, enabled by the tier itself (engine1 does too; enabling twice throws)
s = open(P).read()
if "g_fn16_peer" not in s:
    def rep2(old, new):
        global s
        if s.count(old) != 1:
            sys.exit("anchor x%d: %s" % (s.count(old), old[:100]))
        s = s.replace(old, new)
    rep2("""namespace {

double now_ms() {""", """namespace {

// larry, STRATA_XE_FN16=2 (patch-fn16k.py): every expert card can read the dense card's memory (peer access on)
bool g_fn16_peer = true;

double now_ms() {""")
    rep2("""    if (dense_dev && dense_card < 0) { err = "xe tier: the dense half's card is not among STRATA_XE_DEVICES"; return false; }
    im.dense = dense_card;
""", """    if (dense_dev && dense_card < 0) { err = "xe tier: the dense half's card is not among STRATA_XE_DEVICES"; return false; }
    im.dense = dense_card;
    if (env_i("STRATA_XE_FN16", 0) >= 2 && im.dense >= 0) {   // larry, patch-fn16k.py: the peer reads of FN16's X
        const sycl::device d0 = im.cards[(size_t) im.dense]->dev;
        for (auto& cp : im.cards) {
            if (cp->dense) continue;
            if (!cp->dev.ext_oneapi_can_access_peer(d0)) { g_fn16_peer = false; continue; }
            try { cp->dev.ext_oneapi_enable_peer_access(d0); } catch (const std::exception&) {}   // on already: fine
        }
        std::fprintf(stderr, "xe tier: STRATA_XE_FN16=2: the expert cards read X by peer loads%s\\n",
                     g_fn16_peer ? "" : " - NO PEER ACCESS: the copy engine reads it");
    }
""")
    rep2("""        if (fn16 && !c.dense && fn16_env >= 2 && (((size_t) n * H) % 8) == 0) {""",
         """        if (fn16 && !c.dense && fn16_env >= 2 && g_fn16_peer && (((size_t) n * H) % 8) == 0) {""")
    open(P, "w").write(s)
    print("patched (peer access)", P)
