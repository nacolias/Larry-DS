#!/usr/bin/env python3
# patch-streamgrow.py [TIER_CPP] - larry: an expert card's stream buffer (the streamed experts of one chunk-call round:
# only that card's queues write it - H2D copies, the hot ring's DMAs - and its kernels read it) in card-local memory
# (local_alloc: zeMemAllocDevice, resident on that card only - SYCL's malloc_device makes UR map every allocation onto
# all P2P peers, ~30 ms for a few hundred MiB here) and grown by half its size at least: a prompt without the prefetch
# (8K tokens from empty) regrew it call by call to the next call's exact need - 33 regrowths in an 8K prompt after a
# 16K one, ~1.1 s under the submit lock (STRATA_XE_PROF "buffers" lines).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "grow_stream(" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:90]))
    s = s.replace(old, new)


rep("""template <class T>
bool grow_host(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); sycl::free(p, q); }
    p = sycl::malloc_host<T>(need, q);
    cap = p ? need : 0;
    return p != nullptr;
}
""", """template <class T>
bool grow_host(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); sycl::free(p, q); }
    p = sycl::malloc_host<T>(need, q);
    cap = p ? need : 0;
    return p != nullptr;
}
/// larry: the card's stream buffer, card-local (local_alloc) and grown by half its size at least (to `cap`; whole 32 MiB)
bool grow_stream(XeCard& c, size_t need, size_t cap) {
    if (c.stream_bytes >= need) return true;
    const size_t step = (size_t) 32 << 20;
    size_t want = std::max(need, c.stream_bytes + c.stream_bytes / 2);
    want = std::min((want + step - 1) / step * step, std::max(need, cap));
    if (c.stream) { c.q.wait(); local_free(c.q, c.stream); c.stream = nullptr; c.stream_bytes = 0; }
    c.stream = local_alloc(c.q, want);
    c.stream_bytes = c.stream ? want : 0;
    return c.stream != nullptr;
}
""")
rep("""        if (!grow_dev(c.stream, c.stream_bytes, sreq, c.q)) {
""", """        if (!grow_stream(c, sreq, stream_cap)) {
""")
rep("""        if (c.stream) { c.q.wait(); sycl::free(c.stream, c.q); c.stream = nullptr; c.stream_bytes = 0; }   // regrown on demand
""", """        if (c.stream) { c.q.wait(); local_free(c.q, c.stream); c.stream = nullptr; c.stream_bytes = 0; }   // regrown on demand
""")
rep("""        if (!grow_dev(c.stream, c.stream_bytes, sb, c.q) || !grow_dev(c.cX, c.cX_cap, (size_t) nc * H, c.q) ||
""", """        if (!grow_stream(c, sb, sb) || !grow_dev(c.cX, c.cX_cap, (size_t) nc * H, c.q) ||
""")
rep("""        fr(c.cX); fr(c.cOUT); fr(c.cOUT_host); fr(c.cS); fr(c.cW); fr(c.cjobs); fr(c.cjobs_host); fr(c.dq_host); fr(c.cH); fr(c.cY); fr(c.stream);
""", """        fr(c.cX); fr(c.cOUT); fr(c.cOUT_host); fr(c.cS); fr(c.cW); fr(c.cjobs); fr(c.cjobs_host); fr(c.dq_host); fr(c.cH); fr(c.cY);
        local_free(c.q, c.stream);   // larry: card-local (grow_stream)
""")
if "sycl::free(c.stream" in s or "grow_dev(c.stream" in s:
    sys.exit("a SYCL alloc / free of c.stream left")
open(P, "w").write(s)
print("patched", P)
