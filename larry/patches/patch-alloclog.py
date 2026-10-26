#!/usr/bin/env python3
# patch-alloclog.py [TIER_CPP] - larry (diagnostic, always on for slow ones): a tier allocation that takes > 30 ms is
# logged (what, MiB, the queue wait before the old buffer's free, the allocation itself).  abwT75: in a fresh engine's
# first 16K-column batch, layer 14's calls had expert-card threads idle 1.0 / 2.1 s between their own steps (STRATA_GPU_PTL
# host stamps) - where the per-call buffers reach new high-water marks and are grown mid-prompt (grow_dev / grow_host /
# grow_loc / grow_stream wait for their queue, free, allocate; pinned staging re-allocated).  STRATA_XE_ALLOC_LOG=1:
# every growth.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_ALLOC_LOG" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""template <class T>
bool grow_dev(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); addr_log("tier usm free", q, p, 0); sycl::free(p, q); }
    p = sycl::malloc_device<T>(need, q);
    cap = p ? need : 0;
    if (p) addr_log("tier usm alloc", q, p, need * sizeof(T));
    return p != nullptr;
}
template <class T>
bool grow_host(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); sycl::free(p, q); }
    p = sycl::malloc_host<T>(need, q);
    cap = p ? need : 0;
    return p != nullptr;
}
/// larry: grow_dev in card-local memory (local_alloc / local_free): buffers no other card's queue touches
template <class T>
bool grow_loc(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); local_free(q, p); }
    p = reinterpret_cast<T*>(local_alloc(q, need * sizeof(T)));
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
""", """// larry, STRATA_XE_ALLOC_LOG (patch-alloclog.py): a growth over 30 ms (or every one with =1) on stderr
void alloc_note(const char* what, size_t bytes, double t0, double t1) {
    static const bool all = env_i("STRATA_XE_ALLOC_LOG", 0) != 0;
    const double t2 = now_ms();
    if (all || t2 - t0 > 30.0)
        std::fprintf(stderr, "xe tier: alloc %s %.1f MiB: wait %.1f ms, free + alloc %.1f ms\\n", what, bytes / 1048576.0, t1 - t0, t2 - t1);
}
template <class T>
bool grow_dev(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    const double t0 = now_ms();
    if (p) q.wait();
    const double t1 = now_ms();
    if (p) { addr_log("tier usm free", q, p, 0); sycl::free(p, q); }
    p = sycl::malloc_device<T>(need, q);
    cap = p ? need : 0;
    if (p) addr_log("tier usm alloc", q, p, need * sizeof(T));
    alloc_note("device", need * sizeof(T), t0, t1);
    return p != nullptr;
}
template <class T>
bool grow_host(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    const double t0 = now_ms();
    if (p) q.wait();
    const double t1 = now_ms();
    if (p) sycl::free(p, q);
    p = sycl::malloc_host<T>(need, q);
    cap = p ? need : 0;
    alloc_note("host (pinned)", need * sizeof(T), t0, t1);
    return p != nullptr;
}
/// larry: grow_dev in card-local memory (local_alloc / local_free): buffers no other card's queue touches
template <class T>
bool grow_loc(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    const double t0 = now_ms();
    if (p) q.wait();
    const double t1 = now_ms();
    if (p) local_free(q, p);
    p = reinterpret_cast<T*>(local_alloc(q, need * sizeof(T)));
    cap = p ? need : 0;
    alloc_note("card-local", need * sizeof(T), t0, t1);
    return p != nullptr;
}
/// larry: the card's stream buffer, card-local (local_alloc) and grown by half its size at least (to `cap`; whole 32 MiB)
bool grow_stream(XeCard& c, size_t need, size_t cap) {
    if (c.stream_bytes >= need) return true;
    const size_t step = (size_t) 32 << 20;
    size_t want = std::max(need, c.stream_bytes + c.stream_bytes / 2);
    want = std::min((want + step - 1) / step * step, std::max(need, cap));
    const double t0 = now_ms();
    if (c.stream) c.q.wait();
    const double t1 = now_ms();
    if (c.stream) { local_free(c.q, c.stream); c.stream = nullptr; c.stream_bytes = 0; }
    c.stream = local_alloc(c.q, want);
    c.stream_bytes = c.stream ? want : 0;
    alloc_note("stream", want, t0, t1);
    return c.stream != nullptr;
}
""")
# the per-call pinned staging (ZCOPY / DENSE_LITE) and the GEMM tables' pinned copy
rep("""            if (c.nw_cap < need) {
                std::lock_guard<PrioMutex> lk(g_submit);
                c.q.wait();   // nothing of this call reads the old staging yet; the previous call's combine waited
                if (c.nw_host) sycl::free(c.nw_host, c.q);
                c.nw_host = sycl::malloc_host<uint8_t>(need + need / 4, c.q);
                c.nw_cap = c.nw_host ? need + need / 4 : 0;
            }
""", """            if (c.nw_cap < need) {
                std::lock_guard<PrioMutex> lk(g_submit);
                const double ta0 = now_ms();   // larry, STRATA_XE_ALLOC_LOG
                c.q.wait();   // nothing of this call reads the old staging yet; the previous call's combine waited
                const double ta1 = now_ms();
                if (c.nw_host) sycl::free(c.nw_host, c.q);
                c.nw_host = sycl::malloc_host<uint8_t>(need + need / 4, c.q);
                c.nw_cap = c.nw_host ? need + need / 4 : 0;
                alloc_note("call staging (pinned)", need + need / 4, ta0, ta1);
            }
""")
open(P, "w").write(s)
print("patched", P)
