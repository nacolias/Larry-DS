#!/usr/bin/env python3
# patch-zcopy.py [TIER_CPP] - larry, STRATA_XE_ZCOPY=1 (default 0 until measured): a prompt call's small uploads (the
# GEMM experts' row lists and slot map, xe_dqgemm's job / tile tables, each job batch's jobs + slot map) by a KERNEL
# reading pinned host memory, not a queue memcpy - and on the expert cards from pinned per-call copies (the dense
# card's STRATA_XE_DENSE_LITE staging), so none of the host waits the shared buffers needed.  A card has one user copy
# engine, FIFO across its queues and 90-100 % busy with the prefetch's staging in a prompt (bench/speed/engmon.sh):
# every one of those memcpys queued behind megabytes of DMA, and the expert cards' card threads waited for them ~40
# times a call (two host waits a job batch) - a call's GEMM phase ran ~88 ms for what bench/dqb/bench_dqgemm does in
# ~30 (96 experts x 200 tokens, 42-47 TFLOPS).  The end-of-call wait stays.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_ZCOPY" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


# the helper, before hot_put's comment block
rep("""/// STRATA_XE_HOTRING: `bytes` of host memory at `src` to `dst` on queue `q` in ring chunks:""",
    """/// larry, STRATA_XE_ZCOPY (patch-zcopy.py): `bytes` (a multiple of 4) of PINNED host memory at `src` to `dst` by a kernel
/// on `q` - the compute engine reads the host memory (zero-copy); the card's one copy engine stays with the bulk DMA
bool zcopy_on() { static const bool on = env_i("STRATA_XE_ZCOPY", 0) != 0; return on; }
sycl::event zc_copy(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return {};
    if (bytes % 16 == 0 && (((uintptr_t) dst | (uintptr_t) src) % 16) == 0) {
        using v4 = sycl::vec<uint32_t, 4>;
        const v4* sv = (const v4*) src;
        v4* dv = (v4*) dst;
        return q.parallel_for(sycl::range<1>(bytes / 16), [=](sycl::id<1> i) { dv[i] = sv[i]; });
    }
    const uint32_t* sv = (const uint32_t*) src;
    uint32_t* dv = (uint32_t*) dst;
    return q.parallel_for(sycl::range<1>(bytes / 4), [=](sycl::id<1> i) { dv[i] = sv[i]; });
}
/// larry: a small upload from pinned host memory - a kernel (STRATA_XE_ZCOPY) or the queue's memcpy
sycl::event up_pinned(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    return zcopy_on() ? zc_copy(q, dst, src, bytes) : q.memcpy(dst, src, bytes);
}

/// STRATA_XE_HOTRING: `bytes` of host memory at `src` to `dst` on queue `q` in ring chunks:""")
# zc: the expert cards stage like nowait
rep("""        const bool nowait = lite && c.dense && cmb > 0 && streamed[(size_t) c.idx].empty() && kept[(size_t) c.idx].empty();
        size_t nw_off = 0;
        bool out_init = !gather_init;   // larry, STRATA_XE_GATHER_INIT: false = the next gather writes the part
        if (nowait) {""", """        const bool nowait = lite && c.dense && cmb > 0 && streamed[(size_t) c.idx].empty() && kept[(size_t) c.idx].empty();
        // larry, STRATA_XE_ZCOPY (patch-zcopy.py): an expert card's uploads from pinned per-call copies too (no host waits);
        // single-round calls only (the staging below is sized for one pass: nothing streamed in the call)
        const bool zc = zcopy_on() && !c.dense && streamed[(size_t) c.idx].empty() && kept[(size_t) c.idx].empty();
        const bool stg = nowait || zc;
        size_t nw_off = 0;
        bool out_init = !gather_init;   // larry, STRATA_XE_GATHER_INIT: false = the next gather writes the part
        if (stg) {""")
rep("""        auto nw_take = [&](size_t bytes) -> uint8_t* {
            uint8_t* p = c.nw_host + nw_off;
            nw_off += (bytes + 255) & ~(size_t) 255;
            return p;
        };""", """        auto nw_take = [&](size_t bytes) -> uint8_t* {
            uint8_t* p = c.nw_host + nw_off;
            nw_off += (bytes + 255) & ~(size_t) 255;
            if (zc && nw_off > c.nw_cap) {   // larry, patch-zcopy: never past the staging (sized for one pass)
                std::fprintf(stderr, "xe tier: card %d: per-call staging overrun (%zu of %zu bytes) - aborting\\n", c.idx, nw_off, c.nw_cap);
                std::abort();
            }
            return p;
        };""")
# GEMM path 1 (dq) and path 2: both blocks are identical text -> replace both
old_g = """                if (nowait) {   // larry, STRATA_XE_DENSE_LITE: from pinned per-call copies, no wait
                    uint8_t* hp = nw_take(ptok.size() * 4);
                    uint8_t* hs = nw_take(S.size() * 4);
                    std::memcpy(hp, ptok.data(), ptok.size() * 4);
                    std::memcpy(hs, S.data(), S.size() * 4);
                    std::lock_guard<PrioMutex> lk(g_submit);
                    c.q.memcpy(dtok, hp, ptok.size() * 4);
                    c.q.memcpy(c.cS, hs, S.size() * 4);
                } else {"""
new_g = """                if (stg) {   // larry, STRATA_XE_DENSE_LITE / STRATA_XE_ZCOPY: from pinned per-call copies, no wait
                    uint8_t* hp = nw_take(ptok.size() * 4);
                    uint8_t* hs = nw_take(S.size() * 4);
                    std::memcpy(hp, ptok.data(), ptok.size() * 4);
                    std::memcpy(hs, S.data(), S.size() * 4);
                    std::lock_guard<PrioMutex> lk(g_submit);
                    up_pinned(c.q, dtok, hp, ptok.size() * 4);
                    up_pinned(c.q, c.cS, hs, S.size() * 4);
                } else {"""
rep(old_g, new_g, 2)
# dq tables
rep("""                        c.q.memcpy(djobs + jo0, hjobs + jo0, (jo - jo0) * sizeof(xe::DqJob));""",
    """                        up_pinned(c.q, djobs + jo0, hjobs + jo0, (jo - jo0) * sizeof(xe::DqJob));   // larry, ZCOPY""")
rep("""                            c.q.memcpy(dtiles + to, htiles + to, gt[g].size() * sizeof(xe::DqTile));""",
    """                            up_pinned(c.q, dtiles + to, htiles + to, gt[g].size() * sizeof(xe::DqTile));   // larry, ZCOPY""")
# job batches
rep("""                    if (!nowait) c.q.wait();   // cjobs / cjobs_host and cS are read by the previous batch / the GEMM gather
                    if (call_tl) { if (tl.b == 0) tl.b = now_ms() - t0; tl.nj += (int64_t) nb; }
                    XeJob* jh = nowait ? (XeJob*) nw_take(nb * sizeof(XeJob)) : c.cjobs_host;   // larry, STRATA_XE_DENSE_LITE""",
    """                    if (!stg) c.q.wait();   // cjobs / cjobs_host and cS are read by the previous batch / the GEMM gather
                    if (call_tl) { if (tl.b == 0) tl.b = now_ms() - t0; tl.nj += (int64_t) nb; }
                    XeJob* jh = stg ? (XeJob*) nw_take(nb * sizeof(XeJob)) : c.cjobs_host;   // larry, STRATA_XE_DENSE_LITE / ZCOPY""")
rep("""                        const int32_t* sp = S.data();
                        if (nowait) {   // larry, STRATA_XE_DENSE_LITE: a per-batch pinned copy, no wait
                            uint8_t* hs = nw_take(S.size() * 4);
                            std::memcpy(hs, S.data(), S.size() * 4);
                            sp = (const int32_t*) hs;
                        }
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.cjobs, jh, nb * sizeof(XeJob));
                        c.q.memcpy(c.cS, sp, S.size() * 4);
                    }
                    if (!nowait) c.q.wait();""", """                        const int32_t* sp = S.data();
                        if (stg) {   // larry, STRATA_XE_DENSE_LITE / ZCOPY: a per-batch pinned copy, no wait
                            uint8_t* hs = nw_take(S.size() * 4);
                            std::memcpy(hs, S.data(), S.size() * 4);
                            sp = (const int32_t*) hs;
                        }
                        std::lock_guard<PrioMutex> lk(g_submit);
                        if (stg) {   // pinned sources: a kernel under ZCOPY
                            up_pinned(c.q, c.cjobs, jh, nb * sizeof(XeJob));
                            up_pinned(c.q, c.cS, sp, S.size() * 4);
                        } else {
                            c.q.memcpy(c.cjobs, jh, nb * sizeof(XeJob));
                            c.q.memcpy(c.cS, sp, S.size() * 4);
                        }
                    }
                    if (!stg) c.q.wait();""")
open(P, "w").write(s)
print("patched", P)
