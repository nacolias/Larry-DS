#!/usr/bin/env python3
# patch-engdirect.py [ENGINE_DIR] - larry, STRATA_ENGRAM_DIRECT=1 (default 0 until measured): a prompt batch's Engram
# reads from a PAGED table (engram-L<layer>.epg: layer 1) by O_DIRECT, when no prefetch warmed them.  A fresh 16K
# batch reads ~200K random rows (a 4 KiB page each) before layer 1 can run: 250-400 ms of layer-1 wait
# (STRATA_ENG_PROF, 2026-10-25), ~400K reads/s through the page cache - every miss allocates a page on a box whose
# page cache is full (32 GB; allocstall_normal 213M since boot), so each read also pays reclaim.  O_DIRECT reads
# skip the page cache.  A batch whose rows a prefetch (Ds41Engram::prefetch, the next batch's) read keeps the
# buffered path: those reads warmed the page cache for it.  Decode passes (<= 64 tokens) stay buffered.  Idempotent.
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine"
H, C = D + "/ds41_engram.hpp", D + "/ds41_engram.cpp"
h, s = open(H).read(), open(C).read()
if "STRATA_ENGRAM_DIRECT" in s:
    print("already patched"); sys.exit(0)


def rep(txt, old, new, n=1):
    if txt.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (txt.count(old), n, old[:100]))
    return txt.replace(old, new)


h = rep(h, """    struct Table { int fd_w = -1, fd_s = -1; uint64_t off_w = 0, off_s = 0; int64_t rows = 0;""",
        """    struct Table { int fd_w = -1, fd_s = -1; uint64_t off_w = 0, off_s = 0; int64_t rows = 0;
                   int fd_d = -1;   // larry, STRATA_ENGRAM_DIRECT: a paged table's O_DIRECT fd""")
h = rep(h, """    uint64_t bytes_ = 0, reads_ = 0;""", """    uint64_t bytes_ = 0, reads_ = 0;
    int64_t pf_lo_ = -1, pf_hi_ = -1;   // larry, STRATA_ENGRAM_DIRECT: the positions the last prefetch read""")

s = rep(s, """static inline ssize_t tread(int fd, const std::shared_ptr<Ds41Engram::RawMap> & m, void * buf, size_t n, uint64_t off) {
    return m ? m->read(buf, n, off) : ::pread(fd, buf, n, (off_t) off);
}
""", """static inline ssize_t tread(int fd, const std::shared_ptr<Ds41Engram::RawMap> & m, void * buf, size_t n, uint64_t off) {
    return m ? m->read(buf, n, off) : ::pread(fd, buf, n, (off_t) off);
}

// larry, STRATA_ENGRAM_DIRECT (patch-engdirect.py): n bytes at off through an O_DIRECT fd - the enclosing 4 KiB
// pages (a paged group never leaves its page) into this thread's aligned buffer, then the asked range; -1 on failure
static bool engram_direct_on() {
    static const bool on = std::getenv("STRATA_ENGRAM_DIRECT") && std::atoi(std::getenv("STRATA_ENGRAM_DIRECT")) != 0;
    return on;
}
static ssize_t dread(int fd, void * buf, size_t n, uint64_t off) {
    static thread_local uint8_t * pg = nullptr;
    if (!pg && posix_memalign((void **) &pg, 4096, 8192) != 0) { pg = nullptr; return -1; }
    const uint64_t a0 = off & ~(uint64_t) 4095, a1 = (off + n + 4095) & ~(uint64_t) 4095;
    if (a1 - a0 > 8192 || ::pread(fd, pg, (size_t) (a1 - a0), (off_t) a0) != (ssize_t) (a1 - a0)) return -1;
    std::memcpy(buf, pg + (off - a0), n);
    return (ssize_t) n;
}
""")
s = rep(s, """        if (t.fd_w >= 0) ::close(t.fd_w);
""", """        if (t.fd_w >= 0) ::close(t.fd_w);
        if (t.fd_d >= 0) ::close(t.fd_d);
""")
s = rep(s, """            t.rm_w = raw_map(pp, fd);
            tables_.push_back(t);""", """            t.rm_w = raw_map(pp, fd);
            if (engram_direct_on() && !t.rm_w) t.fd_d = ::open(pp.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);   // larry
            tables_.push_back(t);""")
# the batch's direct flag, decided in start_into
s = rep(s, """    const bool pool_dq = nthreads_ > 0 && n > 64;   // larry: a prompt batch's cached rows go to the pool
""", """    const bool pool_dq = nthreads_ > 0 && n > 64;   // larry: a prompt batch's cached rows go to the pool
    // larry, STRATA_ENGRAM_DIRECT: a prompt batch no prefetch read goes around the page cache
    const bool direct = engram_direct_on() && n > 64 && !(pf_lo_ >= 0 && pos0 >= pf_lo_ && pos0 + n <= pf_hi_);
""")
s = rep(s, """        t->fn = [this, bp, dim, srsz, g0, ng, c0, c1, kCachedChunk](int64_t a, int64_t c) {""",
        """        t->fn = [this, bp, dim, srsz, g0, ng, c0, c1, kCachedChunk, direct](int64_t a, int64_t c) {""")
s = rep(s, """                if (G.len_w > sizeof bw || G.len_s > sizeof bs ||
                    tread(T.fd_w, T.rm_w, bw, G.len_w, G.off_w) != (ssize_t) G.len_w ||
                    (G.len_s && tread(T.fd_s, T.rm_s, bs, G.len_s, G.off_s) != (ssize_t) G.len_s)) { bp->bad.fetch_add(1); continue; }
                bp->bytes.fetch_add((uint64_t) G.len_w + G.len_s);
                const auto & uq = bp->uniq[(size_t) G.e];
                const auto & pk = bp->packed[(size_t) G.e];""", """                const bool dw = direct && T.fd_d >= 0 && G.len_w <= sizeof bw &&
                                dread(T.fd_d, bw, G.len_w, G.off_w) == (ssize_t) G.len_w;   // larry: else buffered
                if (G.len_w > sizeof bw || G.len_s > sizeof bs ||
                    (!dw && tread(T.fd_w, T.rm_w, bw, G.len_w, G.off_w) != (ssize_t) G.len_w) ||
                    (G.len_s && tread(T.fd_s, T.rm_s, bs, G.len_s, G.off_s) != (ssize_t) G.len_s)) { bp->bad.fetch_add(1); continue; }
                bp->bytes.fetch_add((uint64_t) G.len_w + G.len_s);
                const auto & uq = bp->uniq[(size_t) G.e];
                const auto & pk = bp->packed[(size_t) G.e];""")
s = rep(s, """    if (tables_.empty() || n <= 0 || nthreads_ <= 0) return nullptr;
    auto b = std::make_shared<Batch>();
    b->n = n;
    // hashing / sorting""", """    if (tables_.empty() || n <= 0 || nthreads_ <= 0) return nullptr;
    pf_lo_ = pos0;   // larry, STRATA_ENGRAM_DIRECT: this batch's reads stay buffered (the prefetch warms its pages)
    pf_hi_ = pos0 + n;
    auto b = std::make_shared<Batch>();
    b->n = n;
    // hashing / sorting""")
open(H, "w").write(h)
open(C, "w").write(s)
print("patched", H, C)
