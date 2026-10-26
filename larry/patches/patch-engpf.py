#!/usr/bin/env python3
# patch-engpf.py - larry: Engram reads off the prompt's critical path.
#  * Ds41Engram::prefetch: the NEXT prompt batch's rows read into the row cache (no output) on the I/O pool while this
#    batch computes - a 16K batch's reads otherwise stalled every card at its first Engram layer (~0.9 s a 31.6K prompt).
#  * start(): a prompt batch's cached rows are dequantized by the I/O pool with the reads, not on the caller (once the
#    next batch is prefetched ~all of its rows are cached: ~0.8 GB of f32 written by the dense thread otherwise).
#  * Ds41Dense::engram_prefetch (STRATA_ENG_PREFETCH, default 1), waited before the next start(); the generator calls
#    it after each layer-major batch's begin_batch.
# Idempotent.
import sys
EN = "/opt/ai/dsv41/xolotl/tools/v41/engine/"


def rep(path, old, new, cnt=1):
    s = open(path).read()
    if new.strip() in s:
        print(f"{path}: already"); return
    if s.count(old) != cnt: sys.exit(f"{path}: anchor x{s.count(old)}: {old[:90]!r}")
    open(path, "w").write(s.replace(old, new))
    print(f"{path}: patched")


H = EN + "ds41_engram.hpp"
rep(H, """    std::shared_ptr<Batch> start(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err);
    bool wait(const std::shared_ptr<Batch> & b, std::string & err);
""", """    std::shared_ptr<Batch> start(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err);
    bool wait(const std::shared_ptr<Batch> & b, std::string & err);
    /// larry: warm the row cache with the rows of positions pos0..pos0+n-1 (every slot; their tokens set): the reads
    /// queue on the I/O pool behind any batch already started, fill the cache (until full; past that they still warm
    /// the page cache) and write no output.  Null when nothing is left to read.  wait() it before the next start().
    std::shared_ptr<Batch> prefetch(int64_t pos0, int n, std::string & err);
""")

C = EN + "ds41_engram.cpp"
rep(C, """    std::shared_ptr<Task> read_task;
    std::atomic<int> bad { 0 };
    std::atomic<uint64_t> bytes { 0 };
};
""", """    std::shared_ptr<Task> read_task;
    std::atomic<int> bad { 0 };
    std::atomic<uint64_t> bytes { 0 };
    // larry: a prompt batch's cached rows (slot, unique index, cache entry), dequantized by the pool with the reads
    struct CachedRow { uint32_t e, k, idx; };
    std::vector<CachedRow> cached;
};
""")
rep(C, """        // row cache: a cached row is dequantized into its slots now and never read; groups span uncached rows only
        std::vector<char> hit(u, 0);
        if (cache_cap_) {
            float * oe = out[(size_t) e].data();
            for (size_t k = 0; k < u; ++k) {
                if ((uint64_t) uq[k] >= (uint64_t) T.rows) continue;
                auto it = cache_idx_.find(((uint64_t) e << 32) | uq[k]);
                if (it == cache_idx_.end()) { ++cache_miss_; continue; }
                ++cache_hits_;
                hit[k] = 1;
""", """        // row cache: a cached row is never read - dequantized into its slots now, or (a prompt batch) by the pool with
        // the reads; groups span uncached rows only
        std::vector<char> hit(u, 0);
        if (cache_cap_) {
            std::lock_guard<std::mutex> lk(cache_mu_);   // larry: a prefetch's reads may still be filling it
            float * oe = out[(size_t) e].data();
            for (size_t k = 0; k < u; ++k) {
                if ((uint64_t) uq[k] >= (uint64_t) T.rows) continue;
                auto it = cache_idx_.find(((uint64_t) e << 32) | uq[k]);
                if (it == cache_idx_.end()) { ++cache_miss_; continue; }
                ++cache_hits_;
                hit[k] = 1;
                if (pool_dq) { b->cached.push_back({ (uint32_t) e, (uint32_t) k, it->second }); continue; }
""")
rep(C, """    auto b = std::make_shared<Batch>();
    b->n = n;
    b->out = &out;
    const int64_t cols = cols_, dim = dim_, srsz = dim / 32, m = n * cols;
""", """    auto b = std::make_shared<Batch>();
    b->n = n;
    b->out = &out;
    const int64_t cols = cols_, dim = dim_, srsz = dim / 32, m = n * cols;
    const bool pool_dq = nthreads_ > 0 && n > 64;   // larry: a prompt batch's cached rows go to the pool
""")
rep(C, """    if (!b->groups.empty()) {
        auto t = std::make_shared<Task>();
        t->n = (int64_t) b->groups.size();
        Batch * bp = b.get();
        t->fn = [this, bp, dim, srsz](int64_t a, int64_t c) {
            uint8_t bw[4096 + 512], bs[1024];
            auto & outv = *bp->out;
            for (int64_t g = a; g < c; ++g) {
                const ReadGroup & G = bp->groups[(size_t) g];""", """    const int64_t ng = (int64_t) b->groups.size();
    const int64_t kCachedChunk = 512;   // cached rows per pool item
    const int64_t nc = ((int64_t) b->cached.size() + kCachedChunk - 1) / kCachedChunk;
    if (ng + nc > 0) {
        auto t = std::make_shared<Task>();
        t->n = ng + nc;
        Batch * bp = b.get();
        t->fn = [this, bp, dim, srsz, ng, kCachedChunk](int64_t a, int64_t c) {
            uint8_t bw[4096 + 512], bs[1024];
            auto & outv = *bp->out;
            for (int64_t g = a; g < c; ++g) {
                if (g >= ng) {   // larry: a chunk of cached rows, dequantized from the cache pool (never evicted)
                    const size_t q0 = (size_t) ((g - ng) * kCachedChunk);
                    const size_t q1 = std::min(bp->cached.size(), q0 + (size_t) kCachedChunk);
                    for (size_t q = q0; q < q1; ++q) {
                        const Batch::CachedRow & cr = bp->cached[q];
                        const auto & pk = bp->packed[cr.e];
                        const auto & rb = bp->run_begin[cr.e];
                        float * oe = outv[cr.e].data();
                        const uint8_t * w = &cache_pool_[(size_t) cr.idx * (size_t) (dim + srsz)];
                        const uint8_t * sc = w + dim;
                        float * o0 = oe + (size_t) (uint32_t) pk[rb[cr.k]] * (size_t) dim;
                        for (int64_t qq = 0; qq < srsz; ++qq) {
                            const float f = std::ldexp(1.0f, (int) sc[qq] - 127);
                            for (int64_t x = 0; x < 32; ++x) o0[qq * 32 + x] = fp8_e4m3(w[qq * 32 + x]) * f;
                        }
                        for (uint32_t p = rb[cr.k] + 1; p < rb[cr.k + 1]; ++p)
                            std::memcpy(oe + (size_t) (uint32_t) pk[p] * (size_t) dim, o0, (size_t) dim * sizeof(float));
                    }
                    continue;
                }
                const ReadGroup & G = bp->groups[(size_t) g];""")

# prefetch(): after wait()
rep(C, """bool Ds41Engram::rows(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err) {""",
    """std::shared_ptr<Ds41Engram::Batch> Ds41Engram::prefetch(int64_t pos0, int n, std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    // (a full row cache still gains: the reads warm the block device's page cache for the batch's own preads)
    if (nel == 0 || n <= 0 || nthreads_ <= 0) return nullptr;
    auto b = std::make_shared<Batch>();
    b->n = n;
    const int64_t cols = cols_, dim = dim_, srsz = dim / 32, m = n * cols;
    b->uniq.resize((size_t) nel);
    std::vector<int64_t> ids((size_t) m);
    std::vector<uint32_t> r((size_t) m);
    for (int64_t e = 0; e < nel; ++e) {
        const Table & T = tables_[(size_t) e];
        for (int64_t t = 0; t < n; ++t) hash(pos0 + t, (int) e, &ids[(size_t) (t * cols)]);
        for (int64_t i = 0; i < m; ++i) r[(size_t) i] = (uint32_t) ids[(size_t) i];
        std::sort(r.begin(), r.end());
        auto & uq = b->uniq[(size_t) e];
        {
            std::lock_guard<std::mutex> lk(cache_mu_);
            for (int64_t i = 0; i < m; ++i) {
                const uint32_t x = r[(size_t) i];
                if ((i > 0 && r[(size_t) i - 1] == x) || (uint64_t) x >= (uint64_t) T.rows) continue;
                if (!cache_idx_.count(((uint64_t) e << 32) | x)) uq.push_back(x);
            }
        }
        const uint64_t bw = T.off_w, bs = T.off_s;
        size_t i = 0;
        while (i < uq.size()) {   // page-coalesced, as start()
            const uint64_t page = (bw + (uint64_t) uq[i] * (uint64_t) dim) & ~(uint64_t) 4095;
            size_t j = i + 1;
            while (j < uq.size() && ((bw + (uint64_t) uq[j] * (uint64_t) dim) & ~(uint64_t) 4095) == page) ++j;
            const uint64_t w0 = bw + (uint64_t) uq[i] * (uint64_t) dim;
            const uint64_t w1 = bw + (uint64_t) uq[j - 1] * (uint64_t) dim + (uint64_t) dim;
            const uint64_t s0 = bs + (uint64_t) uq[i] * (uint64_t) srsz;
            const uint64_t s1 = bs + (uint64_t) uq[j - 1] * (uint64_t) srsz + (uint64_t) srsz;
            b->groups.push_back(ReadGroup{ e, w0, s0, (uint32_t) (w1 - w0), (uint32_t) (s1 - s0), (uint32_t) i, (uint32_t) (j - i) });
            i = j;
        }
    }
    if (b->groups.empty()) return nullptr;
    auto t = std::make_shared<Task>();
    t->n = (int64_t) b->groups.size();
    Batch * bp = b.get();
    t->fn = [this, bp, dim, srsz](int64_t a, int64_t c) {
        uint8_t bw[4096 + 512], bs[1024];
        for (int64_t g = a; g < c; ++g) {
            const ReadGroup & G = bp->groups[(size_t) g];
            const Table & T = tables_[(size_t) G.e];
            if (G.len_w > sizeof bw || G.len_s > sizeof bs ||
                tread(T.fd_w, T.rm_w, bw, G.len_w, G.off_w) != (ssize_t) G.len_w ||
                tread(T.fd_s, T.rm_s, bs, G.len_s, G.off_s) != (ssize_t) G.len_s) { bp->bad.fetch_add(1); continue; }
            bp->bytes.fetch_add((uint64_t) G.len_w + G.len_s);
            const auto & uq = bp->uniq[(size_t) G.e];
            const uint32_t r0 = uq[G.u0];
            std::lock_guard<std::mutex> lk(cache_mu_);
            for (uint32_t k = 0; k < G.count && cache_used_ < cache_cap_; ++k) {
                const uint32_t row = uq[G.u0 + k];
                if (!cache_idx_.emplace(((uint64_t) G.e << 32) | row, (uint32_t) cache_used_).second) continue;
                uint8_t * d = &cache_pool_[cache_used_ * (size_t) (dim + srsz)];
                std::memcpy(d, bw + (uint64_t) (row - r0) * (uint64_t) dim, (size_t) dim);
                std::memcpy(d + dim, bs + (uint64_t) (row - r0) * (uint64_t) srsz, (size_t) srsz);
                ++cache_used_;
            }
        }
    };
    b->read_task = t;
    fire(t, std::min<int64_t>(t->n, 8 * nthreads_));
    (void) err;
    return b;
}

bool Ds41Engram::rows(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err) {""")

DH = EN + "ds41_dense.hpp"
rep(DH, """    bool begin_batch(const int * tids, int B, int pos0);
""", """    bool begin_batch(const int * tids, int B, int pos0);
    /// larry: the NEXT layer-major batch's Engram rows (tokens tids[0..n) at positions pos0..) read into the Engram row
    /// cache while this batch computes (STRATA_ENG_PREFETCH, default 1); the next pass's inputs wait for it first.
    void engram_prefetch(const int * tids, int n, int pos0);
""")
D = EN + "ds41_dense.cpp"
rep(D, """    std::shared_ptr<Ds41Engram::Batch> eng_batch;
""", """    std::shared_ptr<Ds41Engram::Batch> eng_batch;
    std::shared_ptr<Ds41Engram::Batch> eng_pf;   // larry: the next batch's rows being read into the row cache
""")
rep(D, """    if (im.engram.enabled()) {
        if (!engram_ready(im)) return false;    // flush a still-pending batch from a previous pass first
        for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);""", """    if (im.engram.enabled()) {
        if (!engram_ready(im)) return false;    // flush a still-pending batch from a previous pass first
        if (im.eng_pf) {   // larry: the prefetched rows are in the cache once it is done (start() reads the cache)
            std::string e2;
            im.engram.wait(im.eng_pf, e2);   // best effort: a failed read is read again below
            im.eng_pf.reset();
        }
        for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);""")
rep(D, """bool Ds41Dense::begin_batch(const int * tids, int B, int pos0) {""", """void Ds41Dense::engram_prefetch(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    static const bool on = env_on("STRATA_ENG_PREFETCH", true);
    if (!on || !im.engram.enabled() || n <= 64 || im.eng_pf || pos0 < 0 || pos0 + n > im.cfg.ctx_max) return;
    for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);   // rewritten by that batch's own inputs
    std::string e2;
    im.eng_pf = im.engram.prefetch(pos0, n, e2);
}

bool Ds41Dense::begin_batch(const int * tids, int B, int pos0) {""")
G = EN + "ds41_generate.cpp"
rep(G, """                // enc-skip: columns whose global position is before `pos0 + n - ER` are phase A (layers past layer 20""",
    """                // larry: the next batch's Engram rows go into the row cache while this one computes
                if (c0 + B < n) dense.engram_prefetch(ids + c0 + B, std::min(pb_cap, n - c0 - B), pos0 + c0 + B);
                // enc-skip: columns whose global position is before `pos0 + n - ER` are phase A (layers past layer 20""")
