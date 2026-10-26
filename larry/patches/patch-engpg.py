#!/usr/bin/env python3
# patch-engpg.py [ENGINE_DIR] - larry, two changes to the Engram row reader (ds41_engram.cpp / .hpp):
#  (1) paged tables: engram-L<layer>.epg in any --engram dir is used before the safetensors shard - the table as 4 KiB
#      pages of 15 rows, each its 256 weight bytes, its 8 scale bytes and 8 zero bytes (bench/engram/engram_epg.py
#      writes it from the shard; the shard keeps the scales and the weights 98 GB apart: two random reads a row, one
#      here).  A 16K-token prompt's first Engram layer waited 2.4-3.0 s for ~600K reads on the slower NVMe.
#  (2) a read task per Engram slot, fired in slot order, and wait_slot(): the dense half waits for the first Engram
#      layer's rows only at layer 1 (the second layer's are read meanwhile and waited at layer 14).
# Idempotent.
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine"
H, C = D + "/ds41_engram.hpp", D + "/ds41_engram.cpp"
h, c = open(H).read(), open(C).read()
if "wait_slot" in h:
    print("already patched"); sys.exit(0)


def rep(s, old, new, n=1):
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:90]))
    return s.replace(old, new)


# ---------------------------------------------------------------- hpp
h = rep(h, """// rows_reference() keeps the original per-row random reader so the bench can diff the two byte for byte.
""", """// rows_reference() keeps the original per-row random reader so the bench can diff the two byte for byte.
// larry: a table may instead be an engram-L<layer>.epg (bench/engram/engram_epg.py: 4 KiB pages of 15 rows of weights +
// scales) - one read a row - found in the same dirs before the shards.
""")
h = rep(h, """    bool wait(const std::shared_ptr<Batch> & b, std::string & err);
""", """    bool wait(const std::shared_ptr<Batch> & b, std::string & err);
    /// larry: wait for slot e's rows of a start()ed batch only (each slot's reads are their own task, queued in slot
    /// order: the first Engram layer's rows are read first).  wait() still waits for the whole batch.
    bool wait_slot(const std::shared_ptr<Batch> & b, int e, std::string & err);
""")
h = rep(h, """    struct Table { int fd_w = -1, fd_s = -1; uint64_t off_w = 0, off_s = 0; int64_t rows = 0;
                   std::shared_ptr<RawMap> rm_w, rm_s; };
""", """    struct Table { int fd_w = -1, fd_s = -1; uint64_t off_w = 0, off_s = 0; int64_t rows = 0;
                   std::shared_ptr<RawMap> rm_w, rm_s;
                   // larry: a paged table (engram-L<layer>.epg on fd_w / rm_w): after a header page, 15 rows of 272 bytes
                   // (dim weight bytes, dim / 32 scale bytes, zeros) a 4 KiB page - a row's scales follow its weights
                   bool paged = false;
                   uint64_t row_off(uint64_t r, uint64_t dim) const { return paged ? 4096 * (1 + r / 15) + 272 * (r % 15) : off_w + r * dim; }
                   uint64_t page_of(uint64_t r, uint64_t dim) const { return paged ? r / 15 : (off_w + r * dim) >> 12; }
                   uint64_t stride(uint64_t dim) const { return paged ? 272 : dim; } };
""")

# ---------------------------------------------------------------- cpp: open()
c = rep(c, """    if (index.empty()) { err = "engram: no model.safetensors.index.json in the given dirs"; return false; }
    for (int64_t e = 0; e < nel; ++e) {
        const std::string base = "layers." + std::to_string(g.engram_layers[(size_t) e]) + ".engram.embed.";
        Table t;
""", """    for (int64_t e = 0; e < nel; ++e) {
        // larry: a paged copy of the table (engram-L<layer>.epg) in any of the dirs first
        std::string pp;
        for (const auto & d : dirs) {
            const std::string cand = d + "/engram-L" + std::to_string(g.engram_layers[(size_t) e]) + ".epg";
            if (exists(cand)) { pp = cand; break; }
        }
        if (!pp.empty()) {
            const int fd = ::open(pp.c_str(), O_RDONLY);
            if (fd < 0) { err = "engram: cannot open " + pp; return false; }
            uint8_t hd[40] = {};
            uint64_t prow = 0;
            uint32_t f[6] = {};   // row bytes, rows a page, weight bytes, scale bytes, page bytes, layer
            struct stat st;
            const bool okh = ::pread(fd, hd, sizeof hd, 0) == (ssize_t) sizeof hd && ::fstat(fd, &st) == 0;
            std::memcpy(&prow, hd + 8, 8);
            std::memcpy(f, hd + 16, sizeof f);
            if (!okh || std::memcmp(hd, "EPG1\\0\\0\\0\\0", 8) != 0 || prow != (uint64_t) g.engram_rows[(size_t) e] || f[0] != 272 ||
                f[1] != 15 || f[2] != (uint32_t) dim_ || f[3] != (uint32_t) (dim_ / 32) || f[4] != 4096 ||
                f[5] != (uint32_t) g.engram_layers[(size_t) e] || (uint64_t) st.st_size != 4096ull * (1 + (prow + 14) / 15)) {
                ::close(fd);
                err = "engram: " + pp + " is not a paged table of this model's layer " + std::to_string(g.engram_layers[(size_t) e]);
                return false;
            }
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
            Table t;
            t.paged = true;
            t.fd_w = fd;
            t.rows = (int64_t) prow;
            t.rm_w = raw_map(pp, fd);
            tables_.push_back(t);
            std::fprintf(stderr, "engram: layer %d: paged table %s (a read a row)\\n", (int) g.engram_layers[(size_t) e], pp.c_str());
            continue;
        }
        if (index.empty()) { err = "engram: no model.safetensors.index.json in the given dirs"; return false; }
        const std::string base = "layers." + std::to_string(g.engram_layers[(size_t) e]) + ".engram.embed.";
        Table t;
""")

# ---------------------------------------------------------------- cpp: rows_reference()
c = rep(c, """                ::pread(T.fd_w, w, (size_t) dim_, (off_t) (T.off_w + (uint64_t) r * (uint64_t) dim_)) != (ssize_t) dim_ ||
                ::pread(T.fd_s, s, (size_t) (dim_ / 32), (off_t) (T.off_s + (uint64_t) r * (uint64_t) (dim_ / 32))) != (ssize_t) (dim_ / 32)) {
""", """                ::pread(T.fd_w, w, (size_t) dim_, (off_t) T.row_off((uint64_t) r, (uint64_t) dim_)) != (ssize_t) dim_ ||
                ::pread(T.paged ? T.fd_w : T.fd_s, s, (size_t) (dim_ / 32),
                        (off_t) (T.paged ? T.row_off((uint64_t) r, (uint64_t) dim_) + (uint64_t) dim_
                                         : T.off_s + (uint64_t) r * (uint64_t) (dim_ / 32))) != (ssize_t) (dim_ / 32)) {
""")

# ---------------------------------------------------------------- cpp: Batch (a task per slot)
c = rep(c, """    std::shared_ptr<Task> read_task;
""", """    std::shared_ptr<Task> read_task;                  // prefetch(): every slot's reads
    std::vector<std::shared_ptr<Task>> slot_tasks;    // larry, start(): slot e's reads + cached rows
""")

# ---------------------------------------------------------------- cpp: start_into() groups (paged) + slot ranges
c = rep(c, """    b->uniq.resize((size_t) nel); b->packed.resize((size_t) nel); b->run_begin.resize((size_t) nel);
    for (int64_t e = 0; e < nel; ++e) {
        if (!b->outp[(size_t) e]) continue;   // larry, engine1: this slot not wanted by this call (outp[e] null)
""", """    b->uniq.resize((size_t) nel); b->packed.resize((size_t) nel); b->run_begin.resize((size_t) nel);
    std::vector<int64_t> gb((size_t) nel, 0), ge((size_t) nel, 0), cb((size_t) nel, 0), ce((size_t) nel, 0);   // larry: slot ranges
    for (int64_t e = 0; e < nel; ++e) {
        gb[(size_t) e] = ge[(size_t) e] = (int64_t) b->groups.size();
        cb[(size_t) e] = ce[(size_t) e] = (int64_t) b->cached.size();
        if (!b->outp[(size_t) e]) continue;   // larry, engine1: this slot not wanted by this call (outp[e] null)
""")
c = rep(c, """        size_t i = 0;
        while (i < u) {
            if ((uint64_t) uq[i] >= (uint64_t) T.rows || hit[i]) { ++i; continue; }   // out of range (bad set) / cached
            const uint64_t pw = bw + (uint64_t) uq[i] * (uint64_t) dim;
            const uint64_t page = pw & ~(uint64_t) 4095;
            size_t j = i + 1;
            while (j < u && (uint64_t) uq[j] < (uint64_t) T.rows && !hit[j] &&
                   ((bw + (uint64_t) uq[j] * (uint64_t) dim) & ~(uint64_t) 4095) == page) ++j;
            const uint64_t w0 = bw + (uint64_t) uq[i] * (uint64_t) dim;
            const uint64_t w1 = bw + (uint64_t) uq[j - 1] * (uint64_t) dim + (uint64_t) dim;
            const uint64_t s0 = bs + (uint64_t) uq[i] * (uint64_t) srsz;
            const uint64_t s1 = bs + (uint64_t) uq[j - 1] * (uint64_t) srsz + (uint64_t) srsz;
            b->groups.push_back(ReadGroup{ e, w0, s0, (uint32_t) (w1 - w0), (uint32_t) (s1 - s0), (uint32_t) i, (uint32_t) (j - i) });
            i = j;
        }
    }
""", """        size_t i = 0;
        (void) bw;
        while (i < u) {
            if ((uint64_t) uq[i] >= (uint64_t) T.rows || hit[i]) { ++i; continue; }   // out of range (bad set) / cached
            const uint64_t page = T.page_of(uq[i], (uint64_t) dim);
            size_t j = i + 1;
            while (j < u && (uint64_t) uq[j] < (uint64_t) T.rows && !hit[j] && T.page_of(uq[j], (uint64_t) dim) == page) ++j;
            // larry: a paged table's span holds the rows' scales too (len_s 0: no second read)
            const uint64_t w0 = T.row_off(uq[i], (uint64_t) dim);
            const uint64_t w1 = T.row_off(uq[j - 1], (uint64_t) dim) + (uint64_t) (T.paged ? dim + srsz : dim);
            const uint64_t s0 = T.paged ? 0 : bs + (uint64_t) uq[i] * (uint64_t) srsz;
            const uint64_t s1 = T.paged ? 0 : bs + (uint64_t) uq[j - 1] * (uint64_t) srsz + (uint64_t) srsz;
            b->groups.push_back(ReadGroup{ e, w0, s0, (uint32_t) (w1 - w0), (uint32_t) (s1 - s0), (uint32_t) i, (uint32_t) (j - i) });
            i = j;
        }
        ge[(size_t) e] = (int64_t) b->groups.size();
        ce[(size_t) e] = (int64_t) b->cached.size();
    }
""")

# ---------------------------------------------------------------- cpp: start_into() tasks: one per slot
old_task_head = """    const int64_t ng = (int64_t) b->groups.size();
    const int64_t kCachedChunk = 512;   // cached rows per pool item
    const int64_t nc = ((int64_t) b->cached.size() + kCachedChunk - 1) / kCachedChunk;
    if (ng + nc > 0) {
        auto t = std::make_shared<Task>();
        t->n = ng + nc;
        Batch * bp = b.get();
        t->fn = [this, bp, dim, srsz, ng, kCachedChunk](int64_t a, int64_t c) {
            uint8_t bw[4096 + 512], bs[1024];
            for (int64_t g = a; g < c; ++g) {
                if (g >= ng) {   // larry: a chunk of cached rows, dequantized from the cache pool (never evicted)
                    const size_t q0 = (size_t) ((g - ng) * kCachedChunk);
                    const size_t q1 = std::min(bp->cached.size(), q0 + (size_t) kCachedChunk);
"""
new_task_head = """    const int64_t kCachedChunk = 512;   // cached rows per pool item
    b->slot_tasks.assign((size_t) nel, nullptr);
    // larry: a task per slot (its read groups, then its cached rows), fired in slot order - the FIFO pool reads the
    // first Engram layer's rows first, and wait_slot(b, 0) returns before the second layer's are read
    for (int64_t e = 0; e < nel; ++e) {
    const int64_t g0 = gb[(size_t) e], ng = ge[(size_t) e] - gb[(size_t) e];
    const int64_t c0 = cb[(size_t) e], c1 = ce[(size_t) e];
    const int64_t nc = (c1 - c0 + kCachedChunk - 1) / kCachedChunk;
    if (ng + nc > 0) {
        auto t = std::make_shared<Task>();
        t->n = ng + nc;
        Batch * bp = b.get();
        t->fn = [this, bp, dim, srsz, g0, ng, c0, c1, kCachedChunk](int64_t a, int64_t c) {
            uint8_t bw[4096 + 512], bs[1024];
            for (int64_t gi = a; gi < c; ++gi) {
                if (gi >= ng) {   // larry: a chunk of cached rows, dequantized from the cache pool (never evicted)
                    const size_t q0 = (size_t) (c0 + (gi - ng) * kCachedChunk);
                    const size_t q1 = std::min((size_t) c1, q0 + (size_t) kCachedChunk);
"""
c = rep(c, old_task_head, new_task_head)
c = rep(c, """                    continue;
                }
                const ReadGroup & G = bp->groups[(size_t) g];
                const Table & T = tables_[(size_t) G.e];
""", """                    continue;
                }
                const ReadGroup & G = bp->groups[(size_t) (g0 + gi)];
                const Table & T = tables_[(size_t) G.e];
""")
# both read sites (start_into, prefetch_run): a paged group has no scale read
c = rep(c, """tread(T.fd_s, T.rm_s, bs, G.len_s, G.off_s) != (ssize_t) G.len_s) { bp->bad.fetch_add(1); continue; }""",
        """(G.len_s && tread(T.fd_s, T.rm_s, bs, G.len_s, G.off_s) != (ssize_t) G.len_s)) { bp->bad.fetch_add(1); continue; }""", 2)
c = rep(c, """                const uint32_t r0 = uq[G.u0];      // the group's first (lowest-offset) row: the buffer's origin
                for (uint32_t k = 0; k < G.count; ++k) {
                    const uint32_t uu = G.u0 + k;
                    const uint32_t row = uq[uu];
                    const uint8_t * w = bw + (uint64_t) (row - r0) * (uint64_t) dim;
                    const uint8_t * s = bs + (uint64_t) (row - r0) * (uint64_t) srsz;
""", """                const uint32_t r0 = uq[G.u0];      // the group's first (lowest-offset) row: the buffer's origin
                const uint64_t wst = T.stride((uint64_t) dim);
                for (uint32_t k = 0; k < G.count; ++k) {
                    const uint32_t uu = G.u0 + k;
                    const uint32_t row = uq[uu];
                    const uint8_t * w = bw + (uint64_t) (row - r0) * wst;
                    const uint8_t * s = T.paged ? w + dim : bs + (uint64_t) (row - r0) * (uint64_t) srsz;
""")
c = rep(c, """        b->read_task = t;
        if (nthreads_ <= 0 || t->n < 4) {            // tiny batch: run on the caller, no queue round-trip.  (Was n <= 2 *
                                                     // nthreads: a decode token's ~48 preads then ran serially, ~11 ms cold.)
            t->parts = 1;
            t->pending.store(1);
            t->fn(0, t->n);
            t->done();
        } else {
            fire(t, std::min<int64_t>(t->n, 8 * nthreads_));
        }
    }
    return b;
}
""", """        b->slot_tasks[(size_t) e] = t;
        if (nthreads_ <= 0 || t->n < 4) {            // tiny batch: run on the caller, no queue round-trip.  (Was n <= 2 *
                                                     // nthreads: a decode token's ~48 preads then ran serially, ~11 ms cold.)
            t->parts = 1;
            t->pending.store(1);
            t->fn(0, t->n);
            t->done();
        } else {
            fire(t, std::min<int64_t>(t->n, 8 * nthreads_));
        }
    }
    }
    return b;
}
""")

# ---------------------------------------------------------------- cpp: wait() / wait_slot()
c = rep(c, """    if (b->th.joinable()) b->th.join();   // a prefetch: its planner fired and waited its reads
    if (b->read_task) b->read_task->wait();
    reads_ += (uint64_t) b->groups.size();
""", """    if (b->th.joinable()) b->th.join();   // a prefetch: its planner fired and waited its reads
    if (b->read_task) b->read_task->wait();
    for (auto & t : b->slot_tasks) if (t) t->wait();
    reads_ += (uint64_t) b->groups.size();
""")
c = rep(c, """std::shared_ptr<Ds41Engram::Batch> Ds41Engram::prefetch(int64_t pos0, int n, std::string & err) {
""", """bool Ds41Engram::wait_slot(const std::shared_ptr<Batch> & b, int e, std::string & err) {
    if (!b) { err = "engram: null batch"; return false; }
    if (b->th.joinable() || b->read_task) return wait(b, err);   // a prefetch: no per-slot tasks
    if (e >= 0 && e < (int) b->slot_tasks.size() && b->slot_tasks[(size_t) e]) b->slot_tasks[(size_t) e]->wait();
    if (b->bad.load()) { err = "engram: " + std::to_string(b->bad.load()) + " row reads failed"; return false; }
    return true;
}

std::shared_ptr<Ds41Engram::Batch> Ds41Engram::prefetch(int64_t pos0, int n, std::string & err) {
""")

# ---------------------------------------------------------------- cpp: prefetch_run() groups + reads (paged)
c = rep(c, """        const uint64_t bw = T.off_w, bs = T.off_s;
        size_t i = 0;
        while (i < uq.size()) {   // page-coalesced, as start()
            const uint64_t page = (bw + (uint64_t) uq[i] * (uint64_t) dim) & ~(uint64_t) 4095;
            size_t j = i + 1;
            while (j < uq.size() && ((bw + (uint64_t) uq[j] * (uint64_t) dim) & ~(uint64_t) 4095) == page) ++j;
            const uint64_t w0 = bw + (uint64_t) uq[i] * (uint64_t) dim;
            const uint64_t w1 = bw + (uint64_t) uq[j - 1] * (uint64_t) dim + (uint64_t) dim;
            const uint64_t s0 = bs + (uint64_t) uq[i] * (uint64_t) srsz;
            const uint64_t s1 = bs + (uint64_t) uq[j - 1] * (uint64_t) srsz + (uint64_t) srsz;
""", """        const uint64_t bs = T.off_s;
        size_t i = 0;
        while (i < uq.size()) {   // page-coalesced, as start() (larry: a paged table's span has the scales too)
            const uint64_t page = T.page_of(uq[i], (uint64_t) dim);
            size_t j = i + 1;
            while (j < uq.size() && T.page_of(uq[j], (uint64_t) dim) == page) ++j;
            const uint64_t w0 = T.row_off(uq[i], (uint64_t) dim);
            const uint64_t w1 = T.row_off(uq[j - 1], (uint64_t) dim) + (uint64_t) (T.paged ? dim + srsz : dim);
            const uint64_t s0 = T.paged ? 0 : bs + (uint64_t) uq[i] * (uint64_t) srsz;
            const uint64_t s1 = T.paged ? 0 : bs + (uint64_t) uq[j - 1] * (uint64_t) srsz + (uint64_t) srsz;
""")
c = rep(c, """                std::memcpy(d, bw + (uint64_t) (row - r0) * (uint64_t) dim, (size_t) dim);
                std::memcpy(d + dim, bs + (uint64_t) (row - r0) * (uint64_t) srsz, (size_t) srsz);
""", """                const uint8_t * w = bw + (uint64_t) (row - r0) * T.stride((uint64_t) dim);
                std::memcpy(d, w, (size_t) dim);
                std::memcpy(d + dim, T.paged ? w + dim : bs + (uint64_t) (row - r0) * (uint64_t) srsz, (size_t) srsz);
""")

open(H, "w").write(h)
open(C, "w").write(c)
print("patched", H, C)
