#!/usr/bin/env python3
# patch-engdrop.py [XOLOTL_DIR] - larry, 2026-10-26: STRATA_ENGRAM_DROP=1 (engine, default 0): a prompt batch (> 64
# tokens) drops the Engram weight pages it has read from the page cache (posix_fadvise DONTNEED on the group's pages).
# A prompt's Engram reads put 12-14 GB into the page cache per 110K prompt tokens (memtrace3.bt, abwT138 / abwT139);
# on a box whose RAM the engine fills, every new page costs reclaim, and reclaim runs the xe shrinker, which writes
# the engine's own buffers out to swap (abwT139 d3: 1.8 GB backed up during its first prompts, a 1.6 s stall).  A
# weight page holds 15-16 random rows, read once a batch (the row cache keeps the rows); the prefetch's reads stay
# cached until the batch reads them, then go.  The scale pages (8 bytes a row, shared by 512 rows) stay cached.
# Decode passes (<= 64 tokens) keep the buffered path.  RawMap tables (root only: /dev/dm-0) are not dropped.
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_engram.cpp")
s = open(P).read()
if "STRATA_ENGRAM_DROP" in s:
    print("already patched", P)
    sys.exit(0)
pairs = [(
"""Ds41Engram::~Ds41Engram() {""",
"""// larry, STRATA_ENGRAM_DROP (patch-engdrop.py): a prompt batch drops the weight pages it read from the page cache
static bool engram_drop_on() {
    static const bool on = std::getenv("STRATA_ENGRAM_DROP") && std::atoi(std::getenv("STRATA_ENGRAM_DROP")) != 0;
    return on;
}
static void page_drop(int fd, uint64_t off, size_t n) {
    const uint64_t a0 = off & ~(uint64_t) 4095, a1 = (off + n + 4095) & ~(uint64_t) 4095;
    ::posix_fadvise(fd, (off_t) a0, (off_t) (a1 - a0), POSIX_FADV_DONTNEED);
}

Ds41Engram::~Ds41Engram() {"""), (
"""    const bool direct = engram_direct_on() && n > 64 && !(pf_lo_ >= 0 && pos0 >= pf_lo_ && pos0 + n <= pf_hi_);
""",
"""    const bool direct = engram_direct_on() && n > 64 && !(pf_lo_ >= 0 && pos0 >= pf_lo_ && pos0 + n <= pf_hi_);
    const bool drop = engram_drop_on() && n > 64;   // larry, STRATA_ENGRAM_DROP
"""), (
"""        t->fn = [this, bp, dim, srsz, g0, ng, c0, c1, kCachedChunk, direct](int64_t a, int64_t c) {""",
"""        t->fn = [this, bp, dim, srsz, g0, ng, c0, c1, kCachedChunk, direct, drop](int64_t a, int64_t c) {"""), (
"""                    (G.len_s && tread(T.fd_s, T.rm_s, bs, G.len_s, G.off_s) != (ssize_t) G.len_s)) { bp->bad.fetch_add(1); continue; }
                bp->bytes.fetch_add((uint64_t) G.len_w + G.len_s);
                const auto & uq = bp->uniq[(size_t) G.e];
                const auto & pk = bp->packed[(size_t) G.e];""",
"""                    (G.len_s && tread(T.fd_s, T.rm_s, bs, G.len_s, G.off_s) != (ssize_t) G.len_s)) { bp->bad.fetch_add(1); continue; }
                if (drop && !dw && !T.rm_w) page_drop(T.fd_w, G.off_w, G.len_w);   // larry, STRATA_ENGRAM_DROP
                bp->bytes.fetch_add((uint64_t) G.len_w + G.len_s);
                const auto & uq = bp->uniq[(size_t) G.e];
                const auto & pk = bp->packed[(size_t) G.e];""")]
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:80]))
    s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
