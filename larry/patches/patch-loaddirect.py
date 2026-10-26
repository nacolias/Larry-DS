#!/usr/bin/env python3
# patch-loaddirect.py [XOLOTL_DIR] - larry, 2026-10-26: STRATA_XE_LOAD_DIRECT (tier, default 1; 0 = the old buffered
# reads).  The tier's bulk reads (read_all: the experts placed in VRAM, the RAM arena's, the replicas) took the blob
# source's buffered read() - ~150 GB of the model file through the page cache at every start (memtrace3.bt, abwT138:
# 182-186 GB of page-cache insertions by the engine while it loaded, 79.5M pages of them the GGUF).  On a box whose
# RAM the engine fills that flood pushed the arena's own pages out to swap (a first 8K prompt then swapped 7.2 GB back
# in: 17 s) and had the xe shrinker back up the engine's buffers (49 GB written out and read back in one load).  Now
# they go through the source's O_DIRECT twin (read_staged, the prefetch's path: per-thread aligned scratch), which
# also reads faster (4.65 vs 2.25 GB/s on the scattered expert pattern, ds41_moe.cpp).  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/xe/ds41_xe_tier.cpp")
s = open(P).read()
if "STRATA_XE_LOAD_DIRECT" in s:
    print("already patched", P)
    sys.exit(0)
old = """template <class Sink>
void read_all(Ds41XeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& list, int nth, Sink sink) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < nth; ++i)
        th.emplace_back([&]() {
            uint8_t* buf = sycl::malloc_host<uint8_t>((size_t) im.blob_max, im.ctx);
            for (;;) {
                const size_t k = next.fetch_add(1);
                if (k >= list.size()) break;
                im.blobs->read(list[k].first, list[k].second, buf);
                sink(k, buf);
            }
            sycl::free(buf, im.ctx);
        });
    for (auto& t : th) t.join();
}
"""
new = """template <class Sink>
void read_all(Ds41XeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& list, int nth, Sink sink) {
    // larry, STRATA_XE_LOAD_DIRECT (default 1; patch-loaddirect.py): through the source's O_DIRECT twin (read_staged,
    // a per-thread aligned scratch) - the buffered read() put ~150 GB through the page cache at every start, which
    // swapped the arena out and had the xe shrinker back up the engine's buffers (abwT138)
    static const bool direct = env_i("STRATA_XE_LOAD_DIRECT", 1) != 0;
    const size_t sc = direct ? (im.blobs->read_staged_scratch() + 4095) & ~(size_t) 4095 : 0;
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < nth; ++i)
        th.emplace_back([&]() {
            uint8_t* buf = sycl::malloc_host<uint8_t>((size_t) im.blob_max, im.ctx);
            uint8_t* scr = sc ? (uint8_t*) std::aligned_alloc(4096, sc) : nullptr;
            for (;;) {
                const size_t k = next.fetch_add(1);
                if (k >= list.size()) break;
                if (scr) im.blobs->read_staged(list[k].first, list[k].second, buf, scr, sc);
                else im.blobs->read(list[k].first, list[k].second, buf);
                sink(k, buf);
            }
            std::free(scr);
            sycl::free(buf, im.ctx);
        });
    for (auto& t : th) t.join();
}
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
