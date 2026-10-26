#!/usr/bin/env python3
# patch-tpevict2.py [TIER_CPP] - larry, after patch-tpevict.py: the evictable experts' host copies are made when they are
# evicted, not at load.  At load they took ~9 GiB more of the arena (88.4 GiB): MemAvailable fell to ~10 GiB, larry
# swapped, and a fresh 1M-context agent replay's prompts ran 147 s (99 s on the 262K config) - the cards' copies were
# only needed past ~262K tokens of context.  Now evict_slabs reads the slab's experts from the model file into a
# pageable buffer first (read_all, 8 threads: ~3 GiB a card, once), they become arena experts, and only then is the
# slab freed; with less than STRATA_XE_TP_EVICT_MEM_GIB (4) of MemAvailable left over it evicts nothing (the flex
# region gives the room, as before).  The free-VRAM poll after a free stops once the reading stops rising.  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "evict_bufs" in s:
    print("already patched"); sys.exit(0)
if "STRATA_XE_TP_EVICT" not in s:
    sys.exit("patch-tpevict.py first")


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:90]))
    s = s.replace(old, new)


rep("""    std::vector<size_t> evict_from;
    int64_t evicted_n = 0;
""", """    std::vector<size_t> evict_from;
    int64_t evicted_n = 0;
    std::vector<uint8_t*> evict_bufs;  ///< the evicted experts' host copies (read from the model file at eviction)
""")

# no arena copies at load
rep("""    // larry, engine1, STRATA_XE_TP_EVICT: the cards' last slabs over the deferred VRAM beyond the flex's share are
    // evictable - their experts get arena copies too (after the arena's own experts: pageable)
""", """    // larry, engine1, STRATA_XE_TP_EVICT: the cards' last slabs over the deferred VRAM beyond the flex's share are
    // evictable - their host copies are read from the model file when they are evicted (evict_slabs)
""")
rep("""        im.evict_from[ci] = start[start.size() - nslab];
        for (size_t i = im.evict_from[ci]; i < on[ci].size(); ++i) evictable.push_back(on[ci][i]);
    }
""", """        im.evict_from[ci] = start[start.size() - nslab];
    }
""")
rep("""    std::vector<std::pair<int32_t, int32_t>> evictable;
""", "")
rep("""    const size_t n_rest = rest.size();
    rest.insert(rest.end(), evictable.begin(), evictable.end());   // STRATA_XE_TP_EVICT: their copies, last
    for (size_t ri = 0; ri < rest.size(); ++ri) {
        const auto& le = rest[ri];
        const int64_t b = (int64_t) im.bbytes(le.first, le.second);
        if (abytes + b > cap) { if (ri < n_rest) ++im.file_n; continue; }
""", """    for (const auto& le : rest) {
        const int64_t b = (int64_t) im.bbytes(le.first, le.second);
        if (abytes + b > cap) { ++im.file_n; continue; }
""")
rep("""        bool all = true;
        int64_t eb = 0;
        for (size_t i = im.evict_from[ci]; i < im.card_set[ci].size(); ++i) {
            const auto& le = im.card_set[ci][i];
            all = all && im.aptr[(size_t) im.key(le.first, le.second)];
            eb += (int64_t) im.bbytes(le.first, le.second);
        }
        if (!all) {
            std::fprintf(stderr, "xe tier: card %d: its evictable experts have no arena copies (RAM cap) - none evictable\\n", cp->idx);
            im.evict_from[ci] = im.card_set[ci].size();
            continue;
        }
""", """        int64_t eb = 0;
        for (size_t i = im.evict_from[ci]; i < im.card_set[ci].size(); ++i)
            eb += (int64_t) im.bbytes(im.card_set[ci][i].first, im.card_set[ci][i].second);
""")

# evict_slabs: read the slab's experts from the file first; memory guard; the poll
rep("""        if (k < from) break;     // the slab holds experts without arena copies
        for (size_t i = k; i < set.size(); ++i) {
            const auto& le = set[i];
            const size_t key = (size_t) im.key(le.first, le.second);
            im.card_of[key] = -1;
""", """        if (k < from) break;     // the slab holds experts that are not evictable
        // their host copies first, from the model file (pageable; the arena's own blobs stay where they are)
        std::vector<std::pair<int32_t, int32_t>> list(set.begin() + (std::ptrdiff_t) k, set.end());
        std::vector<size_t> off(list.size());
        size_t tot = 0;
        for (size_t i = 0; i < list.size(); ++i) { off[i] = tot; tot += (im.bbytes(list[i].first, list[i].second) + 4095) & ~(size_t) 4095; }
        const int64_t mem_keep = (int64_t) (env_d("STRATA_XE_TP_EVICT_MEM_GIB", 4.0) * 1073741824.0);
        const int64_t avail = mem_available_bytes();
        if (avail > 0 && avail - (int64_t) tot < mem_keep) {
            std::fprintf(stderr, "xe tier: card %d: no eviction - %.2f GiB of host memory available, %.2f GiB needed\\n", c.idx,
                         avail / 1073741824.0, tot / 1073741824.0);
            break;
        }
        uint8_t* hb = (uint8_t*) std::aligned_alloc(4096, tot);
        if (!hb) break;
        read_all(im, list, 8, [&](size_t i, const uint8_t* buf) { std::memcpy(hb + off[i], buf, im.bbytes(list[i].first, list[i].second)); });
        im.evict_bufs.push_back(hb);
        for (size_t i = 0; i < list.size(); ++i) {
            const size_t key = (size_t) im.key(list[i].first, list[i].second);
            im.aptr[key] = hb + off[i];
            im.apin[key] = 0;
        }
        im.arena_bytes += (int64_t) tot;
        im.arena_n += (int64_t) list.size();
        for (size_t i = k; i < set.size(); ++i) {
            const auto& le = set[i];
            const size_t key = (size_t) im.key(le.first, le.second);
            im.card_of[key] = -1;
""")
rep("""    // the driver may report the freed pages a little later
    for (int i = 0; i < 100 && c.free_vram() < need; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
""", """    // the driver may report the freed pages a little later: until it does or the reading stops rising
    for (int i = 0, same = 0; i < 100 && c.free_vram() < need && same < 3; ++i) {
        const int64_t f0 = c.free_vram();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        same = c.free_vram() == f0 ? same + 1 : 0;
    }
""")
# close: free the copies with the arena
rep("""    for (uint8_t* p : im.achunks) {""", """    for (uint8_t* p : im.evict_bufs) std::free(p);   // STRATA_XE_TP_EVICT: the evicted experts' copies
    im.evict_bufs.clear();
    for (uint8_t* p : im.achunks) {""")
open(P, "w").write(s)
print("patched", P)
