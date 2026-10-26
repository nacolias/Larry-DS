#!/usr/bin/env python3
# patch-pinstream.py [TIER_CPP] - larry, STRATA_XE_PIN_STREAM=1 (default 0 until measured): the pinned arena chunks for
# what prompts STREAM.  The arena is laid out in ranked order and its first STRATA_XE_PINNED_GIB are pinned - which is
# the dense card's whole cache (21.25 GiB of the 23.95) plus a little.  With STRATA_XE_SEED_TAIL_LAYERS=L the cache's
# layers >= L stay on card 0 through a prompt batch (phase B's calls are < STRATA_XE_DENSE_OFFLOAD columns: card 0
# computes them) - pinned, they were only ever read by the seed.  Phase B streams the NEXT-ranked experts of layers
# >= L instead, all pageable: through the stagers' rings at ~14 GB/s (STRATA_XE_HOT_PROF: the stagers wait on the
# rings' DMAs more than they copy), where a pinned expert's DMA reads RAM straight (22-28 GB/s, no CPU copy).  Now the
# pinned budget holds the cache's layers < L (phase A streams them, the regrow refills them) and then the next-ranked
# experts of layers >= L; everything else keeps its ranked order.  The seed reads the cache's layers >= L through the
# bounce (~1.5 s more at load, and on a full re-seed after a batch over STRATA_SERVE_KEEP_MAX).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_PIN_STREAM" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    // ---- the arena: the rest, in ranked order, up to the RAM budget, in pinned chunks (no blob straddles one)
""", """    // larry, STRATA_XE_PIN_STREAM (see patch-pinstream.py): the arena's head re-ordered for the pinned budget - the
    // dense card's predicted cache (the ranked arena's first STRATA_XE_PIN_STREAM_CACHE_GIB, default its seed budget)
    // below layer L = STRATA_XE_SEED_TAIL_LAYERS, then the next-ranked experts of layers >= L while the budget lasts
    if (env_i("STRATA_XE_PIN_STREAM", 0) && im.dense >= 0) {
        const int tail_l = env_i("STRATA_XE_SEED_TAIL_LAYERS", 0);
        XeCard& dc = *im.cards[(size_t) im.dense];
        double cgib = env_d("STRATA_XE_PIN_STREAM_CACHE_GIB", 0.0);
        if (cgib <= 0) {   // dense_budget's caps as the seed will see them (the slot budget is set later, at the seed)
            cgib = env_d("STRATA_XE_DENSE_GIB", 10.0) + im.dense_bonus_gib;
            const int64_t fr = dc.free_vram();
            if (fr > 0) cgib = std::min(cgib, (fr + dc.used) / kGiB - env_d("STRATA_XE_DENSE_RESERVE_GIB", 8.0));
        }
        if (tail_l > 0 && tail_l < NL && cgib > 0) {
            const int64_t cbud = (int64_t) (cgib * kGiB), pbud = (int64_t) (env_d("STRATA_XE_PINNED_GIB", 16.0) * kGiB);
            std::vector<char> in_cache(rest.size(), 0), taken(rest.size(), 0);
            int64_t cu = 0;
            for (size_t i = 0; i < rest.size(); ++i) {   // seed_dense's walk: the next expert that still fits
                const int64_t b = (int64_t) im.bbytes(rest[i].first, rest[i].second);
                if (cu + b > cbud) continue;
                cu += b;
                in_cache[i] = 1;
            }
            std::vector<std::pair<int32_t, int32_t>> ord;
            ord.reserve(rest.size());
            int64_t pu = 0, n_lo = 0, n_hi = 0, b_hi = 0, n_kept = 0, b_kept = 0;
            for (size_t i = 0; i < rest.size(); ++i)
                if (in_cache[i] && rest[i].first < tail_l) {
                    ord.push_back(rest[i]);
                    taken[i] = 1;
                    pu += (int64_t) im.bbytes(rest[i].first, rest[i].second);
                    ++n_lo;
                }
            const int64_t b_lo = pu;
            for (size_t i = 0; i < rest.size() && pu < pbud; ++i)
                if (!in_cache[i] && rest[i].first >= tail_l) {
                    const int64_t b = (int64_t) im.bbytes(rest[i].first, rest[i].second);
                    ord.push_back(rest[i]);
                    taken[i] = 1;
                    pu += b;
                    b_hi += b;
                    ++n_hi;
                }
            for (size_t i = 0; i < rest.size(); ++i) {
                if (taken[i]) continue;
                if (in_cache[i]) { ++n_kept; b_kept += (int64_t) im.bbytes(rest[i].first, rest[i].second); }
                ord.push_back(rest[i]);
            }
            rest.swap(ord);
            std::fprintf(stderr, "xe tier: STRATA_XE_PIN_STREAM: the pinned arena = the dense card's cache below layer %d (%lld experts, "
                         "%.2f GiB), then %lld next-ranked experts of layers %d.. (%.2f GiB); the cache's %lld of layers %d.. "
                         "(%.2f GiB) pageable (cache predicted %.2f GiB)\\n", tail_l, (long long) n_lo, b_lo / kGiB, (long long) n_hi,
                         tail_l, b_hi / kGiB, (long long) n_kept, tail_l, b_kept / kGiB, cu / kGiB);
        }
    }
    // ---- the arena: the rest, in ranked order, up to the RAM budget, in pinned chunks (no blob straddles one)
""")
open(P, "w").write(s)
print("patched", P)
