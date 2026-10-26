#!/usr/bin/env python3
# patch-seedtail.py [TIER_CPP] - larry, STRATA_XE_SEED_TAIL_LAYERS=L (default 0 = off): the dense card's cache holds the
# same experts, laid out with layers >= L first and layers < L last (each group in rank order), so a keep-cache shrink
# for a prompt batch (it frees the last slabs) gives back lower-layer experts first.  With enc-skip (L = its cut layer,
# 20) the upper layers run only the batch's last ~2.7K columns and are bound by the prefetch's staging of their arena
# experts (~1.4 GB a layer at ~16 GB/s): every one of their experts kept on the dense card is one not staged.  The lower
# layers' calls are long (8K tokens) and, with STRATA_XE_DENSE_OFFLOAD, take the dense card's experts from their pinned
# arena copies anyway.  The regrow after the batch puts back exactly what the shrink took (the same reordered list).
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_SEED_TAIL_LAYERS" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    XeCard& c = *im.cards[(size_t) im.dense];
    im.seed_ranked = ranked;
    c.budget = dense_budget(im, c);
    if (c.budget <= 0) return true;
    if (seed_dense(im, c, ranked, "seeded", err) < 0) return false;
""", """    XeCard& c = *im.cards[(size_t) im.dense];
    im.seed_ranked = ranked;
    c.budget = dense_budget(im, c);
    if (c.budget <= 0) return true;
    // larry, STRATA_XE_SEED_TAIL_LAYERS (see patch-seedtail.py): the cache's set as the ranked seed takes it, laid out
    // with the layers >= L first; then everything else in rank order (a regrow into more room continues as before)
    static const int tail_l = env_i("STRATA_XE_SEED_TAIL_LAYERS", 0);
    if (tail_l > 0 && tail_l < im.g.n_layers) {
        std::vector<std::pair<int32_t, int32_t>> hi, lo, rest;
        int64_t used = c.used;
        for (const auto& le : ranked) {
            const bool ok = le.first >= 0 && le.first < im.g.n_layers && le.second >= 0 && le.second < im.g.n_experts;
            const size_t key = ok ? (size_t) im.key(le.first, le.second) : 0;
            if (!ok || im.card_of[key] >= 0 || !im.aptr[key]) { rest.push_back(le); continue; }
            const int64_t b = (int64_t) im.bbytes(le.first, le.second);
            if (used + b > c.budget) { rest.push_back(le); continue; }
            used += b;
            (le.first >= tail_l ? hi : lo).push_back(le);
        }
        std::vector<std::pair<int32_t, int32_t>> order;
        order.reserve(ranked.size());
        order.insert(order.end(), hi.begin(), hi.end());
        order.insert(order.end(), lo.begin(), lo.end());
        order.insert(order.end(), rest.begin(), rest.end());
        std::fprintf(stderr, "xe tier: dense card cache laid out layers %d.. first: %zu experts, then %zu below layer %d\\n", tail_l,
                     hi.size(), lo.size(), tail_l);
        im.seed_ranked = order;
    }
    if (seed_dense(im, c, im.seed_ranked, "seeded", err) < 0) return false;
""")
open(P, "w").write(s)
print("patched", P)
