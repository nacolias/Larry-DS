#!/usr/bin/env python3
# patch-litopk-eng.py [XOLOTL_DIR] - larry, 2026-10-28: the engine's side of patch-litopk.py.
#  - ds41_dense.cpp, STRATA_LI_TOPK=1 (default 0; with STRATA_LI_NVIS): an index source that ranks every visible row (not
#    the candidate layer, not the candidates' users - in a long prompt layers 2 / 8 / 14) takes its ids from
#    ggml_lightning_indexer_topk, one op a span (no [cap, n] score tensor, no column slices); the ids' canonical order and
#    their mask as before.
#  - ds41_generate.cpp, STRATA_LI_REMOTE=N (default 0 = off; with STRATA_LI_TOPK): a served prompt of >= N tokens shares
#    those calls' queries with the expert cards - their buffers sized for its end (its last cap bucket, every such layer, a
#    span's share) before its first batch, freed when it is read.
# Needs patch-litopk.py.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
D = os.path.join(X, "tools/v41/engine/ds41_dense.cpp")
G = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
d = open(D).read()
g = open(G).read()
if "STRATA_LI_TOPK" in d:
    print("already patched")
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


d = rep(d, """                if (li_chunk > 0 && cap * n * 4 > li_chunk) ncol = std::min<int64_t>(n, std::max<int64_t>(16, (li_chunk / (cap * 4)) & ~(int64_t) 15));
""", """                if (li_chunk > 0 && cap * n * 4 > li_chunk) ncol = std::min<int64_t>(n, std::max<int64_t>(16, (li_chunk / (cap * 4)) & ~(int64_t) 15));
                // larry, STRATA_LI_TOPK (patch-litopk.py, default 0; with STRATA_LI_NVIS): a source that ranks every visible row
                // (not the candidate layer, not the candidates' users) takes its ids from ggml_lightning_indexer_topk - the
                // scores and their top-k in one op that never holds [cap, n] scores in the graph (and can hand a share of its
                // queries to the expert cards: ggml_backend_sycl_li_remote) - one call a span
                static const bool li_topk_on = env_on("STRATA_LI_TOPK", false) && env_on("STRATA_LI_NVIS", false);
                const bool topk_op = li_topk_on && im.sycl && !L.uses_cand && !L.cand_src && idxk->type == GGML_TYPE_BF16 &&
                                     idxk->nb[1] == 256 && cap % 32 == 0 && im.ID == 128 && im.IH == 32;
                if (topk_op) ncol = n;
""", "ncol")
d = rep(d, """                    ggml_tensor * vis = nullptr, * sc = nullptr;
                    if (li_nvis && im.sycl) {
""", """                    ggml_tensor * vis = nullptr, * sc = nullptr;
                    if (topk_op) {   // (patch-litopk.py) the ids straight from the scores
                        // (the rows its last query sees are written by now: (position + 1) / ratio of it)
                        sel = ggml_lightning_indexer_topk(gc, iqs, ksrc, iws, nvs, (int) K, (int) (((int64_t) im.pos0 + qc0 + nq) / r));
                    } else {
                    if (li_nvis && im.sycl) {
""", "topk op")
d = rep(d, """                    sel = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                                     // I32 [K, nq]
""", """                    sel = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                                     // I32 [K, nq]
                    }   // (patch-litopk.py)
""", "top_k close")

g = rep(g, """            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
""", """            // larry, STRATA_LI_REMOTE=N (patch-litopk.py, default 0; with STRATA_LI_TOPK): a prompt of >= N tokens shares its
            // indexer's queries with the expert cards - their buffers sized for its end (its last ratio-2 cap bucket, every
            // source that ranks all rows, a span's share) before its first batch, freed once it is read
            static const int64_t lir_min = std::getenv("STRATA_LI_REMOTE") ? std::atoll(std::getenv("STRATA_LI_REMOTE")) : 0;
            bool lir = false;
            if (lir_min > 0 && !by_step && dh > 0 && total >= lir_min) {
                int64_t nkv = 1024;
                while (nkv < (int64_t) (total + 64) / 2 + 1) nkv <<= 1;
                int nlay = 0;
                for (int l = 0; l < n_layer; ++l) if (G.is_index_source(l) && (G.cand_layer < 0 || l < G.cand_layer)) ++nlay;
                lir = nlay > 0 && ggml_backend_sycl_li_remote_prepare(ggml_backend_sycl_get_queue(dense.backend()), nkv, nlay, (SB + 3) / 4 + 8);
                if (lir) ggml_backend_sycl_li_remote(1);
            }
            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
            if (lir) ggml_backend_sycl_li_remote_release();   // (patch-litopk.py)
""", "feed")
open(D, "w").write(d)
open(G, "w").write(g)
print("patched", D, G)
