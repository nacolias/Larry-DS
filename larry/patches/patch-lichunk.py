#!/usr/bin/env python3
# patch-lichunk.py [DENSE_CPP] - larry: the prompt path's lightning indexer in column slices.  Its [cap, n] tensors
# (the visibility mask chain, the F16 mask, the scores, the candidate mask and its sum, top-k's workspace) grow with
# the context: at a 1M-token context one 512-column span of a ratio-1 indexer layer needed ~9 GiB on card 0 and the
# last batch of a 984K-token prompt failed (UR OUT_OF_RESOURCES in DSV4_HC_PRE, free VRAM 10.47 -> 1.46 GiB at layer 39).
# Every query column ranks on its own, so slices of at most STRATA_LI_CHUNK_MIB (512) MiB a [cap, cols] F32 tensor
# give the same ids; their outputs are pushed slice by slice (STRATA_CPY_LAST expands the sources in that order, so
# a slice's temporaries are freed before the next one's).  Up to cap 262,144 rows (n 512) one slice = the old graph.
# Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_LI_CHUNK_MIB" in s:
    print("already patched"); sys.exit(0)
old = """            } else if (L.idx_src) {
                // ---- indexer (model.py Indexer.forward): scores over the visible rows of the shared index keys
                ggml_tensor * iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                iq = b.rope(ggml_reshape_3d(gc, iq, im.ID, im.IH, n), pos, r);
                if (im.cfg.act_quant) iq = fake_quant(gc, cont_if(gc, iq), Ds41Quant::FP4_POW2_32);
                ggml_tensor * iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                iw = ggml_scale(gc, iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));         // softmax_scale * heads^-0.5
                ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, cap, idxk->nb[1], idxk->nb[1], 0);
                // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], n, (size_t) col0 * 4), 1, n);
                ggml_tensor * ramp = ggml_view_2d(gc, im.neg_ramp, cap, 1, (size_t) cap * 4, 0);
                ggml_tensor * vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, n)), nvs)));
                ggml_tensor * sc = ggml_lightning_indexer(gc, iq, ksrc, iw, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, n]
                sc = ggml_reshape_2d(gc, sc, cap, n);
                if (L.uses_cand) {   // rank only inside cand_layer's candidate blocks
                    const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                    ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, n), NEG_INF);
                    ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, n), 0.0f);
                    m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, n, im.cand->nb[1], (size_t) col0 * im.cand->nb[1])));
                    m = ggml_view_2d(gc, m, cap, n, m->nb[2], 0);
                    sc = ggml_add(gc, sc, m);
                }
                if (L.cand_src) {   // model.py select_candidate_blocks: block score = max, newest block pinned
                    const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                    GGML_ASSERT(cap % im.CBS == 0);
                    ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);   // [nblk, n]
                    // pin: + 1e30 at block (nvis - 1) / CBS of each query
                    ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                    ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, n, (size_t) col0 * 4), 1, n);
                    ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                    bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                    ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));                     // I32 [ncb, n]
                    outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, n, im.cand->nb[1], (size_t) col0 * im.cand->nb[1])));
                }
                sel = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                                     // I32 [K, n]
"""
new = """            } else if (L.idx_src) {
                // ---- indexer (model.py Indexer.forward): scores over the visible rows of the shared index keys
                ggml_tensor * iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                iq = b.rope(ggml_reshape_3d(gc, iq, im.ID, im.IH, n), pos, r);
                if (im.cfg.act_quant) iq = fake_quant(gc, cont_if(gc, iq), Ds41Quant::FP4_POW2_32);
                ggml_tensor * iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                iw = ggml_scale(gc, iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));         // softmax_scale * heads^-0.5
                ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, cap, idxk->nb[1], idxk->nb[1], 0);
                ggml_tensor * ramp = ggml_view_2d(gc, im.neg_ramp, cap, 1, (size_t) cap * 4, 0);
                // larry: the [cap, n] tensors below in column slices of at most STRATA_LI_CHUNK_MIB (512) MiB an F32 one -
                // at a 1M-token context a 512-column span took ~9 GiB on card 0 and a 984K prompt's last batch failed.
                // Each query column ranks on its own: the same ids.  A slice's outputs are pushed before the next slice
                // is built (STRATA_CPY_LAST expands their sources in that order, so its temporaries are freed first).
                static const int64_t li_chunk = (std::getenv("STRATA_LI_CHUNK_MIB") ? std::atoll(std::getenv("STRATA_LI_CHUNK_MIB")) : 512) << 20;
                int64_t ncol = n;
                if (li_chunk > 0 && cap * n * 4 > li_chunk) ncol = std::min<int64_t>(n, std::max<int64_t>(16, (li_chunk / (cap * 4)) & ~(int64_t) 15));
                std::vector<ggml_tensor *> sel_parts, selm_parts;
                for (int64_t q0 = 0; q0 < n; q0 += ncol) {
                    const int64_t nq = std::min<int64_t>(ncol, n - q0);
                    const int64_t qc0 = col0 + q0;   // the slice's first column of the state
                    ggml_tensor * iqs = nq == n ? iq : ggml_view_3d(gc, iq, im.ID, im.IH, nq, iq->nb[1], iq->nb[2], (size_t) q0 * iq->nb[2]);
                    ggml_tensor * iws = nq == n ? iw : ggml_view_2d(gc, iw, iw->ne[0], nq, iw->nb[1], (size_t) q0 * iw->nb[1]);
                    // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                    ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], nq, (size_t) qc0 * 4), 1, nq);
                    ggml_tensor * vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, nq)), nvs)));
                    ggml_tensor * sc = ggml_lightning_indexer(gc, iqs, ksrc, iws, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, nq]
                    sc = ggml_reshape_2d(gc, sc, cap, nq);
                    if (L.uses_cand) {   // rank only inside cand_layer's candidate blocks
                        const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                        ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, nq), NEG_INF);
                        ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, nq), 0.0f);
                        m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], (size_t) qc0 * im.cand->nb[1])));
                        m = ggml_view_2d(gc, m, cap, nq, m->nb[2], 0);
                        sc = ggml_add(gc, sc, m);
                    }
                    if (L.cand_src) {   // model.py select_candidate_blocks: block score = max, newest block pinned
                        const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                        GGML_ASSERT(cap % im.CBS == 0);
                        ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);   // [nblk, nq]
                        // pin: + 1e30 at block (nvis - 1) / CBS of each query
                        ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                        ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, nq, (size_t) qc0 * 4), 1, nq);
                        ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                        bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                        ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));                     // I32 [ncb, nq]
                        outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], (size_t) qc0 * im.cand->nb[1])));
                    }
                    sel = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                                     // I32 [K, nq]
"""
if s.count(old) != 1:
    sys.exit("anchor 1 x%d" % s.count(old))
s = s.replace(old, new)
old2 = """                sel = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sel, 1, K, n),
                                                        ggml_argsort(gc, ggml_cast(gc, sel, GGML_TYPE_F32),
                                                                     GGML_SORT_ORDER_ASC)), K, n);
                selm = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, n), sel), K, n);   // vis at sel
                outs.push_back(ggml_cpy(gc, sel, ggml_view_2d(gc, L.sel, K, n, L.sel->nb[1], (size_t) col0 * L.sel->nb[1])));
                outs.push_back(ggml_cpy(gc, selm, ggml_view_2d(gc, L.selm, K, n, L.selm->nb[1], (size_t) col0 * L.selm->nb[1])));
            } else {
"""
new2 = """                    sel = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sel, 1, K, nq),
                                                            ggml_argsort(gc, ggml_cast(gc, sel, GGML_TYPE_F32),
                                                                         GGML_SORT_ORDER_ASC)), K, nq);
                    selm = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, nq), sel), K, nq);   // vis at sel
                    outs.push_back(ggml_cpy(gc, sel, ggml_view_2d(gc, L.sel, K, nq, L.sel->nb[1], (size_t) qc0 * L.sel->nb[1])));
                    outs.push_back(ggml_cpy(gc, selm, ggml_view_2d(gc, L.selm, K, nq, L.selm->nb[1], (size_t) qc0 * L.selm->nb[1])));
                    sel_parts.push_back(sel);
                    selm_parts.push_back(selm);
                }
                sel = sel_parts[0];
                selm = selm_parts[0];
                for (size_t i = 1; i < sel_parts.size(); ++i) {   // the span's ids / masks for its attention
                    sel = ggml_concat(gc, sel, sel_parts[i], 1);
                    selm = ggml_concat(gc, selm, selm_parts[i], 1);
                }
            } else {
"""
if s.count(old2) != 1:
    sys.exit("anchor 2 x%d" % s.count(old2))
s = s.replace(old2, new2)
open(P, "w").write(s)
print("patched", P)
