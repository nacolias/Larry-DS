// with_shexp = false (larry, decode): the shared expert is left out; attn_router runs it as its own graph (build_shexp)
// after the router readback, so the dense card computes it while the expert cards compute the routed experts
static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0,
                                bool with_shexp = true, int64_t max_nodes = 8192, bool dec = false) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, max_nodes, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH, NH = im.NH, SWA = im.SWA, RAW = im.RAW;
    const int64_t r = L.ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    std::vector<ggml_tensor *> outs, late;    // late: ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    // ---- Engram (model.py Engram.forward), before the block, on the hc state
    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);          // [(HC+1) D, n]
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = ggml_mul(gc, im.BL(il, "engram_q_norm.weight"), im.BL(il, "engram_k_norm.weight"));   // [D, HC]
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn));    // [1, HC, n]
        dot = ggml_scale(gc, dot, 1.0f / std::sqrt((float) D));
        // gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))  (signed sqrt, as the training kernel)
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        ggml_tensor * gv = ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate);
        x = ggml_add(gc, x, gv);
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));   // the block input (residual) = state + Engram
    }

    // ---- attention hc: mixes of the block input; its pre-mix is the previous FFN's (pre_state)
    ggml_tensor * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr, * xn = nullptr;
    const bool hcf = b.hc_fused(n);
    if (hcf) {
        ggml_tensor * fa = ggml_dsv41_hc_fa(gc, x, im.BL(il, "hc_attn_fn.weight"), b.cols2(im.pre_state, HC, n, col0),
                                            nullptr, nullptr, nullptr);
        ggml_tensor * fb = ggml_dsv41_hc_fb(gc, fa, im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
                                            im.BL(il, "attn_norm.weight"), (float) G.rms_eps, (float) G.hc_eps, (int) G.hc_iters);
        b.hc_fb_views(fb, n, &xn, &a_pre, &a_post, &a_comb);
    } else {
        b.hc_mixes(x, im.BL(il, "hc_attn_fn.weight"), im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
                   &a_pre, &a_post, &a_comb);
        ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));         // [D, n]
        xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));
    }

    // ---- q (no per-head norm in V4.1)
    ggml_tensor * qr = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_q_a.weight"), xn), im.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * q = ggml_reshape_3d(gc, ggml_mul_mat(gc, im.BL(il, "attn_q_b.weight"), qr), DH, NH, n);
    q = b.rope(q, pos, r);

    // ---- window K (= V): [ring | this pass's rows] -> each query's SWA rows; the ring keeps the last RAW
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);   // model.py _window_kv
    ggml_tensor * wsrc = ggml_concat(gc, L.raw->type == GGML_TYPE_F32 ? L.raw : ggml_cast(gc, L.raw, GGML_TYPE_F32), kv2, 1);     // [DH, RAW + n]
    ggml_tensor * widx_v = b.v1(im.i_widx, SWA * n, (size_t) col0 * SWA * 4);
    ggml_tensor * Kw = ggml_reshape_3d(gc, ggml_get_rows(gc, wsrc, widx_v), DH, SWA, n);
    ggml_tensor * Mw = b.cols2(im.i_wmask, SWA, n, col0);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV (kv sources write it; every ratio > 0 layer reads its source's)
    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    ggml_tensor * kg_comp = nullptr, * kg_sel = nullptr;   // Kc's source rows and their ids (ggml_dsv41_kgather)
    ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;   // this pass's writes (later reads depend on them)
    if (r) {
        if (L.kv_src) {
            ggml_tensor * latent = nullptr;                                                  // [DH, n]
            if (r == 1) {   // model.py Compressor.forward: ratio 1 = a plain projection, no gate
                latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            } else {        // ratio 2: softmax-gated pooling over the group's 2 states, per dim (fp32)
                ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
                ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
                ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
                ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);                     // [DH, 2 + n]
                ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
                ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
                ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
                ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
                vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));                       // [2, DH, n]
                vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
                latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
                const int64_t k = std::min<int64_t>(n, 2);
                late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
                late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
            }
            latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
            ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
            ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
            // indexer keys from the UNROTATED latent (model.py Indexer.forward, owns_k)
            ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
            ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
            if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
            idxk_w = ggml_set_rows(gc, L.idxk, ik, cslot);
            outs.push_back(idxk_w);
            ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
            if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
            comp_w = ggml_set_rows(gc, L.comp, lr, cslot);
            outs.push_back(comp_w);
        }
        if (cap > 0) {
            const L_t & KS = im.ly[(size_t) L.kvl];
            const int64_t K = std::min<int64_t>(cap, im.ITOP);
            ggml_tensor * comp = comp_w ? comp_w : KS.comp;
            ggml_tensor * idxk = idxk_w ? idxk_w : KS.idxk;
            ggml_tensor * sel = nullptr, * selm = nullptr;
            if (L.idx_src) {
                // ---- indexer (model.py Indexer.forward): scores over the visible rows of the shared index keys
                ggml_tensor * iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                iq = b.rope(ggml_reshape_3d(gc, iq, im.ID, im.IH, n), pos, r);
                if (im.cfg.act_quant) iq = fake_quant(gc, ggml_cont(gc, iq), Ds41Quant::FP4_POW2_32);
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
                // Canonicalise the top-k order.  ggml_cuda_op_top_k asks CUB's DeviceTopK for
                // output_ordering::unsorted, and CUB hands the k ids back in an order its atomically assigned
                // output slots produce - run-to-run unstable, while the SET is stable (measured: order-only on
                // every differing row).  The ids drive the gather of the compressed rows, so that order is the
                // flash-attn's accumulation order; left alone the whole run stops being reproducible from the
                // first query whose compressed rows are attended (the SWA window's 128).  Sort ascending.
                sel = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sel, 1, K, n),
                                                        ggml_argsort(gc, ggml_cast(gc, sel, GGML_TYPE_F32),
                                                                     GGML_SORT_ORDER_ASC)), K, n);
                selm = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, n), sel), K, n);   // vis at sel
                outs.push_back(ggml_cpy(gc, sel, ggml_view_2d(gc, L.sel, K, n, L.sel->nb[1], (size_t) col0 * L.sel->nb[1])));
                outs.push_back(ggml_cpy(gc, selm, ggml_view_2d(gc, L.selm, K, n, L.selm->nb[1], (size_t) col0 * L.selm->nb[1])));
            } else {
                const L_t & IS = im.ly[(size_t) L.idxl];
                sel = ggml_view_2d(gc, IS.sel, K, n, IS.sel->nb[1], (size_t) col0 * IS.sel->nb[1]);
                selm = ggml_view_2d(gc, IS.selm, K, n, IS.selm->nb[1], (size_t) col0 * IS.selm->nb[1]);
            }
            Kc = ggml_reshape_3d(gc, ggml_get_rows(gc, comp, ggml_cont(gc, ggml_reshape_1d(gc, ggml_cont(gc, sel), K * n))), DH, K, n);
            kg_comp = comp;
            kg_sel = sel;
            Mc = ggml_cont(gc, selm);
            // the source's rows must be read AFTER this pass wrote its own (a kv source attends to its new row)
        }
    }

    // ---- attention: per query its window rows + its top-k compressed rows, one flash-attn with the query as batch
    ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;                                 // [DH, M, n]
    ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;                                 // [M, n]
    int64_t M = Kall->ne[1];
    // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256; ggml-sycl's DSV4 prompt path (>= 8
    // queries, fattn-dsv4.cpp in larry's ggml-sycl patch) takes any multiple of 16 - padding there only moves bytes
    const char * mqa_env = std::getenv("GGML_SYCL_DSV4_MQA");
    const bool sycl_mqa = im.sycl && n >= 8 && M % 16 == 0 && !(mqa_env && mqa_env[0] == '0');
    // larry: ggml-sycl's DSV4 decode / verify attention (< 8 queries, fattn-dsv4-dec.hpp) reads f32 keys of any count -
    // no padding, no key cast (GGML_SYCL_DSV4_DEC=0: the generic path)
    const char * dec_env = std::getenv("GGML_SYCL_DSV4_DEC");
    const bool sycl_dec = im.sycl && !sycl_mqa && DH == 512 && NH % 32 == 0 && !(dec_env && dec_env[0] == '0');
    if (!im.cpu && !sycl_mqa && !sycl_dec && M % 256) {
        const int64_t pad = 256 - M % 256;
        Kall = ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
        Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, n), NEG_INF), 0);
        M += pad;
    }
    // larry: STRATA_KGATHER (default 1): the prompt path's keys gathered straight into f16 by one op - the get_rows /
    // concat / cast chain behind Kall wrote and re-read ~2 GB per 512-query pass (~40% of a long prompt's card-0 time);
    // Kw / Kc / Kall are then unused and drop out of the graph
    static const bool kgather = env_on("STRATA_KGATHER", true);
    ggml_tensor * k4 = nullptr;
    static const bool dec_fuse = env_on("STRATA_DEC_FUSE", true);
    if (sycl_mqa && kgather) {
        k4 = ggml_dsv41_kgather(gc, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, (int) SWA);
        GGML_ASSERT(k4->ne[1] == M);
    } else if (sycl_dec && kgather && dec_fuse && (!Kc || kg_sel->nb[0] == 4)) {
        // larry: decode / verify: the window rows, the compressed rows and their concat as one f32 gather
        k4 = ggml_dsv41_kgather2(gc, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, (int) SWA, GGML_TYPE_F32);
        GGML_ASSERT(k4->ne[1] == M);
    } else {
        k4 = ggml_reshape_4d(gc, Kall, DH, M, 1, n);
        if (!sycl_dec) k4 = ggml_cast(gc, k4, GGML_TYPE_F16);
    }
    ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);    // [DH, 1, NH, n]
    // the mask stays f16 (ggml_flash_attn_ext asserts it; it is M halves)
    ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, n), GGML_TYPE_F16);
    ggml_tensor * o = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(o, im.BL(il, "attn_sinks.weight"));
    o = ggml_reshape_3d(gc, o, DH, NH, n);
    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    ggml_tensor * attn = nullptr;
    // larry, STRATA_TP_OPROJ (decode / verify graphs): wo_a and wo_b as one op that ggml-sycl hands to the xe tier, which
    // splits wo_a's head groups (and wo_b's matching columns) over the four cards and adds the partial sums on this one
    static const bool tp_oproj = env_on("STRATA_TP_OPROJ", false);
    if (dec && tp_oproj && im.sycl && n <= 8) {
        attn = ggml_dsv41_oproj(gc, o, wo_a, im.BL(il, "attn_output_b.weight"));                  // [D, n]
    } else {
        // grouped output LoRA (model.py: einsum over groups) then wo_b
        o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);           // [OGD, n, OG]
        ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);          // [OL, OG, n]
        attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]
    }

    // ---- hc_post(attention) -> FFN hc: its mixes give the next layer's pre-mix; its input uses the attention's
    ggml_tensor * h = nullptr, * fn = nullptr;                                              // [D, HC, n], [D, n]
    ggml_tensor * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    if (hcf) {
        ggml_tensor * fa = ggml_dsv41_hc_fa(gc, x, im.BL(il, "hc_ffn_fn.weight"), a_pre, attn, a_post, a_comb);
        h = ggml_view_3d(gc, fa, D, HC, n, (size_t) D * 4, (size_t) (HC * D) * 4, 0);
        ggml_tensor * fb = ggml_dsv41_hc_fb(gc, fa, im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"),
                                            im.BL(il, "ffn_norm.weight"), (float) G.rms_eps, (float) G.hc_eps, (int) G.hc_iters);
        b.hc_fb_views(fb, n, &fn, &f_pre, &f_post, &f_comb);
    } else {
        h = ggml_dsv4_hc_post(gc, attn, x, a_post, a_comb);
        b.hc_mixes(h, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"),
                   &f_pre, &f_post, &f_comb);
        fn = b.rms_w(ggml_dsv4_hc_pre(gc, h, a_pre), im.BL(il, "ffn_norm.weight"));
    }

    // ---- router (model.py Gate): sqrt(softplus), top-k on score + bias, weights from the raw scores
    ggml_tensor * rl = ggml_mul_mat(gc, im.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_sqrt(gc, ggml_softplus(gc, rl));                                  // [NE, n]
    // selection = score + exp_probs_b + the selection-only rbias (soft prune / cache bonus); the weights below come
    // from the unbiased scores, as model.py's Gate - all-zero rbias (the default) = the model's routing bit for bit
    ggml_tensor * s0 = ggml_add(gc, sc, im.BL(il, "exp_probs_b.bias"));                        // the model's own selection score
    ggml_tensor * rb = ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1);
    ggml_tensor * sel = nullptr;
    // SMART route bias (opt-in; the flat bias changes ~every choice it can: -14 pt API top-1, repetition loops):
    //   STRATA_RB_PROTECT=P  the exact top-P experts are always kept (+1e4); the bias only competes for the other slots
    //   STRATA_RB_MARGIN=d   the bias only applies to experts whose exact score is within d of the cutoff (midpoint of the
    //                        exact NU-th / NU+1-th scores); experts confidently above it (> cut + d) are always kept -
    //                        only near-ties can flip
    // Both may be set.  Weights still come from the unbiased scores.  Unset = the flat bias, bit for bit as before.
    static const int rb_protect = std::getenv("STRATA_RB_PROTECT") ? std::atoi(std::getenv("STRATA_RB_PROTECT")) : 0;
    static const float rb_margin = std::getenv("STRATA_RB_MARGIN") ? (float) std::atof(std::getenv("STRATA_RB_MARGIN")) : -1.0f;
    if (rb_protect > 0 || rb_margin >= 0.0f) {
        const int kt = (int) im.NU + 1;
        ggml_tensor * ti = ggml_argsort_top_k(gc, s0, kt);                                         // [kt, n]
        ggml_tensor * tv = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s0, 1, im.NE, n), ti), kt, n);   // [kt, n] desc
        auto row = [&](int r) { return ggml_view_2d(gc, tv, 1, n, tv->nb[1], (size_t) r * tv->nb[0]); };   // [1, n]
        ggml_tensor * like = ggml_new_tensor_2d(gc, GGML_TYPE_F32, im.NE, n);
        ggml_tensor * protect = nullptr;
        if (rb_protect > 0) {   // s0 >= the P-th largest exact score
            ggml_tensor * vp = ggml_repeat(gc, ggml_cont(gc, row(std::min(rb_protect, (int) im.NU) - 1)), like);
            protect = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, vp), 1.0f, 1e-6f));
        }
        ggml_tensor * gate = nullptr;
        if (rb_margin >= 0.0f) {
            ggml_tensor * cut = ggml_repeat(gc, ggml_scale(gc, ggml_add(gc, ggml_cont(gc, row((int) im.NU - 1)), ggml_cont(gc, row((int) im.NU))), 0.5f), like);
            ggml_tensor * dv = ggml_sub(gc, s0, cut);                                              // [NE, n]
            ggml_tensor * above = ggml_step(gc, ggml_scale_bias(gc, dv, 1.0f, -rb_margin));          // dv > d
            ggml_tensor * below = ggml_step(gc, ggml_scale_bias(gc, dv, -1.0f, -rb_margin));         // dv < -d
            protect = protect ? ggml_clamp(gc, ggml_add(gc, protect, above), 0.0f, 1.0f) : above;
            gate = ggml_scale_bias(gc, ggml_clamp(gc, ggml_add(gc, above, below), 0.0f, 1.0f), -1.0f, 1.0f);   // in the band
        }
        ggml_tensor * unprot = ggml_scale_bias(gc, protect, -1.0f, 1.0f);
        if (gate) unprot = ggml_mul(gc, unprot, gate);
        sel = ggml_add(gc, ggml_add(gc, s0, ggml_mul(gc, ggml_repeat(gc, rb, like), unprot)), ggml_scale(gc, protect, 1e4f));
    } else {
        sel = ggml_add(gc, s0, rb);
    }
    ggml_tensor * ids = nullptr, * wts = nullptr;
    // larry: ggml-sycl's fused Gate (GGML_OP_DSV41_ROUTER: one kernel for the ~11 below) with the flat bias (the default;
    // STRATA_ROUTER_FUSED=0: the composite).  Its ids are floats; the cpy into o_ids converts them.
    static const bool fused_router = [] { const char * v = std::getenv("STRATA_ROUTER_FUSED"); return !(v && v[0] == '0'); }();
    if (im.sycl && fused_router && rb_protect <= 0 && rb_margin < 0.0f) {
        ggml_tensor * ro = ggml_dsv41_router(gc, rl, im.BL(il, "exp_probs_b.bias"), ggml_reshape_1d(gc, rb, im.NE), (int) im.NU,
                                             G.norm_topk, (float) G.weights_scale);   // [2 NU, n]
        ids = ggml_view_2d(gc, ro, im.NU, n, ro->nb[1], 0);
        wts = ggml_view_2d(gc, ro, im.NU, n, ro->nb[1], (size_t) im.NU * 4);
    } else {
        ids = ggml_argsort_top_k(gc, sel, (int) im.NU);   // [NU, n]
        wts = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, im.NE, n), ids), im.NU, n);
        if (G.norm_topk) wts = ggml_div(gc, wts, ggml_scale_bias(gc, ggml_sum_rows(gc, wts), 1.0f, 1e-20f));
        wts = ggml_scale(gc, wts, (float) G.weights_scale);
    }

    // ---- shared expert
    if (with_shexp) outs.push_back(ggml_cpy(gc, shexp_ffn(im, gc, il, fn), b.cols2(im.shexp, D, n, col0)));

    outs.push_back(ggml_cpy(gc, h, b.cols3(im.hap, D, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, f_post, b.cols2(im.post_f, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, ggml_reshape_3d(gc, f_comb, HC, HC, n), b.cols3(im.comb_f, HC, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, fn, b.cols2(im.o_fn, D, n, col0)));
    outs.push_back(ggml_cpy(gc, ids, b.cols2(im.o_ids, im.NU, n, col0)));
    outs.push_back(ggml_cpy(gc, wts, b.cols2(im.o_wts, im.NU, n, col0)));
    outs.push_back(ggml_cpy(gc, f_pre, b.cols2(im.pre_state, HC, n, col0)));   // reads of pre_state happen above
    // larry, STRATA_CPY_LAST (default 1): every output copy's source first, then the copies - consecutive plain copies
    // run as one batched kernel in ggml-sycl (each used to sit right after its source: its own command)
    static const bool cpy_last = env_on("STRATA_CPY_LAST", true);
    if (cpy_last) for (ggml_tensor * t : outs) if (t->op == GGML_OP_CPY && t->src[0]) ggml_build_forward_expand(gf, t->src[0]);
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// enc-skip: the persistent-KV half of build_attn for one span.  Runs hc_pre / attn_norm, writes the raw window K/V
// ring, and on a kv source layer writes the compressed-KV row + indexer-key row (+ the ratio-2 compressor state).
// Everything else build_attn does (q, the attention + output LoRA, hc_post, the router, the shared expert, the indexer
// selection / candidate blocks, and the pre_state / hap / shexp / o_* writes) only ever flows into THIS layer's own
