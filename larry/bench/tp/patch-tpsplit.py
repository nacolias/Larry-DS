#!/usr/bin/env python3
# patch-tpsplit.py - larry, STRATA_TP_OPROJ=2 (on top of patch-tpoproj.py): the decode / verify attention graph in two
# kept halves around the attention output projection, so both stay replayable SYCL graphs (GGML_SYCL_GRAPH_CACHE=1:
# the attention graph's ~38 submissions took ~210 us of host time a layer, a replay ~35) and the projection's four-card
# split is queued between them (ggml_backend_sycl_oproj -> the xe tier).  Each expert card's part of the split is
# recorded once per layer and width as a SYCL graph too (STRATA_TP_GRAPH, default 1): one submission a card.
# STRATA_TP_OPROJ=1 stays the in-graph op (eager graphs only).  Idempotent.
import sys

ROOT = "/opt/ai/dsv41/xolotl"
GG = ROOT + "/third_party/llama.cpp/ggml"
FILES = {
    "sh": GG + "/include/ggml-sycl.h",
    "sc": GG + "/src/ggml-sycl/ggml-sycl.cpp",
    "dense": ROOT + "/tools/v41/engine/ds41_dense.cpp",
    "dh": ROOT + "/tools/v41/engine/ds41_dense.hpp",
    "tc": ROOT + "/tools/v41/engine/xe/ds41_xe_tier.cpp",
    "gen": ROOT + "/tools/v41/engine/ds41_generate.cpp",
}
src = {k: open(p).read() for k, p in FILES.items()}
if "ggml_backend_sycl_oproj(" in src["sh"]:
    print("already"); sys.exit(0)
if "GGML_OP_DSV41_OPROJ" not in src["sc"]:
    sys.exit("patch-tpoproj.py first")


def rep(k, old, new, cnt=1):
    s = src[k]
    if s.count(old) != cnt:
        sys.exit(f"{k}: anchor x{s.count(old)}: {old[:100]!r}")
    src[k] = s.replace(old, new)


# ---------------------------------------------------------------- ggml-sycl.h
rep("sh", """GGML_BACKEND_API void ggml_backend_sycl_set_oproj_hook(ggml_sycl_larry_oproj_hook_t fn, void * user);
""", """GGML_BACKEND_API void ggml_backend_sycl_set_oproj_hook(ggml_sycl_larry_oproj_hook_t fn, void * user);
// larry: DSV41_OPROJ outside a graph (STRATA_TP_OPROJ=2: the decode attention graph in two halves around it) - wo_a the
// 2-D weight read as OG slices of OL rows, o / dst device pointers of n columns (OGD * OG floats apart / dst_col_stride
// floats apart).  Ordered on the backend's queue (the launcher above if it takes it); false if the shape is not covered.
GGML_BACKEND_API bool ggml_backend_sycl_oproj(ggml_backend_t backend, const struct ggml_tensor * wo_a, int64_t OGD, int64_t OL,
                                              int64_t OG, const struct ggml_tensor * wo_b, const float * o, int64_t n,
                                              float * dst, int64_t dst_col_stride);
""")

# ---------------------------------------------------------------- ggml-sycl.cpp: the op's body on plain pointers
rep("sc", """static bool larry_oproj(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * o = dst->src[0];
    const ggml_tensor * wa = dst->src[1];
    const ggml_tensor * wb = dst->src[2];
    const int64_t OGD = wa->ne[0], OL = wa->ne[1], OG = wa->ne[2], D = wb->ne[1], KA = OL * OG;
    const int64_t n = dst->ne[1];
    if (n < 1 || n > 8 || wa->type != GGML_TYPE_Q6_K || wb->type != GGML_TYPE_Q6_K || OGD % QK_K || KA % QK_K) return false;
    queue_ptr st = ctx.stream();
    if (!g_larry_slice_planar.count(wa->data)) {   // wo_a per slice (STRATA_PREREORDER_SLICES did it at load), once
        const ggml_tensor * par = wa->view_src ? wa->view_src : wa;
        if (!reorder_qw_q6_k_moe((uint8_t *) par->data, wa->nb[2], OG, st)) return false;
        g_larry_slice_planar.insert(wa->data);
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(wb->extra);
    if (!extra) return false;
    if (!extra->optimized_feature.reorder) {   // wo_b whole (STRATA_PREREORDER did it at load), once
        if (!reorder_qw(wb, st)) return false;
        extra->optimized_feature.reorder = true;
    }
    const float * x = (const float *) o->data;
    float * y = (float *) dst->data;
    const int64_t y_cs = (int64_t) (dst->nb[1] / 4);
    if (g_larry_oproj_hook &&
        g_larry_oproj_hook(st, wa->data, wa->nb[2], wb->data, OGD, OL, OG, D, (int) n, x, OGD * OG, y, y_cs, g_larry_oproj_user))
        return true;""", """// wa: wo_a's 2-D weight (its slices: sb bytes apart); x / y: n columns (OGD * OG / y_cs floats apart)
static bool larry_oproj_core(ggml_backend_sycl_context & ctx, const ggml_tensor * wa, size_t sb, int64_t OGD, int64_t OL,
                             int64_t OG, const ggml_tensor * wb, const float * x, int64_t n, float * y, int64_t y_cs) {
    const int64_t D = wb->ne[1], KA = OL * OG;
    if (n < 1 || n > 8 || wa->type != GGML_TYPE_Q6_K || wb->type != GGML_TYPE_Q6_K || OGD % QK_K || KA % QK_K) return false;
    queue_ptr st = ctx.stream();
    if (!g_larry_slice_planar.count(wa->data)) {   // wo_a per slice (STRATA_PREREORDER_SLICES did it at load), once
        if (!reorder_qw_q6_k_moe((uint8_t *) wa->data, sb, OG, st)) return false;
        g_larry_slice_planar.insert(wa->data);
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(wb->extra);
    if (!extra) return false;
    if (!extra->optimized_feature.reorder) {   // wo_b whole (STRATA_PREREORDER did it at load), once
        if (!reorder_qw(wb, st)) return false;
        extra->optimized_feature.reorder = true;
    }
    if (g_larry_oproj_hook &&
        g_larry_oproj_hook(st, wa->data, sb, wb->data, OGD, OL, OG, D, (int) n, x, OGD * OG, y, y_cs, g_larry_oproj_user))
        return true;""")
rep("sc", """    b.ncols = (int) n; b.xq = aq.get(); b.xd = ad.get();
    return fastmv::launch(*st, fastmv::WType::Q6K_R8, b);
}
""", """    b.ncols = (int) n; b.xq = aq.get(); b.xd = ad.get();
    return fastmv::launch(*st, fastmv::WType::Q6K_R8, b);
}

static bool larry_oproj(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * wa = dst->src[1];   // the 3-D view: its parent holds the slices
    return larry_oproj_core(ctx, wa->view_src ? wa->view_src : wa, wa->nb[2], wa->ne[0], wa->ne[1], wa->ne[2], dst->src[2],
                            (const float *) dst->src[0]->data, dst->ne[1], (float *) dst->data, (int64_t) (dst->nb[1] / 4));
}

bool ggml_backend_sycl_oproj(ggml_backend_t backend, const ggml_tensor * wo_a, int64_t OGD, int64_t OL, int64_t OG,
                             const ggml_tensor * wo_b, const float * o, int64_t n, float * dst, int64_t dst_col_stride) {
    if (!backend || !wo_a || !wo_b || wo_a->ne[0] != OGD || wo_a->ne[1] != OL * OG) return false;
    auto * ctx = static_cast<ggml_backend_sycl_context *>(backend->context);
    return larry_oproj_core(*ctx, wo_a, (size_t) OL * wo_a->nb[1], OGD, OL, OG, wo_b, o, n, dst, dst_col_stride);
}
""")

# ---------------------------------------------------------------- ds41_dense.hpp / .cpp
rep("dh", """    void set_pre_wait_hook(std::function<void(int, int)> h);
""", """    void set_pre_wait_hook(std::function<void(int, int)> h);
    /// larry, STRATA_TP_OPROJ=2: the attention output projection of a decode / verify layer, queued between the two
    /// halves of its attention graph (layer, width, wo_a, wo_b, o: OGD * OG floats a column, dst: n_embd a column)
    void set_oproj_hook(std::function<bool(int, int, const ggml_tensor *, const ggml_tensor *, const float *, float *)> h);
""")
rep("dense", """    std::function<void(int, int)> pre_wait;   // larry: set_pre_wait_hook
""", """    std::function<void(int, int)> pre_wait;   // larry: set_pre_wait_hook
    std::function<bool(int, int, const ggml_tensor *, const ggml_tensor *, const float *, float *)> oproj_hook;   // larry
""")
rep("dense", """void Ds41Dense::set_pre_wait_hook(std::function<void(int, int)> h) { p_->pre_wait = std::move(h); }
""", """void Ds41Dense::set_pre_wait_hook(std::function<void(int, int)> h) { p_->pre_wait = std::move(h); }
void Ds41Dense::set_oproj_hook(std::function<bool(int, int, const ggml_tensor *, const ggml_tensor *, const float *, float *)> h) {
    p_->oproj_hook = std::move(h);
}
""")
rep("dense", """    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
""", """    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
    // larry, STRATA_TP_OPROJ=2: between a decode layer's two attention halves (<= 8 columns) - the de-RoPEd attention
    // output, the attention hc mixes (hc_fb's pre | post | comb planes) and the output projection
    ggml_tensor * o_att = nullptr;        // [OGD * OG, <= 8]
    ggml_tensor * o_mix = nullptr;        // [2 HC + HC * HC, <= 8]
    ggml_tensor * o_proj = nullptr;       // [D, <= 8]
""")
rep("dense", """    im.o_fn = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
""", """    im.o_fn = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    {
        const int64_t NO = std::min<int64_t>(NC, 8);
        im.o_att = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.OGD * im.OG, NO);
        im.o_mix = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * HC + HC * HC, NO);
        im.o_proj = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NO);
    }
""")
# build_attn: the part parameter, the shared declarations hoisted, part 2 skips the front
rep("dense", """                                bool with_shexp = true, int64_t max_nodes = 8192, bool dec = false) {""",
    """                                bool with_shexp = true, int64_t max_nodes = 8192, bool dec = false, int part = 0) {""")
rep("dense", """    std::vector<ggml_tensor *> outs, late;    // late: ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    // ---- Engram (model.py Engram.forward), before the block, on the hc state
    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
""", """    std::vector<ggml_tensor *> outs, late;    // late: ring / state write-backs, after every read
    // larry, STRATA_TP_OPROJ=2 (decode widths, fused hc): part 1 ends with the de-RoPEd attention output and the attention
    // hc mixes in o_att / o_mix (every ring / state write-back with it: no read of them follows); the projection runs
    // between the two graphs; part 2 starts from o_proj.  Part 0: the whole layer.
    ggml_tensor * x = nullptr, * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr, * attn = nullptr, * fb_a = nullptr;
    const bool hcf = b.hc_fused(n);
    GGML_ASSERT(part == 0 || (hcf && col0 == 0 && n <= im.o_att->ne[1]));
    if (part == 2) {
        x = b.cols3(im.x_state, D, HC, n, col0);
        ggml_tensor * mx = ggml_view_1d(gc, im.o_mix, (2 * HC + HC * HC) * n, 0);
        a_pre = ggml_view_2d(gc, mx, HC, n, (size_t) HC * 4, 0);
        a_post = ggml_view_2d(gc, mx, HC, n, (size_t) HC * 4, (size_t) (HC * n) * 4);
        a_comb = ggml_view_3d(gc, mx, HC, HC, n, (size_t) HC * 4, (size_t) (HC * HC) * 4, (size_t) (2 * HC * n) * 4);
        attn = b.cols2(im.o_proj, D, n, 0);
    } else {
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    // ---- Engram (model.py Engram.forward), before the block, on the hc state
    x = b.cols3(im.x_state, D, HC, n, col0);
""")
rep("dense", """    // ---- attention hc: mixes of the block input; its pre-mix is the previous FFN's (pre_state)
    ggml_tensor * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr, * xn = nullptr;
    const bool hcf = b.hc_fused(n);
    if (hcf) {
        ggml_tensor * fa = ggml_dsv41_hc_fa(gc, x, im.BL(il, "hc_attn_fn.weight"), b.cols2(im.pre_state, HC, n, col0),
                                            nullptr, nullptr, nullptr);
        ggml_tensor * fb = ggml_dsv41_hc_fb(gc, fa, im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
                                            im.BL(il, "attn_norm.weight"), (float) G.rms_eps, (float) G.hc_eps, (int) G.hc_iters);
        b.hc_fb_views(fb, n, &xn, &a_pre, &a_post, &a_comb);""", """    // ---- attention hc: mixes of the block input; its pre-mix is the previous FFN's (pre_state)
    ggml_tensor * xn = nullptr;
    if (hcf) {
        ggml_tensor * fa = ggml_dsv41_hc_fa(gc, x, im.BL(il, "hc_attn_fn.weight"), b.cols2(im.pre_state, HC, n, col0),
                                            nullptr, nullptr, nullptr);
        ggml_tensor * fb = ggml_dsv41_hc_fb(gc, fa, im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
                                            im.BL(il, "attn_norm.weight"), (float) G.rms_eps, (float) G.hc_eps, (int) G.hc_iters);
        fb_a = fb;
        b.hc_fb_views(fb, n, &xn, &a_pre, &a_post, &a_comb);""")
rep("dense", """    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
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
""", """    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
    if (part == 1) {
        outs.push_back(ggml_cpy(gc, o, b.cols2(im.o_att, im.OGD * im.OG, n, 0)));
        outs.push_back(ggml_cpy(gc, ggml_view_1d(gc, fb_a, (2 * HC + HC * HC) * n, (size_t) (D * n) * 4),
                                ggml_view_1d(gc, im.o_mix, (2 * HC + HC * HC) * n, 0)));
        for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
        for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
        return gf;
    }
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    // larry, STRATA_TP_OPROJ=1 (decode / verify graphs): wo_a and wo_b as one op that ggml-sycl hands to the xe tier, which
    // splits wo_a's head groups (and wo_b's matching columns) over the four cards and adds the partial sums on this one
    static const int tp_oproj = std::getenv("STRATA_TP_OPROJ") ? std::atoi(std::getenv("STRATA_TP_OPROJ")) : 0;
    if (dec && tp_oproj == 1 && im.sycl && n <= 8) {
        attn = ggml_dsv41_oproj(gc, o, wo_a, im.BL(il, "attn_output_b.weight"));                  // [D, n]
    } else {
        // grouped output LoRA (model.py: einsum over groups) then wo_b
        o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);           // [OGD, n, OG]
        ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);          // [OL, OG, n]
        attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]
    }
    }   // part != 2
""")
# attn_router: the two halves and the projection between them
rep("dense", """    if (cached) {
        const Impl::GKey key { il, 1, cap, (int64_t) n };
        auto it = im.graphs.find(key);
        // a kept graph is a few hundred nodes: room for 2048 (the gctx holds hundreds of them: layers x caps x widths)
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, n, 0, !split_shexp, 2048, true)).first;
        gf = it->second;
    } else {""", """    // larry, STRATA_TP_OPROJ=2: the kept graph in two halves, the output projection queued between them (oproj_hook)
    static const int tp_mode = std::getenv("STRATA_TP_OPROJ") ? std::atoi(std::getenv("STRATA_TP_OPROJ")) : 0;
    static const bool hc_fuse = env_on("STRATA_HC_FUSE", true);
    static const int64_t hc_maxn = std::getenv("STRATA_HC_FUSE_MAXN") ? std::atoll(std::getenv("STRATA_HC_FUSE_MAXN")) : 8;
    const bool halves = cached && tp_mode == 2 && im.sycl && im.oproj_hook && hc_fuse && n <= hc_maxn && n <= im.o_att->ne[1];
    ggml_cgraph * gf2 = nullptr;
    if (halves) {
        const Impl::GKey ka { il, 11, cap, (int64_t) n }, kb { il, 12, cap, (int64_t) n };
        auto ia = im.graphs.find(ka);
        if (ia == im.graphs.end()) ia = im.graphs.emplace(ka, build_attn(im, im.gctx, il, n, 0, !split_shexp, 2048, true, 1)).first;
        auto ib = im.graphs.find(kb);
        if (ib == im.graphs.end()) ib = im.graphs.emplace(kb, build_attn(im, im.gctx, il, n, 0, !split_shexp, 2048, true, 2)).first;
        gf = ia->second;
        gf2 = ib->second;
    } else if (cached) {
        const Impl::GKey key { il, 1, cap, (int64_t) n };
        auto it = im.graphs.find(key);
        // a kept graph is a few hundred nodes: room for 2048 (the gctx holds hundreds of them: layers x caps x widths)
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, n, 0, !split_shexp, 2048, true)).first;
        gf = it->second;
    } else {""")
rep("dense", """    const bool ok = im.compute(gf, cached, async);
    if (gc) ggml_free(gc);
    if (!ok) return false;
    const double tp1 = g_decprof_on ? dprof_now() : 0;""", """    bool ok = im.compute(gf, cached, async);
    if (ok && gf2) {
        ok = im.oproj_hook(il, (int) n, im.BL(il, "attn_output_a.weight"), im.BL(il, "attn_output_b.weight"),
                           (const float *) im.o_att->data, (float *) im.o_proj->data);
        if (!ok) im.err = "attention output projection failed";
        ok = ok && im.compute(gf2, cached, async);
    }
    if (gc) ggml_free(gc);
    if (!ok) return false;
    const double tp1 = g_decprof_on ? dprof_now() : 0;""")

# ---------------------------------------------------------------- the tier: each expert card's part as a recorded graph
rep("tc", """        std::vector<int8_t*> xq, aq;      ///< part i >= 1: its groups' int8 o, its int8 wo_a output (8 columns)
        std::vector<float*> xd, oa, ad, y;
    };""", """        std::vector<int8_t*> xq, aq;      ///< part i >= 1: its groups' int8 o, its int8 wo_a output (8 columns)
        std::vector<float*> xd, oa, ad, y;
        /// STRATA_TP_GRAPH (default 1): part i's commands for a width, recorded once (g[i][ncols])
        std::vector<std::array<std::shared_ptr<sycl::ext::oneapi::experimental::command_graph<
            sycl::ext::oneapi::experimental::graph_state::executable>>, 9>> g;
    };""")
rep("tc", """    q0.wait();
    sycl::free(stage, q0);
    if (!ok) return Ds41XeImpl::OSplit();   // (what was allocated stays: a few layers' worth at most, then nothing splits)
    s.ok = true;""", """    q0.wait();
    sycl::free(stage, q0);
    if (!ok) return Ds41XeImpl::OSplit();   // (what was allocated stays: a few layers' worth at most, then nothing splits)
    s.g.resize((size_t) np);
    s.ok = true;""")
rep("tc", """        for (int i = 1; i < s.np; ++i) {
            XeCard& c = *s.c[(size_t) i - 1];
            const size_t r0 = (size_t) s.g0[i] * n;   // its groups' rows of the quantised o
            int8_t* xq = s.xq[(size_t) i];
            const int8_t* xq_src = im.os_xq + r0 * OGD;
            c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.memcpy(xq, xq_src, (size_t) gp * n * OGD); });
            c.q.memcpy(s.xd[(size_t) i], im.os_xd + r0 * (OGD / 32), (size_t) gp * n * (OGD / 32) * sizeof(float));
            os_mv(c.q, s.wa[(size_t) i], OGD, OL, gp, sb, n, xq, s.xd[(size_t) i], s.oa[(size_t) i], KB, OL);
            fastmv::quant_x(c.q, s.oa[(size_t) i], KB, KB, n, 1, KB, s.aq[(size_t) i], s.ad[(size_t) i]);
            os_mv(c.q, s.wb[(size_t) i], KB, D, 1, 0, n, s.aq[(size_t) i], s.ad[(size_t) i], s.y[(size_t) i], D, D * n);
            ev.push_back(c.q.memcpy(im.os_slots + (size_t) (i - 1) * 8 * D, s.y[(size_t) i], (size_t) n * D * sizeof(float)));
        }""", """        static const bool rec = env_i("STRATA_TP_GRAPH", 1) != 0;
        for (int i = 1; i < s.np; ++i) {
            XeCard& c = *s.c[(size_t) i - 1];
            const size_t r0 = (size_t) s.g0[i] * n;   // its groups' rows of the quantised o
            int8_t* xq = s.xq[(size_t) i];
            const int8_t* xq_src = im.os_xq + r0 * OGD;
            // its copies of its groups' int8 o, both products, its partial sum to the dense card (the first copy waits
            // for the dense card's quant_x unless recorded: a replay carries that dependency instead)
            auto enqueue = [&](const sycl::event* dep) {
                if (dep) c.q.submit([&](sycl::handler& h) { h.depends_on(*dep); h.memcpy(xq, xq_src, (size_t) gp * n * OGD); });
                else c.q.memcpy(xq, xq_src, (size_t) gp * n * OGD);
                c.q.memcpy(s.xd[(size_t) i], im.os_xd + r0 * (OGD / 32), (size_t) gp * n * (OGD / 32) * sizeof(float));
                os_mv(c.q, s.wa[(size_t) i], OGD, OL, gp, sb, n, xq, s.xd[(size_t) i], s.oa[(size_t) i], KB, OL);
                fastmv::quant_x(c.q, s.oa[(size_t) i], KB, KB, n, 1, KB, s.aq[(size_t) i], s.ad[(size_t) i]);
                os_mv(c.q, s.wb[(size_t) i], KB, D, 1, 0, n, s.aq[(size_t) i], s.ad[(size_t) i], s.y[(size_t) i], D, D * n);
                return c.q.memcpy(im.os_slots + (size_t) (i - 1) * 8 * D, s.y[(size_t) i], (size_t) n * D * sizeof(float));
            };
            if (rec) {
                namespace sx = sycl::ext::oneapi::experimental;
                auto& eg = s.g[(size_t) i][(size_t) n];
                if (!eg) {
                    sx::command_graph<sx::graph_state::modifiable> g(c.q.get_context(), c.q.get_device());
                    g.begin_recording(c.q);
                    enqueue(nullptr);
                    g.end_recording();
                    eg = std::make_shared<sx::command_graph<sx::graph_state::executable>>(g.finalize());
                }
                sx::command_graph<sx::graph_state::executable>& gr = *eg;
                ev.push_back(c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.ext_oneapi_graph(gr); }));
            } else {
                ev.push_back(enqueue(&e0));
            }
        }""")

# ---------------------------------------------------------------- ds41_generate.cpp: the dense hook
rep("gen", """    if (!mc.cpu_only && std::getenv("STRATA_TP_OPROJ") && std::atoi(std::getenv("STRATA_TP_OPROJ")) != 0)
        ggml_backend_sycl_set_oproj_hook(&Ds41XeTier::oproj_hook, &tier);
""", """    if (!mc.cpu_only && std::getenv("STRATA_TP_OPROJ") && std::atoi(std::getenv("STRATA_TP_OPROJ")) != 0) {
        ggml_backend_sycl_set_oproj_hook(&Ds41XeTier::oproj_hook, &tier);
        // =2: between the two halves of a decode layer's attention graph (both stay replayable SYCL graphs)
        const auto & gg = dense.geom();
        const int64_t ogd = gg.n_head * gg.d_head / gg.o_groups, ol = gg.o_lora, og = gg.o_groups;
        dense.set_oproj_hook([&dense, ogd, ol, og](int, int n, const ggml_tensor * wa, const ggml_tensor * wb, const float * o, float * y) {
            return ggml_backend_sycl_oproj(dense.backend(), wa, ogd, ol, og, wb, o, n, y, wb->ne[1]);
        });
    }
""")

for k, p in FILES.items():
    open(p, "w").write(src[k])
print("patched")
