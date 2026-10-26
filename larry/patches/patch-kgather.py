# patch-kgather.py - ggml (larry): GGML_OP_DSV41_KGATHER, DeepSeek-V4.1's per-query attention key set in ONE op
# (ggml-sycl kernel + a CPU reference): out[t] = [ f16(win rows at widx[t]) | f16(comp rows at sel[t]) ], F16
# [DH, SWA + K, 1, n] - the prompt path's get_rows (window) + get_rows (compressed) + concat + cast chain, which wrote
# and re-read ~2 GB per 512-query pass.  Plus GGML_SYCL_LARRY_F16GEMM: prompt-sized mul_mats on the fp16 (XMX) path.
root = '/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/'
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80])
        s = s.replace(old, new)
    open(path, 'w').write(s)

h = root + 'include/ggml.h'
if 'GGML_OP_DSV41_KGATHER' not in open(h).read():
    patch(h, [('        GGML_OP_DSV41_ROUTER,\n', '        GGML_OP_DSV41_ROUTER,\n        GGML_OP_DSV41_KGATHER,\n'),
              ('''            bool                  norm,
            float                 scale);

    // custom operators''', '''            bool                  norm,
            float                 scale);

    // larry: DeepSeek-V4.1's per-query attention keys in one op.  win F32 [DH, W] rows, widx I32 [swa * n] contiguous
    // (rows of win), comp F32 / F16 / BF16 [DH, C] rows and sel I32 [K, n] (rows of comp; both null: no compressed
    // rows).  Result F16 [DH, swa + K, 1, n]: query t's swa window rows, then its K compressed rows.
    GGML_API struct ggml_tensor * ggml_dsv41_kgather(
            struct ggml_context * ctx,
            struct ggml_tensor  * win,
            struct ggml_tensor  * widx,
            struct ggml_tensor  * comp,
            struct ggml_tensor  * sel,
            int                   swa);

    // custom operators''')])

c = root + 'src/ggml.c'
if 'DSV41_KGATHER' not in open(c).read():
    s = open(c).read()
    s = s.replace('    "DSV41_ROUTER",\n', '    "DSV41_ROUTER",\n    "DSV41_KGATHER",\n', 1)
    s = s.replace('    "dsv41_router(logits, bias, sel_bias)",\n', '    "dsv41_router(logits, bias, sel_bias)",\n    "dsv41_kgather(win, widx, comp, sel)",\n', 1)
    assert s.count('static_assert(GGML_OP_COUNT == 104, "GGML_OP_COUNT != 104");') == 2
    s = s.replace('static_assert(GGML_OP_COUNT == 104, "GGML_OP_COUNT != 104");', 'static_assert(GGML_OP_COUNT == 105, "GGML_OP_COUNT != 105");')
    old = '''    result->op     = GGML_OP_DSV41_ROUTER;
    result->src[0] = logits;
    result->src[1] = bias;
    result->src[2] = sel_bias;
    return result;
}
'''
    assert s.count(old) == 1
    s = s.replace(old, old + '''
struct ggml_tensor * ggml_dsv41_kgather(
        struct ggml_context * ctx,
        struct ggml_tensor  * win,
        struct ggml_tensor  * widx,
        struct ggml_tensor  * comp,
        struct ggml_tensor  * sel,
        int                   swa) {
    GGML_ASSERT(win->type == GGML_TYPE_F32 && win->nb[0] == 4 && win->ne[2] == 1 && win->ne[3] == 1);
    GGML_ASSERT(widx->type == GGML_TYPE_I32 && ggml_is_contiguous(widx) && swa > 0 && ggml_nelements(widx) % swa == 0);
    GGML_ASSERT((comp == NULL) == (sel == NULL));
    const int64_t DH = win->ne[0], n = ggml_nelements(widx) / swa;
    int64_t K = 0;
    if (comp) {
        GGML_ASSERT(comp->type == GGML_TYPE_F32 || comp->type == GGML_TYPE_F16 || comp->type == GGML_TYPE_BF16);
        GGML_ASSERT(comp->ne[0] == DH && comp->nb[0] == ggml_type_size(comp->type));
        GGML_ASSERT(sel->type == GGML_TYPE_I32 && sel->nb[0] == 4 && sel->ne[1] == n && sel->ne[2] == 1 && sel->ne[3] == 1);
        K = sel->ne[0];
    }
    struct ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DH, swa + K, 1, n);
    ggml_set_op_params_i32(result, 0, swa);
    result->op     = GGML_OP_DSV41_KGATHER;
    result->src[0] = win;
    result->src[1] = widx;
    result->src[2] = comp;
    result->src[3] = sel;
    return result;
}
''', 1)
    open(c, 'w').write(s)

cc = root + 'src/ggml-cpu/ggml-cpu.c'
if 'GGML_OP_DSV41_KGATHER' not in open(cc).read():
    patch(cc, [('''        case GGML_OP_DSV41_ROUTER:
            {
                ggml_compute_forward_dsv41_router(params, tensor);
            } break;''', '''        case GGML_OP_DSV41_ROUTER:
            {
                ggml_compute_forward_dsv41_router(params, tensor);
            } break;
        case GGML_OP_DSV41_KGATHER:
            {
                ggml_compute_forward_dsv41_kgather(params, tensor);
            } break;'''),
               ('''        case GGML_OP_DSV41_ROUTER:
            {
                n_tasks = 1;
            } break;''', '''        case GGML_OP_DSV41_ROUTER:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DSV41_KGATHER:
            {
                n_tasks = n_threads;
            } break;''')])
oh = root + 'src/ggml-cpu/ops.h'
if 'dsv41_kgather' not in open(oh).read():
    patch(oh, [('void ggml_compute_forward_dsv41_router(const struct ggml_compute_params * params, struct ggml_tensor * dst);',
                'void ggml_compute_forward_dsv41_router(const struct ggml_compute_params * params, struct ggml_tensor * dst);\n'
                'void ggml_compute_forward_dsv41_kgather(const struct ggml_compute_params * params, struct ggml_tensor * dst);')])
oc = root + 'src/ggml-cpu/ops.cpp'
if 'ggml_compute_forward_dsv41_kgather' not in open(oc).read():
    s = open(oc).read()
    s += '''
// ggml_compute_forward_dsv41_kgather (larry): the reference of ggml-sycl's fused per-query attention-key gather
void ggml_compute_forward_dsv41_kgather(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * win = dst->src[0];
    const ggml_tensor * widx = dst->src[1];
    const ggml_tensor * comp = dst->src[2];
    const ggml_tensor * sel = dst->src[3];
    const int64_t DH = dst->ne[0], M = dst->ne[1], n = dst->ne[3];
    const int64_t SWA = ggml_get_op_params_i32(dst, 0);
    std::vector<float> tmp((size_t) DH);
    for (int64_t row = params->ith; row < n * M; row += params->nth) {
        const int64_t m = row % M, t = row / M;
        const float * src = nullptr;
        if (m < SWA) {
            src = (const float *) ((const char *) win->data + (size_t) ((const int32_t *) widx->data)[t * SWA + m] * win->nb[1]);
        } else {
            const int32_t r = *(const int32_t *) ((const char *) sel->data + (size_t) t * sel->nb[1] + (size_t) (m - SWA) * 4);
            const char * s = (const char *) comp->data + (size_t) r * comp->nb[1];
            if (comp->type == GGML_TYPE_F32) src = (const float *) s;
            else { ggml_get_type_traits(comp->type)->to_float(s, tmp.data(), DH); src = tmp.data(); }
        }
        ggml_fp16_t * o = (ggml_fp16_t *) dst->data + (size_t) row * DH;
        for (int64_t d = 0; d < DH; ++d) o[d] = GGML_CPU_FP32_TO_FP16(src[d]);
    }
}
'''
    open(oc, 'w').write(s)

sy = root + 'src/ggml-sycl/ggml-sycl.cpp'
if 'GGML_OP_DSV41_KGATHER' not in open(sy).read():
    patch(sy, [('''        case GGML_OP_DSV41_ROUTER:
            ggml_sycl_op_dsv41_router(ctx, dst);
            break;''', '''        case GGML_OP_DSV41_ROUTER:
            ggml_sycl_op_dsv41_router(ctx, dst);
            break;
        case GGML_OP_DSV41_KGATHER:
            ggml_sycl_op_dsv41_kgather(ctx, dst);
            break;'''),
               ('''        case GGML_OP_DSV41_ROUTER:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->type == GGML_TYPE_F32;''',
                '''        case GGML_OP_DSV41_ROUTER:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV41_KGATHER:
            return op->ne[0] % 8 == 0 && op->src[0]->nb[1] % 16 == 0 && (!op->src[2] || op->src[2]->nb[1] % 16 == 0);'''),
               ('''#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    bool use_fp16 = false;
#endif''', '''#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    // larry: GGML_SYCL_LARRY_F16GEMM (default 1) - prompt-sized products (>= 32 columns) of F16 / quantized weights on
    // the fp16 path (src0 / src1 in f16, XMX GEMM, f32 accumulation and result); the f32 path has no XMX (a long
    // prompt's dense GEMMs were ~28% of card 0's time).  Products asking for GGML_PREC_F32 stay on the f32 path.
    static const int larry_f16 = ggml_sycl_get_env("GGML_SYCL_LARRY_F16GEMM", 1);
    bool use_fp16 = larry_f16 && src1_ncols >= 32;
#endif''')])
fh = root + 'src/ggml-sycl/dsv41-fused.hpp'
if 'dsv41_kgather' not in open(fh).read():
    patch(fh, [('void ggml_sycl_op_dsv41_router(ggml_backend_sycl_context & ctx, ggml_tensor * dst);',
                'void ggml_sycl_op_dsv41_router(ggml_backend_sycl_context & ctx, ggml_tensor * dst);\n'
                '// larry: DeepSeek-V4.1\'s per-query attention keys in f16 (GGML_OP_DSV41_KGATHER)\n'
                'void ggml_sycl_op_dsv41_kgather(ggml_backend_sycl_context & ctx, ggml_tensor * dst);')])
fc = root + 'src/ggml-sycl/dsv41-fused.cpp'
if 'ggml_sycl_op_dsv41_kgather' not in open(fc).read():
    s = open(fc).read()
    s += '''
// GGML_OP_DSV41_KGATHER (larry): query t's window rows (win at widx) then its compressed rows (comp at sel), each
// converted to f16 once - one work-item per 8 values of an output row (16-byte loads and stores)
void ggml_sycl_op_dsv41_kgather(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    const ggml_tensor * win = dst->src[0];
    const ggml_tensor * widx = dst->src[1];
    const ggml_tensor * comp = dst->src[2];
    const ggml_tensor * sel = dst->src[3];
    const int64_t DH = dst->ne[0], M = dst->ne[1], n = dst->ne[3];
    const int64_t SWA = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(DH % 8 == 0 && win->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16 && ggml_is_contiguous(dst));
    GGML_ASSERT(win->nb[1] % 16 == 0 && (!comp || comp->nb[1] % 16 == 0));
    const char * pw = (const char *) win->data;
    const size_t wnb = win->nb[1];
    const int32_t * pwi = (const int32_t *) widx->data;
    const char * pc = comp ? (const char *) comp->data : nullptr;
    const size_t cnb = comp ? comp->nb[1] : 0;
    const int ct = !comp ? 0 : comp->type == GGML_TYPE_BF16 ? 1 : comp->type == GGML_TYPE_F16 ? 2 : 0;
    const char * ps = sel ? (const char *) sel->data : nullptr;
    const size_t snb = sel ? sel->nb[1] : 0;
    sycl::half * po = (sycl::half *) dst->data;
    const int64_t V = DH / 8;
    const size_t total = (size_t) n * M * V;
    ctx.stream()->parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int64_t v = (int64_t) (i % V), row = (int64_t) (i / V), m = row % M, t = row / M;
        float f[8];
        if (m < SWA) {
            const float * s = (const float *) (pw + (size_t) pwi[t * SWA + m] * wnb) + v * 8;
            const sycl::float4 a = *(const sycl::float4 *) s, b = *(const sycl::float4 *) (s + 4);
            f[0] = a.x(); f[1] = a.y(); f[2] = a.z(); f[3] = a.w(); f[4] = b.x(); f[5] = b.y(); f[6] = b.z(); f[7] = b.w();
        } else {
            const int32_t r = *(const int32_t *) (ps + (size_t) t * snb + (size_t) (m - SWA) * 4);
            const char * s = pc + (size_t) r * cnb;
            if (ct == 1) {
                const sycl::uint4 u = *(const sycl::uint4 *) (s + v * 16);
                const uint32_t w[4] = { u.x(), u.y(), u.z(), u.w() };
#pragma unroll
                for (int j = 0; j < 4; ++j) { f[2 * j] = sycl::bit_cast<float>(w[j] << 16); f[2 * j + 1] = sycl::bit_cast<float>(w[j] & 0xffff0000u); }
            } else if (ct == 2) {
                const sycl::vec<sycl::half, 8> hv = *(const sycl::vec<sycl::half, 8> *) (s + v * 16);
#pragma unroll
                for (int j = 0; j < 8; ++j) f[j] = (float) hv[j];
            } else {
                const float * sf = (const float *) s + v * 8;
                const sycl::float4 a = *(const sycl::float4 *) sf, b = *(const sycl::float4 *) (sf + 4);
                f[0] = a.x(); f[1] = a.y(); f[2] = a.z(); f[3] = a.w(); f[4] = b.x(); f[5] = b.y(); f[6] = b.z(); f[7] = b.w();
            }
        }
        sycl::vec<sycl::half, 8> o;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = (sycl::half) f[j];
        *(sycl::vec<sycl::half, 8> *) (po + (size_t) row * DH + v * 8) = o;
    });
}
'''
    open(fc, 'w').write(s)
print('ggml patched')

# ---- the engine: the prompt path's key set through ggml_dsv41_kgather (STRATA_KGATHER, default 1)
d = '/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp'
if 'STRATA_KGATHER' not in open(d).read():
    patch(d, [('''    ggml_tensor * Kw = ggml_reshape_3d(gc, ggml_get_rows(gc, wsrc, b.v1(im.i_widx, SWA * n, (size_t) col0 * SWA * 4)), DH, SWA, n);''',
               '''    ggml_tensor * widx_v = b.v1(im.i_widx, SWA * n, (size_t) col0 * SWA * 4);
    ggml_tensor * Kw = ggml_reshape_3d(gc, ggml_get_rows(gc, wsrc, widx_v), DH, SWA, n);'''),
              ('''    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;''', '''    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    ggml_tensor * kg_comp = nullptr, * kg_sel = nullptr;   // Kc's source rows and their ids (ggml_dsv41_kgather)
    ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;'''),
              ('''            Kc = ggml_reshape_3d(gc, ggml_get_rows(gc, comp, ggml_cont(gc, ggml_reshape_1d(gc, ggml_cont(gc, sel), K * n))), DH, K, n);''',
               '''            Kc = ggml_reshape_3d(gc, ggml_get_rows(gc, comp, ggml_cont(gc, ggml_reshape_1d(gc, ggml_cont(gc, sel), K * n))), DH, K, n);
            kg_comp = comp;
            kg_sel = sel;'''),
              ('''    ggml_tensor * k4 = ggml_reshape_4d(gc, Kall, DH, M, 1, n);
    if (!sycl_dec) k4 = ggml_cast(gc, k4, GGML_TYPE_F16);''',
               '''    // larry: STRATA_KGATHER (default 1): the prompt path's keys gathered straight into f16 by one op - the get_rows /
    // concat / cast chain behind Kall wrote and re-read ~2 GB per 512-query pass (~40% of a long prompt's card-0 time);
    // Kw / Kc / Kall are then unused and drop out of the graph
    static const bool kgather = env_on("STRATA_KGATHER", true);
    ggml_tensor * k4 = nullptr;
    if (sycl_mqa && kgather) {
        k4 = ggml_dsv41_kgather(gc, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, (int) SWA);
        GGML_ASSERT(k4->ne[1] == M);
    } else {
        k4 = ggml_reshape_4d(gc, Kall, DH, M, 1, n);
        if (!sycl_dec) k4 = ggml_cast(gc, k4, GGML_TYPE_F16);
    }''')])
print('engine patched')
