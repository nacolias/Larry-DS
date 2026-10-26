#!/usr/bin/env python3
# patch-sdpa.py [GGML_DIR] - larry, 2026-10-25: DeepSeek-V4.1's prompt attention as ONE oneDNN graph SDPA (the fused
# micro-kernel; bench/fa/test_sdpa_dsv4: 1.04 ms vs the oneMKL path's 2.30 for a 512-query span at 640 keys, 0.47 vs
# 0.95 at 128).  Two ops:
#   ggml_dsv41_kgather_pad(win, widx, comp, sel, swa, dp) - KGATHER into rows of dp (> 512) halves, zero past 512, with
#     one more key row after the M real ones, [0 .. 0, 1 (at 512), 0 ..] = the sink, then zero rows to a multiple of 16
#   ggml_dsv41_attn_sdpa(q, kp, mask, sinks, scale) - Q' = [scale * q, the head's sink, 0 ..] (the sink key scores the
#     sink; as a value it only touches dim 512, dropped), the mask with 0 for the sink row and -inf past it; a full span
#     of GGML_SYCL_DSV4_SDPA_N (512) queries through the compiled partition (one per shape, cached), any other count
#     through the oneMKL path over the same buffers (row stride dp, the sinks in the softmax).
# The new op is ggml-sycl only (the CPU aborts on it and on a padded KGATHER).  Idempotent.
import sys

L = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml"


def patch(path, pairs, marker):
    s = open(path).read()
    if marker in s:
        print("already patched", path); return
    for old, new, n in pairs:
        if s.count(old) != n:
            sys.exit("%s: anchor x%d (want %d): %s" % (path, s.count(old), n, old[:100]))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


# ---- ggml.h
patch(L + "/include/ggml.h", [
    ("""        GGML_OP_DSV41_SEL_FIN,
""", """        GGML_OP_DSV41_SEL_FIN,
        GGML_OP_DSV41_ATTN_SDPA,
""", 1),
    ("""    // larry: the same with the result type F16 or F32
    GGML_API struct ggml_tensor * ggml_dsv41_kgather2(""", """    // larry: KGATHER into rows of dp (> DH, a multiple of 16) halves, zero past DH; after the swa + K real rows one more,
    // [0 .. 0, 1 (at DH), 0 ..] (ggml_dsv41_attn_sdpa's sink key), then zero rows up to a multiple of 16.  Result F16
    // [dp, Mp, 1, n]; op param 1 = the real rows (swa + K).
    GGML_API struct ggml_tensor * ggml_dsv41_kgather_pad(
            struct ggml_context * ctx,
            struct ggml_tensor  * win,
            struct ggml_tensor  * widx,
            struct ggml_tensor  * comp,
            struct ggml_tensor  * sel,
            int                   swa,
            int                   dp);
    // larry: DeepSeek-V4.1's prompt attention (ggml-sycl only): q F32 [DH, NH, n] RoPE'd, contiguous; kp =
    // ggml_dsv41_kgather_pad's keys (= values); mask F32 / F16 [M, n] (0 / -inf, any column stride); sinks F32 [NH].
    // Result F32 [DH, NH, n] (not de-RoPE'd).  One oneDNN SDPA for a full span (GGML_SYCL_DSV4_SDPA_N queries).
    GGML_API struct ggml_tensor * ggml_dsv41_attn_sdpa(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * kp,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * sinks,
            float                 scale);
    // larry: the same with the result type F16 or F32
    GGML_API struct ggml_tensor * ggml_dsv41_kgather2(""", 1),
], "GGML_OP_DSV41_ATTN_SDPA")

# ---- ggml.c
patch(L + "/src/ggml.c", [
    ("""    "DSV41_SEL_FIN",
""", """    "DSV41_SEL_FIN",
    "DSV41_ATTN_SDPA",
""", 1),
    ("""    "dsv41_sel_fin(ids, nvis, sel, selm)",
""", """    "dsv41_sel_fin(ids, nvis, sel, selm)",
    "dsv41_attn_sdpa(q, kp, mask, sinks)",
""", 1),
    ("""static_assert(GGML_OP_COUNT == 117, "GGML_OP_COUNT != 117");""", """static_assert(GGML_OP_COUNT == 118, "GGML_OP_COUNT != 118");""", 2),
    ("""struct ggml_tensor * ggml_dsv41_kgather2(""", """struct ggml_tensor * ggml_dsv41_kgather_pad(
        struct ggml_context * ctx,
        struct ggml_tensor  * win,
        struct ggml_tensor  * widx,
        struct ggml_tensor  * comp,
        struct ggml_tensor  * sel,
        int                   swa,
        int                   dp) {
    struct ggml_tensor * r = ggml_dsv41_kgather(ctx, win, widx, comp, sel, swa);
    const int64_t DH = r->ne[0], M = r->ne[1], n = r->ne[3];
    GGML_ASSERT(dp > DH && dp % 16 == 0);
    const int64_t Mp = (M + 1 + 15) / 16 * 16;
    struct ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dp, Mp, 1, n);
    ggml_set_op_params_i32(result, 0, swa);
    ggml_set_op_params_i32(result, 1, (int32_t) M);
    result->op = GGML_OP_DSV41_KGATHER;
    for (int i = 0; i < 4; ++i) result->src[i] = r->src[i];
    return result;
}

struct ggml_tensor * ggml_dsv41_attn_sdpa(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * kp,
        struct ggml_tensor  * mask,
        struct ggml_tensor  * sinks,
        float                 scale) {
    GGML_ASSERT(q->type == GGML_TYPE_F32 && ggml_is_contiguous(q) && q->ne[3] == 1);
    GGML_ASSERT(kp->type == GGML_TYPE_F16 && kp->op == GGML_OP_DSV41_KGATHER && kp->ne[3] == q->ne[2] && kp->ne[2] == 1);
    const int64_t DH = q->ne[0], NH = q->ne[1], n = q->ne[2];
    const int64_t M = ggml_get_op_params_i32(kp, 1);
    GGML_ASSERT(M > 0 && kp->ne[0] > DH && kp->ne[1] > M && kp->src[0]->ne[0] == DH);
    GGML_ASSERT((mask->type == GGML_TYPE_F32 || mask->type == GGML_TYPE_F16) && mask->ne[0] >= M && ggml_nrows(mask) == n);
    GGML_ASSERT(sinks && sinks->type == GGML_TYPE_F32 && ggml_nelements(sinks) == NH);
    struct ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DH, NH, n);
    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, (int32_t) M);
    result->op     = GGML_OP_DSV41_ATTN_SDPA;
    result->src[0] = q;
    result->src[1] = kp;
    result->src[2] = mask;
    result->src[3] = sinks;
    return result;
}

struct ggml_tensor * ggml_dsv41_kgather2(""", 1),
], "ggml_dsv41_attn_sdpa")

# ---- ggml-cpu: the new op aborts; a padded KGATHER asserts
patch(L + "/src/ggml-cpu/ggml-cpu.c", [
    ("""        case GGML_OP_DSV41_SEL_FIN:
            {
                GGML_ABORT("DSV41_KV_FIN / _ATTN_DEC / _MVQ / _RMS_Q8 are ggml-sycl ops (no CPU implementation)");""",
     """        case GGML_OP_DSV41_SEL_FIN:
        case GGML_OP_DSV41_ATTN_SDPA:
            {
                GGML_ABORT("DSV41_KV_FIN / _ATTN_DEC / _MVQ / _RMS_Q8 / _ATTN_SDPA are ggml-sycl ops (no CPU implementation)");""", 1),
    ("""        case GGML_OP_DSV41_SEL_FIN:
            {
                n_tasks = 1;""", """        case GGML_OP_DSV41_SEL_FIN:
        case GGML_OP_DSV41_ATTN_SDPA:
            {
                n_tasks = 1;""", 1),
], "GGML_OP_DSV41_ATTN_SDPA")
patch(L + "/src/ggml-cpu/ops.cpp", [
    ("""    const int64_t SWA = ggml_get_op_params_i32(dst, 0);
    std::vector<float> tmp((size_t) DH);""", """    const int64_t SWA = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(ggml_get_op_params_i32(dst, 1) == 0 && "ggml_dsv41_kgather_pad is ggml-sycl only");   // larry, patch-sdpa
    std::vector<float> tmp((size_t) DH);""", 1),
], "ggml_dsv41_kgather_pad is ggml-sycl only")

# ---- ggml-sycl: the padded KGATHER
G = L + "/src/ggml-sycl"
patch(G + "/dsv41-fused.cpp", [
    ("""    const int64_t DH = dst->ne[0], M = dst->ne[1], n = dst->ne[3];
    const int64_t SWA = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(DH % 8 == 0 && win->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));""",
     """    const int64_t DH = dst->ne[0], M = dst->ne[1], n = dst->ne[3];
    const int64_t SWA = ggml_get_op_params_i32(dst, 0);
    // larry, patch-sdpa (ggml_dsv41_kgather_pad): rows of DP > DH halves, the M real ones, then the sink row, then zeros
    const int64_t MR = ggml_get_op_params_i32(dst, 1);
    if (MR > 0) {
        const int64_t DS = win->ne[0], DP = DH, MP = M;
        GGML_ASSERT(dst->type == GGML_TYPE_F16 && DS % 8 == 0 && DP % 8 == 0 && DP > DS && MP > MR && ggml_is_contiguous(dst));
        GGML_ASSERT(win->type == GGML_TYPE_F32 && win->nb[1] % 16 == 0 && (!comp || comp->nb[1] % 16 == 0));
        const char * pw = (const char *) win->data;
        const size_t wnb = win->nb[1];
        const int32_t * pwi = (const int32_t *) widx->data;
        const char * pc = comp ? (const char *) comp->data : nullptr;
        const size_t cnb = comp ? comp->nb[1] : 0;
        const int ct = !comp ? 0 : comp->type == GGML_TYPE_BF16 ? 1 : comp->type == GGML_TYPE_F16 ? 2 : 0;
        const char * ps = sel ? (const char *) sel->data : nullptr;
        const size_t snb = sel ? sel->nb[1] : 0;
        sycl::half * po = (sycl::half *) dst->data;
        const int64_t VS = DS / 8, VP = DP / 8;
        ctx.stream()->parallel_for(sycl::range<1>((size_t) n * MP * VP), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int64_t v = (int64_t) (i % VP), row = (int64_t) (i / VP), m = row % MP, t = row / MP;
            sycl::vec<sycl::half, 8> o(0.0f);
            if (m < MR && v < VS) {
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
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = (sycl::half) f[j];
            } else if (m == MR && v == VS) {
                o[0] = (sycl::half) 1.0f;   // the sink key: 1 at dim DS
            }
            *(sycl::vec<sycl::half, 8> *) (po + (size_t) row * DP + v * 8) = o;
        });
        return;
    }
    GGML_ASSERT(DH % 8 == 0 && win->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));""", 1),
], "patch-sdpa (ggml_dsv41_kgather_pad)")

# ---- ggml-sycl: the SDPA op (fattn-dsv4.cpp / .hpp), its dispatch
patch(G + "/fattn-dsv4.hpp", [
    ("""#endif  // GGML_SYCL_FATTN_DSV4_HPP""", """// larry, patch-sdpa: GGML_OP_DSV41_ATTN_SDPA - the prompt attention as one oneDNN graph SDPA (fattn-dsv4.cpp)
void ggml_sycl_op_dsv41_attn_sdpa(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_FATTN_DSV4_HPP""", 1),
], "ggml_sycl_op_dsv41_attn_sdpa")
patch(G + "/fattn-dsv4.cpp", [
    ("""#include <cmath>
#include <cstdlib>
""", """#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

#if GGML_SYCL_DNNL
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#endif
""", 1),
    ("""bool ggml_sycl_fattn_dsv4_dec_ok(const ggml_tensor * dst) {""", """// ---- larry, patch-sdpa: GGML_OP_DSV41_ATTN_SDPA
#if GGML_SYCL_DNNL
namespace {
struct SdpaPart {
    dnnl::graph::compiled_partition cp;
    std::vector<dnnl::graph::logical_tensor> ins;
    dnnl::graph::logical_tensor out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_s = 0, id_m = 0;
    bool ok = false;
};
// fattn-onednn.cpp's GQA form: batch = the queries, one KV head, a group of NH = the heads, a query length of 1
SdpaPart sdpa_build(const dnnl::engine & eng, int n, int NH, int Mp, int Dp) try {
    using namespace dnnl::graph;
    using ltype = logical_tensor::layout_type;
    using dt = logical_tensor::data_type;
    const logical_tensor::dims q_sz = {n, 1, NH, 1, Dp}, kv_sz = {n, 1, 1, Mp, Dp}, s_sz = {n, 1, NH, 1, Mp},
                               sc = {1, 1, 1, 1, 1}, msk = {n, 1, 1, 1, Mp}, o_sz = {n, 1, NH, 1, Dp};
    int64_t id = 0;
    SdpaPart E;
    auto query = logical_tensor(id++, dt::f16, q_sz, ltype::strided);
    auto key = logical_tensor(id++, dt::f16, kv_sz, ltype::strided);
    auto score = logical_tensor(id++, dt::f32, s_sz, ltype::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});
    auto scale_t = logical_tensor(id++, dt::f32, sc, ltype::strided);
    auto scaled = logical_tensor(id++, dt::f32, s_sz, ltype::strided);
    auto sdiv = op(id++, op::kind::Divide, "scale_div");
    sdiv.add_inputs({score, scale_t}); sdiv.add_outputs({scaled});
    auto mask = logical_tensor(id++, dt::f16, msk, ltype::strided);
    auto masked = logical_tensor(id++, dt::f32, s_sz, ltype::strided);
    auto madd = op(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask}); madd.add_outputs({masked});
    auto probs = logical_tensor(id++, dt::f16, s_sz, ltype::strided);
    auto smax = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked}); smax.add_outputs({probs});
    auto value = logical_tensor(id++, dt::f16, kv_sz, ltype::strided);
    auto output = logical_tensor(id++, dt::f16, o_sz, ltype::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});
    dnnl::graph::graph g(eng.get_kind());
    g.add_op(bmm1); g.add_op(sdiv); g.add_op(madd); g.add_op(smax); g.add_op(bmm2);
    g.finalize();
    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the DSV4 SDPA graph (n %d, Mp %d, Dp %d) - oneMKL path\\n", __func__, n, Mp, Dp);
        return E;
    }
    E.ins = parts[0].get_input_ports();
    E.out = parts[0].get_output_ports()[0];
    E.cp = parts[0].compile(E.ins, {E.out}, eng);
    E.out = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q = query.get_id(); E.id_k = key.get_id(); E.id_v = value.get_id(); E.id_s = scale_t.get_id(); E.id_m = mask.get_id();
    E.ok = true;
    return E;
} catch (const std::exception & e) {
    GGML_LOG_WARN("%s: oneDNN SDPA build failed (%s) - oneMKL path\\n", __func__, e.what());
    return {};
}
}  // namespace
#endif

void ggml_sycl_op_dsv41_attn_sdpa(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    namespace mkl = oneapi::mkl;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * KP = dst->src[1];
    const ggml_tensor * MK = dst->src[2];
    const ggml_tensor * SK = dst->src[3];
    const int64_t D = Q->ne[0], NH = Q->ne[1], n = Q->ne[2], Dp = KP->ne[0], Mp = KP->ne[1];
    const int M = ggml_get_op_params_i32(dst, 1);
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    dpct::queue_ptr stream = ctx.stream();
    ggml_sycl_pool & pool = ctx.pool();
    // Q' [n][NH][Dp] f16: scale * q, then the head's sink (the sink key scores it), then zeros
    ggml_sycl_pool_alloc<sycl::half> Qp(pool, (size_t) n * NH * Dp);
    sycl::half * qp = Qp.get();
    {
        const float * q = (const float *) Q->data;
        const float * sk = (const float *) SK->data;
        stream->parallel_for(sycl::range<1>((size_t) n * NH * (Dp / 4)), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int64_t v = (int64_t) (i % (size_t) (Dp / 4)), bh = (int64_t) (i / (size_t) (Dp / 4)), d = v * 4;
            sycl::half4 o(0.0f);
            if (d < D) {
                const sycl::float4 x = *(const sycl::float4 *) (q + bh * D + d) * scale;
                o = sycl::half4((sycl::half) x.x(), (sycl::half) x.y(), (sycl::half) x.z(), (sycl::half) x.w());
            } else if (d == D) {
                o[0] = (sycl::half) sk[bh % NH];
            }
            *(sycl::half4 *) (qp + bh * Dp + d) = o;
        });
    }
    const bool m16 = MK->type == GGML_TYPE_F16;
    const size_t mstride = MK->nb[1] / ggml_type_size(MK->type);
    const void * mkd = MK->data;
#if GGML_SYCL_DNNL
    static const int sdpa_n = env_int("GGML_SYCL_DSV4_SDPA_N", 512);
    if (n == sdpa_n && D % 4 == 0 && Dp % 4 == 0) {
        static std::mutex mu;
        static std::map<std::tuple<const void *, int64_t, int64_t, int64_t, int64_t>, SdpaPart> parts;
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        SdpaPart * P = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu);
            const auto key = std::make_tuple((const void *) stream, n, NH, Mp, Dp);
            auto it = parts.find(key);
            if (it == parts.end()) {
                const auto t0 = std::chrono::steady_clock::now();
                it = parts.emplace(key, sdpa_build(eng, (int) n, (int) NH, (int) Mp, (int) Dp)).first;
                GGML_LOG_INFO("%s: oneDNN SDPA (n %lld, %lld heads, %lld keys, %lld dims) %s in %.0f ms\\n", __func__, (long long) n,
                              (long long) NH, (long long) Mp, (long long) Dp, it->second.ok ? "compiled" : "NOT available",
                              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            P = &it->second;
        }
        if (P->ok) {
            ggml_sycl_pool_alloc<sycl::half> Ms(pool, (size_t) n * Mp);
            ggml_sycl_pool_alloc<sycl::half> Op(pool, (size_t) n * NH * Dp);
            ggml_sycl_pool_alloc<float> one(pool, 1);
            sycl::half * ms = Ms.get();
            float * onep = one.get();
            stream->parallel_for(sycl::range<1>((size_t) n * Mp), [=](sycl::id<1> id) {
                const size_t i = id[0];
                const int64_t j = (int64_t) (i % (size_t) Mp), b = (int64_t) (i / (size_t) Mp);
                float x = -INFINITY;
                if (j < M) x = m16 ? (float) ((const sycl::half *) mkd)[b * mstride + j] : ((const float *) mkd)[b * mstride + j];
                else if (j == M) x = 0.0f;
                ms[i] = (sycl::half) x;
                if (i == 0) *onep = 1.0f;   // Q' carries the scale: the graph divides by 1
            });
            std::vector<dnnl::graph::tensor> tin;
            tin.reserve(P->ins.size());
            for (auto & lt : P->ins) {
                const size_t lid = lt.get_id();
                void * ptr = lid == P->id_q ? (void *) qp : (lid == P->id_k || lid == P->id_v) ? KP->data :
                             lid == P->id_s ? (void *) onep : (void *) ms;
                tin.emplace_back(lt, eng, ptr);
            }
            dnnl::graph::tensor tout(P->out, eng, Op.get());
            P->cp.execute(strm, tin, {tout});
            // O' f16 [n][NH][Dp] -> dst f32 [n][NH][D]
            const sycl::half * op = Op.get();
            float * o = (float *) dst->data;
            stream->parallel_for(sycl::range<1>((size_t) n * NH * (D / 4)), [=](sycl::id<1> id) {
                const size_t i = id[0];
                const int64_t v = (int64_t) (i % (size_t) (D / 4)), bh = (int64_t) (i / (size_t) (D / 4));
                const sycl::half4 h = *(const sycl::half4 *) (op + bh * Dp + v * 4);
                *(sycl::float4 *) (o + bh * D + v * 4) = sycl::float4((float) h[0], (float) h[1], (float) h[2], (float) h[3]);
            });
            return;
        }
    }
#endif
    // the oneMKL path over the same buffers: S_b = Q'_b [NH x D] . K'_b^T (the first M rows, row stride Dp), the
    // softmax with the mask and the sinks, O_b = P_b . K'_b -> dst
    ggml_sycl_pool_alloc<float> S(pool, (size_t) n * NH * M);
    ggml_sycl_pool_alloc<sycl::half> P(pool, (size_t) n * NH * M);
    float * s = S.get();
    sycl::half * p = P.get();
    const sycl::half * kp = (const sycl::half *) KP->data;
    mkl::blas::row_major::gemm_batch(*stream, mkl::transpose::nontrans, mkl::transpose::trans, NH, M, D, 1.0f, qp, Dp,
                                     NH * Dp, kp, Dp, Mp * Dp, 0.0f, s, M, NH * M, n);
    {
        const float * sk = (const float *) SK->data;
        constexpr int SGW = 8;
        const size_t rows = (size_t) n * NH;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((rows + SGW - 1) / SGW * SGW * 16), sycl::range<1>(SGW * 16)),
                             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const size_t row = (size_t) it.get_group(0) * SGW + sg.get_group_linear_id();
            if (row >= rows) return;
            const int lane = (int) sg.get_local_linear_id();
            const int64_t b = (int64_t) (row / NH), h = (int64_t) (row % NH);
            const float * sr = s + (int64_t) row * M;
            auto mv = [&](int64_t m) { return m16 ? (float) ((const sycl::half *) mkd)[b * mstride + m] : ((const float *) mkd)[b * mstride + m]; };
            float mx = sk[h];
            for (int64_t m = lane; m < M; m += 16) mx = sycl::fmax(mx, sr[m] + mv(m));
            mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t m = lane; m < M; m += 16) sum += sycl::exp(sr[m] + mv(m) - mx);
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>()) + sycl::exp(sk[h] - mx);
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            sycl::half * pr = p + (int64_t) row * M;
            for (int64_t m = lane; m < M; m += 16) pr[m] = (sycl::half) (sycl::exp(sr[m] + mv(m) - mx) * inv);
        });
    }
    mkl::blas::row_major::gemm_batch(*stream, mkl::transpose::nontrans, mkl::transpose::nontrans, NH, D, M, 1.0f, p, M,
                                     NH * M, kp, Dp, Mp * Dp, 0.0f, (float *) dst->data, D, NH * D, n);
}

bool ggml_sycl_fattn_dsv4_dec_ok(const ggml_tensor * dst) {""", 1),
], "patch-sdpa: GGML_OP_DSV41_ATTN_SDPA")
patch(G + "/ggml-sycl.cpp", [
    ("""#include "ggml-sycl/dsv41-fused.hpp"
""", """#include "ggml-sycl/dsv41-fused.hpp"
#include "ggml-sycl/fattn-dsv4.hpp"   // larry, patch-sdpa: ggml_sycl_op_dsv41_attn_sdpa
""", 1),
    ("""        case GGML_OP_DSV41_ATTN_DEC:
            ggml_sycl_op_dsv41_attn_dec(ctx, dst);
            break;""", """        case GGML_OP_DSV41_ATTN_DEC:
            ggml_sycl_op_dsv41_attn_dec(ctx, dst);
            break;
        case GGML_OP_DSV41_ATTN_SDPA:   // larry, patch-sdpa
            ggml_sycl_op_dsv41_attn_sdpa(ctx, dst);
            break;""", 1),
    ("""        case GGML_OP_DSV41_ATTN_DEC:
            return op->ne[2] < 8 && op->src[1]->nb[1] % 16 == 0 && op->src[2]->nb[1] % 16 == 0 &&""",
     """        case GGML_OP_DSV41_ATTN_SDPA:   // larry, patch-sdpa
            return op->src[0]->ne[0] % 4 == 0 && op->src[1]->ne[0] % 16 == 0;
        case GGML_OP_DSV41_ATTN_DEC:
            return op->ne[2] < 8 && op->src[1]->nb[1] % 16 == 0 && op->src[2]->nb[1] % 16 == 0 &&""", 1),
], "GGML_OP_DSV41_ATTN_SDPA")
