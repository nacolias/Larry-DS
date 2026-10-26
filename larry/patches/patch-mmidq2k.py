#!/usr/bin/env python3
# patch-mmidq2k.py - larry, GGML_SYCL_LARRY_MMID_Q2K (default 1): mul_mat_id of Q2_K experts for a few tokens (the
# DSpark draft block: 5 tokens x 6 experts) as ONE device-side launch - the routed path below it copies the ids to the
# host and drains the queue per call (9 drains a draft), then runs one ggml mat-vec per expert at 28-49 GB/s.
# Activations int8 per 32 (fastmv::quant_x, as q8_1 rounds them); every (token, slot) pair's rows are dotted with dp4a:
# a 16-value group's sum is d * sc * sum(q x) - dmin * m * sum(x), scaled by the 32-block's activation scale.
# Idempotent.
import sys
S = "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(S).read()
if "GGML_SYCL_LARRY_MMID_Q2K" in s:
    print("already"); sys.exit(0)

KERNEL = r'''// larry, GGML_SYCL_LARRY_MMID_Q2K: Q2_K experts x a few tokens, one launch (see ggml_sycl_mul_mat_id).  Pair p =
// token t * n_ids + slot u runs expert ids[t][u] on activation row t (ne11 == 1) or t * n_ids + u; a sub-group owns
// R rows of one pair, a lane 64-value units (one 128-half's 16 qs bytes, quarter lh: values 32 j + 16 lh + i, j = 0..3).
static bool larry_mmid_q2k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_LARRY_MMID_Q2K", 1);
    if (!on || src0->type != GGML_TYPE_Q2_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    const int64_t K = src0->ne[0], N = src0->ne[1], n_as = src0->ne[2];
    const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2], n_ids = ids->ne[0];
    if (K % QK_K || src1->ne[0] != K || ne12 < 1 || ne12 > 8 || n_ids < 1 || n_ids > 8 || ids->ne[1] != ne12) return false;
    if (ne11 != 1 && ne11 != n_ids) return false;
    if (src1->nb[0] != 4 || src1->ne[3] != 1 || dst->ne[0] != N || dst->nb[0] != 4) return false;
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != 4) return false;
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (extra && extra->optimized_feature.reorder) return false;   // the standard block layout only
    if (src0->nb[1] != (size_t) (K / QK_K) * sizeof(block_q2_K)) return false;
    dpct::queue_ptr st = ctx.stream();
    const int64_t nrows_x = ne11 * ne12;
    ggml_sycl_pool_alloc<int8_t> xq(ctx.pool(), (size_t) (nrows_x * K));
    ggml_sycl_pool_alloc<float> xd(ctx.pool(), (size_t) (nrows_x * K / 32));
    fastmv::quant_x(*st, (const float *) src1->data, (int64_t) (src1->nb[1] / 4), (int64_t) (src1->nb[2] / 4), (int) ne11, ne12,
                    K, xq.get(), xd.get());
    constexpr int R = 2, SGW = 8;   // rows per sub-group, sub-groups per work-group
    const int64_t rows_wg = (int64_t) R * SGW, nrg = (N + rows_wg - 1) / rows_wg, P = ne12 * n_ids;
    const char * w = (const char *) src0->data;
    const size_t nb02 = src0->nb[2], nb01 = src0->nb[1];
    const int32_t * idp = (const int32_t *) ids->data;
    const int64_t ids_ld = (int64_t) (ids->nb[1] / 4);
    float * y = (float *) dst->data;
    const int64_t y1 = (int64_t) (dst->nb[1] / 4), y2 = (int64_t) (dst->nb[2] / 4);
    const int8_t * XQ = xq.get();
    const float * XD = xd.get();
    const int64_t bpr = K / QK_K, upr = 4 * bpr;   // 64-value units a row
    st->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) P, (size_t) (nrg * SGW * 16)), sycl::range<2>(1, (size_t) SGW * 16)),
                     [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t p = it.get_global_id(0);
        const int64_t t = p / n_ids, u = p % n_ids;
        const int32_t e = idp[t * ids_ld + u];
        const sycl::sub_group sg = it.get_sub_group();
        const int sgi = (int) sg.get_group_linear_id(), ln = (int) sg.get_local_linear_id();
        const int64_t row0 = ((int64_t) it.get_group(1) * SGW + sgi) * R;
        if (e < 0 || e >= n_as || row0 >= N) return;   // a whole sub-group leaves together
        const int64_t xr = ne11 == 1 ? t : t * n_ids + u;
        const int8_t * xq_r = XQ + xr * K;
        const float * xd_r = XD + xr * (K / 32);
        const char * we = w + (size_t) e * nb02;
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.0f;
        for (int64_t un = ln; un < upr; un += 16) {
            const int64_t b = un >> 2;
            const int h = (int) ((un >> 1) & 1), lh = (int) (un & 1);
            // the unit's activations: values 256 b + 128 h + 32 j + 16 lh + 0..15 (4 dwords each j), their scales
            const int8_t * xb = xq_r + b * QK_K + 128 * h + 16 * lh;
            uint32_t xv[4][4];
            float xs[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int k = 0; k < 4; ++k) xv[j][k] = *reinterpret_cast<const uint32_t *>(xb + 32 * j + 4 * k);
                xs[j] = xd_r[(b * QK_K + 128 * h + 32 * j) / 32];
            }
            int sx[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int a = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) a = dpct::dp4a(0x01010101, (int) xv[j][k], a);
                sx[j] = a;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int64_t row = sycl::min(row0 + r, N - 1);
                const block_q2_K * blk = reinterpret_cast<const block_q2_K *>(we + (size_t) row * nb01) + b;
                const uint32_t * qs = reinterpret_cast<const uint32_t *>(blk->qs + 32 * h + 16 * lh);
                const uint32_t q0 = qs[0], q1 = qs[1], q2 = qs[2], q3 = qs[3];
                const uint32_t * scw = reinterpret_cast<const uint32_t *>(blk->scales);
                const uint32_t s0 = scw[2 * h], s1 = scw[2 * h + 1];   // scales[8 h .. 8 h + 7]
                const sycl::float2 dm = blk->dm.convert<float, sycl::rounding_mode::automatic>();
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t scb = (((j < 2) ? s0 : s1) >> (8 * ((2 * j + lh) % 4))) & 0xFF;   // scales[8 h + 2 j + lh]
                    const int qd = dpct::dp4a((int) ((q0 >> (2 * j)) & 0x03030303u), (int) xv[j][0],
                                   dpct::dp4a((int) ((q1 >> (2 * j)) & 0x03030303u), (int) xv[j][1],
                                   dpct::dp4a((int) ((q2 >> (2 * j)) & 0x03030303u), (int) xv[j][2],
                                   dpct::dp4a((int) ((q3 >> (2 * j)) & 0x03030303u), (int) xv[j][3], 0))));
                    sum += xs[j] * (dm.x() * (float) (scb & 0xF) * (float) qd - dm.y() * (float) (scb >> 4) * (float) sx[j]);
                }
                acc[r] += sum;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
            if (ln == 0 && row0 + r < N) y[t * y2 + u * y1 + row0 + r] = v;
        }
    });
    return true;
}

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) try {'''

old = """static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) try {"""
if s.count(old) != 1: sys.exit("mul_mat_id anchor")
s = s.replace(old, KERNEL, 1)
old2 = """    if (ne12 == 1) {
        if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            return;
        }
    }"""
new2 = """    if (ne12 == 1) {
        if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            return;
        }
    }
    if (ne12 > 1 && larry_mmid_q2k(ctx, src0, src1, ids, dst)) {   // larry: no host round trip for a few tokens
        return;
    }"""
if s.count(old2) != 1: sys.exit("fused anchor")
s = s.replace(old2, new2)
open(S, "w").write(s)
print("patched")
