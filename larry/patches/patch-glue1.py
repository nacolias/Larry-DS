#!/usr/bin/env python3
# patch-glue1.py [GGML_SYCL_DIR] - larry, the prompt path's dense-span glue kernels (2026-10-25 unitrace: a 512-column
# span ~7.4 ms on card 0, ~1.2 ms of it in these four), each behind a switch:
#  (1) dsv4-hc.cpp hc_post: one work-item per (dim, token) writes all hc outputs - the per-output form read the
#      residual state (D x 4 x n floats, 42 MB a span) once per output (4x): ~466 us -> ~roofline.  Same sums in the same
#      order (bit-identical).  GGML_SYCL_HC_POST_V2=1 (default 0 until measured).
#  (2) rope.cpp: a ROPE / ROPE_BACK done in place (src0 == dst, same strides) whose rotated dims are a part of the row
#      (DeepSeek-V4: the last 64 of 512 - ggml_rope_set_offset) launches only over those pairs; out of place it copies
#      the other dims as before.  Bit-identical (the other dims are untouched in place).
#  (3) fattn-dsv4.cpp (the DSV4 MQA prompt attention), GGML_SYCL_DSV4_MQA_V2=1: Q -> f16 four floats a work-item when
#      Q is contiguous ([D] rows, heads, queries), and the softmax one sub-group a row (no work-group reductions: M <=
#      1024 keys, 64 a lane in registers).  Same math; the softmax's sum in another order.
# Idempotent.
import sys

G = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl"


def patch(path, pairs, mark):
    s = open(path).read()
    if mark in s:
        print("already patched", path)
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit("anchor x%d in %s: %s" % (s.count(old), path, old[:100]))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


# ---- (1) hc_post
patch(G + "/dsv4-hc.cpp", [("""    const int64_t nr = n_embd * hc * n_tokens;
    const int64_t block_size = 256;
    const int64_t num_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            const int64_t ir = item.get_global_id(0);
            if (ir >= nr) {
                return;
            }

            const int64_t i0   = ir % n_embd;
            const int64_t idst = (ir / n_embd) % hc;
            const int64_t it   = ir / (n_embd * hc);
""", """    const int64_t block_size = 256;
    // larry, GGML_SYCL_HC_POST_V2 (default 0 until measured): one work-item per (dim, token) for all hc outputs - the
    // per-output form read the residual hc times (a 512-token span: 4 x 42 MB).  The same sums, in the same order.
    static const bool v2 = std::getenv("GGML_SYCL_HC_POST_V2") && std::atoi(std::getenv("GGML_SYCL_HC_POST_V2")) != 0;
    if (v2 && hc <= 8) {
        const int64_t nt = n_embd * n_tokens;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((nt + block_size - 1) / block_size * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item) {
                const int64_t ir = item.get_global_id(0);
                if (ir >= nt) {
                    return;
                }
                const int64_t i0 = ir % n_embd;
                const int64_t it = ir / n_embd;
                const float xv = x[i0*sx0 + it*sx1];
                float rv[8];
                for (int64_t isrc = 0; isrc < hc; ++isrc) {
                    rv[isrc] = residual[i0*sr0 + isrc*sr1 + it*sr2];
                }
                for (int64_t idst = 0; idst < hc; ++idst) {
                    float sum = xv * post[idst*sp0 + it*sp1];
                    for (int64_t isrc = 0; isrc < hc; ++isrc) {
                        sum += rv[isrc] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                    }
                    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
                }
            });
        return;
    }
    const int64_t nr = n_embd * hc * n_tokens;
    const int64_t num_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            const int64_t ir = item.get_global_id(0);
            if (ir >= nr) {
                return;
            }

            const int64_t i0   = ir % n_embd;
            const int64_t idst = (ir / n_embd) % hc;
            const int64_t it   = ir / (n_embd * hc);
""")], "GGML_SYCL_HC_POST_V2")

# ---- (2) rope in place over the rotated pairs only
patch(G + "/rope.cpp", [
    ("""static void rope_norm(const T *x, D *dst, const int ne00, const int ne01,
                      const int ne02, const int s01, const int s02,
                      const int s03, const int s1, const int s2, const int s3,
                      const int n_dims, const int n_offs, const int32_t *pos,
                      const float freq_scale, const float ext_factor,
                      const float attn_factor, const rope_corr_dims corr_dims,
                      const float theta_scale, const float *freq_factors,
                      const int64_t *row_indices, const int set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }
""", """static void rope_norm(const T *x, D *dst, const int ne00, const int ne01,
                      const int ne02, const int s01, const int s02,
                      const int s03, const int s1, const int s2, const int s3,
                      const int n_dims, const int n_offs, const int32_t *pos,
                      const float freq_scale, const float ext_factor,
                      const float attn_factor, const rope_corr_dims corr_dims,
                      const float theta_scale, const float *freq_factors,
                      const int64_t *row_indices, const int set_rows_stride,
                      const int i0_lo, const int i0_end) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    // larry: [i0_lo, i0_end) - the whole row, or (in place) only the rotated pairs
    const int i0 = i0_lo + 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                                item_ct1.get_local_id(1));

    if (i0 >= i0_end) {
        return;
    }
"""),
    ("""               const float *freq_factors, const int64_t *row_indices,
               const int set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, false>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, true>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}""", """               const float *freq_factors, const int64_t *row_indices,
               const int set_rows_stride, dpct::queue_ptr stream, const int i0_lo = 0, int i0_end = -1) {
    GGML_ASSERT(ne00 % 2 == 0);
    if (i0_end < 0) i0_end = ne00;
    const dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (i0_end - i0_lo + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, false>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride, i0_lo, i0_end);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, true>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride, i0_lo, i0_end);
            });
    }
}"""),
    ("""        GGML_SYCL_DEBUG("%s: norm path\\n", __func__);
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, float, float>(
                (const float *)src0_d, (float *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);""", """        GGML_SYCL_DEBUG("%s: norm path\\n", __func__);
        // larry: in place (same buffer, same strides) with the rotated dims a part of the row (DeepSeek-V4: the last 64
        // of 512, ggml_rope_set_offset) - only those pairs; the other dims are already where they belong
        const bool part = src0_d == dst_d && s01 == s1 && s02 == s2 && s03 == s3 && set_rows_stride == 0 &&
                          src0->type == dst_type && (n_offs > 0 || n_dims < ne00) && n_offs % 2 == 0 && n_dims % 2 == 0 &&
                          n_offs + n_dims <= ne00;
        const int i0_lo = part ? n_offs : 0, i0_end = part ? n_offs + n_dims : (int) ne00;
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, float, float>(
                (const float *)src0_d, (float *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream, i0_lo, i0_end);"""),
    ("""        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);""", """        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream, i0_lo, i0_end);"""),
], "only those pairs; the other dims")

# ---- (3) the DSV4 MQA prompt attention's Q conversion and softmax
patch(G + "/fattn-dsv4.cpp", [
    ("""    // 1. Q (any strides) -> Qh[b][h][d] = half(scale * Q)
    {""", """    // larry, GGML_SYCL_DSV4_MQA_V2 (default off): Q four floats a work-item when contiguous, the softmax a sub-group a row
    static const int v2 = env_int("GGML_SYCL_DSV4_MQA_V2", 0);
    const bool q_contig = Q->nb[0] == 4 && Q->nb[2] == (size_t) D * 4 && Q->nb[3] == (size_t) NH * D * 4 && D % 4 == 0;
    // 1. Q (any strides) -> Qh[b][h][d] = half(scale * Q)
    if (v2 && q_contig) {
        const sycl::float4 * q4 = (const sycl::float4 *) Q->data;
        sycl::half4 * h4 = (sycl::half4 *) qh;
        const size_t total4 = (size_t) n * NH * D / 4;
        stream->parallel_for(sycl::range<1>(total4), [=](sycl::id<1> id) {
            const sycl::float4 v = q4[id[0]] * scale;
            h4[id[0]] = sycl::half4((sycl::half) v.x(), (sycl::half) v.y(), (sycl::half) v.z(), (sycl::half) v.w());
        });
    } else {"""),
    ("""    // 3. softmax over M with the mask row of the query and the head's sink; P in fp16
    {""", """    // 3. softmax over M with the mask row of the query and the head's sink; P in fp16
    constexpr int SM_PL = 64;   // GGML_SYCL_DSV4_MQA_V2: keys a lane (M <= 1024)
    if (v2 && M <= 16 * SM_PL) {
        const sycl::half * mk = (const sycl::half *) mask->data;
        const size_t mstride = mask->nb[3] / 2;
        const float * sk = sinks ? (const float *) sinks->data : nullptr;
        constexpr int SGW = 8;   // sub-groups (rows) a work-group
        const size_t rows = (size_t) n * NH;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((rows + SGW - 1) / SGW * SGW * 16), sycl::range<1>(SGW * 16)),
                             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const size_t row = (size_t) it.get_group(0) * SGW + sg.get_group_linear_id();   // b * NH + h
            if (row >= rows) return;
            const int lane = (int) sg.get_local_linear_id();
            const int64_t b = (int64_t) (row / NH), h = (int64_t) (row % NH);
            const float * sr = s + (int64_t) row * M;
            const sycl::half * mr = mk + b * mstride;
            float v[SM_PL];
            float mx = sk ? sk[h] : -1e30f;
#pragma unroll
            for (int i = 0; i < SM_PL; ++i) {
                const int64_t m = lane + 16 * i;
                v[i] = m < M ? sr[m] + (float) mr[m] : -1e30f;
                mx = sycl::fmax(mx, v[i]);
            }
            mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < SM_PL; ++i) {
                v[i] = lane + 16 * i < M ? sycl::exp(v[i] - mx) : 0.0f;
                sum += v[i];
            }
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
            if (sk) sum += sycl::exp(sk[h] - mx);
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            sycl::half * pr = p + (int64_t) row * M;
#pragma unroll
            for (int i = 0; i < SM_PL; ++i) {
                const int64_t m = lane + 16 * i;
                if (m < M) pr[m] = (sycl::half) (v[i] * inv);
            }
        });
    } else {"""),
], "GGML_SYCL_DSV4_MQA_V2")
