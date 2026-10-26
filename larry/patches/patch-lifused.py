#!/usr/bin/env python3
# patch-lifused.py [LIGHTNING_INDEXER_CPP] - larry, GGML_SYCL_LI_FUSED (default 0): the prompt indexer as ONE fused XMX
# kernel.  li_gemm writes S = Q [B*H x E] . K^T in f32 (31K deep: 512 queries x 32 heads x ~16K keys = ~1 GB a call)
# and reads it back for the relu-weighted head sum: 5.7 ms a call, ~10 % of card 0's time in a 31K prompt's second
# batch (abwT110), memory-bound (S in fp16 alone: 3.5 ms, abwT111).  Here S never leaves the registers: a sub-group =
# one query's 32 heads (4 row tiles of 8) x 32 keys (2 column tiles of 16), K = 128 in 8 steps, fp16 x fp16 -> f32 as
# li_gemm's GEMM; B = keys^T loaded col_major straight from the (fp16-converted) key rows; an accumulator lane holds
# its key's column (element i = row i: bench/dqb/coord_probe), so the head sum is lane-local; 32-key steps whose mask
# is -inf for the whole query are skipped (cap = the next power of two of the visible rows: up to half the keys).
# bench/li/li_probe.cpp (card 1, 512 queries; li_gemm -> fused): 16K keys 4.36 -> 0.63 ms, 32K 8.71 -> 1.22, 64K
# 17.4 -> 2.33, 256K 69.4 -> 11.2 (~110 TFLOPS); max |diff| / max |ref| 2e-7.  Work-group: 8 queries (16 past 128K
# keys) x 128 keys (256 to 64K, 512 beyond) - the sweep's best (GGML_SYCL_LI_FUSED_QW / _KR).
# GGML_SYCL_LI_FUSED_CHECK=N: the first N calls also through li_gemm into scratch, the largest difference logged.
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/lightning-indexer.cpp"
s = open(P).read()
if "GGML_SYCL_LI_FUSED" in s:
    print("already patched"); sys.exit(0)

block = r'''
// ---- larry, GGML_SYCL_LI_FUSED (patch-lifused.py, default 0): prompt batches as ONE fused XMX kernel - S stays in
// registers.  A sub-group = one query (32 heads = 4 row tiles of 8) x 32 keys (2 column tiles of 16) a step, K = 128 in
// 8 steps, fp16 x fp16 -> f32, B = keys^T col_major from the key rows; an accumulator lane holds its key's column
// (element i = row i), so the relu-weighted head sum is lane-local.  Steps whose 32 keys are all masked for the query
// write -inf without the multiplies.

using lif_A = li_mx::joint_matrix<sycl::sub_group, sycl::half, li_mx::use::a, 8, 16, li_mx::layout::row_major>;
using lif_B = li_mx::joint_matrix<sycl::sub_group, sycl::half, li_mx::use::b, 16, 16, li_mx::layout::col_major>;
using lif_C = li_mx::joint_matrix<sycl::sub_group, float, li_mx::use::accumulator, 8, 16>;

static int lif_mode() {
    static const int v = [] { const char * e = std::getenv("GGML_SYCL_LI_FUSED"); return e && *e ? std::atoi(e) : 0; }();
    return v;
}

static bool li_fused_ok(const ggml_tensor * dst) {
    static const int min_b = [] { const char * v = std::getenv("GGML_SYCL_LI_FUSED_MIN"); return v && *v ? std::atoi(v) : 16; }();
    if (!lif_mode()) return false;
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    if (q->ne[0] != 128 || q->ne[1] != 32 || q->ne[2] < min_b || q->ne[3] != 1 || k->ne[0] != 128 || k->ne[1] != 1 ||
        k->ne[3] != 1 || m->ne[3] != 1 || q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) ||
        m->nb[0] != sizeof(sycl::half) || w->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) return false;
    if (w->nb[1] % sizeof(float) || m->nb[1] % sizeof(sycl::half) || dst->nb[1] % sizeof(float)) return false;
    return k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16 || k->type == GGML_TYPE_F32;
}

#define LIF_HSUM(s, x0, x1, x2, x3)                                                                        \
    {                                                                                                     \
        auto v0 = sycl::ext::oneapi::detail::get_wi_data(sg, x0);                                         \
        auto v1 = sycl::ext::oneapi::detail::get_wi_data(sg, x1);                                         \
        auto v2 = sycl::ext::oneapi::detail::get_wi_data(sg, x2);                                         \
        auto v3 = sycl::ext::oneapi::detail::get_wi_data(sg, x3);                                         \
        _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                   \
            s += sycl::select_from_group(sg, wl0, i) * sycl::fmax((float) v0[i], 0.0f);                   \
            s += sycl::select_from_group(sg, wl0, 8 + i) * sycl::fmax((float) v1[i], 0.0f);               \
            s += sycl::select_from_group(sg, wl1, i) * sycl::fmax((float) v2[i], 0.0f);                   \
            s += sycl::select_from_group(sg, wl1, 8 + i) * sycl::fmax((float) v3[i], 0.0f);               \
        }                                                                                                 \
    }

// qh: fp16 [B][32][128]; kh: the keys in fp16 [NKVp][128] (zero rows past NKV, NKVp a multiple of 32)
static void lif_run(const sycl::half * qh, const sycl::half * kh, int64_t NKV, const float * w, int64_t ldw,
                    const sycl::half * m, int64_t ldm, float * od, int64_t ldo, int64_t B, int QW, int64_t KR, queue_ptr stream) {
    const int64_t nkr = (NKV + KR - 1) / KR, nqb = (B + QW - 1) / QW;
    stream->parallel_for(sycl::nd_range<1>((size_t) (nqb * nkr * QW * 16), (size_t) (QW * 16)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t g = (int64_t) it.get_group(0);
        const int64_t kr = g % nkr, b = (g / nkr) * QW + (int64_t) sg.get_group_linear_id();
        if (b >= B) return;
        const float wl0 = w[b * ldw + lane], wl1 = w[b * ldw + 16 + lane];
        const sycl::half * qr = qh + b * 32 * 128;
        const sycl::half * mr = m + b * ldm;
        float * orow = od + b * ldo;
        const int64_t j1 = sycl::min(NKV, (kr + 1) * KR);
        for (int64_t j0 = kr * KR; j0 < j1; j0 += 32) {
            const int64_t ja = j0 + lane, jb = j0 + 16 + lane;
            const float ma = ja < NKV ? (float) mr[ja] : -INFINITY;
            const float mb = jb < NKV ? (float) mr[jb] : -INFINITY;
            if (sycl::all_of_group(sg, ma == -INFINITY && mb == -INFINITY)) {
                if (ja < NKV) orow[ja] = -INFINITY;
                if (jb < NKV) orow[jb] = -INFINITY;
                continue;
            }
            lif_C c00, c01, c10, c11, c20, c21, c30, c31;
            li_mx::joint_matrix_fill(sg, c00, 0.0f); li_mx::joint_matrix_fill(sg, c01, 0.0f);
            li_mx::joint_matrix_fill(sg, c10, 0.0f); li_mx::joint_matrix_fill(sg, c11, 0.0f);
            li_mx::joint_matrix_fill(sg, c20, 0.0f); li_mx::joint_matrix_fill(sg, c21, 0.0f);
            li_mx::joint_matrix_fill(sg, c30, 0.0f); li_mx::joint_matrix_fill(sg, c31, 0.0f);
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                lif_A a0, a1, a2, a3;
                lif_B b0, b1;
                li_mx::joint_matrix_load(sg, a0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qr + 0 * 1024 + ks * 16), 128);
                li_mx::joint_matrix_load(sg, a1, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qr + 1 * 1024 + ks * 16), 128);
                li_mx::joint_matrix_load(sg, a2, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qr + 2 * 1024 + ks * 16), 128);
                li_mx::joint_matrix_load(sg, a3, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qr + 3 * 1024 + ks * 16), 128);
                li_mx::joint_matrix_load(sg, b0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kh + j0 * 128 + ks * 16), 128);
                li_mx::joint_matrix_load(sg, b1, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kh + (j0 + 16) * 128 + ks * 16), 128);
                li_mx::joint_matrix_mad(sg, c00, a0, b0, c00); li_mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                li_mx::joint_matrix_mad(sg, c10, a1, b0, c10); li_mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                li_mx::joint_matrix_mad(sg, c20, a2, b0, c20); li_mx::joint_matrix_mad(sg, c21, a2, b1, c21);
                li_mx::joint_matrix_mad(sg, c30, a3, b0, c30); li_mx::joint_matrix_mad(sg, c31, a3, b1, c31);
            }
            float s0 = 0.0f, s1 = 0.0f;
            LIF_HSUM(s0, c00, c10, c20, c30)
            LIF_HSUM(s1, c01, c11, c21, c31)
            if (ja < NKV) orow[ja] = s0 + ma;
            if (jb < NKV) orow[jb] = s1 + mb;
        }
    });
}

static bool li_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    const int64_t B = q->ne[2], NKV = k->ne[2], NKVp = (NKV + 31) / 32 * 32;
    const size_t nbq1 = q->nb[1], nbq2 = q->nb[2], nbk2 = k->nb[2];
    const ggml_type kt = k->type;
    queue_ptr stream = ctx.stream();
    ggml_sycl_pool & pool = ctx.pool();
    ggml_sycl_pool_alloc<sycl::half> Qh(pool, (size_t) B * 32 * 128);
    ggml_sycl_pool_alloc<sycl::half> Kh(pool, (size_t) NKVp * 128);
    sycl::half * qh = Qh.get();
    sycl::half * kh = Kh.get();
    const char * qd = (const char *) q->data;
    const char * kd = (const char *) k->data;
    stream->parallel_for(sycl::range<1>((size_t) B * 32 * 128), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int64_t e = (int64_t) (i % 128), hh = (int64_t) ((i / 128) % 32), b = (int64_t) (i / 4096);
        qh[i] = (sycl::half) *(const float *) (qd + b * nbq2 + hh * nbq1 + e * sizeof(float));
    });
    stream->parallel_for(sycl::range<1>((size_t) NKVp * 128), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t j = i / 128, e = i % 128;
        float v = 0.0f;
        if (j < NKV) {
            const char * row = kd + j * nbk2;
            if (kt == GGML_TYPE_F16) v = (float) ((const sycl::half *) row)[e];
            else if (kt == GGML_TYPE_BF16) v = (float) ((const sycl::ext::oneapi::bfloat16 *) row)[e];
            else v = ((const float *) row)[e];
        }
        kh[i] = (sycl::half) v;
    });
    static const int qw_env = [] { const char * v = std::getenv("GGML_SYCL_LI_FUSED_QW"); return v && *v ? std::atoi(v) : 0; }();
    static const int64_t kr_env = [] { const char * v = std::getenv("GGML_SYCL_LI_FUSED_KR"); return v && *v ? std::atoll(v) : 0; }();
    // the li_probe sweep's best (card 1, 512 queries): 8 queries x 128 keys a work-group to 16K keys, 256 to 64K, 512
    // beyond (16 queries past 128K)
    const int qw = qw_env > 0 ? qw_env : NKV > 131072 ? 16 : 8;
    const int64_t KR = kr_env > 0 ? (kr_env + 31) / 32 * 32 : NKV <= 16384 ? 128 : NKV <= 65536 ? 256 : 512;
    lif_run(qh, kh, NKV, (const float *) w->data, (int64_t) (w->nb[1] / sizeof(float)), (const sycl::half *) m->data,
            (int64_t) (m->nb[1] / sizeof(sycl::half)), (float *) dst->data, (int64_t) (dst->nb[1] / sizeof(float)), B, qw, KR, stream);
    static const int check = [] { const char * v = std::getenv("GGML_SYCL_LI_FUSED_CHECK"); return v && *v ? std::atoi(v) : 0; }();
    static int nchk = 0;
    if (check && nchk < check) {   // debug: the same call through li_gemm into scratch (dst keeps ours), the largest difference
        ++nchk;
        const size_t n = (size_t) B * NKV;
        std::vector<float> a(n), r(n);
        ggml_sycl_pool_alloc<char> ref(pool, (size_t) B * dst->nb[1]);
        stream->wait();
        for (int64_t b = 0; b < B; ++b)
            stream->memcpy(a.data() + b * NKV, (const char *) dst->data + b * dst->nb[1], NKV * sizeof(float)).wait();
        void * keep = dst->data;
        dst->data = ref.get();
        li_gemm(ctx, dst);
        dst->data = keep;
        stream->wait();
        for (int64_t b = 0; b < B; ++b)
            stream->memcpy(r.data() + b * NKV, ref.get() + b * dst->nb[1], NKV * sizeof(float)).wait();
        double md = 0, mx = 0;
        int64_t bad = 0;
        for (size_t i = 0; i < n; ++i) {
            const bool fa = std::isfinite(a[i]), fr = std::isfinite(r[i]);
            if (fa != fr || (!fa && a[i] != r[i])) { ++bad; continue; }
            if (!fr) continue;
            md = std::max(md, (double) std::fabs(a[i] - r[i]));
            mx = std::max(mx, (double) std::fabs(r[i]));
        }
        GGML_LOG_INFO("larry li_fused check %d: %lld queries x %lld keys (KR %lld): max |diff| %.3e of max %.3e, %lld non-finite mismatches\n",
                      nchk, (long long) B, (long long) NKV, (long long) KR, md, mx, (long long) bad);
    }
    return true;
}
'''

anchor = "void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {\n"
if s.count(anchor) != 1:
    sys.exit("anchor x%d" % s.count(anchor))
s = s.replace(anchor, block.lstrip("\n") + "\n" + anchor)
old = """    if (li_gemm_ok(dst)) {   // larry: prompt batches as one XMX GEMM"""
new = """    if (li_fused_ok(dst) && li_fused(ctx, dst)) return;   // larry: prompt batches, one fused XMX kernel (S in registers)
    if (li_gemm_ok(dst)) {   // larry: prompt batches as one XMX GEMM"""
if s.count(old) != 1:
    sys.exit("dispatch anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
