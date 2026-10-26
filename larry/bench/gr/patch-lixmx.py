#!/usr/bin/env python3
# patch-lixmx.py - larry, GGML_SYCL_LI_XMX (default 0 for the A/B): the lightning indexer of decode / verify passes
# (<= 8 queries, BF16 keys, 32 heads x 128) as ONE XMX kernel.  li_gemm (GGML_SYCL_LI_GEMM_MIN=1 sends these to it)
# converts K to fp16 and writes S = Q.K^T [B*32 x NKV] f32 to memory and reads it back for the head sum: at 98K keys
# and 4 queries ~25 MB + 25 MB + 2 x 50 MB a layer.  Here each sub-group computes 8 key rows x 32 heads of a query
# with bf16 DPAS (A = the keys as stored, B = the query packed once per call), and the epilogue (relu x head weight,
# the sum over heads, + mask) reads the tile from local memory - S never leaves the card's registers / SLM.  The keys
# and queries are FP4 grid points (act_quant) in BF16, so every product is exact; the f32 sums run in another order.
# GGML_SYCL_LI_XMX_CHECK=1 (debug): the first 32 calls also run the existing path and print the largest difference.
# Usage: patch-lixmx.py [lightning-indexer.cpp]   Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/lightning-indexer.cpp"
s = open(P).read()
if "GGML_SYCL_LI_XMX" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


KERNEL = r'''
// ---- larry, GGML_SYCL_LI_XMX (default 0): decode / verify batches (<= 8 queries) with BF16 keys as ONE XMX kernel -
// S tiles stay in registers / local memory (li_gemm writes S [B*H x NKV] f32 and reads it back), keys read once.
// Sub-group: 8 key rows x 32 heads of one query = 2 accumulators (heads 0-15, 16-31), K = 128 in 8 bf16 DPAS steps;
// A = the keys as stored (row-major), B = the query packed once per call (VNNI pairs, 16 x 16 tiles).  Epilogue: the
// tile to local memory, lane i < 8 sums its key row over the heads (relu x weight) + the mask.
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace li_mx = sycl::ext::oneapi::experimental::matrix;
using li_bf16 = sycl::ext::oneapi::bfloat16;

static bool li_xmx_ok(const ggml_tensor * dst) {
    static const int on = [] { const char * v = std::getenv("GGML_SYCL_LI_XMX"); return v && *v ? std::atoi(v) : 0; }();
    if (!on) return false;
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    return k->type == GGML_TYPE_BF16 && q->type == GGML_TYPE_F32 && q->ne[0] == 128 && q->ne[1] == 32 && q->ne[2] >= 1 &&
           q->ne[2] <= 8 && q->ne[3] == 1 && k->ne[0] == 128 && k->ne[1] == 1 && k->ne[3] == 1 && k->nb[0] == 2 &&
           k->nb[2] % 64 == 0 && k->ne[2] % 8 == 0 && ((uintptr_t) k->data) % 64 == 0 && q->nb[0] == 4;
}

// q [128, 32, B] f32 -> qp: per (query b, head tile nt, dim step ks) a 16 x 16 B tile in the packed layout - element
// (k2, n) = the bf16 pair (q[b][16 nt + n][16 ks + 2 k2], q[b][16 nt + n][16 ks + 2 k2 + 1]); 256 bf16 a tile.
static void li_xmx_pack_q(const char * q, size_t nbq1, size_t nbq2, int64_t B, li_bf16 * qp, queue_ptr stream) {
    stream->parallel_for(sycl::range<1>((size_t) B * 2 * 8 * 8 * 16), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int n = (int) (i % 16), k2 = (int) ((i / 16) % 8), ks = (int) ((i / 128) % 8), nt = (int) ((i / 1024) % 2);
        const int64_t b = i / 2048;
        const float * row = (const float *) (q + b * nbq2 + (size_t) (16 * nt + n) * nbq1);
        li_bf16 * t = qp + ((b * 2 + nt) * 8 + ks) * 256;
        t[(k2 * 16 + n) * 2 + 0] = (li_bf16) row[16 * ks + 2 * k2];
        t[(k2 * 16 + n) * 2 + 1] = (li_bf16) row[16 * ks + 2 * k2 + 1];
    });
}

static void li_xmx_run(const li_bf16 * kd, int64_t ld_k, int64_t NKV, const li_bf16 * qp, const char * w, size_t nbw1,
                       const char * m, size_t nbm1, float * od, int64_t ld_o, int64_t B, queue_ptr stream) {
    constexpr int SG = 16, SGPWG = 8, TPS = 4;   // sub-groups a work-group, 8-row key tiles a sub-group
    const int64_t ntiles = NKV / 8;
    const int64_t nsg = (ntiles + TPS - 1) / TPS;
    const int64_t nwg = (nsg + SGPWG - 1) / SGPWG;
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> scr(sycl::range<1>((size_t) SGPWG * 256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nwg * SGPWG * SG), sycl::range<1>(SGPWG * SG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int sgi = (int) sg.get_group_linear_id();
            const int lane = (int) sg.get_local_linear_id();
            const int64_t gsg = (int64_t) it.get_group(0) * SGPWG + sgi;
            auto sp = scr.get_multi_ptr<sycl::access::decorated::no>() + (size_t) sgi * 256;
            for (int tt = 0; tt < TPS; ++tt) {
                const int64_t tile = gsg * TPS + tt;
                if (tile >= ntiles) break;   // (uniform across the sub-group)
                const int64_t row0 = tile * 8;
                for (int64_t b = 0; b < B; ++b) {
                    li_mx::joint_matrix<sycl::sub_group, float, li_mx::use::accumulator, 8, 16> c0, c1;
                    li_mx::joint_matrix_fill(sg, c0, 0.0f);
                    li_mx::joint_matrix_fill(sg, c1, 0.0f);
                    for (int ks = 0; ks < 8; ++ks) {
                        li_mx::joint_matrix<sycl::sub_group, li_bf16, li_mx::use::a, 8, 16, li_mx::layout::row_major> ma;
                        li_mx::joint_matrix<sycl::sub_group, li_bf16, li_mx::use::b, 16, 16, li_mx::layout::ext_intel_packed> mb0, mb1;
                        li_mx::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(kd + row0 * ld_k + ks * 16), ld_k);
                        li_mx::joint_matrix_load(sg, mb0, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(qp + ((b * 2 + 0) * 8 + ks) * 256), 32);
                        li_mx::joint_matrix_load(sg, mb1, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(qp + ((b * 2 + 1) * 8 + ks) * 256), 32);
                        li_mx::joint_matrix_mad(sg, c0, ma, mb0, c0);
                        li_mx::joint_matrix_mad(sg, c1, ma, mb1, c1);
                    }
                    li_mx::joint_matrix_store(sg, c0, sp, 16, li_mx::layout::row_major);         // [8 rows][16 heads]
                    li_mx::joint_matrix_store(sg, c1, sp + 128, 16, li_mx::layout::row_major);   // heads 16-31
                    sycl::group_barrier(sg);
                    if (lane < 8) {
                        const float * wb = (const float *) (w + b * nbw1);
                        float acc = 0.0f;
                        for (int n = 0; n < 16; ++n) acc += sycl::fmax(sp[lane * 16 + n], 0.0f) * wb[n];
                        for (int n = 0; n < 16; ++n) acc += sycl::fmax(sp[128 + lane * 16 + n], 0.0f) * wb[16 + n];
                        const sycl::half * mb = (const sycl::half *) (m + b * nbm1);
                        od[b * ld_o + row0 + lane] = acc + (float) mb[row0 + lane];
                    }
                    sycl::group_barrier(sg);
                }
            }
        });
    });
}

static bool li_gemm_ok(const ggml_tensor * dst);
static void li_gemm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

static bool li_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    const int64_t B = q->ne[2], NKV = k->ne[2];
    queue_ptr stream = ctx.stream();
    ggml_sycl_pool_alloc<li_bf16> QP(ctx.pool(), (size_t) B * 4096);
    li_xmx_pack_q((const char *) q->data, q->nb[1], q->nb[2], B, QP.get(), stream);
    li_xmx_run((const li_bf16 *) k->data, (int64_t) (k->nb[2] / 2), NKV, QP.get(), (const char *) w->data, w->nb[1],
               (const char *) m->data, m->nb[1], (float *) dst->data, (int64_t) (dst->nb[1] / sizeof(float)), B, stream);
    static const int check = [] { const char * v = std::getenv("GGML_SYCL_LI_XMX_CHECK"); return v && *v ? std::atoi(v) : 0; }();
    static int nchk = 0;
    if (check && nchk < 32) {   // debug: the same call through li_gemm into a copy of dst, the largest difference
        ++nchk;
        const size_t n = (size_t) B * NKV;
        std::vector<float> a(n), r(n);
        stream->wait();
        for (int64_t b = 0; b < B; ++b)
            stream->memcpy(a.data() + b * NKV, (const char *) dst->data + b * dst->nb[1], NKV * sizeof(float)).wait();
        li_gemm(ctx, dst);
        stream->wait();
        for (int64_t b = 0; b < B; ++b)
            stream->memcpy(r.data() + b * NKV, (const char *) dst->data + b * dst->nb[1], NKV * sizeof(float)).wait();
        double md = 0, mx = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(r[i]) || !std::isfinite(a[i])) { if (std::isfinite(r[i]) != std::isfinite(a[i])) md = INFINITY; continue; }
            md = std::max(md, (double) std::fabs(a[i] - r[i]));
            mx = std::max(mx, (double) std::fabs(r[i]));
        }
        GGML_LOG_INFO("larry li_xmx check %d: %lld queries x %lld keys: max |diff| %.3e of max %.3e\n", nchk, (long long) B,
                      (long long) NKV, md, mx);
    }
    return true;
}
'''

rep("""void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    if (li_gemm_ok(dst)) {   // larry: prompt batches as one XMX GEMM""", KERNEL + """
void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    if (li_xmx_ok(dst) && li_xmx(ctx, dst)) return;   // larry: decode / verify batches, one fused XMX kernel
    if (li_gemm_ok(dst)) {   // larry: prompt batches as one XMX GEMM""")
for inc in ["#include <cmath>\n", "#include <vector>\n", "#include <algorithm>\n"]:
    if inc not in s:
        rep('#include "lightning-indexer.hpp"\n', '#include "lightning-indexer.hpp"\n' + inc)
open(P, "w").write(s)
print("patched")
