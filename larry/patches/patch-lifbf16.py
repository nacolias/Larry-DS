#!/usr/bin/env python3
# patch-lifbf16.py [XOLOTL_DIR] - larry, 2026-10-28 (optimization #5, long contexts): the fused prompt indexer
# (GGML_SYCL_LI_FUSED, patch-lifused.py) reads BF16 keys as they are stored - GGML_SYCL_LI_FUSED=2.  Mode 1 converts the
# whole key table to fp16 on every call (at 512K keys 256 MiB read + written, ~0.45 ms of a ~8 ms call; a 967K prompt's
# three indexer layers make ~4K such calls); mode 2 runs the same XMX kernel in bf16 x bf16 -> f32 on the keys in place
# (rows of 128, a multiple of 32 of them - the prompt's cap buckets) and converts only the queries.  The values are the
# indexer's FP4-rounded q and k, exact in bf16 as in fp16, so the scores should be the same bits:
# GGML_SYCL_LI_FUSED_CHECK16=N runs the first N mode-2 calls through mode 1 too and counts the floats that differ.  Other
# key types / shapes keep mode 1.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "third_party/llama.cpp/ggml/src/ggml-sycl/lightning-indexer.cpp")
s = open(P).read()
if "lif_run_t" in s:
    print("already patched", P)
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


s = rep(s, """// qh: fp16 [B][32][128]; kh: the keys in fp16 [NKVp][128] (zero rows past NKV, NKVp a multiple of 32)
static void lif_run(const sycl::half * qh, const sycl::half * kh, int64_t NKV, const float * w, int64_t ldw,
                    const sycl::half * m, int64_t ldm, const float * nvis, float * od, int64_t ldo, int64_t B, int QW, int64_t KR,
                    queue_ptr stream) {
""", """// qh: fp16 [B][32][128]; kh: the keys in fp16 [NKVp][128] (zero rows past NKV, NKVp a multiple of 32)
// larry (patch-lifbf16.py): T = sycl::half (those copies, ld_k 128) or li_bf16 (GGML_SYCL_LI_FUSED=2: the queries in bf16,
// the BF16 keys read where they are, rows ld_k apart, NKV a multiple of 32)
template <typename T>
static void lif_run_t(const T * qh, const T * kh, int64_t ld_k, int64_t NKV, const float * w, int64_t ldw,
                      const sycl::half * m, int64_t ldm, const float * nvis, float * od, int64_t ldo, int64_t B, int QW, int64_t KR,
                      queue_ptr stream) {
    using lif_A = li_mx::joint_matrix<sycl::sub_group, T, li_mx::use::a, 8, 16, li_mx::layout::row_major>;
    using lif_B = li_mx::joint_matrix<sycl::sub_group, T, li_mx::use::b, 16, 16, li_mx::layout::col_major>;
""", "template header")
s = rep(s, """        const sycl::half * qr = qh + b * 32 * 128;
        const sycl::half * mr = m + b * ldm;""", """        const T * qr = qh + b * 32 * 128;
        const sycl::half * mr = m + b * ldm;""", "qr")
s = rep(s, """(kh + j0 * 128 + ks * 16), 128);""", """(kh + j0 * ld_k + ks * 16), ld_k);""", "b0 load")
s = rep(s, """(kh + (j0 + 16) * 128 + ks * 16), 128);""", """(kh + (j0 + 16) * ld_k + ks * 16), ld_k);""", "b1 load")
s = rep(s, """    lif_run(qh, kh, NKV, (const float *) w->data,""", """    lif_run_t<sycl::half>(qh, kh, 128, NKV, (const float *) w->data,""", "fp16 call")

# the bf16 branch at the top of li_fused (before the fp16 copies are allocated)
s = rep(s, """    const ggml_type kt = k->type;
    queue_ptr stream = ctx.stream();
    ggml_sycl_pool & pool = ctx.pool();
    ggml_sycl_pool_alloc<sycl::half> Qh(pool, (size_t) B * 32 * 128);
""", """    const ggml_type kt = k->type;
    queue_ptr stream = ctx.stream();
    ggml_sycl_pool & pool = ctx.pool();
    // larry (patch-lifbf16.py, GGML_SYCL_LI_FUSED=2): BF16 keys in rows of 128, a multiple of 32 of them, read as they are
    if (lif_mode() == 2 && kt == GGML_TYPE_BF16 && nbk2 == 256 && NKV % 32 == 0 && ((uintptr_t) k->data) % 64 == 0) {
        ggml_sycl_pool_alloc<li_bf16> Qb(pool, (size_t) B * 32 * 128);
        li_bf16 * qb = Qb.get();
        const char * qd = (const char *) q->data;
        stream->parallel_for(sycl::range<1>((size_t) B * 32 * 128), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int64_t e = (int64_t) (i % 128), hh = (int64_t) ((i / 128) % 32), b = (int64_t) (i / 4096);
            qb[i] = (li_bf16) *(const float *) (qd + b * nbq2 + hh * nbq1 + e * sizeof(float));
        });
        static const int qw_env2 = [] { const char * v = std::getenv("GGML_SYCL_LI_FUSED_QW"); return v && *v ? std::atoi(v) : 0; }();
        static const int64_t kr_env2 = [] { const char * v = std::getenv("GGML_SYCL_LI_FUSED_KR"); return v && *v ? std::atoll(v) : 0; }();
        const int qw2 = qw_env2 > 0 ? qw_env2 : NKV > 131072 ? 16 : 8;
        const int64_t KR2 = kr_env2 > 0 ? (kr_env2 + 31) / 32 * 32 : NKV <= 16384 ? 128 : NKV <= 65536 ? 256 : 512;
        const bool nvm2 = dst->op_params[0] == 1;
        const float * wd = (const float *) w->data;
        const int64_t ldw = (int64_t) (w->nb[1] / sizeof(float)), ldo = (int64_t) (dst->nb[1] / sizeof(float));
        const sycl::half * md = nvm2 ? nullptr : (const sycl::half *) m->data;
        const int64_t ldm = nvm2 ? 0 : (int64_t) (m->nb[1] / sizeof(sycl::half));
        const float * nv = nvm2 ? (const float *) m->data : nullptr;
        lif_run_t<li_bf16>(qb, (const li_bf16 *) k->data, 128, NKV, wd, ldw, md, ldm, nv, (float *) dst->data, ldo, B, qw2, KR2, stream);
        static const int check16 = [] { const char * v = std::getenv("GGML_SYCL_LI_FUSED_CHECK16"); return v && *v ? std::atoi(v) : 0; }();
        static int nchk16 = 0;
        if (check16 && nchk16 < check16) {   // debug: the same call through mode 1 (fp16 copies) into scratch, compared bit for bit
            ++nchk16;
            ggml_sycl_pool_alloc<sycl::half> Q16(pool, (size_t) B * 32 * 128);
            ggml_sycl_pool_alloc<sycl::half> K16(pool, (size_t) NKV * 128);
            ggml_sycl_pool_alloc<char> ref(pool, (size_t) B * dst->nb[1]);
            sycl::half * q16 = Q16.get();
            sycl::half * k16 = K16.get();
            const li_bf16 * kb = (const li_bf16 *) k->data;
            stream->parallel_for(sycl::range<1>((size_t) B * 32 * 128), [=](sycl::id<1> id) { q16[id[0]] = (sycl::half) (float) qb[id[0]]; });
            stream->parallel_for(sycl::range<1>((size_t) NKV * 128), [=](sycl::id<1> id) { k16[id[0]] = (sycl::half) (float) kb[id[0]]; });
            lif_run_t<sycl::half>(q16, k16, 128, NKV, wd, ldw, md, ldm, nv, (float *) ref.get(), ldo, B, qw2, KR2, stream);
            stream->wait();
            std::vector<float> a((size_t) NKV), r((size_t) NKV);
            int64_t differ = 0;
            double md2 = 0;
            for (int64_t b = 0; b < B; ++b) {
                stream->memcpy(a.data(), (const char *) dst->data + b * dst->nb[1], NKV * sizeof(float)).wait();
                stream->memcpy(r.data(), ref.get() + b * dst->nb[1], NKV * sizeof(float)).wait();
                for (int64_t j = 0; j < NKV; ++j)
                    if (std::memcmp(&a[(size_t) j], &r[(size_t) j], sizeof(float)) != 0) {
                        ++differ;
                        if (std::isfinite(a[(size_t) j]) && std::isfinite(r[(size_t) j]))
                            md2 = std::max(md2, (double) std::fabs(a[(size_t) j] - r[(size_t) j]));
                    }
            }
            GGML_LOG_INFO("larry li_fused bf16 check %d: %lld queries x %lld keys: %lld of %lld scores differ from the fp16 path "
                          "(max |diff| %.3e)\\n", nchk16, (long long) B, (long long) NKV, (long long) differ, (long long) (B * NKV), md2);
        }
        return true;
    }
    ggml_sycl_pool_alloc<sycl::half> Qh(pool, (size_t) B * 32 * 128);
""", "bf16 branch")
# the old type aliases are now the template's own
s = rep(s, """using lif_A = li_mx::joint_matrix<sycl::sub_group, sycl::half, li_mx::use::a, 8, 16, li_mx::layout::row_major>;
using lif_B = li_mx::joint_matrix<sycl::sub_group, sycl::half, li_mx::use::b, 16, 16, li_mx::layout::col_major>;
""", "", "old aliases")
if "#include <cstring>" not in s:
    s = "#include <cstring>\n" + s
open(P, "w").write(s)
print("patched", P)
