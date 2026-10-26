#!/usr/bin/env python3
# patch-mmidq2k2.py - GGML_SYCL_LARRY_MMID_Q2K_CHECK=1 (diagnostic): larry_mmid_q2k into a scratch buffer, the routed
# path into dst, the two compared on the host (first 24 calls printed).  Idempotent.
import sys
S = "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(S).read()
if "GGML_SYCL_LARRY_MMID_Q2K_CHECK" in s: print("already"); sys.exit(0)
def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"anchor x{s.count(old)}: {old[:80]!r}")
    s = s.replace(old, new)
rep("""static bool larry_mmid_q2k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst) {""", """static thread_local bool g_larry_mmid_nofast = false;   // the check mode's reference call
static bool larry_mmid_q2k(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst, float * y_over = nullptr) {""")
rep("""    float * y = (float *) dst->data;
    const int64_t y1 = (int64_t) (dst->nb[1] / 4), y2 = (int64_t) (dst->nb[2] / 4);
    const int8_t * XQ = xq.get();""", """    float * y = y_over ? y_over : (float *) dst->data;
    const int64_t y1 = (int64_t) (dst->nb[1] / 4), y2 = (int64_t) (dst->nb[2] / 4);
    const int8_t * XQ = xq.get();""")
rep("""    if (ne12 > 1 && larry_mmid_q2k(ctx, src0, src1, ids, dst)) {   // larry: no host round trip for a few tokens
        return;
    }""", """    if (ne12 > 1 && !g_larry_mmid_nofast) {   // larry: no host round trip for a few tokens
        static const int check = ggml_sycl_get_env("GGML_SYCL_LARRY_MMID_Q2K_CHECK", 0);
        if (check && ggml_is_contiguous(dst)) {
            ggml_sycl_pool_alloc<float> yb(ctx.pool(), (size_t) ggml_nelements(dst));
            if (larry_mmid_q2k(ctx, src0, src1, ids, dst, yb.get())) {
                g_larry_mmid_nofast = true;
                ggml_sycl_mul_mat_id(ctx, dst);
                g_larry_mmid_nofast = false;
                static int nchk = 0;
                if (nchk < 24) {
                    ++nchk;
                    const size_t n = (size_t) ggml_nelements(dst);
                    std::vector<float> a(n), b(n);
                    stream->memcpy(a.data(), dst->data, n * 4).wait();
                    stream->memcpy(b.data(), yb.get(), n * 4).wait();
                    double md = 0, mx = 0;
                    for (size_t i = 0; i < n; ++i) { md = std::max(md, (double) std::fabs(a[i] - b[i])); mx = std::max(mx, (double) std::fabs(a[i])); }
                    GGML_LOG_INFO("larry mmid q2k check %d: %s [%lld x %lld x %lld] max |diff| %.3e, max |ref| %.3e\\n", nchk, dst->name,
                                  (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], md, mx);
                }
                return;
            }
        } else if (larry_mmid_q2k(ctx, src0, src1, ids, dst)) {
            return;
        }
    }""")
open(S, "w").write(s)
print("patched")
