#!/usr/bin/env python3
# patch-sdpaprof.py [FATTN_DSV4_CPP] - larry, GGML_SYCL_SDPA_PROF=1 (diagnostic, default off): the DSV41 SDPA op's
# host side step by step (ms summed, printed every 400 calls): Q' alloc + kernel, the engine / stream / partition
# lookups, the mask's allocs + kernel, the oneDNN tensors, execute, the output kernel.  GGML_SYCL_OP_PROF showed
# ~1.8 ms of host time a call before execute with the card idle.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/fattn-dsv4.cpp"
s = open(P).read()
if "GGML_SYCL_SDPA_PROF" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""void ggml_sycl_op_dsv41_attn_sdpa(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    namespace mkl = oneapi::mkl;""", """// larry, GGML_SYCL_SDPA_PROF (patch-sdpaprof.py): host ms per step, summed
static const bool g_sdpa_prof = std::getenv("GGML_SYCL_SDPA_PROF") != nullptr;
static double g_sdpa_t[8] = {};
static long g_sdpa_n = 0;
static double g_sdpa_last = 0;
static double sdpa_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void sdpa_lap(int i) {
    if (!g_sdpa_prof) return;
    const double t = sdpa_now();
    g_sdpa_t[i] += t - g_sdpa_last;
    g_sdpa_last = t;
}

void ggml_sycl_op_dsv41_attn_sdpa(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    namespace mkl = oneapi::mkl;
    if (g_sdpa_prof) g_sdpa_last = sdpa_now();""")
rep("""    const bool m16 = MK->type == GGML_TYPE_F16;
    const size_t mstride = MK->nb[1] / ggml_type_size(MK->type);
    const void * mkd = MK->data;
#if GGML_SYCL_DNNL
    static const int sdpa_n = env_int("GGML_SYCL_DSV4_SDPA_N", 512);
    if (n == sdpa_n && D % 4 == 0 && Dp % 4 == 0) {""", """    const bool m16 = MK->type == GGML_TYPE_F16;
    const size_t mstride = MK->nb[1] / ggml_type_size(MK->type);
    const void * mkd = MK->data;
    sdpa_lap(0);   // Q' alloc + its kernel
#if GGML_SYCL_DNNL
    static const int sdpa_n = env_int("GGML_SYCL_DSV4_SDPA_N", 512);
    if (n == sdpa_n && D % 4 == 0 && Dp % 4 == 0) {""")
rep("""            P = &it->second;
        }
        if (P->ok) {""", """            P = &it->second;
        }
        sdpa_lap(1);   // the engine / stream / partition lookups
        if (P->ok) {""")
rep("""                if (i == 0) *onep = 1.0f;   // Q' carries the scale: the graph divides by 1
            });""", """                if (i == 0) *onep = 1.0f;   // Q' carries the scale: the graph divides by 1
            });
            sdpa_lap(2);   // the mask's allocs + kernel""")
rep("""            dnnl::graph::tensor tout(P->out, eng, Op.get());
            if (larry_opprof::on())""", """            dnnl::graph::tensor tout(P->out, eng, Op.get());
            sdpa_lap(3);   // the oneDNN tensors
            if (larry_opprof::on())""")
rep("""            if (larry_opprof::on()) larry_opprof::mark(*stream, ctx.device, "SDPA: oneDNN execute");""",
    """            if (larry_opprof::on()) larry_opprof::mark(*stream, ctx.device, "SDPA: oneDNN execute");
            sdpa_lap(4);   // execute (+ the profiler's marks)""")
rep("""                *(sycl::float4 *) (o + bh * D + v * 4) = sycl::float4((float) h[0], (float) h[1], (float) h[2], (float) h[3]);
            });
            return;""", """                *(sycl::float4 *) (o + bh * D + v * 4) = sycl::float4((float) h[0], (float) h[1], (float) h[2], (float) h[3]);
            });
            sdpa_lap(5);   // the output kernel
            if (g_sdpa_prof && ++g_sdpa_n % 400 == 0) {
                fprintf(stderr, "larry sdpa prof (%ld calls): host ms a call: Q' %.3f, lookups %.3f, mask %.3f, tensors %.3f, execute %.3f, out %.3f\\n",
                        g_sdpa_n, g_sdpa_t[0] / g_sdpa_n, g_sdpa_t[1] / g_sdpa_n, g_sdpa_t[2] / g_sdpa_n, g_sdpa_t[3] / g_sdpa_n,
                        g_sdpa_t[4] / g_sdpa_n, g_sdpa_t[5] / g_sdpa_n);
            }
            return;""")
if "#include <chrono>" not in s:
    s = "#include <chrono>\n" + s
open(P, "w").write(s)
print("patched", P)
