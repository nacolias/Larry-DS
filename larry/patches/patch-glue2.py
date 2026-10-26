#!/usr/bin/env python3
# patch-glue2.py [GGML_SYCL_DIR] - larry: GGML_SYCL_DSV4_MQA_V2 (patch-glue1.py) measured SLOWER (16K: card 0's dense
# part 235 -> 272 ms a layer, abwT25 h2) - its softmax kept 64 keys a lane in registers (spilled).  Split in two switches:
# GGML_SYCL_DSV4_MQA_QV=1 the vectorised Q conversion, GGML_SYCL_DSV4_MQA_SM=1 the sub-group softmax, now in three
# passes over the row (max, sum, write - the row is read from the cache, nothing held in registers).  _V2 = both.
# Idempotent.
import sys

G = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl"
P = G + "/fattn-dsv4.cpp"
s = open(P).read()
if "GGML_SYCL_DSV4_MQA_QV" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    static const int v2 = env_int("GGML_SYCL_DSV4_MQA_V2", 0);
    const bool q_contig = Q->nb[0] == 4 && Q->nb[2] == (size_t) D * 4 && Q->nb[3] == (size_t) NH * D * 4 && D % 4 == 0;
    // 1. Q (any strides) -> Qh[b][h][d] = half(scale * Q)
    if (v2 && q_contig) {""", """    static const int v2 = env_int("GGML_SYCL_DSV4_MQA_V2", 0);
    // larry, patch-glue2: the two halves of _V2 on their own switches
    static const int qv = env_int("GGML_SYCL_DSV4_MQA_QV", v2), smv = env_int("GGML_SYCL_DSV4_MQA_SM", v2);
    const bool q_contig = Q->nb[0] == 4 && Q->nb[2] == (size_t) D * 4 && Q->nb[3] == (size_t) NH * D * 4 && D % 4 == 0;
    // 1. Q (any strides) -> Qh[b][h][d] = half(scale * Q)
    if (qv && q_contig) {""")
rep("""    constexpr int SM_PL = 64;   // GGML_SYCL_DSV4_MQA_V2: keys a lane (M <= 1024)
    if (v2 && M <= 16 * SM_PL) {""", """    if (smv) {   // larry, GGML_SYCL_DSV4_MQA_SM: a sub-group a row, three passes over it (max, sum, P)""")
rep("""            const float * sr = s + (int64_t) row * M;
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
            }""", """            const float * sr = s + (int64_t) row * M;
            const sycl::half * mr = mk + b * mstride;
            float mx = sk ? sk[h] : -1e30f;
            for (int64_t m = lane; m < M; m += 16) mx = sycl::fmax(mx, sr[m] + (float) mr[m]);
            mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t m = lane; m < M; m += 16) sum += sycl::exp(sr[m] + (float) mr[m] - mx);
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
            if (sk) sum += sycl::exp(sk[h] - mx);
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            sycl::half * pr = p + (int64_t) row * M;
            for (int64_t m = lane; m < M; m += 16) pr[m] = (sycl::half) (sycl::exp(sr[m] + (float) mr[m] - mx) * inv);""")
open(P, "w").write(s)
print("patched", P)
