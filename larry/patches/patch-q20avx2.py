#!/usr/bin/env python3
# patch-q20avx2.py [GGML_DIR] - larry: an AVX2 ggml_vec_dot_q2_0_q8_0.  On x86 ggml-cpu had none (arch-fallback.h mapped
# the scalar generic - a shift / mask / multiply-add a weight), and Q2_0 is 33 % of DeepSeek-V4.1 GSQ's routed gate / up
# matrices and 44 % of its down ones: the CPU pool decoded them at ~3 GB/s a core (light experts of prompt calls,
# every arena expert of a decode pass).  Per 32-weight half: its 8 bytes shuffled to 32 (each byte four times), masked
# to its own two bits, both nibbles looked up to 0..3 (pshufb), then sum(q * y) - sum(y) by maddubs / madd: the
# generic's integer sums exactly, and its float steps in its own order - the same bits (bench/dqb/q20_parity.c).
# Idempotent.
import sys

G = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml"
fb = G + "/src/ggml-cpu/arch-fallback.h"
qx = G + "/src/ggml-cpu/arch/x86/quants.c"
a = open(fb).read()
q = open(qx).read()
if "larry, patch-q20avx2" in q:
    print("already patched"); sys.exit(0)
old_fb = """#elif defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
// quants.c
#define ggml_vec_dot_q2_0_q8_0_generic ggml_vec_dot_q2_0_q8_0
"""
new_fb = """#elif defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
// quants.c
// (larry, patch-q20avx2: ggml_vec_dot_q2_0_q8_0 has an x86 form in arch/x86/quants.c)
"""
if a.count(old_fb) != 1:
    sys.exit("arch-fallback anchor x%d" % a.count(old_fb))
a = a.replace(old_fb, new_fb)
anchor = "void ggml_vec_dot_q1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {\n"
if q.count(anchor) != 1:
    sys.exit("quants.c anchor x%d" % q.count(anchor))
kernel = r"""// larry, patch-q20avx2: Q2_0 (64 weights a block, four 2-bit codes a byte in order, {0..3} -> {-1..2}) x Q8_0.  Per
// 32-weight half: its 8 bytes shuffled to 32 (each byte four times), masked to its own two bits, both nibbles looked up
// to 0..3, then sum(q * y) - sum(y) by maddubs / madd - the generic's integer sums exactly; the float steps are the
// generic's own, in its order, so the result is the same bits.
void ggml_vec_dot_q2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK2_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

#if defined(__AVX2__)
    const block_q2_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;
    const __m256i rep = _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
    const __m256i msk = _mm256_setr_epi8(0x03, 0x0C, 0x30, (char) 0xC0, 0x03, 0x0C, 0x30, (char) 0xC0,
                                         0x03, 0x0C, 0x30, (char) 0xC0, 0x03, 0x0C, 0x30, (char) 0xC0,
                                         0x03, 0x0C, 0x30, (char) 0xC0, 0x03, 0x0C, 0x30, (char) 0xC0,
                                         0x03, 0x0C, 0x30, (char) 0xC0, 0x03, 0x0C, 0x30, (char) 0xC0);
    // a nibble holding a code at bits 0-1 or 2-3 -> the code: 1, 2, 3 -> 1, 2, 3 and 4, 8, 12 -> 1, 2, 3
    const __m256i lut = _mm256_setr_epi8(0, 1, 2, 3, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0,
                                         0, 1, 2, 3, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
    const __m256i lo4 = _mm256_set1_epi8(0x0F);
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);

    // larry: GGML_Q20_GENERIC=1 - the scalar generic (an A/B switch; the same bits either way)
    static int force_generic = -1;
    if (force_generic < 0) force_generic = getenv("GGML_Q20_GENERIC") != NULL;
    if (force_generic) { ggml_vec_dot_q2_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc); return; }

    // one 32-weight half: the code bytes (0..3) dotted with its Q8_0 block as sum((q - 1) * y), 8 int32 partials
    #define Q20_HALF(QS8, YB) ({ \
        const __m256i r_ = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i *) (QS8))), rep), msk); \
        const __m256i q_ = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(r_, lo4)), \
                                           _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(r_, 4), lo4))); \
        const __m256i y_ = _mm256_loadu_si256((const __m256i *) (YB)->qs); \
        _mm256_madd_epi16(_mm256_sub_epi16(_mm256_maddubs_epi16(q_, y_), _mm256_maddubs_epi16(ones8, y_)), ones16); })

    float sumf = 0.0f;

    int i = 0;
    for (; i + 1 < nb; i += 2) {   // two blocks: their four halves reduced together, then the generic's float steps
        const __m256i v0 = Q20_HALF(&x[i].qs[0], &y[i * 2 + 0]);
        const __m256i v1 = Q20_HALF(&x[i].qs[8], &y[i * 2 + 1]);
        const __m256i v2 = Q20_HALF(&x[i + 1].qs[0], &y[i * 2 + 2]);
        const __m256i v3 = Q20_HALF(&x[i + 1].qs[8], &y[i * 2 + 3]);
        const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(v0, v1), _mm256_hadd_epi32(v2, v3));
        const __m128i t = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));   // the four sums
        const int sb0 = _mm_cvtsi128_si32(t), sb1 = _mm_extract_epi32(t, 1), sb2 = _mm_extract_epi32(t, 2), sb3 = _mm_extract_epi32(t, 3);
        {
            const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);
            float sumi = 0.0f;
            sumi += GGML_CPU_FP16_TO_FP32(y[i * 2 + 0].d) * sb0;
            sumi += GGML_CPU_FP16_TO_FP32(y[i * 2 + 1].d) * sb1;
            sumf += d0 * sumi;
        }
        {
            const float d0 = GGML_CPU_FP16_TO_FP32(x[i + 1].d);
            float sumi = 0.0f;
            sumi += GGML_CPU_FP16_TO_FP32(y[i * 2 + 2].d) * sb2;
            sumi += GGML_CPU_FP16_TO_FP32(y[i * 2 + 3].d) * sb3;
            sumf += d0 * sumi;
        }
    }
    for (; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 2; k++) {
            const int sumi_block = hsum_i32_8(Q20_HALF(&x[i].qs[k * 8], &y[i * 2 + k]));
            sumi += GGML_CPU_FP16_TO_FP32(y[i * 2 + k].d) * sumi_block;
        }
        sumf += d0 * sumi;
    }
    #undef Q20_HALF

    *s = sumf;
#else
    ggml_vec_dot_q2_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

"""
q = q.replace(anchor, kernel + anchor)
open(fb, "w").write(a)
open(qx, "w").write(q)
print("patched", fb, qx)
