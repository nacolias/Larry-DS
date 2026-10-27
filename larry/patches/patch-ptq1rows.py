#!/usr/bin/env python3
# patch-ptq1rows.py [XOLOTL_DIR] - larry, 2026-10-27 (optimization #1, the CPU pool): PTQ1_0 (ternary) rows - the CPU
# pool's one compute-bound format (bench_cpu_pool: 22.0 / 16.8 / 14.7 GB/s at 1 / 2 / 3 tokens against a 30.7 GB/s
# streaming read).  Bit-identical to the engine's rows_avx2 (bench/cpu/ptq1_rows_bench.cpp compares every output):
#   - the block's trits decoded in 8-bit lanes (x = byte ^ 0x80; 3x mod 256 keeps the ^ 0x80, so trit n of all 26 code
#     bytes is two compares of x_n = 3^n x mod 256: -1 below 86, +1 from 171), 5 vectors then shuffled into element
#     order, instead of 16-bit multiplies per trit (the decode alone 6.8 -> 5.0 ns a block on a Zen 2 core);
#   - each token's fp16 activation scales converted once a call, not once a row.
# The per-token half (sign, maddubs, madd, the 4 fmadds into the block's sum - the engine's rounding) is unchanged.
# STRATA_PTQ1_ROWS1=1: the old decode (A/B).  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "src/kernels/cpu/ptq1_0.cpp")
s = open(P).read()
if "runpack2" in s:
    print("already patched", P)
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


s = rep(s, """__attribute__((target("avx2,fma,f16c"))) inline float rhsum(__m256 v) {""",
"""// larry (patch-ptq1rows.py): the same 4 x 32 trits, decoded in 8-bit lanes - the block's code bytes as [0..15 | 12..27]
// (two 16-byte loads inside the block), x0 = byte ^ 0x80, x_{n+1} = 3 x_n (mod 256: x_n = (byte * 3^n mod 256) ^ 0x80),
// trit n = (x_n < -42) - (x_n > 42) negated = -1 below 86, +1 from 171 (digit = floor(3 v / 256)), then into element order
__attribute__((target("avx2,fma,f16c"))) inline void runpack2(const uint8_t* blk, __m256i dg[4]) {
    const __m256i b = _mm256_set_m128i(_mm_loadu_si128((const __m128i*) (blk + 12)), _mm_loadu_si128((const __m128i*) blk));
    const __m256i lo = _mm256_set1_epi8(-42), hi = _mm256_set1_epi8(42);
    __m256i x = _mm256_xor_si256(b, _mm256_set1_epi8((char) 0x80));
    __m256i T[5];
    for (int n = 0; n < 5; ++n) {
        T[n] = _mm256_sub_epi8(_mm256_cmpgt_epi8(lo, x), _mm256_cmpgt_epi8(x, hi));
        x = _mm256_add_epi8(x, _mm256_add_epi8(x, x));
    }
    // T[n]: low lane trit n of bytes 0..15; high lane trit n of bytes 12..27 (16..23 at 4..11, 24 / 25 at 12 / 13)
    dg[0] = _mm256_permute2x128_si256(T[0], T[1], 0x20);
    dg[1] = _mm256_permute2x128_si256(T[2], T[3], 0x20);
    const __m256i s0 = _mm256_srli_si256(T[0], 4), s1 = _mm256_srli_si256(T[1], 4);
    const __m256i s2 = _mm256_srli_si256(T[2], 4), s3 = _mm256_srli_si256(T[3], 4), s4 = _mm256_srli_si256(T[4], 4);
    dg[2] = _mm256_permute2x128_si256(T[4], _mm256_unpacklo_epi64(s0, s1), 0x30);
    const __m256i q = _mm256_unpacklo_epi32(_mm256_unpackhi_epi16(s0, s1), _mm256_unpackhi_epi16(s2, s3));
    dg[3] = _mm256_permute2x128_si256(_mm256_unpacklo_epi64(s2, s3), _mm256_unpacklo_epi64(s4, q), 0x31);
}
__attribute__((target("avx2,fma,f16c"))) inline void rcvt_scales(const Q8* y, int nb, float* out) {
    for (int i = 0; i < nb; ++i) out[i] = _cvtsh_ss(y[i].d);
}
__attribute__((target("avx2,fma,f16c"))) inline float rhsum(__m256 v) {""", "runpack2")
s = rep(s, """template <int NT>
__attribute__((target("avx2,fma,f16c"))) void rows_avx2(const uint8_t* w, size_t rb, int n, const Q8* const* y, float* const* out, int r0, int r1) {
    const __m256i ones16 = _mm256_set1_epi16(1), ones8 = _mm256_set1_epi8(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * rb;
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < n / kPtq1Block; ++b) {
            const uint8_t* blk = x + (size_t) b * kPtq1Bytes;
            __m256i dg[4];
            runpack(blk, dg);
            uint16_t hd;
            std::memcpy(&hd, blk + 26, 2);
            const __m256 vdw = _mm256_set1_ps(_cvtsh_ss(hd));   // F16C: exact, as h2f
            for (int t = 0; t < NT; ++t) {
                const Q8* a = y[t] + b * 4;
                __m256 bs = _mm256_setzero_ps();
                for (int k = 0; k < 4; ++k) {
                    const __m256i sq = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) a[k].qs), dg[k]);
                    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, sq), ones16);   // |q| <= 127
                    bs = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(_cvtsh_ss(a[k].d)), bs);
                }
                acc[t] = _mm256_fmadd_ps(bs, vdw, acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = rhsum(acc[t]);
    }
}""", """template <int NT>
__attribute__((target("avx2,fma,f16c"))) void rows_avx2(const uint8_t* w, size_t rb, int n, const Q8* const* y, float* const* out, int r0, int r1) {
    const __m256i ones16 = _mm256_set1_epi16(1), ones8 = _mm256_set1_epi8(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * rb;
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < n / kPtq1Block; ++b) {
            const uint8_t* blk = x + (size_t) b * kPtq1Bytes;
            __m256i dg[4];
            runpack(blk, dg);
            uint16_t hd;
            std::memcpy(&hd, blk + 26, 2);
            const __m256 vdw = _mm256_set1_ps(_cvtsh_ss(hd));   // F16C: exact, as h2f
            for (int t = 0; t < NT; ++t) {
                const Q8* a = y[t] + b * 4;
                __m256 bs = _mm256_setzero_ps();
                for (int k = 0; k < 4; ++k) {
                    const __m256i sq = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) a[k].qs), dg[k]);
                    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, sq), ones16);   // |q| <= 127
                    bs = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(_cvtsh_ss(a[k].d)), bs);
                }
                acc[t] = _mm256_fmadd_ps(bs, vdw, acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = rhsum(acc[t]);
    }
}
// larry (patch-ptq1rows.py): rows_avx2 with runpack2 and the activations' scales converted beforehand (ds[t][i] = the
// scale of token t's 32-value block i) - the same arithmetic in the same order, so the same bits
template <int NT>
__attribute__((target("avx2,fma,f16c"))) void rows2_avx2(const uint8_t* w, size_t rb, int n, const Q8* const* y, const float* const* ds,
                                                         float* const* out, int r0, int r1) {
    const __m256i ones16 = _mm256_set1_epi16(1), ones8 = _mm256_set1_epi8(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * rb;
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < n / kPtq1Block; ++b) {
            const uint8_t* blk = x + (size_t) b * kPtq1Bytes;
            __m256i dg[4];
            runpack2(blk, dg);
            uint16_t hd;
            std::memcpy(&hd, blk + 26, 2);
            const __m256 vdw = _mm256_set1_ps(_cvtsh_ss(hd));
            for (int t = 0; t < NT; ++t) {
                const Q8* a = y[t] + b * 4;
                const float* d = ds[t] + b * 4;
                __m256 bs = _mm256_setzero_ps();
                for (int k = 0; k < 4; ++k) {
                    const __m256i sq = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) a[k].qs), dg[k]);
                    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, sq), ones16);
                    bs = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_broadcast_ss(d + k), bs);
                }
                acc[t] = _mm256_fmadd_ps(bs, vdw, acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = rhsum(acc[t]);
    }
}""", "rows2")
s = rep(s, """    if (avx2) {
        // tokens in groups of <= 4 (the accumulators stay in registers)
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int g = nt - t0 < 4 ? nt - t0 : 4;
            const Q8* y[4];
            for (int t = 0; t < g; ++t) y[t] = (const Q8*) act[t0 + t];
            switch (g) {
                case 1: rows_avx2<1>(w, row_bytes, n, y, out + t0, r0, r1); break;
                case 2: rows_avx2<2>(w, row_bytes, n, y, out + t0, r0, r1); break;
                case 3: rows_avx2<3>(w, row_bytes, n, y, out + t0, r0, r1); break;
                default: rows_avx2<4>(w, row_bytes, n, y, out + t0, r0, r1); break;
            }
        }
        return;
    }""", """    if (avx2) {
        // larry (patch-ptq1rows.py): runpack2 + the scales converted once a call (STRATA_PTQ1_ROWS1=1: the old decode)
        static const bool old_rows = std::getenv("STRATA_PTQ1_ROWS1") != nullptr;
        const int nb = n / 32;
        thread_local std::vector<float> dsv;
        // tokens in groups of <= 4 (the accumulators stay in registers)
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int g = nt - t0 < 4 ? nt - t0 : 4;
            const Q8* y[4];
            for (int t = 0; t < g; ++t) y[t] = (const Q8*) act[t0 + t];
            if (!old_rows) {
                dsv.resize((size_t) g * (size_t) nb);
                const float* ds[4];
                for (int t = 0; t < g; ++t) { rcvt_scales(y[t], nb, dsv.data() + (size_t) t * nb); ds[t] = dsv.data() + (size_t) t * nb; }
                switch (g) {
                    case 1: rows2_avx2<1>(w, row_bytes, n, y, ds, out + t0, r0, r1); break;
                    case 2: rows2_avx2<2>(w, row_bytes, n, y, ds, out + t0, r0, r1); break;
                    case 3: rows2_avx2<3>(w, row_bytes, n, y, ds, out + t0, r0, r1); break;
                    default: rows2_avx2<4>(w, row_bytes, n, y, ds, out + t0, r0, r1); break;
                }
                continue;
            }
            switch (g) {
                case 1: rows_avx2<1>(w, row_bytes, n, y, out + t0, r0, r1); break;
                case 2: rows_avx2<2>(w, row_bytes, n, y, out + t0, r0, r1); break;
                case 3: rows_avx2<3>(w, row_bytes, n, y, out + t0, r0, r1); break;
                default: rows_avx2<4>(w, row_bytes, n, y, out + t0, r0, r1); break;
            }
        }
        return;
    }""", "dispatch")
if "#include <vector>" not in s:
    s = rep(s, "#include <cstring>\n", "#include <cstdlib>\n#include <cstring>\n#include <vector>\n", "includes")
open(P, "w").write(s)
print("patched", P)
