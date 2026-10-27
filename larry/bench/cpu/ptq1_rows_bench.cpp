// bench/cpu/ptq1_rows_bench.cpp - larry, 2026-10-27: PTQ1_0 (ternary) rows kernels on the CPU - the engine's rows_avx2
// (src/kernels/cpu/ptq1_0.cpp, copied here as the reference) against rows2_avx2: the same arithmetic after the decode
// (a sub-block's 8 lanes of 4-weight int32 sums -> fp32 * the activation's scale, fmadd into the block's sum, * the
// block's scale into the row's 8 accumulators, one horizontal sum), so the results must be bit-identical; the decode in
// 8-bit lanes (x = byte ^ 0x80; 3x mod 256 keeps the ^ 0x80, so trit n of every byte is two compares of x_n = 3^n x
// mod 256), the activation scales converted once a call, not once a row.  Random rows and activations; every output
// compared bit for bit; then ns a row / GB/s of weights, one thread, at 1-4 tokens.
//   g++ -O2 -mavx2 -mfma -mf16c -o ptq1_rows_bench ptq1_rows_bench.cpp && ./ptq1_rows_bench [n] [rows] [iters]
#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

constexpr int kBlock = 128, kBytes = 28;   // 26 bytes of 5-trit codes + an fp16 scale
struct Q8 { uint16_t d; int8_t qs[32]; };

// ---------------------------------------------------------------- packing (ptq1_0.cpp's pack_block), for test data
static uint16_t f2h(float f) { return _cvtss_sh(f, 0); }
static void pack_block(const int8_t* t, float d, uint8_t* blk) {
    for (int m = 0; m < 16; ++m) {
        unsigned q = 0;
        for (int n = 0; n < 5; ++n) q = q * 3 + (unsigned) (t[m + 16 * n] + 1);
        blk[m] = (uint8_t) ((q * 256 + 242) / 243);
    }
    for (int m = 0; m < 8; ++m) {
        unsigned q = 0;
        for (int n = 0; n < 5; ++n) q = q * 3 + (unsigned) (t[80 + m + 8 * n] + 1);
        blk[16 + m] = (uint8_t) ((q * 256 + 242) / 243);
    }
    for (int h = 0; h < 2; ++h) {
        unsigned q = 0;
        for (int n = 0; n < 4; ++n) q = q * 3 + (unsigned) (t[120 + h + 2 * n] + 1);
        q *= 3;
        blk[24 + h] = (uint8_t) ((q * 256 + 242) / 243);
    }
    const uint16_t hd = f2h(d);
    std::memcpy(blk + 26, &hd, 2);
}

// ---------------------------------------------------------------- the engine's kernel (reference, verbatim logic)
__attribute__((target("avx2,fma,f16c"))) inline __m256i rtrit16(__m256i w, __m256i pw) {
    const __m256i p = _mm256_and_si256(_mm256_mullo_epi16(w, pw), _mm256_set1_epi16(0xff));
    return _mm256_srli_epi16(_mm256_add_epi16(p, _mm256_add_epi16(p, p)), 8);
}
__attribute__((target("avx2,fma,f16c"))) inline __m256i rpack(__m256i a, __m256i b) {
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0xD8);
}
__attribute__((target("avx2,fma,f16c"))) inline void runpack(const uint8_t* blk, __m256i dg[4]) {
    const __m256i w = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) blk));
    const __m256i t0 = rtrit16(w, _mm256_set1_epi16(1)), t1 = rtrit16(w, _mm256_set1_epi16(3));
    const __m256i t2 = rtrit16(w, _mm256_set1_epi16(9)), t3 = rtrit16(w, _mm256_set1_epi16(27));
    const __m256i t4 = rtrit16(w, _mm256_set1_epi16(81));
    const __m128i v8 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (blk + 16)));
    const __m256i w2 = _mm256_set_m128i(v8, v8);
    const __m256i u01 = rtrit16(w2, _mm256_setr_epi16(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i u23 = rtrit16(w2, _mm256_setr_epi16(9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27));
    const uint32_t h0 = blk[24], h1 = blk[25];
    const __m256i w3 = _mm256_set_m128i(_mm_set1_epi32((int) (h0 | h1 << 16)), v8);
    const __m256i u4 = rtrit16(w3, _mm256_setr_epi16(81, 81, 81, 81, 81, 81, 81, 81, 1, 1, 3, 3, 9, 9, 27, 27));
    const __m256i one8 = _mm256_set1_epi8(1);
    dg[0] = _mm256_sub_epi8(rpack(t0, t1), one8);
    dg[1] = _mm256_sub_epi8(rpack(t2, t3), one8);
    dg[2] = _mm256_sub_epi8(rpack(t4, u01), one8);
    dg[3] = _mm256_sub_epi8(rpack(u23, u4), one8);
}
__attribute__((target("avx2,fma,f16c"))) inline float rhsum(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}
template <int NT>
__attribute__((target("avx2,fma,f16c"))) void rows_avx2(const uint8_t* w, size_t rb, int n, const Q8* const* y, float* const* out, int r0, int r1) {
    const __m256i ones16 = _mm256_set1_epi16(1), ones8 = _mm256_set1_epi8(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * rb;
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < n / kBlock; ++b) {
            const uint8_t* blk = x + (size_t) b * kBytes;
            __m256i dg[4];
            runpack(blk, dg);
            uint16_t hd;
            std::memcpy(&hd, blk + 26, 2);
            const __m256 vdw = _mm256_set1_ps(_cvtsh_ss(hd));
            for (int t = 0; t < NT; ++t) {
                const Q8* a = y[t] + b * 4;
                __m256 bs = _mm256_setzero_ps();
                for (int k = 0; k < 4; ++k) {
                    const __m256i sq = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) a[k].qs), dg[k]);
                    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, sq), ones16);
                    bs = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(_cvtsh_ss(a[k].d)), bs);
                }
                acc[t] = _mm256_fmadd_ps(bs, vdw, acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = rhsum(acc[t]);
    }
}

// ---------------------------------------------------------------- rows2: the decode in 8-bit lanes, scales once a call
// the block's 26 code bytes as [bytes 0..15 | bytes 12..27] (two 16-byte loads inside the block: no read past it); then
// x0 = byte ^ 0x80, x_{n+1} = 3 x_n (mod 256, 8-bit adds) - x_n = (byte * 3^n mod 256) ^ 0x80 - and trit n of each byte
// = (x_n < -42) - (x_n > 42) negated: -1 below 86, +1 from 171, 0 between (digit = floor(3 v / 256))
__attribute__((target("avx2,fma,f16c"))) inline void runpack2(const uint8_t* blk, __m256i dg[4]) {
    const __m256i b = _mm256_set_m128i(_mm_loadu_si128((const __m128i*) (blk + 12)), _mm_loadu_si128((const __m128i*) blk));
    const __m256i lo = _mm256_set1_epi8(-42), hi = _mm256_set1_epi8(42);
    __m256i x = _mm256_xor_si256(b, _mm256_set1_epi8((char) 0x80));
    __m256i T[5];
    for (int n = 0; n < 5; ++n) {
        T[n] = _mm256_sub_epi8(_mm256_cmpgt_epi8(lo, x), _mm256_cmpgt_epi8(x, hi));   // -1 / 0 / +1
        x = _mm256_add_epi8(x, _mm256_add_epi8(x, x));
    }
    // T[n]: low lane = trit n of bytes 0..15; high lane = trit n of bytes 12..27 (16..23 at 4..11, 24 / 25 at 12 / 13)
    dg[0] = _mm256_permute2x128_si256(T[0], T[1], 0x20);   // elements 0..31: trits 0, 1 of bytes 0..15
    dg[1] = _mm256_permute2x128_si256(T[2], T[3], 0x20);   // 32..63: trits 2, 3
    const __m256i s0 = _mm256_srli_si256(T[0], 4), s1 = _mm256_srli_si256(T[1], 4);
    const __m256i s2 = _mm256_srli_si256(T[2], 4), s3 = _mm256_srli_si256(T[3], 4), s4 = _mm256_srli_si256(T[4], 4);
    // 64..95: trit 4 of bytes 0..15 | trits 0, 1 of bytes 16..23
    dg[2] = _mm256_permute2x128_si256(T[4], _mm256_unpacklo_epi64(s0, s1), 0x30);
    // 96..127: trits 2, 3 of bytes 16..23 | trit 4 of bytes 16..23, then trits 0..3 of bytes 24, 25 (120 + h + 2 n)
    const __m256i q = _mm256_unpacklo_epi32(_mm256_unpackhi_epi16(s0, s1), _mm256_unpackhi_epi16(s2, s3));
    dg[3] = _mm256_permute2x128_si256(_mm256_unpacklo_epi64(s2, s3), _mm256_unpacklo_epi64(s4, q), 0x31);
}
template <int NT>
__attribute__((target("avx2,fma,f16c"))) void rows2_avx2(const uint8_t* w, size_t rb, int n, const Q8* const* y, const float* const* ds,
                                                         float* const* out, int r0, int r1) {
    const __m256i ones16 = _mm256_set1_epi16(1), ones8 = _mm256_set1_epi8(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * rb;
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < n / kBlock; ++b) {
            const uint8_t* blk = x + (size_t) b * kBytes;
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
}

// ---------------------------------------------------------------- the test
template <class F> static double time_ns(F f, int iters) {
    f();
    double best = 1e300;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 5120, R = argc > 2 ? std::atoi(argv[2]) : 2048, iters = argc > 3 ? std::atoi(argv[3]) : 7;
    const size_t rb = (size_t) (n / kBlock) * kBytes;
    std::mt19937 rng(42);
    std::vector<uint8_t> w(rb * (size_t) R);
    std::uniform_int_distribution<int> tri(-1, 1);
    std::uniform_real_distribution<float> sc(0.001f, 0.05f);
    int8_t t[kBlock];
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < n / kBlock; ++b) {
            for (int j = 0; j < kBlock; ++j) t[j] = (int8_t) tri(rng);
            pack_block(t, sc(rng), w.data() + (size_t) r * rb + (size_t) b * kBytes);
        }
    // every byte value too (the decode's corners): a row of blocks whose code bytes run 0..255
    for (int b = 0; b < n / kBlock && b < 10; ++b)
        for (int j = 0; j < 26; ++j) w[(size_t) b * kBytes + j] = (uint8_t) (b * 26 + j);
    const int NB = n / 32;
    std::vector<std::vector<Q8>> act(4, std::vector<Q8>((size_t) NB));
    std::vector<std::vector<float>> dsc(4, std::vector<float>((size_t) NB));
    std::uniform_int_distribution<int> qv(-127, 127);
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < NB; ++i) {
            act[k][(size_t) i].d = f2h(sc(rng));
            for (int j = 0; j < 32; ++j) act[k][(size_t) i].qs[j] = (int8_t) qv(rng);
            dsc[k][(size_t) i] = _cvtsh_ss(act[k][(size_t) i].d);
        }
    const Q8* y[4];
    const float* ds[4];
    for (int k = 0; k < 4; ++k) { y[k] = act[k].data(); ds[k] = dsc[k].data(); }
    std::vector<std::vector<float>> o1(4, std::vector<float>((size_t) R)), o2(4, std::vector<float>((size_t) R));
    float* p1[4];
    float* p2[4];
    for (int k = 0; k < 4; ++k) { p1[k] = o1[k].data(); p2[k] = o2[k].data(); }
    bool ok = true;
    for (int nt = 1; nt <= 4; ++nt) {
        auto ref = [&] {
            switch (nt) { case 1: rows_avx2<1>(w.data(), rb, n, y, p1, 0, R); break; case 2: rows_avx2<2>(w.data(), rb, n, y, p1, 0, R); break;
                          case 3: rows_avx2<3>(w.data(), rb, n, y, p1, 0, R); break; default: rows_avx2<4>(w.data(), rb, n, y, p1, 0, R); }
        };
        auto neu = [&] {
            switch (nt) { case 1: rows2_avx2<1>(w.data(), rb, n, y, ds, p2, 0, R); break; case 2: rows2_avx2<2>(w.data(), rb, n, y, ds, p2, 0, R); break;
                          case 3: rows2_avx2<3>(w.data(), rb, n, y, ds, p2, 0, R); break; default: rows2_avx2<4>(w.data(), rb, n, y, ds, p2, 0, R); }
        };
        ref();
        neu();
        long diff = 0;
        for (int k = 0; k < nt; ++k)
            for (int r = 0; r < R; ++r) diff += std::memcmp(&o1[(size_t) k][(size_t) r], &o2[(size_t) k][(size_t) r], 4) != 0;
        ok = ok && diff == 0;
        const double a = time_ns(ref, iters), b = time_ns(neu, iters);
        const double bytes = (double) rb * R;
        std::printf("nt %d: %ld of %d outputs differ | engine %.0f us = %.2f GB/s | rows2 %.0f us = %.2f GB/s (x%.2f)\n", nt, diff, nt * R,
                    a / 1e3, bytes / a, b / 1e3, bytes / b, a / b);
    }
    // the parts alone: the decode of every block (results folded so nothing is dropped), and the per-token half with
    // one block's digits for all (no decode)
    {
        const int nb = R * (n / kBlock);
        __m256i sink = _mm256_setzero_si256();
        auto dec1 = [&] { for (int i = 0; i < nb; ++i) { __m256i d[4]; runpack(w.data() + (size_t) i * kBytes, d);
                                                       sink = _mm256_xor_si256(sink, _mm256_xor_si256(_mm256_xor_si256(d[0], d[1]), _mm256_xor_si256(d[2], d[3]))); } };
        auto dec2 = [&] { for (int i = 0; i < nb; ++i) { __m256i d[4]; runpack2(w.data() + (size_t) i * kBytes, d);
                                                       sink = _mm256_xor_si256(sink, _mm256_xor_si256(_mm256_xor_si256(d[0], d[1]), _mm256_xor_si256(d[2], d[3]))); } };
        const double a = time_ns(dec1, iters), b = time_ns(dec2, iters);
        std::printf("decode alone: engine %.1f ns a block, rows2 %.1f ns a block (x%.2f) [%d]\n", a / nb, b / nb, a / b, _mm256_movemask_epi8(sink) & 1);
    }
    std::printf("%s\n", ok ? "bit-identical" : "DIFFERENT");
    return ok ? 0 : 1;
}
