// q20_parity.c - larry: patch-q20avx2's AVX2 ggml_vec_dot_q2_0_q8_0 against ggml's scalar generic (both from the
// patched libggml-cpu.a): random Q2_0 rows x Q8_0 activations at the expert shapes (n = 2048: down, 4096 / 5120:
// gate / up), every result compared bit for bit; then the time a row of each.  q20_parity [rows]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

void ggml_vec_dot_q2_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q2_0_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_0(const float * x, void * y, int64_t k);

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t) (rs >> 16); }

int main(int argc, char ** argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 4000;
    const int ns[3] = {2048, 4096, 5120};
    long bad = 0, total = 0;
    for (int t = 0; t < 3; ++t) {
        const int n = ns[t], nb = n / QK2_0;
        block_q2_0 * w = malloc(sizeof(block_q2_0) * (size_t) nb * rows);
        block_q8_0 * a = malloc(sizeof(block_q8_0) * (size_t) (n / QK8_0));
        float * xf = malloc(sizeof(float) * (size_t) n);
        for (int i = 0; i < n; ++i) xf[i] = ((int) (rnd() % 20001) - 10000) * 1e-3f * ((i % 7 == 0) ? 8.f : 1.f);
        quantize_row_q8_0(xf, a, n);
        for (long b = 0; b < (long) nb * rows; ++b) {
            w[b].d = ggml_fp32_to_fp16(((int) (rnd() % 2001) - 1000) * 1e-4f);
            for (int j = 0; j < QK2_0 / 4; ++j) w[b].qs[j] = (uint8_t) rnd();
        }
        float * r0 = malloc(sizeof(float) * rows), * r1 = malloc(sizeof(float) * rows);
        for (int r = 0; r < rows; ++r) {
            ggml_vec_dot_q2_0_q8_0_generic(n, &r0[r], 0, w + (size_t) r * nb, 0, a, 0, 1);
            ggml_vec_dot_q2_0_q8_0(n, &r1[r], 0, w + (size_t) r * nb, 0, a, 0, 1);
            if (memcmp(&r0[r], &r1[r], 4) != 0) { if (bad < 5) printf("n %d row %d: generic %.9g avx2 %.9g\n", n, r, r0[r], r1[r]); ++bad; }
            ++total;
        }
        double t0 = now_s();
        float acc = 0;
        for (int rep = 0; rep < 3; ++rep)
            for (int r = 0; r < rows; ++r) { float v; ggml_vec_dot_q2_0_q8_0_generic(n, &v, 0, w + (size_t) r * nb, 0, a, 0, 1); acc += v; }
        const double tg = (now_s() - t0) / (3.0 * rows);
        t0 = now_s();
        for (int rep = 0; rep < 3; ++rep)
            for (int r = 0; r < rows; ++r) { float v; ggml_vec_dot_q2_0_q8_0(n, &v, 0, w + (size_t) r * nb, 0, a, 0, 1); acc += v; }
        const double tv = (now_s() - t0) / (3.0 * rows);
        const double rb = (double) sizeof(block_q2_0) * nb;
        printf("n %5d: generic %.2f us a row (%.2f GB/s of weights), avx2 %.3f us (%.2f GB/s): %.1fx  (sum %g)\n", n, tg * 1e6,
               rb / tg / 1e9, tv * 1e6, rb / tv / 1e9, tg / tv, acc);
        free(w); free(a); free(xf); free(r0); free(r1);
    }
    printf("%ld of %ld results differ\n", bad, total);
    return bad != 0;
}
