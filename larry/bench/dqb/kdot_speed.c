// kdot_speed.c - larry: one core's rate (GB/s of weights) in ggml-cpu's vec_dot for the expert formats the CPU pool
// meets (Q3_K x Q8_K, Q2_0 x Q8_0) at n = 2048 / 5120, rows round-robin over a 256 MiB matrix (out of the caches)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-cpu.h"

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(void) {
    const enum ggml_type types[2] = {GGML_TYPE_Q3_K, GGML_TYPE_Q2_0};
    const int ns[2] = {2048, 5120};
    for (int ti = 0; ti < 2; ++ti)
        for (int ni = 0; ni < 2; ++ni) {
            const enum ggml_type t = types[ti];
            const int n = ns[ni];
            const struct ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(t);
            const size_t rb = ggml_row_size(t, n);
            const int rows = (int) ((256u << 20) / rb);
            float * xf = malloc(sizeof(float) * (size_t) n);
            for (int i = 0; i < n; ++i) xf[i] = (float) ((i * 2654435761u) % 2001) / 1000.f - 1.f;
            uint8_t * w = malloc(rb * (size_t) rows);
            uint8_t * row = malloc(rb);
            ggml_quantize_chunk(t, xf, row, 0, 1, n, NULL);
            for (int r = 0; r < rows; ++r) memcpy(w + (size_t) r * rb, row, rb);
            const size_t ab = ggml_row_size(tc->vec_dot_type, n);
            uint8_t * a = malloc(ab);
            ggml_get_type_traits_cpu(tc->vec_dot_type)->from_float(xf, a, n);
            float acc = 0;
            double t0 = now_s();
            for (int rep = 0; rep < 4; ++rep)
                for (int r = 0; r < rows; ++r) { float v; tc->vec_dot(n, &v, 0, w + (size_t) r * rb, 0, a, 0, 1); acc += v; }
            const double dt = (now_s() - t0) / (4.0 * rows);
            printf("%-6s n %5d: %.3f us a row, %.2f GB/s of weights (%g)\n", ggml_type_name(t), n, dt * 1e6, rb / dt / 1e9, acc);
            free(xf); free(w); free(row); free(a);
        }
    return 0;
}
