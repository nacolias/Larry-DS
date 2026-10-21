// tools/v41/engine/xe/xe_kernels.hpp - SYCL kernels for DeepSeek-V4.1's routed experts on Intel Arc (Xe2, B70).
//
// The GSQ mixed-tier experts come in three formats, per expert and per projection class:
//   Q3_K   (ggml 11)  : 256 weights / 110 bytes  (hmask[32] qs[64] scales[12] d)
//   Q2_0   (ggml 42)  :  64 weights /  18 bytes  (d, qs[16]: 2 bits each, value = q - 1)
//   PTQ1_0 (type 143) : 128 weights /  28 bytes  (base-3 trits, 5 per byte; d at byte 26)
// An expert's blob is [gate rows | up rows | down rows] (Ds41FileBlobs::read), gate/up rows n_embd wide, down rows
// n_ff wide.  The kernels take the activations in F32 (no 8-bit activation rounding: closer to the F32 reference than
// the CPU / CUDA tiers, which round to Q8).
//
// A call is a list of JOBS: one expert applied to up to 8 tokens of the call.
//   gate_up : H[j][s][r] = silu(min(g, L)) * clamp(u, -L, L), g/u = gate/up row r of job j's expert . X[tok_j[s]]
//   down    : OUT[t][i] (+)= sum_j sum_{s: tok_j[s] = t} w_j[s] * (down row i of job j's expert . H[j][s])
//             (decode / verify: a call has at most 8 tokens), or
//   down_y  : Y[j][s][i] = down row i . H[j][s] (prompt chunks: any number of tokens), then
//   gather  : OUT[t][i] = sum_k w_tk * Y[slot_tk][i] over the token's k = 0..top_k-1 entries, in routing order
//             (deterministic: no atomics).
// One sub-group (16 lanes) per weight row; the lanes split each block so that consecutive lanes read consecutive
// activations.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::v41::xe {

inline constexpr int kQ3K = 11, kQ20 = 42, kPtq1 = 143;
inline constexpr int kMaxJobTok = 8;

struct XeJob {
    const uint8_t* blob = nullptr;   ///< device pointer to the expert's blob
    uint32_t up_off = 0, down_off = 0;
    uint16_t gu_type = 0, d_type = 0;
    uint16_t gu_row = 0, d_row = 0;  ///< bytes per gate/up row and per down row
    uint8_t nt = 0;                  ///< tokens of the call this job serves (1..8)
    uint8_t pad[3] = {};
    int32_t tok[kMaxJobTok] = {};    ///< their indices in the call's X (and OUT for the decode form)
    float w[kMaxJobTok] = {};        ///< their routing weights (the decode form)
};

/// Whether the kernels can run this ggml type.
inline bool xe_type_ok(int t) { return t == kQ3K || t == kQ20 || t == kPtq1; }

/// Decode / verify: gate_up + down for `nj` jobs of a call of `ntok` <= 8 tokens.  X: [ntok][n_embd] f32,
/// H: scratch [nj][8][n_ff] f32, OUT: [ntok][n_embd] f32, overwritten (accumulate = false) or added to.
/// `jobs` is device-readable (device or host USM).  Enqueued on the in-order queue `q`; returns the last event.
sycl::event xe_experts(sycl::queue& q, const XeJob* jobs, int nj, int ntok, const float* X, float* H, float* OUT,
                       int n_embd, int n_ff, float swiglu_limit, bool accumulate);

/// Prompt chunks: gate_up + down_y for `nj` jobs (each <= 8 tokens).  Y: [nj][8][n_embd] f32.
sycl::event xe_experts_y(sycl::queue& q, const XeJob* jobs, int nj, const float* X, float* H, float* Y, int n_embd,
                         int n_ff, float swiglu_limit);

/// OUT[t][i] (+)= sum_{k < top_k} W[t*top_k + k] * Y[S[t*top_k + k]][i] (S = job * 8 + slot, -1 = skip).
sycl::event xe_gather(sycl::queue& q, const float* Y, const int32_t* S, const float* W, int ntok, int top_k,
                      int n_embd, float* OUT, bool accumulate);

/// OUT[i] = sum_k parts[k * n + i] for k < nparts, plus OUT[i] when `accumulate`.
sycl::event xe_sum_parts(sycl::queue& q, const float* parts, int nparts, int n, float* out, bool accumulate);

/// Dequantise `nrows` rows of `type` (n weights each, `row_bytes` apart) into F32 (the parity test).
sycl::event xe_dequant_rows(sycl::queue& q, int type, const uint8_t* rows, int nrows, int row_bytes, int n, float* y);

// ---- the planar int8 path (xe_planar.cpp): experts stored repacked on the cards, int8 activations, dp4a ----

/// int8 activation rows of n values: q[row * n] (int8), d[row * n / 32] (scale: x ~ d * q), s[row * n / 16] (sum of q
/// over 16 values).
struct XeQ8 {
    int8_t* q = nullptr;
    float* d = nullptr;
    int32_t* s = nullptr;
};

/// The file layout of an expert blob (gate | up | down) -> the planar layout, out of place, same size and offsets.
sycl::event xe_repack_blob(sycl::queue& q, const uint8_t* src, uint8_t* dst, int gu_type, int d_type, size_t up_off,
                           size_t down_off, int n_embd, int n_ff);
/// Quantises `nrows` rows of n f32 values (n % 32 == 0) into `out`.
sycl::event xe_quant_rows(sycl::queue& q, const float* x, int nrows, int n, XeQ8 out);
/// xe_experts on PLANAR experts: X quantised (xe_quant_rows of the call's ntok rows), H = scratch [nj][8] rows of n_ff.
/// max_nt = the most tokens any of the jobs has (0: ntok).
sycl::event xe_experts_q8(sycl::queue& q, const XeJob* jobs, int nj, int ntok, XeQ8 X, XeQ8 H, float* OUT, int n_embd,
                          int n_ff, float swiglu_limit, bool accumulate, int max_nt = 0);

/// larry (xe_xmx.cpp): the same on the XMX engines (fp16 DPAS) - xt = xe_xt of the call's rows ([n_embd][8] fp16,
/// rows >= ntok zero), ht = scratch [nj][n_ff][8] fp16.  The weights stream once whatever a job's tokens.
sycl::event xe_xt(sycl::queue& q, const float* x, int ntok, int n, sycl::half* xt);
sycl::event xe_experts_x(sycl::queue& q, const XeJob* jobs, int nj, int ntok, const sycl::half* xt, sycl::half* ht,
                         float* OUT, int n_embd, int n_ff, float swiglu_limit, bool accumulate);

/// xe_experts_y on PLANAR experts (Y: [nj][8][n_embd]; H scratch as above).
sycl::event xe_experts_y_q8(sycl::queue& q, const XeJob* jobs, int nj, XeQ8 X, XeQ8 H, float* Y, int n_embd, int n_ff,
                            float swiglu_limit, int max_nt);
/// xe_dequant_rows_f16 of a PLANAR matrix (nrows rows of n values).
sycl::event xe_dequant_planar_f16(sycl::queue& q, int type, const uint8_t* mat, int nrows, int n, sycl::half* y);

// ---- the prompt path's GEMM form (an expert with many tokens): fp16 operands for oneMKL's XMX GEMMs ----

/// The same dequantisation into fp16 (row-major [nrows][n]).
sycl::event xe_dequant_rows_f16(sycl::queue& q, int type, const uint8_t* rows, int nrows, int row_bytes, int n, sycl::half* y);
/// Xg[i][c] = X[tok[i]][c] in fp16, i < m (X: [*][n] f32).
sycl::event xe_gather_rows_f16(sycl::queue& q, const float* X, const int32_t* tok, int m, int n, sycl::half* Xg);
/// Hh[i][r] = silu(min(G[i][r], L)) * clamp(G[i][F + r], -L, L) in fp16 (G: [m][2F] f32 = gate | up).
sycl::event xe_swiglu_f16(sycl::queue& q, const float* G, int m, int F, float L, sycl::half* Hh);

/// larry (xe_dqgemm.cpp): C[i][n0 + c] = A[i] . W[n0 + c] for one expert matrix W ([N rows][K] packed, planar with NB
/// blocks or the file layout with row_bytes a row), A: [m][lda] fp16 (rows padded to a multiple of 8 - the tiles
/// store whole 8-row blocks), C: [m][ldc] f32.  Device-readable.
struct DqJob {
    const uint8_t* W = nullptr;
    const sycl::half* A = nullptr;
    float* C = nullptr;
    int64_t NB = 0;
    int32_t m = 0, lda = 0, ldc = 0, row_bytes = 0;
};
/// One work-group's tile: job, its first token row (multiple of 128) and first output row (multiple of 128).
struct DqTile { int32_t job = 0, m0 = 0, n0 = 0, pad = 0; };
/// Every tile of `tiles` (all its jobs one quant type, one layout), K values a row (a multiple of 64).
sycl::event xe_dqgemm(sycl::queue& q, int type, bool planar, const DqJob* jobs, const DqTile* tiles, int ntiles, int K);

}  // namespace strata::v41::xe
