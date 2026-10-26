// ggml-sycl/fastmv.hpp - larry (DeepSeek-V4.1 on Intel Arc B70): mat-vec kernels for the decode / verify shapes of the
// DSV4.1 dense half (1..8 columns), written for bandwidth on Xe2.
//
// ggml-sycl's DMMV / MMVQ kernels reach 400-590 GB/s on large matrices but 150-300 GB/s on the dense half's small ones
// (5120x512, 5120x1280, 2304x5120, the 8 grouped 4096x1024 of wo_a run as 8 launches) and ~14 GB/s on the hyper-
// connection projections (F16, 20480 wide, 24 rows: one row per work-group).  These kernels:
//   * read ggml-sycl's REORDERED Q6_K layout (whole tensor: ql[128] | qh[64] | scales[16] | d per block, each plane
//     contiguous), so every load is aligned and wide; F16 / F32 rows directly;
//   * give each lane 32-value units with consecutive lanes on consecutive bytes; LPR lanes per row (4 / 8 / 16) so a
//     short row still keeps every lane busy;
//   * split K over KS sub-groups of a work-group (reduced in local memory) when the matrix has few rows;
//   * take a batch dimension (wo_a's 8 groups: one launch) and up to 8 columns (DSpark verify passes).
//
// y[g][c][r] = sum_k W[g][r][k] * x[g][c][k]   (x: f32, contiguous in k; any column / batch stride)
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace fastmv {

enum class WType { Q6K_R, Q6K, F16, F32, Q6K_R8 };   // Q6K = ggml's standard block_q6_K layout (210-byte blocks);
                                                    // Q6K_R8 = Q6K_R with int8 activations (xq / xd, dp4a)

struct Args {
    const void * w = nullptr;   // weights
    int64_t K = 0, N = 0;       // row length, rows per batch
    int64_t G = 1;              // batches (src0 ne[2]); src1 batch g uses weight batch g / g_div
    int64_t g_div = 1;
    int64_t n_blocks_total = 0; // Q6K_R: blocks per plane set (a 2-D tensor: all of it; a 3-D one: one weight batch,
                                // ggml-sycl's reorder_qw_q6_k_moe lays every batch out as its own ql|qh|scales|d planes)
    size_t w_row_bytes = 0;     // F16/F32: bytes per row
    size_t w_batch_bytes = 0;   // bytes per weight batch (Q6K_R: the plane set's base advances by this)
    const float * x = nullptr;
    int64_t x_col_stride = 0, x_batch_stride = 0;   // floats
    float * y = nullptr;
    int64_t y_col_stride = 0, y_batch_stride = 0;   // floats
    int ncols = 1;
    // Q6K_R8: the activations quantised by quant_x (same strides as x, in elements): x ~ xd[col][k / 32] * xq[col][k]
    const int8_t * xq = nullptr;
    const float * xd = nullptr;
};

// for G batches of ncols columns (row r = g * ncols + c, K % 32 == 0):
//   xq[r * K + k] = round(x[g * x_batch_stride + c * x_col_stride + k] / xd[r * K / 32 + k / 32])
sycl::event quant_x(sycl::queue & q, const float * x, int64_t x_col_stride, int64_t x_batch_stride, int ncols, int64_t G,
                    int64_t K, int8_t * xq, float * xd);

// ggml-sycl's per-slice reordered Q6_K (one slice: ql | qh | scales | d planes over nrows x K) -> f16 rows [nrows][K]
sycl::event dequant_q6k_planar_f16(sycl::queue & q, const void * slice, int64_t nrows, int64_t K, sycl::half * y);
// f16 copy of ncols columns (column stride in floats): y[c * K + k] = x[c * x_col_stride + k]
sycl::event cols_to_f16(sycl::queue & q, const float * x, int64_t x_col_stride, int64_t ncols, int64_t K, sycl::half * y);

// launch; returns false when the shape is not covered (caller falls back)
bool launch(sycl::queue & q, WType t, const Args & a);

// larry: the XMX path for 2..8 columns of a Q6K_R weight (planar).  quant_xT: G batches x ncols columns (as quant_x)
// -> xT[g][k][8] (fp16 integers, the int8 rounding of quant_x; columns >= ncols zero) and xdT[g][k / 32][8] scales.
sycl::event quant_xT(sycl::queue & q, const float * x, int64_t x_col_stride, int64_t x_batch_stride, int ncols, int64_t G,
                     int64_t K, sycl::half * xT, float * xdT);
// whether launch_xmx takes this shape (the latency-bound ones), and the launch (a.xq / a.xd unused)
bool xmx_wants(const Args & a);
bool launch_xmx(sycl::queue & q, const Args & a, const sycl::half * xT, const float * xdT);

}  // namespace fastmv
