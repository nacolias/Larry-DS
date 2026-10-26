#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
namespace strata::v41::xe {
inline constexpr int kQ3K = 11, kQ20 = 42, kPtq1 = 143;
struct DqJob {
    const uint8_t* W = nullptr;
    const sycl::half* A = nullptr;
    float* C = nullptr;
    int64_t NB = 0;
    int32_t m = 0, lda = 0, ldc = 0, row_bytes = 0;
};
struct DqTile { int32_t job = 0, m0 = 0, n0 = 0, pad = 0; };
sycl::event xe_dqgemm(sycl::queue& q, int type, bool planar, const DqJob* jobs, const DqTile* tiles, int ntiles, int K);
}  // namespace strata::v41::xe
