// repack_bench.cpp - larry: xe_repack_blob's time for one expert of each class (file layout -> planar), on card 1
#include "../../xolotl/tools/v41/engine/xe/xe_kernels.hpp"
#include <cstdio>
#include <vector>
namespace xe = strata::v41::xe;
int main() {
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const int H = 5120, F = 2304;
    for (int t : {xe::kQ3K, xe::kQ20, xe::kPtq1}) {
        const int qk = t == xe::kQ3K ? 256 : t == xe::kQ20 ? 64 : 128, bb = t == xe::kQ3K ? 110 : t == xe::kQ20 ? 18 : 28;
        const size_t gu = (size_t) F * (H / qk) * bb, dn = (size_t) H * (F / qk) * bb, tot = 2 * gu + dn;
        uint8_t* src = sycl::malloc_device<uint8_t>(tot, q);
        uint8_t* dst = sycl::malloc_device<uint8_t>(tot, q);
        q.memset(src, 7, tot).wait();
        xe::xe_repack_blob(q, src, dst, t, t, gu, 2 * gu, H, F).wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 10; ++r) xe::xe_repack_blob(q, src, dst, t, t, gu, 2 * gu, H, F);
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 10;
        std::printf("type %3d: %.1f MB expert repacked in %.1f us (%.1f GB/s)\n", t, tot / 1e6, us, tot / us / 1e3);
        sycl::free(src, q); sycl::free(dst, q);
    }
    return 0;
}
