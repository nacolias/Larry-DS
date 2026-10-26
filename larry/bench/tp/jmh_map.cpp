// jmh_map.cpp - larry: the fp16 A (8x16) / B (16x16) / float C (8x16) work-item layouts as the expert kernels use them:
// lane l sets A element r = 100 l + r (if l == 0 only: K index 0), B = 1 at K index 0 of every column, so C[m][n] =
// the A value of (row m, K 0); then each lane prints what its C elements hold.  And the reverse: lane 0 only, A
// element r = 1 for r == R, to see which row of C lights up.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
namespace mx = sycl::ext::oneapi::experimental::matrix;
int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    float* out = sycl::malloc_device<float>(16 * 8, q);
    float* cst = sycl::malloc_device<float>(8 * 16, q);
    for (int mode = 0; mode < 2; ++mode) {
        q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            sycl::sub_group sg = it.get_sub_group();
            const int ln = (int) sg.get_local_linear_id();
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16, mx::layout::row_major> ma;
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> mb;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> mc;
            mx::joint_matrix_fill(sg, mc, 0.0f);
            auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
            for (int r = 0; r < 8; ++r) wa[r] = (sycl::half) (mode == 0 ? (ln == 0 ? (float) (r + 1) : 0.0f) : (float) (ln * 8 + r));
            auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
            for (int i = 0; i < 16; ++i) wb[i] = (sycl::half) (mode == 0 ? (i == 0 ? 1.0f : 0.0f) : (i == 0 && ln == 0 ? 1.0f : 0.0f));
            mx::joint_matrix_mad(sg, mc, ma, mb, mc);
            auto wc = sycl::ext::oneapi::detail::get_wi_data(sg, mc);
            for (int i = 0; i < 8; ++i) out[ln * 8 + i] = wc[i];
            mx::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(cst), 16,
                                   mx::layout::row_major);
        }).wait();
        std::vector<float> o(16 * 8), cs(8 * 16);
        q.memcpy(o.data(), out, o.size() * 4).wait();
        q.memcpy(cs.data(), cst, cs.size() * 4).wait();
        std::printf("mode %d: per lane, its 8 C elements (get_wi_data):\n", mode);
        for (int l = 0; l < 4; ++l) { std::printf("  lane %d:", l); for (int i = 0; i < 8; ++i) std::printf(" %5.0f", o[l * 8 + i]); std::printf("\n"); }
        std::printf("  stored C (row-major 8 x 16), column 0..3 of each row:\n");
        for (int m = 0; m < 8; ++m) { std::printf("  row %d:", m); for (int n = 0; n < 4; ++n) std::printf(" %5.0f", cs[m * 16 + n]); std::printf("\n"); }
    }
    return 0;
}
