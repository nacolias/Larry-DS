// jmh_var.cpp - larry: which joint_matrix usage loses A's rows 1..7: A filled (a) inline, (b) through a function taking
// the matrix by reference, (c) as (b) with A reused for two products and an array of two accumulators.  A[m][k] =
// m + 1 at k = 0 (lane 0), B = 1 at k = 0: every C column should read 1..8.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
namespace mx = sycl::ext::oneapi::experimental::matrix;
using MatA = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16, mx::layout::row_major>;
using MatB = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major>;
using Acc = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16>;
inline void load_a(sycl::sub_group sg, MatA& ma, const sycl::half* xt, int ln) {
    const sycl::vec<sycl::half, 8> a8 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xt + (size_t) ln * 8);
    auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
    for (int r = 0; r < 8; ++r) wa[r] = a8[r];
}
int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::vector<sycl::half> xh(16 * 8, (sycl::half) 0.0f);
    for (int r = 0; r < 8; ++r) xh[r] = (sycl::half) (float) (r + 1);   // K index 0: rows 1..8
    sycl::half* xt = sycl::malloc_device<sycl::half>(16 * 8, q);
    q.memcpy(xt, xh.data(), xh.size() * 2).wait();
    float* out = sycl::malloc_device<float>(4 * 16, q);
    q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        sycl::sub_group sg = it.get_sub_group();
        const int ln = (int) sg.get_local_linear_id();
        MatB mb;
        auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
        for (int i = 0; i < 16; ++i) wb[i] = (sycl::half) (i == 0 ? 1.0f : 0.0f);
        // (a) inline
        {
            MatA ma;
            const sycl::vec<sycl::half, 8> a8 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xt + (size_t) ln * 8);
            auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
            for (int r = 0; r < 8; ++r) wa[r] = a8[r];
            Acc c; mx::joint_matrix_fill(sg, c, 0.0f);
            mx::joint_matrix_mad(sg, c, ma, mb, c);
            auto wc = sycl::ext::oneapi::detail::get_wi_data(sg, c);
            if (ln == 3) for (int i = 0; i < 8; ++i) out[0 * 16 + i] = wc[i];
        }
        // (b) function by reference
        {
            MatA ma;
            load_a(sg, ma, xt, ln);
            Acc c; mx::joint_matrix_fill(sg, c, 0.0f);
            mx::joint_matrix_mad(sg, c, ma, mb, c);
            auto wc = sycl::ext::oneapi::detail::get_wi_data(sg, c);
            if (ln == 3) for (int i = 0; i < 8; ++i) out[1 * 16 + i] = wc[i];
        }
        // (c) function, reused A, an array of accumulators
        {
            MatA ma;
            load_a(sg, ma, xt, ln);
            Acc c[2];
            for (int k = 0; k < 2; ++k) mx::joint_matrix_fill(sg, c[k], 0.0f);
            for (int k = 0; k < 2; ++k) mx::joint_matrix_mad(sg, c[k], ma, mb, c[k]);
            for (int k = 0; k < 2; ++k) {
                auto wc = sycl::ext::oneapi::detail::get_wi_data(sg, c[k]);
                if (ln == 3) for (int i = 0; i < 8; ++i) out[(2 + k) * 16 + i] = wc[i];
            }
        }
    }).wait();
    std::vector<float> o(4 * 16);
    q.memcpy(o.data(), out, o.size() * 4).wait();
    const char* nm[4] = {"(a) inline", "(b) by reference", "(c) array acc 0", "(c) array acc 1"};
    for (int v = 0; v < 4; ++v) { std::printf("%-18s:", nm[v]); for (int i = 0; i < 8; ++i) std::printf(" %4.0f", o[v * 16 + i]); std::printf("\n"); }
    return 0;
}
