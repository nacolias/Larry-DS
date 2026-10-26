// coord_probe.cpp - larry: which (row, col) each work-item holds in a joint_matrix fragment (B half 16x16, A half
// 8x16, accumulator float 8x16) on the B70, and a B fragment built in registers (each lane: its column's 16 values)
// checked against the CPU.
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    int* out = sycl::malloc_shared<int>(3 * 16 * 32 * 2, q);
    q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const int l = (int) it.get_local_id(0);
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> mb;
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16, mx::layout::row_major> ma;
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> mc;
        auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
        for (int i = 0; i < (int) wb.length() && i < 32; ++i) { auto [r, c] = wb[i].get_coord(); out[(0 * 16 + l) * 64 + 2 * i] = (int) r; out[(0 * 16 + l) * 64 + 2 * i + 1] = (int) c; }
        auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
        for (int i = 0; i < (int) wa.length() && i < 32; ++i) { auto [r, c] = wa[i].get_coord(); out[(1 * 16 + l) * 64 + 2 * i] = (int) r; out[(1 * 16 + l) * 64 + 2 * i + 1] = (int) c; }
        auto wc = sycl::ext::oneapi::detail::get_wi_data(sg, mc);
        for (int i = 0; i < (int) wc.length() && i < 32; ++i) { auto [r, c] = wc[i].get_coord(); out[(2 * 16 + l) * 64 + 2 * i] = (int) r; out[(2 * 16 + l) * 64 + 2 * i + 1] = (int) c; }
        if (l == 0) { out[3 * 16 * 64 - 3] = (int) wb.length(); out[3 * 16 * 64 - 2] = (int) wa.length(); out[3 * 16 * 64 - 1] = (int) wc.length(); }
    }).wait();
    const char* nm[3] = {"B half 16x16", "A half 8x16", "C float 8x16"};
    std::printf("lengths: B %d A %d C %d\n", out[3 * 16 * 64 - 3], out[3 * 16 * 64 - 2], out[3 * 16 * 64 - 1]);
    const int len[3] = {out[3 * 16 * 64 - 3], out[3 * 16 * 64 - 2], out[3 * 16 * 64 - 1]};
    for (int f = 0; f < 3; ++f) {
        std::printf("%s:\n", nm[f]);
        for (int l : {0, 1, 2, 15}) {
            std::printf("  lane %2d:", l);
            for (int i = 0; i < len[f] && i < 32; ++i) std::printf(" (%d,%d)", out[(f * 16 + l) * 64 + 2 * i], out[(f * 16 + l) * 64 + 2 * i + 1]);
            std::printf("\n");
        }
    }
    // B built in registers: lane n writes element i (coord (r, n)) = Bref[r][n]; C = A . B vs the CPU
    constexpr int TM = 8, TN = 16, TK = 16;
    std::vector<sycl::half> A(TM * TK), B(TK * TN);
    for (int i = 0; i < TM * TK; ++i) A[i] = (sycl::half) (((i * 7) % 13) / 8.0f - 0.7f);
    for (int i = 0; i < TK * TN; ++i) B[i] = (sycl::half) (((i * 5) % 11) / 8.0f - 0.6f);
    std::vector<float> ref(TM * TN, 0.f);
    for (int m = 0; m < TM; ++m) for (int n = 0; n < TN; ++n) for (int k = 0; k < TK; ++k) ref[m * TN + n] += (float) A[m * TK + k] * (float) B[k * TN + n];
    sycl::half* dA = sycl::malloc_device<sycl::half>(A.size(), q);
    sycl::half* dB = sycl::malloc_device<sycl::half>(B.size(), q);
    float* dC = sycl::malloc_device<float>(TM * TN, q);
    q.memcpy(dA, A.data(), A.size() * 2);
    q.memcpy(dB, B.data(), B.size() * 2).wait();
    q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const int n = (int) it.get_local_id(0);
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ma;
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> mb;
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> mc;
        mx::joint_matrix_fill(sg, mc, 0.0f);
        mx::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dA), TK);
        sycl::half col[TK];
        for (int k = 0; k < TK; ++k) col[k] = dB[k * TN + n];   // this lane's column
        auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
#pragma unroll
        for (int i = 0; i < TK; ++i) wb[i] = col[i];             // assumes element i = (row i, col lane)
        mx::joint_matrix_mad(sg, mc, ma, mb, mc);
        mx::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dC), TN, mx::layout::row_major);
    }).wait();
    std::vector<float> C(TM * TN);
    q.memcpy(C.data(), dC, C.size() * 4).wait();
    double md = 0;
    for (int i = 0; i < TM * TN; ++i) md = std::max(md, (double) std::fabs(C[i] - ref[i]));
    std::printf("B from registers (element i = row i): max |diff| %.3g %s\n", md, md < 1e-2 ? "OK" : "WRONG");
    return 0;
}
