// jm_probe.cpp - joint_matrix (XMX) on the B70: the device's combinations, and a half x half -> float tile product
// with A and B row-major (B loaded with layout::row_major and with ext_intel_packed after a VNNI pack), checked on the CPU.
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto combos = q.get_device().get_info<syclex::info::device::matrix_combinations>();
    for (const auto& c : combos)
        std::printf("combo: M %zu N %zu K %zu (max %zu %zu %zu) a %d b %d c %d d %d\n", c.msize, c.nsize, c.ksize, c.max_msize,
                    c.max_nsize, c.max_ksize, (int) c.atype, (int) c.btype, (int) c.ctype, (int) c.dtype);
    constexpr int TM = 8, TN = 16, TK = 16, KK = 64;   // C[TM x TN] = A[TM x KK] . B[KK x TN]
    std::vector<sycl::half> A(TM * KK), B(KK * TN);
    for (int i = 0; i < TM * KK; ++i) A[i] = (sycl::half) (((i * 7) % 13) / 8.0f - 0.7f);
    for (int i = 0; i < KK * TN; ++i) B[i] = (sycl::half) (((i * 5) % 11) / 8.0f - 0.6f);
    std::vector<float> ref(TM * TN, 0.f);
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
            for (int k = 0; k < KK; ++k) ref[m * TN + n] += (float) A[m * KK + k] * (float) B[k * TN + n];
    sycl::half* dA = sycl::malloc_device<sycl::half>(A.size(), q);
    sycl::half* dB = sycl::malloc_device<sycl::half>(B.size(), q);
    float* dC = sycl::malloc_device<float>(TM * TN, q);
    q.memcpy(dA, A.data(), A.size() * 2);
    q.memcpy(dB, B.data(), B.size() * 2).wait();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            sycl::sub_group sg = it.get_sub_group();
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ma;
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> mb;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> mc;
            mx::joint_matrix_fill(sg, mc, 0.0f);
            for (int k = 0; k < KK; k += TK) {
                mx::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dA + k), KK);
                mx::joint_matrix_load(sg, mb, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dB + k * TN), TN);
                mx::joint_matrix_mad(sg, mc, ma, mb, mc);
            }
            mx::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dC), TN,
                                   mx::layout::row_major);
        });
    }).wait();
    std::vector<float> got(TM * TN);
    q.memcpy(got.data(), dC, got.size() * 4).wait();
    double err = 0;
    for (int i = 0; i < TM * TN; ++i) err = std::max(err, (double) std::fabs(got[i] - ref[i]));
    std::printf("row-major B: max err %.3e %s\n", err, err < 1e-3 ? "ok" : "FAIL");
    return 0;
}
