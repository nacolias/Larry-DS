// jm8_probe.cpp - larry: int8 joint_matrix (XMX) on the B70 - M8 N16 K32, int8 x int8 -> int32, A row-major from
// memory, B written per work-item (get_wi_data: which (k, n) each lane's elements are), checked against the CPU; then
// a DPAS-bound loop's throughput.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

namespace mx = sycl::ext::oneapi::experimental::matrix;
constexpr int M = 8, N = 16, K = 32;

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::vector<int8_t> A(M * K), B(K * N);
    for (int i = 0; i < M * K; ++i) A[i] = (int8_t) ((i * 7 + 3) % 255 - 127);
    for (int i = 0; i < K * N; ++i) B[i] = (int8_t) ((i * 13 + 5) % 9 - 4);   // like Q3_K values
    int8_t* dA = sycl::malloc_device<int8_t>(M * K, q);
    int8_t* dB = sycl::malloc_device<int8_t>(K * N, q);
    int32_t* dC = sycl::malloc_device<int32_t>(M * N, q);
    int32_t* dK = sycl::malloc_device<int32_t>(16 * 64, q);   // per lane: the (k, n) coordinate of each B element
    q.memcpy(dA, A.data(), A.size()).wait();
    q.memcpy(dB, B.data(), B.size()).wait();
    // 1) where each lane's B elements live (and fill them from B[k][n] accordingly)
    q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        sycl::sub_group sg = it.get_sub_group();
        const int ln = (int) sg.get_local_linear_id();
        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, M, K, mx::layout::row_major> ma;
        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, K, N, mx::layout::ext_intel_packed> mb;
        mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, M, N> mc;
        mx::joint_matrix_fill(sg, mc, 0);
        mx::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dA), K);
        auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
        for (int i = 0; i < (int) wb.length(); ++i) {
            auto [r, c] = wb[i].get_coord();   // (row k, column n) of this element
            dK[ln * 64 + i] = (int) (r * 1000 + c);
            wb[i] = dB[r * N + c];
        }
        mx::joint_matrix_mad(sg, mc, ma, mb, mc);
        mx::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dC), N,
                               mx::layout::row_major);
    }).wait();
    std::vector<int32_t> C(M * N), coords(16 * 64);
    q.memcpy(C.data(), dC, C.size() * 4).wait();
    q.memcpy(coords.data(), dK, coords.size() * 4).wait();
    int bad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int ref = 0;
            for (int k = 0; k < K; ++k) ref += (int) A[m * K + k] * (int) B[k * N + n];
            if (ref != C[m * N + n]) ++bad;
        }
    std::printf("int8 M8 N16 K32 (B ext_intel_packed): %d of %d wrong\n", bad, M * N);
    std::printf("lane 0 B elements (k*1000+n):");
    for (int i = 0; i < 32; ++i) std::printf(" %d", coords[i]);
    std::printf("\nlane 5:");
    for (int i = 0; i < 32; ++i) std::printf(" %d", coords[5 * 64 + i]);
    std::printf("\n");
    // 2) throughput: each sub-group 4096 DPAS (k-loop), 4096 work-groups of 1 sub-group
    const int iters = 4096, ngroups = 4096;
    int32_t* dO = sycl::malloc_device<int32_t>((size_t) ngroups * M * N, q);
    auto run = [&] {
        q.parallel_for(sycl::nd_range<1>((size_t) ngroups * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            sycl::sub_group sg = it.get_sub_group();
            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, M, K, mx::layout::row_major> ma;
            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, K, N, mx::layout::ext_intel_packed> mb;
            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, M, N> mc;
            mx::joint_matrix_fill(sg, mc, 0);
            mx::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dA), K);
            mx::joint_matrix_load(sg, mb, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dB), N * 4);
            for (int i = 0; i < iters; ++i) mx::joint_matrix_mad(sg, mc, ma, mb, mc);
            mx::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                       dO + (size_t) it.get_group(0) * M * N), N, mx::layout::row_major);
        });
    };
    run(); q.wait();
    auto t0 = std::chrono::steady_clock::now();
    run(); q.wait();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("throughput: %.1f TOPS int8 (%.0f us)\n", 2.0 * M * N * K * (double) iters * ngroups / s / 1e12, s * 1e6);
    return 0;
}
