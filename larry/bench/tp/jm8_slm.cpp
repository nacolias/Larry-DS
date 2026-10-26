// jm8_slm.cpp - larry: int8 B (K32 x N16, ext_intel_packed) through LOCAL memory, checked: lane n holds column n's 32
// values as 8 dwords (4 consecutive K values a dword) and stores them at (k / 4) * 64 + n * 4 bytes (packed [K/4][N][4]),
// then joint_matrix_load(stride 64 bytes = 16 dwords?); A int8 8 x 32 row-major from global; C int32 against the CPU.
// Tries the load strides 16, 64 and the A load stride 32.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
namespace mx = sycl::ext::oneapi::experimental::matrix;
constexpr int M = 8, N = 16, K = 32;
int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::vector<int8_t> A(M * K), B(K * N);
    for (int i = 0; i < M * K; ++i) A[i] = (int8_t) ((i * 7 + 3) % 251 - 125);
    for (int i = 0; i < K * N; ++i) B[i] = (int8_t) ((i * 13 + 5) % 255 - 127);
    int8_t* dA = sycl::malloc_device<int8_t>(M * K, q);
    int8_t* dB = sycl::malloc_device<int8_t>(K * N, q);   // B[k][n] row-major (the lanes read their column from it)
    int32_t* dC = sycl::malloc_device<int32_t>(M * N, q);
    q.memcpy(dA, A.data(), A.size()).wait();
    q.memcpy(dB, B.data(), B.size()).wait();
    for (int stride : {16, 64}) {
        q.memset(dC, 0, M * N * 4).wait();
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<int8_t, 1> sl(sycl::range<1>(K * N), h);
            h.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                sycl::sub_group sg = it.get_sub_group();
                const int n = (int) sg.get_local_linear_id();
                // lane n: column n, 8 dwords of 4 consecutive K values
                for (int kq = 0; kq < K / 4; ++kq) {
                    uint32_t w = 0;
                    for (int b = 0; b < 4; ++b) w |= (uint32_t) (uint8_t) dB[(4 * kq + b) * N + n] << (8 * b);
                    *reinterpret_cast<uint32_t*>(&sl[(size_t) kq * 64 + n * 4]) = w;
                }
                sycl::group_barrier(sg);
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, M, K, mx::layout::row_major> ma;
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, K, N, mx::layout::ext_intel_packed> mb;
                mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, M, N> mc;
                mx::joint_matrix_fill(sg, mc, 0);
                mx::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dA), K);
                mx::joint_matrix_load(sg, mb, sl.template get_multi_ptr<sycl::access::decorated::no>(), stride);
                mx::joint_matrix_mad(sg, mc, ma, mb, mc);
                mx::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dC), N,
                                       mx::layout::row_major);
            });
        }).wait();
        std::vector<int32_t> C(M * N);
        q.memcpy(C.data(), dC, C.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < M; ++m)
            for (int nn = 0; nn < N; ++nn) {
                int ref = 0;
                for (int k = 0; k < K; ++k) ref += (int) A[m * K + k] * (int) B[k * N + nn];
                if (ref != C[m * N + nn]) ++bad;
            }
        std::printf("B load stride %d: %d of %d wrong (C[0][0] %d)\n", stride, bad, M * N, C[0]);
    }
    return 0;
}
