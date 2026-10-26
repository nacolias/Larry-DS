// jm8_map.cpp - larry: the work-item layout of an int8 B fragment (K32 x N16, ext_intel_packed) on the B70, measured:
// lane L writes value v(L, i) into its element i (get_wi_data, no get_coord), A = one-hot rows (A[m][k_m] = 1), so
// C[m][n] = B[k_m][n] = v of the (lane, element) that holds (k_m, n).  Two passes: v = element index, v = lane.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
namespace mx = sycl::ext::oneapi::experimental::matrix;
constexpr int M = 8, N = 16, K = 32;
int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    int8_t* dA = sycl::malloc_device<int8_t>(M * K, q);
    int32_t* dC = sycl::malloc_device<int32_t>(M * N, q);
    int* dLen = sycl::malloc_device<int>(1, q);
    std::vector<int> elem_of(K * N, -1), lane_of(K * N, -1);
    for (int pass = 0; pass < 2; ++pass)
        for (int k0 = 0; k0 < K; k0 += M) {
            std::vector<int8_t> A(M * K, 0);
            for (int m = 0; m < M; ++m) A[m * K + k0 + m] = 1;
            q.memcpy(dA, A.data(), A.size()).wait();
            q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                sycl::sub_group sg = it.get_sub_group();
                const int ln = (int) sg.get_local_linear_id();
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, M, K, mx::layout::row_major> ma;
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, K, N, mx::layout::ext_intel_packed> mb;
                mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, M, N> mc;
                mx::joint_matrix_fill(sg, mc, 0);
                mx::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dA), K);
                auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
                if (ln == 0) *dLen = (int) wb.length();
                for (int i = 0; i < (int) wb.length(); ++i) wb[i] = (int8_t) (pass == 0 ? i : ln);
                mx::joint_matrix_mad(sg, mc, ma, mb, mc);
                mx::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dC), N,
                                       mx::layout::row_major);
            }).wait();
            std::vector<int32_t> C(M * N);
            q.memcpy(C.data(), dC, C.size() * 4).wait();
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) (pass == 0 ? elem_of : lane_of)[(k0 + m) * N + n] = C[m * N + n];
        }
    int len = 0;
    q.memcpy(&len, dLen, 4).wait();
    std::printf("wi_data length %d\n", len);
    for (int k = 0; k < K; ++k) {
        std::printf("k %2d:", k);
        for (int n = 0; n < N; ++n) std::printf(" %2d.%-2d", lane_of[k * N + n], elem_of[k * N + n]);
        std::printf("\n");
    }
    return 0;
}
