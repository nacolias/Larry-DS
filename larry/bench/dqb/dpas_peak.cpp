// dpas_peak.cpp - larry: the B70's joint_matrix (XMX) fp16 throughput with everything in registers, and with the A
// fragment re-loaded from local memory before every multiply (the dqgemm inner loop's shape).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdio>

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
constexpr int TM = 8, TN = 16, TK = 16;

template <int NACC, int ALOAD>
double run(sycl::queue& q, float* out, int wgs, int iters) {
    auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> As(sycl::range<1>(NACC * TM * TK * 2), h);
        h.parallel_for(sycl::nd_range<1>((size_t) wgs * 128, 128), syclex::properties{sycl::ext::intel::experimental::grf_size<256>},
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const int lid = (int) it.get_local_id(0);
            for (int i = lid; i < NACC * TM * TK * 2; i += 128) As[i] = (sycl::half) (0.001f * (i % 7));
            it.barrier(sycl::access::fence_space::local_space);
            auto asp = As.template get_multi_ptr<sycl::access::decorated::no>();
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[NACC];
            for (int i = 0; i < NACC; ++i) mx::joint_matrix_fill(sg, acc[i], 0.0f);
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ma;
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> mb;
            mx::joint_matrix_fill(sg, ma, (sycl::half) 0.01f);
            mx::joint_matrix_fill(sg, mb, (sycl::half) (0.001f * lid));
            for (int t = 0; t < iters; ++t) {
#pragma unroll
                for (int i = 0; i < NACC; ++i) {
                    if constexpr (ALOAD) mx::joint_matrix_load(sg, ma, asp + (size_t) ((i + t) % NACC) * TM * TK, TK);
                    mx::joint_matrix_mad(sg, acc[i], ma, mb, acc[i]);
                }
            }
            for (int i = 0; i < NACC; ++i)
                mx::joint_matrix_store(sg, acc[i], sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                    out + ((size_t) it.get_group(0) * 8 + sg.get_group_linear_id()) * TM * TN), TN, mx::layout::row_major);
        });
    });
    e.wait();
    const double ns = (double) (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>());
    const double flop = (double) wgs * 8 * iters * NACC * 2.0 * TM * TN * TK;
    return flop / ns / 1e3;   // TFLOPS
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    auto d = q.get_device();
    std::printf("%s: %u CUs, max clock %u MHz\n", d.get_info<sycl::info::device::name>().c_str(),
                d.get_info<sycl::info::device::max_compute_units>(), d.get_info<sycl::info::device::max_clock_frequency>());
    float* out = sycl::malloc_device<float>((size_t) 4096 * 8 * TM * TN, q);
    for (int wgs : {128, 512, 2048, 4096}) {
        run<16, 0>(q, out, wgs, 64);
        std::printf("wgs %4d: regs-only 16 acc %.1f TF | 8 acc %.1f | 16 acc + A load from SLM %.1f | 8 acc + A load %.1f\n", wgs,
                    run<16, 0>(q, out, wgs, 2000), run<8, 0>(q, out, wgs, 4000), run<16, 1>(q, out, wgs, 2000), run<8, 1>(q, out, wgs, 4000));
    }
    return 0;
}
