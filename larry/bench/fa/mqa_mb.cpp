// mqa_mb.cpp - larry (#3): micro-benchmarks for a fused sparse MQA prompt-attention kernel on the B70 (one query's
// 64 heads x its 640 gathered keys of 512 dims): the XMX rate when the B (or A) fragments come from local memory in
// each layout a QK^T / PV product would use, the local memory's own read rate, and per-lane gathers of key-row pieces
// from global memory.  Every kernel: 16 work-groups an Xe-core's worth (512 groups), GRF 256.
//   mqa_mb [device 1]
// Modes (each sub-group 2 x 2 accumulators, A fragments in registers unless noted; "DPAS" = one 8x16x16 multiply):
//   0  B col_major from local memory (QK^T: K^T tiles from row-major key rows)
//   1  B row_major from local memory (PV: V tiles from the same rows - a transposing load)
//   2  B ext_intel_packed from local memory (PV: V tiles from a VNNI copy)
//   3  B by hand: a lane's 16 halves (32 B) from its key row in local memory, written into the fragment
//   4  A row_major from local memory, B in registers (PV: P tiles)
//   5  local memory reads alone: 32 B a lane a load
//   6  global gathers: 32 B a lane from random 1 KB rows (an L2-sized and a VRAM-sized row set)
//   7  B col_major from global memory (K^T tiles from a group's own 64 x 512 half region: 2D block loads)
//   8  B row_major from global memory (V tiles from the same region: VNNI-transforming block loads)
//   9  A row_major from global memory (Q / P tiles), B in registers
//  10  7 + 9: both from global memory (no fragment in registers)
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using half = sycl::half;

constexpr int TM = 8, TN = 16, TK = 16;
constexpr int ROWS = 64, LD = 512 + 16;   // local memory: 64 key rows of 512 halves, rows 528 halves apart
constexpr int SGS = 8, WG = SGS * 16, NWG = 512;
using MA = mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major>;
using MBc = mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major>;
using MBr = mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::row_major>;
using MBp = mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::ext_intel_packed>;
using MC = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN>;

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <int MODE>
static double run(sycl::queue & q, float * out, const uint16_t * grows, const int32_t * gidx, int iters, const half * greg = nullptr) {
    auto launch = [&]() {
        return q.submit([&](sycl::handler & h) {
            sycl::local_accessor<half, 1> sl(sycl::range<1>((size_t) ROWS * LD), h);
            h.parallel_for(sycl::nd_range<1>((size_t) NWG * WG, WG), syclex::properties{sycl::ext::intel::experimental::grf_size<256>},
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int lid = (int) it.get_local_id(0), ln = lid & 15, s = (int) sg.get_group_linear_id();
                for (int i = lid; i < ROWS * LD; i += WG) sl[i] = (half) (float) ((i * 7 + s) % 13 - 6);
                it.barrier(sycl::access::fence_space::local_space);
                auto lp = sl.template get_multi_ptr<sycl::access::decorated::no>();
                MC c00, c01, c10, c11;
                mx::joint_matrix_fill(sg, c00, 0.0f); mx::joint_matrix_fill(sg, c01, 0.0f);
                mx::joint_matrix_fill(sg, c10, 0.0f); mx::joint_matrix_fill(sg, c11, 0.0f);
                const half * gr = greg ? greg + (size_t) it.get_group(0) * 64 * 512 : nullptr;   // the group's 64 KB region
                auto gp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(gr);
                MA a0, a1;
                mx::joint_matrix_load(sg, a0, lp + (size_t) ln * 0, LD);
                mx::joint_matrix_load(sg, a1, lp + (size_t) 8 * LD, LD);
                MBc bc0, bc1;
                mx::joint_matrix_load(sg, bc0, lp + (size_t) 16 * LD, LD);
                mx::joint_matrix_load(sg, bc1, lp + (size_t) 32 * LD, LD);
                float acc = 0.0f;
                uint32_t gx = 0;
                for (int r = 0; r < iters; ++r) {
#pragma unroll 4
                    for (int ks = 0; ks < 32; ++ks) {
                        const int kb = (ks + s) & 31;   // the 16-value step (dims for 0 / 3, a row block + column for 1 / 2 / 4)
                        if constexpr (MODE == 0) {
                            MBc b0, b1;
                            mx::joint_matrix_load(sg, b0, lp + (size_t) (0 * LD + kb * 16), LD);
                            mx::joint_matrix_load(sg, b1, lp + (size_t) (16 * LD + kb * 16), LD);
                            mx::joint_matrix_mad(sg, c00, a0, b0, c00); mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                            mx::joint_matrix_mad(sg, c10, a1, b0, c10); mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                        } else if constexpr (MODE == 1) {
                            MBr b0, b1;
                            const size_t o = (size_t) ((kb & 3) * 16) * LD + (size_t) ((kb >> 2) * 64);
                            mx::joint_matrix_load(sg, b0, lp + o, LD);
                            mx::joint_matrix_load(sg, b1, lp + o + 16, LD);
                            mx::joint_matrix_mad(sg, c00, a0, b0, c00); mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                            mx::joint_matrix_mad(sg, c10, a1, b0, c10); mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                        } else if constexpr (MODE == 2) {
                            MBp b0, b1;   // a VNNI copy: key pair kp's 512 dims x 2 halves = 1024 halves a packed row
                            const size_t o = (size_t) ((kb & 3) * 8) * 1024 + (size_t) ((kb >> 2) * 64);
                            mx::joint_matrix_load(sg, b0, lp + o, 1024);
                            mx::joint_matrix_load(sg, b1, lp + o + 32, 1024);
                            mx::joint_matrix_mad(sg, c00, a0, b0, c00); mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                            mx::joint_matrix_mad(sg, c10, a1, b0, c10); mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                        } else if constexpr (MODE == 3) {
                            MBc b0, b1;
                            const sycl::vec<half, 16> v0 = *reinterpret_cast<const sycl::vec<half, 16> *>(&sl[(size_t) ln * LD + kb * 16]);
                            const sycl::vec<half, 16> v1 = *reinterpret_cast<const sycl::vec<half, 16> *>(&sl[(size_t) (16 + ln) * LD + kb * 16]);
                            auto w0 = sycl::ext::oneapi::detail::get_wi_data(sg, b0);
                            auto w1 = sycl::ext::oneapi::detail::get_wi_data(sg, b1);
#pragma unroll
                            for (int j = 0; j < 16; ++j) { w0[j] = v0[j]; w1[j] = v1[j]; }
                            mx::joint_matrix_mad(sg, c00, a0, b0, c00); mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                            mx::joint_matrix_mad(sg, c10, a1, b0, c10); mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                        } else if constexpr (MODE == 4) {
                            MA x0, x1;
                            const size_t o = (size_t) ((kb & 3) * 16) * LD + (size_t) ((kb >> 2) * 64);
                            mx::joint_matrix_load(sg, x0, lp + o, LD);
                            mx::joint_matrix_load(sg, x1, lp + o + (size_t) 8 * LD, LD);
                            mx::joint_matrix_mad(sg, c00, x0, bc0, c00); mx::joint_matrix_mad(sg, c01, x0, bc1, c01);
                            mx::joint_matrix_mad(sg, c10, x1, bc0, c10); mx::joint_matrix_mad(sg, c11, x1, bc1, c11);
                        } else if constexpr (MODE == 7) {
                            MBc b0, b1;
                            mx::joint_matrix_load(sg, b0, gp + (size_t) ((s & 1) * 32 * 512 + kb * 16), 512);
                            mx::joint_matrix_load(sg, b1, gp + (size_t) ((s & 1) * 32 * 512 + 16 * 512 + kb * 16), 512);
                            mx::joint_matrix_mad(sg, c00, a0, b0, c00); mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                            mx::joint_matrix_mad(sg, c10, a1, b0, c10); mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                        } else if constexpr (MODE == 8) {
                            MBr b0, b1;
                            const size_t o = (size_t) ((kb & 3) * 16) * 512 + (size_t) ((kb >> 2) * 64);
                            mx::joint_matrix_load(sg, b0, gp + o, 512);
                            mx::joint_matrix_load(sg, b1, gp + o + 16, 512);
                            mx::joint_matrix_mad(sg, c00, a0, b0, c00); mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                            mx::joint_matrix_mad(sg, c10, a1, b0, c10); mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                        } else if constexpr (MODE == 9) {
                            MA x0, x1;
                            const size_t o = (size_t) ((kb & 3) * 16) * 512 + (size_t) ((kb >> 2) * 64);
                            mx::joint_matrix_load(sg, x0, gp + o, 512);
                            mx::joint_matrix_load(sg, x1, gp + o + (size_t) 8 * 512, 512);
                            mx::joint_matrix_mad(sg, c00, x0, bc0, c00); mx::joint_matrix_mad(sg, c01, x0, bc1, c01);
                            mx::joint_matrix_mad(sg, c10, x1, bc0, c10); mx::joint_matrix_mad(sg, c11, x1, bc1, c11);
                        } else if constexpr (MODE == 10) {
                            MA x0, x1;
                            MBc b0, b1;
                            const size_t o = (size_t) ((kb & 3) * 16) * 512 + (size_t) ((kb >> 2) * 64);
                            mx::joint_matrix_load(sg, x0, gp + o, 512);
                            mx::joint_matrix_load(sg, x1, gp + o + (size_t) 8 * 512, 512);
                            mx::joint_matrix_load(sg, b0, gp + (size_t) ((s & 1) * 32 * 512 + kb * 16), 512);
                            mx::joint_matrix_load(sg, b1, gp + (size_t) ((s & 1) * 32 * 512 + 16 * 512 + kb * 16), 512);
                            mx::joint_matrix_mad(sg, c00, x0, b0, c00); mx::joint_matrix_mad(sg, c01, x0, b1, c01);
                            mx::joint_matrix_mad(sg, c10, x1, b0, c10); mx::joint_matrix_mad(sg, c11, x1, b1, c11);
                        } else if constexpr (MODE == 5) {
                            const sycl::vec<half, 16> v0 = *reinterpret_cast<const sycl::vec<half, 16> *>(&sl[(size_t) ((ln + kb) & 63) * LD + ((kb * 16 + s * 64) & 511)]);
                            acc += (float) v0[0] + (float) v0[7] + (float) v0[15];
                        } else {
                            const int32_t row = gidx[(size_t) ((it.get_group(0) * 977 + r * 131 + ks * 17 + ln * 5 + s) & 65535)];
                            const sycl::vec<uint32_t, 8> v = *reinterpret_cast<const sycl::vec<uint32_t, 8> *>(grows + (size_t) row * 512 + (size_t) ((kb * 16) & 511));
                            gx ^= v[0] ^ v[3] ^ v[7];
                        }
                    }
                }
                if (out) {
                    auto d00 = sycl::ext::oneapi::detail::get_wi_data(sg, c00);
                    auto d11 = sycl::ext::oneapi::detail::get_wi_data(sg, c11);
                    auto d01 = sycl::ext::oneapi::detail::get_wi_data(sg, c01);
                    auto d10 = sycl::ext::oneapi::detail::get_wi_data(sg, c10);
                    out[it.get_global_id(0)] = (float) d00[0] + (float) d11[1] + (float) d01[2] + (float) d10[3] + acc + (float) (gx & 1);
                }
            });
        });
    };
    launch().wait();
    std::vector<double> ts;
    for (int k = 0; k < 5; ++k) { const double t0 = now_ms(); launch().wait(); ts.push_back(now_ms() - t0); }
    std::sort(ts.begin(), ts.end());
    return ts[2];
}

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 1;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::queue q(devs[(size_t) dev], sycl::property::queue::in_order());
    std::printf("%s: L2 %zu KB, local %zu KB, %u compute units, max clock %u MHz\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                (size_t) q.get_device().get_info<sycl::info::device::global_mem_cache_size>() / 1024,
                (size_t) q.get_device().get_info<sycl::info::device::local_mem_size>() / 1024,
                q.get_device().get_info<sycl::info::device::max_compute_units>(), q.get_device().get_info<sycl::info::device::max_clock_frequency>());
    float * out = sycl::malloc_device<float>((size_t) NWG * WG, q);
    const size_t NR = 262144;   // 256 MB of 1 KB rows
    uint16_t * grows = sycl::malloc_device<uint16_t>(NR * 512, q);
    int32_t * gidx = sycl::malloc_device<int32_t>(65536, q);
    q.memset(grows, 0x3c, NR * 1024).wait();
    const double dpas_total = (double) NWG * SGS * 32 * 4;   // a sub-group's 4 multiplies a step, 32 steps an iteration
    auto rep = [&](const char * name, int iters, double ms) {
        const double n = dpas_total * iters;
        std::printf("  %-58s %8.3f ms  %7.1f TFLOPS  (%.2f clk a DPAS an XVE at 2.8 GHz)\n", name, ms, n * 4096.0 / ms * 1e-9,
                    ms * 1e-3 * 2.8e9 * 256.0 / n);
    };
    const int IT = 64;
    rep("0 B col_major from local (QK^T K^T tiles)", IT, run<0>(q, out, grows, gidx, IT));
    rep("1 B row_major from local (PV V tiles, transposing)", IT, run<1>(q, out, grows, gidx, IT));
    rep("2 B ext_intel_packed from local (PV, a VNNI copy)", IT, run<2>(q, out, grows, gidx, IT));
    rep("3 B by hand: 32 B a lane from its row + wi_data", IT, run<3>(q, out, grows, gidx, IT));
    rep("4 A row_major from local, B in registers (P tiles)", IT, run<4>(q, out, grows, gidx, IT));
    {
        const double ms = run<5>(q, out, grows, gidx, IT);
        const double bytes = (double) NWG * WG * IT * 32 * 32;
        std::printf("  %-58s %8.3f ms  %7.1f GB/s  (%.0f B a clock an Xe-core)\n", "5 local reads, 32 B a lane", ms, bytes / ms * 1e-6,
                    bytes / (ms * 1e-3 * 2.8e9 * 32));
    }
    {
        half * greg = sycl::malloc_device<half>((size_t) NWG * 64 * 512, q);   // 32 MB: a 64 KB region a group
        q.memset(greg, 0, (size_t) NWG * 64 * 512 * 2).wait();
        rep("7 B col_major from global (2D block loads, K^T)", IT, run<7>(q, out, grows, gidx, IT, greg));
        rep("8 B row_major from global (VNNI block loads, V)", IT, run<8>(q, out, grows, gidx, IT, greg));
        rep("9 A row_major from global, B in registers", IT, run<9>(q, out, grows, gidx, IT, greg));
        rep("10 A and B from global", IT, run<10>(q, out, grows, gidx, IT, greg));
        sycl::free(greg, q);
    }
    for (int set : {8192, 262144}) {
        std::vector<int32_t> h(65536);
        std::mt19937 rng(3);
        for (auto & v : h) v = (int32_t) (rng() % (uint32_t) set);
        q.memcpy(gidx, h.data(), h.size() * 4).wait();
        const double ms = run<6>(q, out, grows, gidx, 16);
        const double bytes = (double) NWG * WG * 16 * 32 * 32;
        std::printf("  6 global gathers, 32 B a lane, %6d random 1 KB rows (%3d MB)          %8.3f ms  %7.1f GB/s\n", set, set / 1024, ms,
                    bytes / ms * 1e-6);
    }
    return 0;
}
