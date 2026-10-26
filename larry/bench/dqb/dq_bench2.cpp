// dq_bench2.cpp - larry: the production xe_dqgemm (new) against the previous kernel (SLM-B), valid synthetic experts
// (finite scales), prompt-call shapes; outputs compared (Q2_0 / PTQ1_0 must be bit-identical).
#include "shim/xe_kernels.hpp"
#include "shim/xe_dqgemm_new.cpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
namespace oldk {
using strata::v41::xe::DqJob;
using strata::v41::xe::DqTile;
constexpr int kQ3K = 11, kQ20 = 42, kPtq1 = 143;
#include "dq_kern.inc"
}
using namespace strata::v41;

static double tdiff(sycl::event e) {
    return (double) (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e6;
}

int main(int argc, char** argv) {
    const int NE = argc > 1 ? std::atoi(argv[1]) : 128;
    const int M = argc > 2 ? std::atoi(argv[2]) : 69;
    const int H = 5120, F = 2304;
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const int Mp = (M + 127) / 128 * 128;
    std::mt19937 rng(7);
    double tot_old = 0, tot_new = 0;
    for (int type : {xe::kQ3K, xe::kQ20, xe::kPtq1}) {
        const int qk = type == xe::kQ3K ? 256 : type == xe::kQ20 ? 64 : 128, bb = type == xe::kQ3K ? 110 : type == xe::kQ20 ? 18 : 28;
        for (int phase = 0; phase < 2; ++phase) {
            const int K = phase == 0 ? H : F, N = phase == 0 ? F : H, nj = phase == 0 ? 2 * NE : NE;
            const size_t NB = (size_t) N * (K / qk), mat = NB * bb;
            sycl::half* A = sycl::malloc_device<sycl::half>((size_t) nj * Mp * K, q);
            float* C0 = sycl::malloc_device<float>((size_t) nj * Mp * N, q);
            float* C1 = sycl::malloc_device<float>((size_t) nj * Mp * N, q);
            {
                std::vector<sycl::half> ha((size_t) Mp * K);
                for (auto& x : ha) x = (sycl::half) ((float) (rng() % 2001) / 1000.0f - 1.0f);
                for (int j = 0; j < nj; ++j) q.memcpy(A + (size_t) j * Mp * K, ha.data(), ha.size() * 2);
                q.wait();
            }
            for (int planar = 0; planar < 2; ++planar) {
                std::vector<uint8_t> hw(mat);
                for (auto& x : hw) x = (uint8_t) rng();
                auto setd = [&](size_t off) { const sycl::half d = (sycl::half) (0.001f + 0.009f * (float) (rng() % 1000) / 1000.0f); std::memcpy(&hw[off], &d, 2); };
                for (size_t b = 0; b < NB; ++b) {
                    if (type == xe::kQ20) setd(planar ? 16 * NB + b * 2 : b * 18);
                    else if (type == xe::kPtq1) setd(planar ? 24 * NB + b * 4 + 2 : b * 28 + 26);
                    else setd(planar ? 108 * NB + b * 2 : b * 110 + 108);
                }
                uint8_t* W = sycl::malloc_device<uint8_t>(mat * nj, q);
                for (int j = 0; j < nj; ++j) q.memcpy(W + mat * j, hw.data(), mat);
                std::vector<xe::DqJob> jb(nj), jb1(nj);
                std::vector<xe::DqTile> tl;
                for (int j = 0; j < nj; ++j) {
                    jb[j].W = W + mat * j; jb[j].A = A + (size_t) j * Mp * K; jb[j].C = C0 + (size_t) j * Mp * N;
                    jb[j].NB = (int64_t) NB; jb[j].m = M; jb[j].lda = K; jb[j].ldc = N; jb[j].row_bytes = (K / qk) * bb;
                    jb1[j] = jb[j]; jb1[j].C = C1 + (size_t) j * Mp * N;
                    for (int m0 = 0; m0 < M; m0 += 128)
                        for (int n0 = 0; n0 < N; n0 += 128) tl.push_back({j, m0, n0, 0});
                }
                xe::DqJob* dj = sycl::malloc_device<xe::DqJob>(nj, q);
                xe::DqJob* dj1 = sycl::malloc_device<xe::DqJob>(nj, q);
                xe::DqTile* dt = sycl::malloc_device<xe::DqTile>(tl.size(), q);
                q.memcpy(dj, jb.data(), nj * sizeof(xe::DqJob));
                q.memcpy(dj1, jb1.data(), nj * sizeof(xe::DqJob));
                q.memcpy(dt, tl.data(), tl.size() * sizeof(xe::DqTile)).wait();
                auto old = [&]() {
                    if (type == xe::kQ20) return planar ? oldk::dqgemm_t<xe::kQ20, true, 0>(q, dj, dt, (int) tl.size(), K) : oldk::dqgemm_t<xe::kQ20, false, 0>(q, dj, dt, (int) tl.size(), K);
                    if (type == xe::kPtq1) return planar ? oldk::dqgemm_t<xe::kPtq1, true, 0>(q, dj, dt, (int) tl.size(), K) : oldk::dqgemm_t<xe::kPtq1, false, 0>(q, dj, dt, (int) tl.size(), K);
                    return planar ? oldk::dqgemm_t<xe::kQ3K, true, 0>(q, dj, dt, (int) tl.size(), K) : oldk::dqgemm_t<xe::kQ3K, false, 0>(q, dj, dt, (int) tl.size(), K);
                };
                auto nw = [&]() { return xe::xe_dqgemm(q, type, planar != 0, dj1, dt, (int) tl.size(), K); };
                old().wait(); nw().wait();
                double to = 1e30, tn = 1e30;
                for (int r = 0; r < 5; ++r) { auto e = old(); e.wait(); to = std::min(to, tdiff(e)); auto e2 = nw(); e2.wait(); tn = std::min(tn, tdiff(e2)); }
                std::vector<float> c0((size_t) M * N), c1((size_t) M * N);
                q.memcpy(c0.data(), C0, c0.size() * 4);
                q.memcpy(c1.data(), C1, c1.size() * 4).wait();
                size_t bad = 0, nonfin = 0; double md = 0, mref = 0;
                for (size_t i = 0; i < c0.size(); ++i) {
                    if (!std::isfinite(c0[i]) || !std::isfinite(c1[i])) { ++nonfin; continue; }
                    if (c0[i] != c1[i]) ++bad;
                    md = std::max(md, (double) std::fabs(c0[i] - c1[i]));
                    mref = std::max(mref, (double) std::fabs(c0[i]));
                }
                const double flop = 2.0 * nj * (double) M * K * N;
                std::printf("type %3d %s %s: old %.2f ms  new %.2f ms (%.1f TF, %.2fx)  | %zu differ, max|d| %.3g of max %.3g, %zu non-finite\n", type,
                            planar ? "planar" : "file  ", phase == 0 ? "gate|up" : "down   ", to, tn, flop / tn / 1e9, to / tn, bad, md, mref, nonfin);
                const double w = (type == xe::kQ3K ? (phase ? 0.28 : 0.29) : type == xe::kQ20 ? (phase ? 0.43 : 0.33) : (phase ? 0.29 : 0.37)) * (planar ? 0.63 : 0.37);
                tot_old += w * to; tot_new += w * tn;
                sycl::free(W, q); sycl::free(dj, q); sycl::free(dj1, q); sycl::free(dt, q);
            }
            sycl::free(A, q); sycl::free(C0, q); sycl::free(C1, q);
        }
    }
    std::printf("mix-weighted (tier map shares, 63%% planar): old %.2f ms  new %.2f ms per %d experts of %d tokens (%.2fx)\n", tot_old, tot_new, NE, M, tot_old / tot_new);
    return 0;
}
