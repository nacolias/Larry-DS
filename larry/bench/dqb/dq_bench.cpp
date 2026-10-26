// dq_bench.cpp - larry: xe_dqgemm's kernel on synthetic experts (prompt-call shapes), with parts switched off:
// MODE 0 full, 1 no weight decode (constants), 2 no XMX, 3 no activation staging.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct DqJob { const uint8_t* W; const sycl::half* A; float* C; int64_t NB; int32_t m, lda, ldc, row_bytes; };
struct DqTile { int32_t job, m0, n0, pad; };
constexpr int kQ3K = 11, kQ20 = 42, kPtq1 = 143;
#include "dq_kern.inc"
#include "dq_reg.inc"
#include "dq_reg2.inc"
#include "dq_fast.inc"

template <int MODE, int KC = 64, int GRF = 256, int REG = 0, int PF = 0>
double run(sycl::queue& q, int type, bool planar, const DqJob* jobs, const DqTile* tiles, int ntiles, int K, int reps) {
    auto go = [&]() {
        if constexpr (REG == 5) {
            if (type == kQ20 && planar) return dqgemm_h<kQ20, true>(q, jobs, tiles, ntiles, K);
            if (type == kQ3K && planar) return dqgemm_h<kQ3K, true>(q, jobs, tiles, ntiles, K);
            return dqgemm_f<kQ20, true, 0>(q, jobs, tiles, ntiles, K);
        }
        if constexpr (REG == 4) {
            if (type == kQ20) return planar ? dqgemm_p<kQ20, true, MODE, PF>(q, jobs, tiles, ntiles, K) : dqgemm_p<kQ20, false, MODE, PF>(q, jobs, tiles, ntiles, K);
            if (type == kPtq1) return planar ? dqgemm_p<kPtq1, true, MODE, PF>(q, jobs, tiles, ntiles, K) : dqgemm_p<kPtq1, false, MODE, PF>(q, jobs, tiles, ntiles, K);
            return planar ? dqgemm_p<kQ3K, true, MODE, PF>(q, jobs, tiles, ntiles, K) : dqgemm_p<kQ3K, false, MODE, PF>(q, jobs, tiles, ntiles, K);
        }
        if constexpr (REG == 3) {
            if (type == kQ20) return planar ? dqgemm_f<kQ20, true, MODE>(q, jobs, tiles, ntiles, K) : dqgemm_f<kQ20, false, MODE>(q, jobs, tiles, ntiles, K);
            if (type == kPtq1) return planar ? dqgemm_f<kPtq1, true, MODE>(q, jobs, tiles, ntiles, K) : dqgemm_f<kPtq1, false, MODE>(q, jobs, tiles, ntiles, K);
            return planar ? dqgemm_f<kQ3K, true, MODE>(q, jobs, tiles, ntiles, K) : dqgemm_f<kQ3K, false, MODE>(q, jobs, tiles, ntiles, K);
        }
        if constexpr (REG == 2) {
            if (type == kQ20) return planar ? dqgemm_r2<kQ20, true, MODE, KC>(q, jobs, tiles, ntiles, K) : dqgemm_r2<kQ20, false, MODE, KC>(q, jobs, tiles, ntiles, K);
            if (type == kPtq1) return planar ? dqgemm_r2<kPtq1, true, MODE, KC>(q, jobs, tiles, ntiles, K) : dqgemm_r2<kPtq1, false, MODE, KC>(q, jobs, tiles, ntiles, K);
            return planar ? dqgemm_r2<kQ3K, true, MODE, KC>(q, jobs, tiles, ntiles, K) : dqgemm_r2<kQ3K, false, MODE, KC>(q, jobs, tiles, ntiles, K);
        }
        if constexpr (REG == 1) {
            if (type == kQ20) return planar ? dqgemm_r<kQ20, true, MODE, KC>(q, jobs, tiles, ntiles, K) : dqgemm_r<kQ20, false, MODE, KC>(q, jobs, tiles, ntiles, K);
            if (type == kPtq1) return planar ? dqgemm_r<kPtq1, true, MODE, KC>(q, jobs, tiles, ntiles, K) : dqgemm_r<kPtq1, false, MODE, KC>(q, jobs, tiles, ntiles, K);
            return planar ? dqgemm_r<kQ3K, true, MODE, KC>(q, jobs, tiles, ntiles, K) : dqgemm_r<kQ3K, false, MODE, KC>(q, jobs, tiles, ntiles, K);
        }
        if (type == kQ20) return planar ? dqgemm_t<kQ20, true, MODE, KC, GRF>(q, jobs, tiles, ntiles, K) : dqgemm_t<kQ20, false, MODE, KC, GRF>(q, jobs, tiles, ntiles, K);
        if (type == kPtq1) return planar ? dqgemm_t<kPtq1, true, MODE, KC, GRF>(q, jobs, tiles, ntiles, K) : dqgemm_t<kPtq1, false, MODE, KC, GRF>(q, jobs, tiles, ntiles, K);
        return planar ? dqgemm_t<kQ3K, true, MODE, KC, GRF>(q, jobs, tiles, ntiles, K) : dqgemm_t<kQ3K, false, MODE, KC, GRF>(q, jobs, tiles, ntiles, K);
    };
    go().wait();
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        sycl::event e = go();
        e.wait();
        const double ns = (double) (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                    e.get_profiling_info<sycl::info::event_profiling::command_start>());
        best = std::min(best, ns);
    }
    return best / 1e6;
}

int main(int argc, char** argv) {
    const int NE = argc > 1 ? std::atoi(argv[1]) : 128;     // experts
    const int M = argc > 2 ? std::atoi(argv[2]) : 69;       // tokens an expert
    const int H = 5120, F = 2304;
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const int Mp = (M + 7) & ~7;
    std::mt19937 rng(1);
    for (int type : {kQ3K, kQ20}) {
        const int qk = type == kQ3K ? 256 : type == kQ20 ? 64 : 128, bb = type == kQ3K ? 110 : type == kQ20 ? 18 : 28;
        for (int phase = 0; phase < 2; ++phase) {
            const int K = phase == 0 ? H : F, N = phase == 0 ? F : H, nj = phase == 0 ? 2 * NE : NE;
            const size_t mat = (size_t) N * (K / qk) * bb;
            std::vector<uint8_t> hw(mat);
            for (auto& x : hw) x = (uint8_t) rng();
            // fp16 scales: keep them finite (d bytes are random otherwise)
            uint8_t* W = sycl::malloc_device<uint8_t>(mat * nj, q);
            for (int j = 0; j < nj; ++j) q.memcpy(W + mat * j, hw.data(), mat);
            sycl::half* A = sycl::malloc_device<sycl::half>((size_t) nj * Mp * K, q);
            float* C = sycl::malloc_device<float>((size_t) nj * Mp * N, q);
            {
                std::vector<sycl::half> ha((size_t) Mp * K);
                for (auto& x : ha) x = (sycl::half) ((float) (rng() % 2001) / 1000.0f - 1.0f);
                for (int j = 0; j < nj; ++j) q.memcpy(A + (size_t) j * Mp * K, ha.data(), ha.size() * 2);
                q.wait();
            }
            std::vector<DqJob> jb(nj);
            std::vector<DqTile> tl;
            for (int j = 0; j < nj; ++j) {
                jb[j] = {W + mat * j, A + (size_t) j * Mp * K, C + (size_t) j * Mp * N, (int64_t) N * (K / qk), M, K, N, (K / qk) * bb};
                for (int m0 = 0; m0 < M; m0 += 128)
                    for (int n0 = 0; n0 < N; n0 += 128) tl.push_back({j, m0, n0, 0});
            }
            DqJob* dj = sycl::malloc_device<DqJob>(nj, q);
            DqTile* dt = sycl::malloc_device<DqTile>(tl.size(), q);
            q.memcpy(dj, jb.data(), nj * sizeof(DqJob));
            q.memcpy(dt, tl.data(), tl.size() * sizeof(DqTile)).wait();
            const double flop = 2.0 * nj * (double) M * K * N;
            for (int planar = 1; planar < 2; ++planar) {
                {   // the two forms agree
                    std::vector<float> c0((size_t) Mp * N), c1((size_t) Mp * N);
                    run<0>(q, type, planar, dj, dt, (int) tl.size(), K, 1);
                    q.memcpy(c0.data(), C, c0.size() * 4).wait();
                    q.memset(C, 0, c0.size() * 4).wait();
                    run<0, 64, 256, 5>(q, type, planar, dj, dt, (int) tl.size(), K, 1);
                    q.memcpy(c1.data(), C, c1.size() * 4).wait();
                    size_t bad = 0, fin = 0;
                    for (size_t i = 0; i < (size_t) M * N; ++i) {
                        if (std::isfinite(c0[i])) ++fin;
                        if (!(c0[i] == c1[i]) && !(std::isnan(c0[i]) && std::isnan(c1[i]))) ++bad;
                    }
                    std::printf("  check type %d planar %d phase %d: %zu of %zu differ (%zu finite)\n", type, planar, phase, bad, (size_t) M * N, fin);
                }
                double t[6];
                t[0] = run<0>(q, type, planar, dj, dt, (int) tl.size(), K, 5);
                t[1] = run<0, 64, 256, 3>(q, type, planar, dj, dt, (int) tl.size(), K, 5);
                t[2] = run<0, 64, 256, 5>(q, type, planar, dj, dt, (int) tl.size(), K, 5);
                std::printf("type %3d %s %s: SLM-B %.2f ms | FAST %.2f ms (%.1f TF) | HALF %.2f ms (%.1f TF)\n",
                            type, planar ? "planar" : "file  ", phase == 0 ? "gate|up" : "down   ", t[0],
                            t[1], flop / t[1] / 1e9, t[2], flop / t[2] / 1e9);
            }
            sycl::free(W, q); sycl::free(A, q); sycl::free(C, q); sycl::free(dj, q); sycl::free(dt, q);
        }
    }
    return 0;
}
