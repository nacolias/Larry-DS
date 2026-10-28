// bench_dqgemm_v2.cpp - larry (#3): xe_dqgemm (v1) against xe_dqgemm_v2 (bench/dqb/xe_dqgemm_v2.cpp: weights decoded into
// local memory VNNI-packed, a sub-group 32 tokens x 64 rows) on the same jobs / tiles - outputs compared bit for bit, times.
// (bench_dqgemm.cpp's header follows)
// bench_dqgemm.cpp - larry: xe_dqgemm (the prompt path's expert GEMMs, weights decoded inside the XMX GEMM) in
// isolation at a tier call's shapes: NE experts x M tokens each, gate|up (K = 5120 -> 2 x 2304) and down (K = 2304 ->
// 5120), each quant type in its planar layout (random codes, unit scales).  Prints ms and TFLOPS per phase.  Times are
// GPU event profiling (median of 7 after a warm-up).  Run with the service stopped: bench_dqgemm [NE] [M] [device]
#include "xe_kernels.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::v41;
namespace strata::v41::xe {
sycl::event xe_dqgemm_v2(sycl::queue& q, int type, bool planar, const DqJob* jobs, const DqTile* tiles, int ntiles, int K);
}

static double ev_ms(const sycl::event& e) {
    return (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
            e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
}

int main(int argc, char** argv) {
    const int NE = argc > 1 ? std::atoi(argv[1]) : 96, M = argc > 2 ? std::atoi(argv[2]) : 200, dev = argc > 3 ? std::atoi(argv[3]) : 1;
    const int H = 5120, F = 2304;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::queue q(devs[(size_t) dev], sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    std::printf("%s: %d experts x %d tokens, H %d F %d\n", q.get_device().get_info<sycl::info::device::name>().c_str(), NE, M, H, F);
    const int Mp = (M + 7) & ~7;
    const size_t R = (size_t) NE * Mp;
    sycl::half* X = sycl::malloc_device<sycl::half>(R * H, q);
    float* G = sycl::malloc_device<float>(R * 2 * F, q);
    sycl::half* Hh = sycl::malloc_device<sycl::half>(R * F, q);
    float* Y = sycl::malloc_device<float>(R * H, q);
    float* G2 = sycl::malloc_device<float>(R * 2 * F, q);
    float* Y2 = sycl::malloc_device<float>(R * H, q);
    {
        std::vector<sycl::half> hx(R * (size_t) H);
        std::mt19937 rng(5);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto& v : hx) v = (sycl::half) u(rng);
        q.memcpy(X, hx.data(), hx.size() * 2);
        q.memcpy(Hh, hx.data(), R * (size_t) F * 2).wait();
    }
    struct T { int type; const char* name; int qk; int plane_bytes; int scale_off_plane; int file_block; };
    // planar layouts (xe_dqgemm.cpp Dec<TYPE, true>): bytes a block of qk values, the scale plane last
    const T types[3] = {{xe::kQ3K, "Q3_K", 256, 64 + 32 + 12 + 2, 0, 110}, {xe::kQ20, "Q2_0", 64, 16 + 2, 0, 18},
                        {xe::kPtq1, "PTQ1_0", 128, 16 + 8, 0, 28}};
    const bool planar_run = !(argc > 4 && std::atoi(argv[4]) == 0);   // larry: 0 = the file layout (prefetched experts)
    std::printf("layout: %s\n", planar_run ? "planar (resident experts)" : "file (prefetched / streamed experts)");
    for (const T& t : types) {
        const size_t nb_gu = (size_t) F * (H / t.qk), nb_d = (size_t) H * (F / t.qk);
        const size_t blk = planar_run ? (size_t) t.plane_bytes : (size_t) t.file_block;   // the file layout's blocks (PTQ1: 28)
        const size_t bytes_gu = nb_gu * blk, bytes_d = nb_d * blk;
        // one copy of the weights per expert (distinct memory: no cross-expert cache reuse)
        std::vector<uint8_t*> wg(NE), wu(NE), wd(NE);
        std::vector<uint8_t> hb(std::max(bytes_gu, bytes_d));
        std::mt19937 rng(7);
        for (auto& b : hb) b = (uint8_t) rng();
        auto fix_scales = [&](size_t nb, size_t bytes) {   // the scale planes: fp16 1.0 / small codes
            std::vector<uint8_t> v(hb.begin(), hb.begin() + (long) bytes);
            if (!planar_run) return v;   // the file layout: random scales (timing only)
            if (t.type == xe::kQ20) for (size_t i = 0; i < nb; ++i) { v[16 * nb + 2 * i] = 0x00; v[16 * nb + 2 * i + 1] = 0x3C; }
            if (t.type == xe::kQ3K) for (size_t i = 0; i < nb; ++i) { v[108 * nb + 2 * i] = 0x00; v[108 * nb + 2 * i + 1] = 0x2C; }
            return v;
        };
        const std::vector<uint8_t> hgu = fix_scales(nb_gu, bytes_gu), hd = fix_scales(nb_d, bytes_d);
        for (int e = 0; e < NE; ++e) {
            wg[e] = sycl::malloc_device<uint8_t>(bytes_gu, q); wu[e] = sycl::malloc_device<uint8_t>(bytes_gu, q);
            wd[e] = sycl::malloc_device<uint8_t>(bytes_d, q);
            q.memcpy(wg[e], hgu.data(), bytes_gu); q.memcpy(wu[e], hgu.data(), bytes_gu); q.memcpy(wd[e], hd.data(), bytes_d);
        }
        q.wait();
        // jobs + tiles as the tier builds them
        std::vector<xe::DqJob> jgu, jd;
        std::vector<xe::DqTile> tgu, td;
        for (int e = 0; e < NE; ++e) {
            for (int u = 0; u < 2; ++u) {
                xe::DqJob jb;
                jb.W = u ? wu[e] : wg[e];
                jb.A = X + (size_t) e * Mp * H;
                jb.C = G + (size_t) e * Mp * 2 * F + (size_t) u * F;
                jb.NB = (int64_t) nb_gu; jb.m = M; jb.lda = H; jb.ldc = 2 * F; jb.row_bytes = planar_run ? 0 : (H / t.qk) * t.file_block;
                for (int m0 = 0; m0 < M; m0 += 128) for (int n0 = 0; n0 < F; n0 += 128) tgu.push_back({(int32_t) jgu.size(), m0, n0, 0});
                jgu.push_back(jb);
            }
            xe::DqJob jb;
            jb.W = wd[e]; jb.A = Hh + (size_t) e * Mp * F; jb.C = Y + (size_t) e * Mp * H;
            jb.NB = (int64_t) nb_d; jb.m = M; jb.lda = F; jb.ldc = H; jb.row_bytes = planar_run ? 0 : (F / t.qk) * t.file_block;
            for (int m0 = 0; m0 < M; m0 += 128) for (int n0 = 0; n0 < H; n0 += 128) td.push_back({(int32_t) jd.size(), m0, n0, 0});
            jd.push_back(jb);
        }
        xe::DqJob* djgu = sycl::malloc_device<xe::DqJob>(jgu.size(), q);
        xe::DqJob* djd = sycl::malloc_device<xe::DqJob>(jd.size(), q);
        xe::DqTile* dtgu = sycl::malloc_device<xe::DqTile>(tgu.size(), q);
        xe::DqTile* dtd = sycl::malloc_device<xe::DqTile>(td.size(), q);
        q.memcpy(djgu, jgu.data(), jgu.size() * sizeof(xe::DqJob)); q.memcpy(djd, jd.data(), jd.size() * sizeof(xe::DqJob));
        q.memcpy(dtgu, tgu.data(), tgu.size() * sizeof(xe::DqTile)); q.memcpy(dtd, td.data(), td.size() * sizeof(xe::DqTile)).wait();
        std::vector<double> a, b, a2, b2;
        for (int r = 0; r < 8; ++r) {
            sycl::event e1 = xe::xe_dqgemm(q, t.type, planar_run, djgu, dtgu, (int) tgu.size(), H);
            sycl::event e2 = xe::xe_dqgemm(q, t.type, planar_run, djd, dtd, (int) td.size(), F);
            e2.wait();
            if (r) { a.push_back(ev_ms(e1)); b.push_back(ev_ms(e2)); }
        }
        // v2 into G2 / Y2: the same jobs with C swapped
        std::vector<xe::DqJob> jgu2 = jgu, jd2 = jd;
        for (auto& j : jgu2) j.C = G2 + (j.C - G);
        for (auto& j : jd2) j.C = Y2 + (j.C - Y);
        xe::DqJob* djgu2 = sycl::malloc_device<xe::DqJob>(jgu2.size(), q);
        xe::DqJob* djd2 = sycl::malloc_device<xe::DqJob>(jd2.size(), q);
        q.memcpy(djgu2, jgu2.data(), jgu2.size() * sizeof(xe::DqJob)); q.memcpy(djd2, jd2.data(), jd2.size() * sizeof(xe::DqJob)).wait();
        for (int r = 0; r < 8; ++r) {
            sycl::event e1 = xe::xe_dqgemm_v2(q, t.type, planar_run, djgu2, dtgu, (int) tgu.size(), H);
            sycl::event e2 = xe::xe_dqgemm_v2(q, t.type, planar_run, djd2, dtd, (int) td.size(), F);
            e2.wait();
            if (r) { a2.push_back(ev_ms(e1)); b2.push_back(ev_ms(e2)); }
        }
        // compare: every real token row (not the padding)
        size_t bad = 0, cnt = 0;
        {
            std::vector<float> h1(R * 2 * (size_t) F), h2(R * 2 * (size_t) F), y1(R * (size_t) H), y2(R * (size_t) H);
            q.memcpy(h1.data(), G, h1.size() * 4); q.memcpy(h2.data(), G2, h2.size() * 4);
            q.memcpy(y1.data(), Y, y1.size() * 4); q.memcpy(y2.data(), Y2, y2.size() * 4).wait();
            for (int e = 0; e < NE; ++e)
                for (int i = 0; i < M; ++i) {
                    const size_t r0 = (size_t) e * Mp + i;
                    for (int c = 0; c < 2 * F; ++c) { ++cnt; if (std::memcmp(&h1[r0 * 2 * F + c], &h2[r0 * 2 * F + c], 4)) ++bad; }
                    for (int c = 0; c < H; ++c) { ++cnt; if (std::memcmp(&y1[r0 * H + c], &y2[r0 * H + c], 4)) ++bad; }
                }
        }
        std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); std::sort(a2.begin(), a2.end()); std::sort(b2.begin(), b2.end());
        const double fgu = 2.0 * NE * M * 2.0 * F * H, fd = 2.0 * NE * M * (double) H * F;
        std::printf("%-7s v1 gate|up %6.2f ms (%5.1f TF) down %6.2f ms (%5.1f TF) | v2 gate|up %6.2f ms (%5.1f TF) down %6.2f ms (%5.1f TF) | %zu of %zu values differ\n",
                    t.name, a[3], fgu / a[3] / 1e9, b[3], fd / b[3] / 1e9, a2[3], fgu / a2[3] / 1e9, b2[3], fd / b2[3] / 1e9, bad, cnt);
        std::fflush(stdout);
        sycl::free(djgu2, q); sycl::free(djd2, q);
        for (int e = 0; e < NE; ++e) { sycl::free(wg[e], q); sycl::free(wu[e], q); sycl::free(wd[e], q); }
        sycl::free(djgu, q); sycl::free(djd, q); sycl::free(dtgu, q); sycl::free(dtd, q);
    }
    return 0;
}
