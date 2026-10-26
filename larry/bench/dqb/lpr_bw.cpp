// lpr_bw.cpp - larry: streaming-read bandwidth of the access pattern an XMX mat-vec's B decode wants (each lane of a
// sub-group reads ITS OWN weight row: 16 rows a load message) vs the coalesced pattern (a message covers one row's
// consecutive bytes), at DeepSeek-V4.1's matrix sizes (27.5 MB: q_b / wo_a / wo_b as Q6_K; 6.9 MB: shexp down).
// Buffers rotate over 8 copies so every pass reads DRAM, not the 18 MB L2.
//   build: icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xs "-device bmg-g31" lpr_bw.cpp -o lpr_bw
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

using u4 = sycl::vec<uint32_t, 4>;
constexpr int SG = 16;

// lane-per-row: sub-group s owns rows [16 s, 16 s + 16); lane l reads row 16 s + l, V x 16-byte chunks per step
template <int V>
sycl::event lpr(sycl::queue & q, const uint8_t * buf, size_t rows, size_t row_bytes, uint32_t * out, int wg) {
    const size_t nsg = rows / SG;
    return q.parallel_for(sycl::nd_range<1>(nsg * SG, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const size_t gid = it.get_global_id(0);
        const size_t row = gid;   // = 16 * sub-group + lane
        const u4 * p = reinterpret_cast<const u4 *>(buf + row * row_bytes);
        uint32_t acc = 0;
        for (size_t c = 0; c < row_bytes / 16; c += V) {
            u4 v[V];
#pragma unroll
            for (int i = 0; i < V; ++i) v[i] = p[c + i];
#pragma unroll
            for (int i = 0; i < V; ++i) acc ^= v[i].x() + v[i].y() * 3u + v[i].z() * 5u + v[i].w() * 7u;
        }
        if (acc == 0x12345678u) out[0] = acc;
    });
}

// lane-per-row with the row split over KS sub-groups (small matrices need the parallelism)
template <int V>
sycl::event lprk(sycl::queue & q, const uint8_t * buf, size_t rows, size_t row_bytes, uint32_t * out, int wg, int KS) {
    const size_t nsg = rows / SG;
    const size_t chunks = row_bytes / 16, per = (chunks / KS + V - 1) / V * V;
    return q.parallel_for(sycl::nd_range<1>(nsg * KS * SG, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const size_t gid = it.get_global_id(0);
        const size_t sgi = gid / SG, lane = gid % SG;
        const size_t rg = sgi / KS, ks = sgi % KS;
        const size_t row = rg * SG + lane;
        const u4 * p = reinterpret_cast<const u4 *>(buf + row * row_bytes);
        uint32_t acc = 0;
        const size_t c1 = sycl::min(chunks, (ks + 1) * per);
        for (size_t c = ks * per; c + V <= c1; c += V) {
            u4 v[V];
#pragma unroll
            for (int i = 0; i < V; ++i) v[i] = p[c + i];
#pragma unroll
            for (int i = 0; i < V; ++i) acc ^= v[i].x() + v[i].y() * 3u + v[i].z() * 5u + v[i].w() * 7u;
        }
        if (acc == 0x12345678u) out[0] = acc;
    });
}

// coalesced: sub-group s owns rows [16 s, 16 s + 16) too, but reads them one row at a time, lane l the 16-byte chunk l
template <int V>
sycl::event coal(sycl::queue & q, const uint8_t * buf, size_t rows, size_t row_bytes, uint32_t * out, int wg) {
    const size_t nsg = rows / SG;
    return q.parallel_for(sycl::nd_range<1>(nsg * SG, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const size_t gid = it.get_global_id(0);
        const size_t s = gid / SG, lane = gid % SG;
        uint32_t acc = 0;
        for (size_t r = 16 * s; r < 16 * s + 16; ++r) {
            const u4 * p = reinterpret_cast<const u4 *>(buf + r * row_bytes);
            for (size_t c = lane * V; c < row_bytes / 16; c += SG * V) {
                u4 v[V];
#pragma unroll
                for (int i = 0; i < V; ++i) v[i] = p[c + i];
#pragma unroll
                for (int i = 0; i < V; ++i) acc ^= v[i].x() + v[i].y() * 3u + v[i].z() * 5u + v[i].w() * 7u;
            }
        }
        if (acc == 0x12345678u) out[0] = acc;
    });
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int NB = 8;
    struct Shape { const char * name; size_t rows, row_bytes; };
    // Q6_K bytes: K * 210 / 256 a row; planar planes are contiguous per row block - model as K * 0.8203
    const Shape shapes[] = {
        {"q_b   32768 x 1024 (896 B rows)", 32768, 896},
        {"wo_b   4096 x 8192 (6720 B rows)", 4096, 6720},
        {"wo_a   8192 x 4096 (3392 B rows)", 8192, 3392},
        {"down   4096 x 2048 (1728 B rows)", 4096, 1728},
        {"q_a    1024 x 4096 (3392 B rows)", 1024, 3392},
    };
    uint32_t * out = sycl::malloc_device<uint32_t>(1, q);
    for (const Shape & sh : shapes) {
        const size_t bytes = sh.rows * sh.row_bytes;
        std::vector<uint8_t *> bufs;
        for (int i = 0; i < NB; ++i) {
            uint8_t * b = sycl::malloc_device<uint8_t>(bytes, q);
            q.parallel_for(sycl::range<1>(bytes / 4), [=](sycl::id<1> i) { reinterpret_cast<uint32_t *>(b)[i] = (uint32_t) i * 2654435761u; });
            bufs.push_back(b);
        }
        q.wait();
        auto bench = [&](const char * tag, auto fn) {
            for (int w = 0; w < 3; ++w) fn(bufs[w % NB]);
            q.wait();
            const int reps = 64;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) fn(bufs[r % NB]);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("  %-34s %-18s %8.1f us  %6.0f GB/s\n", sh.name, tag, us, bytes / us / 1e3);
        };
        bench("lpr V=1 wg128", [&](const uint8_t * b) { lpr<1>(q, b, sh.rows, sh.row_bytes, out, 128); });
        bench("lpr V=2 wg128", [&](const uint8_t * b) { lpr<2>(q, b, sh.rows, sh.row_bytes, out, 128); });
        bench("lpr V=4 wg128", [&](const uint8_t * b) { lpr<4>(q, b, sh.rows, sh.row_bytes, out, 128); });
        for (int ks : {2, 4, 8, 16})
            if (sh.row_bytes / 16 / ks >= 4) {
                char t[32]; std::snprintf(t, sizeof t, "lprk V=2 KS=%d", ks);
                bench(t, [&](const uint8_t * b) { lprk<2>(q, b, sh.rows, sh.row_bytes, out, 128, ks); });
            }
        bench("coal V=1 wg128", [&](const uint8_t * b) { coal<1>(q, b, sh.rows, sh.row_bytes, out, 128); });
        bench("coal V=2 wg128", [&](const uint8_t * b) { coal<2>(q, b, sh.rows, sh.row_bytes, out, 128); });
        for (uint8_t * b : bufs) sycl::free(b, q);
    }
    sycl::free(out, q);
    return 0;
}
