// tpmv_bench.cpp - engine1: the TP core's per-card Q6_K mat-vecs (4 cards: q_a+kv, q_b, wo_a, wo_b, shared expert
// gate|up and down) at 1-6 columns, the variants side by side: fastmv's dispatch (what the TP core runs), the
// row-blocked dp4a kernel with R = 2 / 4 rows (normal / 256-register mode), the XMX kernel (fp16 DPAS on the int8
// activations, KS splits), and a plain streaming read of the same bytes (the floor).  Weights rotate through >= 256 MB of
// copies (VRAM, not L2); timing = SYCL graph replays of 64 launches, as the TP core runs its passes.
//
//   tpmv_bench [replays]        (one card; the service must be down)
#include "fastmv.cpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;

struct BQ6 { uint8_t ql[128]; uint8_t qh[64]; int8_t sc[16]; uint16_t d; };
static_assert(sizeof(BQ6) == 210, "q6_K");

static float h2f_(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t f2h_(float f) { return sycl::bit_cast<uint16_t>((sycl::half) f); }

static void deq_q6(const BQ6 & b, float * y) {
    const float d = h2f_(b.d);
    const uint8_t * ql = b.ql; const uint8_t * qh = b.qh; const int8_t * sc = b.sc;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            y[l + 0] = d * sc[is + 0] * (int8_t) (((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            y[l + 32] = d * sc[is + 2] * (int8_t) (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            y[l + 64] = d * sc[is + 4] * (int8_t) (((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            y[l + 96] = d * sc[is + 6] * (int8_t) (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

// the floor: every byte of the weight read once (uint4 loads, one partial a work-group)
static void stream_read(sycl::queue & q, const uint8_t * w, size_t bytes, float * out) {
    const size_t n4 = bytes / 16;
    const size_t wg = 256, per = 8;
    const size_t groups = (n4 + wg * per - 1) / (wg * per);
    q.parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
        const sycl::uint4 * p = reinterpret_cast<const sycl::uint4 *>(w);
        uint32_t acc = 0;
        const size_t base = it.get_group(0) * wg * per + it.get_local_id(0);
#pragma unroll
        for (size_t k = 0; k < per; ++k) {
            const size_t i = base + k * wg;
            if (i < n4) { const sycl::uint4 v = p[i]; acc ^= v.x() ^ v.y() ^ v.z() ^ v.w(); }
        }
        if (acc == 0x12345678u) out[0] = 1.0f;   // keeps the loads
    });
}

int main(int argc, char ** argv) {
    const int replays = argc > 1 ? std::atoi(argv[1]) : 20;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> ub(0, 255);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    struct S { const char * name; int64_t K, N, G; };
    const S shapes[] = {
        {"q_a+kv 5120x1792", 5120, 1792, 1}, {"q_b 1280x8192", 1280, 8192, 1}, {"wo_a 4096x1024 x2", 4096, 1024, 2},
        {"wo_b 2048x5120", 2048, 5120, 1}, {"sh gu 5120x1024", 5120, 1024, 1}, {"sh dn 512x5120", 512, 5120, 1},
        {"sh gu 5120x1536", 5120, 1536, 1}, {"sh dn 768x5120", 768, 5120, 1}};
    const int M = 64;   // launches a graph
    float * sink = sycl::malloc_device<float>(16, q);
    for (const S & s : shapes) {
        const int64_t rows = s.N * s.G, bpr = s.K / 256, nb = rows * bpr, pb = s.N * bpr;
        std::vector<BQ6> blk((size_t) nb);
        for (auto & b : blk) {
            for (auto & v : b.ql) v = (uint8_t) ub(rng);
            for (auto & v : b.qh) v = (uint8_t) ub(rng);
            for (auto & v : b.sc) v = (int8_t) (ub(rng) % 64 - 32);
            b.d = f2h_(0.001f + 0.002f * (ub(rng) / 255.f));
        }
        const size_t wbytes = (size_t) nb * 210;
        std::vector<uint8_t> wdev(wbytes);
        for (int64_t i = 0; i < nb; ++i) {   // ggml-sycl's reorder (per batch): ql | qh | scales | d planes
            const int64_t g = i / pb, j = i % pb;
            uint8_t * P = &wdev[(size_t) g * pb * 210];
            std::memcpy(P + (size_t) j * 128, blk[(size_t) i].ql, 128);
            std::memcpy(P + (size_t) pb * 128 + (size_t) j * 64, blk[(size_t) i].qh, 64);
            std::memcpy(P + (size_t) pb * 192 + (size_t) j * 16, blk[(size_t) i].sc, 16);
            std::memcpy(P + (size_t) pb * 208 + (size_t) j * 2, &blk[(size_t) i].d, 2);
        }
        const int64_t rr[3] = {0, rows / 2, rows - 1};
        std::vector<float> wf((size_t) 3 * s.K);
        for (int k = 0; k < 3; ++k)
            for (int64_t b = 0; b < bpr; ++b) deq_q6(blk[(size_t) (rr[k] * bpr + b)], &wf[(size_t) k * s.K + b * 256]);
        const int ncp = (int) std::min<size_t>(64, std::max<size_t>(2, ((size_t) 256 << 20) / wbytes + 1));
        std::vector<uint8_t *> dws((size_t) ncp);
        for (int c = 0; c < ncp; ++c) { dws[(size_t) c] = sycl::malloc_device<uint8_t>(wbytes, q); q.memcpy(dws[(size_t) c], wdev.data(), wbytes); }
        q.wait();
        std::printf("\n%s: %.2f MB (%d copies)\n", s.name, wbytes / 1e6, ncp);
        // the floor
        {
            sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), q.get_device());
            g.begin_recording(q);
            for (int i = 0; i < M; ++i) stream_read(q, dws[(size_t) (i % ncp)], wbytes, sink);
            g.end_recording(q);
            auto ge = g.finalize();
            q.ext_oneapi_graph(ge).wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < replays; ++r) q.ext_oneapi_graph(ge);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (replays * M);
            std::printf("  %-26s %7.1f us %6.0f GB/s\n", "stream read (floor)", us, wbytes / us / 1e3);
        }
        {
            sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), q.get_device());
            g.begin_recording(q);
            for (int i = 0; i < M; ++i) q.parallel_for(sycl::nd_range<1>(64 * 16, 64), [=](sycl::nd_item<1> it) { if (it.get_global_id(0) == 0) sink[1] += 1.0f; });
            g.end_recording(q);
            auto ge = g.finalize();
            q.ext_oneapi_graph(ge).wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < replays; ++r) q.ext_oneapi_graph(ge);
            q.wait();
            std::printf("  %-26s %7.1f us\n", "the small kernel alone", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (replays * M));
        }
        for (int nc : {1, 5}) {
            std::vector<float> x((size_t) s.G * nc * s.K);
            for (auto & v : x) v = uf(rng);
            float * dx = sycl::malloc_device<float>(x.size(), q);
            float * dy = sycl::malloc_device<float>((size_t) s.G * nc * s.N, q);
            int8_t * dxq = sycl::malloc_device<int8_t>((size_t) s.G * nc * s.K, q);
            float * dxd = sycl::malloc_device<float>((size_t) s.G * nc * s.K / 32, q);
            sycl::half * dxT = sycl::malloc_device<sycl::half>((size_t) s.G * s.K * 8, q);
            float * dxdT = sycl::malloc_device<float>((size_t) s.G * (s.K / 32) * 8, q);
            q.memcpy(dx, x.data(), x.size() * 4).wait();
            fastmv::quant_x(q, dx, s.K, (int64_t) nc * s.K, nc, s.G, s.K, dxq, dxd);
            fastmv::quant_xT(q, dx, s.K, (int64_t) nc * s.K, nc, s.G, s.K, dxT, dxdT);
            q.wait();
            fastmv::Args a;
            a.w = dws[0]; a.K = s.K; a.N = s.N; a.G = s.G; a.g_div = 1; a.n_blocks_total = pb;
            a.w_batch_bytes = (size_t) pb * 210;
            a.x = dx; a.x_col_stride = s.K; a.x_batch_stride = (int64_t) nc * s.K;
            a.y = dy; a.y_col_stride = s.N; a.y_batch_stride = (int64_t) nc * s.N; a.ncols = nc;
            a.xq = dxq; a.xd = dxd;
            struct V { std::string name; std::function<void(const fastmv::Args &)> f; };
            std::vector<V> vs;
            vs.push_back({"dispatch (TP core today)", [&](const fastmv::Args & b) { fastmv::launch(q, fastmv::WType::Q6K_R8, b); }});
            auto add_r8m = [&](int R, bool big) {
                std::string nm = "dp4a rows " + std::to_string(R) + (big ? " 256-reg" : "");
                vs.push_back({nm, [&q, R, big, nc](const fastmv::Args & b) {
                    switch (nc * 100 + R * 10 + (big ? 1 : 0)) {
#define C(NT, R_, B_) case NT * 100 + R_ * 10 + B_: fastmv::pick_r8m<NT, R_, (B_ != 0)>(q, b); break;
                        C(1, 2, 0) C(1, 2, 1) C(1, 4, 0) C(1, 4, 1) C(2, 2, 0) C(2, 2, 1) C(2, 4, 0) C(2, 4, 1)
                        C(3, 2, 0) C(3, 2, 1) C(3, 4, 0) C(3, 4, 1) C(4, 2, 0) C(4, 2, 1) C(4, 4, 0) C(4, 4, 1)
                        C(5, 2, 0) C(5, 2, 1) C(5, 4, 0) C(5, 4, 1) C(6, 2, 0) C(6, 2, 1) C(6, 4, 0) C(6, 4, 1)
#undef C
                    }
                }});
            };
            add_r8m(2, false); add_r8m(2, true); add_r8m(4, false); add_r8m(4, true);
            for (int ks : {1, 2, 4, 8}) {
                if (ks > bpr) continue;
                vs.push_back({"xmx KS " + std::to_string(ks), [&q, ks, dxT, dxdT](const fastmv::Args & b) {
                    switch (ks) {
                        case 1: fastmv::xmxd::run<1>(q, b, dxT, dxdT); break;
                        case 2: fastmv::xmxd::run<2>(q, b, dxT, dxdT); break;
                        case 4: fastmv::xmxd::run<4>(q, b, dxT, dxdT); break;
                        default: fastmv::xmxd::run<8>(q, b, dxT, dxdT); break;
                    }
                }});
            }
            std::printf("  ncols %d:\n", nc);
            for (const V & v : vs) {
                // check rows 0 / mid / last of every column against the dequantised reference
                q.memset(dy, 0, (size_t) s.G * nc * s.N * 4).wait();
                a.w = dws[0];
                v.f(a);
                q.wait();
                std::vector<float> y((size_t) s.G * nc * s.N);
                q.memcpy(y.data(), dy, y.size() * 4).wait();
                double worst = 0;
                for (int j = 0; j < 3; ++j) {
                    const int64_t g = rr[j] / s.N, r = rr[j] % s.N;
                    for (int c = 0; c < nc; ++c) {
                        double ref = 0, mag = 0;
                        for (int64_t k = 0; k < s.K; ++k) {
                            const double p = (double) wf[(size_t) j * s.K + k] * x[(size_t) (g * nc + c) * s.K + k];
                            ref += p; mag += std::fabs(p);
                        }
                        worst = std::max(worst, std::fabs(y[(size_t) (g * nc + c) * s.N + r] - ref) / (mag + 1e-30));
                    }
                }
                double us2[2];
                for (int il = 0; il < 2; ++il) {
                    sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), q.get_device());
                    g.begin_recording(q);
                    for (int i = 0; i < M; ++i) {
                        fastmv::Args b = a; b.w = dws[(size_t) (i % ncp)]; v.f(b);
                        if (il) q.parallel_for(sycl::nd_range<1>(64 * 16, 64), [=](sycl::nd_item<1> it) { if (it.get_global_id(0) == 0) sink[1] += 1.0f; });
                    }
                    g.end_recording(q);
                    auto ge = g.finalize();
                    q.ext_oneapi_graph(ge).wait();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int r = 0; r < replays; ++r) q.ext_oneapi_graph(ge);
                    q.wait();
                    us2[il] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (replays * M);
                }
                const double us = us2[0];
                std::printf("    %-26s %7.1f us %6.0f GB/s  err %.1e %s | with a small kernel between: %7.1f us (+ that kernel)\n", v.name.c_str(), us,
                            wbytes / us / 1e3, worst, worst < 3e-2 ? "ok" : "FAIL", us2[1]);
            }
            // the XMX path's extra kernel (the TP core would have the producers write it)
            {
                sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), q.get_device());
                g.begin_recording(q);
                for (int i = 0; i < M; ++i) fastmv::quant_xT(q, dx, s.K, (int64_t) nc * s.K, nc, s.G, s.K, dxT, dxdT);
                g.end_recording(q);
                auto ge = g.finalize();
                q.ext_oneapi_graph(ge).wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < replays; ++r) q.ext_oneapi_graph(ge);
                q.wait();
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (replays * M);
                std::printf("    %-26s %7.1f us\n", "(quant_xT alone)", us);
            }
            sycl::free(dx, q); sycl::free(dy, q); sycl::free(dxq, q); sycl::free(dxd, q); sycl::free(dxT, q); sycl::free(dxdT, q);
        }
        for (auto * p : dws) sycl::free(p, q);
    }
    sycl::free(sink, q);
    return 0;
}
