// tools/v41/engine/xe/test_ds41_xe.cpp - the Xe expert kernels on REAL DeepSeek-V4.1 GSQ experts against a dequantised
// F32 reference (ggml's to_float; PTQ1_0 through Strata's ptq1_0_dequant) and against the CPU tier's native kernels.
//
//   test_ds41_xe V41-GSQ.gguf [--layers 0,20,39] [--per-class 2] [--bench 200] [--device 0]
//
// Per (layer, expert): rel = ||y - ref|| / ||ref|| for the decode form (1 token), the verify form (3 tokens in one
// job) and the prompt form (experts_y + gather).  Pass: rel < 1e-4 (F32 activations; only summation order differs).
// The CPU native kernels round the activations to 8 bits, so their rel is reported (~1e-3..1e-2), not gated.
#include "ds41_moe.hpp"
#include "xe_kernels.hpp"

#include "ggml.h"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/ptq1_0.hpp"

#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>

#include <chrono>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::v41;
namespace cpu = strata::kernels::cpu;

namespace {

void dequant(int type, const uint8_t* src, float* dst, int64_t n) {
    if (type == xe::kPtq1) { cpu::ptq1_0_dequant(src, dst, n); return; }
    ggml_get_type_traits((ggml_type) type)->to_float(src, dst, n);
}

/// F32 reference: y = Wd (silu(min(Wg x, L)) * clamp(Wu x, -L, L))
void ref_expert(const cpu::NativeFmt& f, const uint8_t* blob, const float* x, float L, float* y) {
    const int64_t E = f.n_embd, F = f.n_ff;
    std::vector<float> row((size_t) std::max(E, F)), h((size_t) F);
    for (int64_t r = 0; r < F; ++r) {
        double g = 0, u = 0;
        dequant(f.gu_type, blob + (size_t) r * f.gu_row, row.data(), E);
        for (int64_t i = 0; i < E; ++i) g += (double) row[(size_t) i] * x[i];
        dequant(f.gu_type, blob + f.up_off + (size_t) r * f.gu_row, row.data(), E);
        for (int64_t i = 0; i < E; ++i) u += (double) row[(size_t) i] * x[i];
        const double gc = std::min(g, (double) L), uc = std::max(-(double) L, std::min(u, (double) L));
        h[(size_t) r] = (float) (gc / (1.0 + std::exp(-gc)) * uc);
    }
    for (int64_t i = 0; i < E; ++i) {
        double s = 0;
        dequant(f.d_type, blob + f.down_off + (size_t) i * f.d_row, row.data(), F);
        for (int64_t r = 0; r < F; ++r) s += (double) row[(size_t) r] * h[(size_t) r];
        y[i] = (float) s;
    }
}

double rel(const float* a, const float* b, int64_t n) {
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) { num += (a[i] - b[i]) * (double) (a[i] - b[i]); den += (double) b[i] * b[i]; }
    return std::sqrt(num / std::max(den, 1e-30));
}

std::vector<int> csv(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_ds41_xe GGUF [--layers a,b] [--per-class N] [--bench N] [--device D]\n"); return 2; }
    std::string gguf = argv[1];
    std::vector<int> layers = {0, 20, 39};
    int per_class = 2, bench = 200, device = 0;
    bool gemm_bench = false;
    int bench_q8 = 0, bench_q8_gu = -1;
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--layers" && i + 1 < argc) layers = csv(argv[++i]);
        else if (k == "--per-class" && i + 1 < argc) per_class = std::atoi(argv[++i]);
        else if (k == "--bench" && i + 1 < argc) bench = std::atoi(argv[++i]);
        else if (k == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (k == "--gemm-bench") gemm_bench = true;
        else if (k == "--bench-q8" && i + 1 < argc) bench_q8 = std::atoi(argv[++i]);
        else if (k == "--bench-q8-gu" && i + 1 < argc) bench_q8_gu = std::atoi(argv[++i]);
    }
    Ds41FileBlobs src(gguf);
    std::string err;
    if (!src.open(err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom& g = src.geom();
    std::vector<cpu::NativeFmt> fl((size_t) g.n_classes());
    for (int64_t c = 0; c < g.n_classes(); ++c)
        if (!ds41_moe_class_layout(g, c, fl[(size_t) c], err)) { std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1; }
    for (int64_t c = 0; c < g.n_classes(); ++c)
        std::printf("class %lld: gate/up %s (%zu B/row), down %s (%zu B/row), %.2f MiB\n", (long long) c,
                    c < (int64_t) g.cls_gu.size() ? ggml_type_name((ggml_type) std::min(g.cls_gu[(size_t) c], 41)) : "?",
                    fl[(size_t) c].gu_row, c < (int64_t) g.cls_d.size() ? ggml_type_name((ggml_type) std::min(g.cls_d[(size_t) c], 41)) : "?",
                    fl[(size_t) c].d_row, fl[(size_t) c].bytes / 1048576.0);
    std::printf("geometry: n_embd %lld n_ff %lld layers %lld experts %lld top-%lld, swiglu limit %g\n", (long long) g.n_embd,
                (long long) g.n_ff, (long long) g.n_layers, (long long) g.n_experts, (long long) g.top_k, (double) g.swiglu_limit);

    std::vector<sycl::device> devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (device >= (int) devs.size()) { std::fprintf(stderr, "no GPU %d\n", device); return 1; }
    sycl::queue q(devs[(size_t) device], sycl::property::queue::in_order());
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int64_t E = g.n_embd, F = g.n_ff;
    const float L = g.swiglu_limit;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int NT = 3;
    std::vector<float> X((size_t) NT * E);
    for (float& v : X) v = nd(rng);
    cpu::ExpertPool pool(0);
    float* dX = sycl::malloc_device<float>((size_t) 8 * E, q);
    float* dH = sycl::malloc_device<float>((size_t) 64 * 8 * F, q);
    float* dO = sycl::malloc_device<float>((size_t) 8 * E, q);
    float* dY = sycl::malloc_device<float>((size_t) 64 * 8 * E, q);
    xe::XeJob* dj = sycl::malloc_device<xe::XeJob>(64, q);
    int32_t* dS = sycl::malloc_device<int32_t>((size_t) 8 * g.top_k, q);
    float* dW = sycl::malloc_device<float>((size_t) 8 * g.top_k, q);
    q.memcpy(dX, X.data(), X.size() * 4).wait();

    int fails = 0, checked = 0;
    std::vector<int> seen_cls((size_t) g.n_classes(), 0);
    std::vector<std::pair<int, int>> bench_set;
    for (int l : layers) {
        if (l < 0 || l >= g.n_layers) continue;
        std::fill(seen_cls.begin(), seen_cls.end(), 0);
        for (int e = 0; e < g.n_experts; ++e) {
            const int c = g.cls(l, e);
            if (seen_cls[(size_t) c] >= per_class) continue;
            ++seen_cls[(size_t) c];
            const cpu::NativeFmt& f = fl[(size_t) c];
            std::vector<uint8_t> blob(f.bytes);
            src.read(l, e, blob.data());
            uint8_t* db = sycl::malloc_device<uint8_t>(f.bytes, q);
            q.memcpy(db, blob.data(), f.bytes).wait();
            if (bench_set.size() < 6) bench_set.push_back({l, e});
            // reference for each of the NT tokens
            std::vector<float> ref((size_t) NT * E), cpuo((size_t) E), gpu((size_t) NT * E);
            for (int t = 0; t < NT; ++t) ref_expert(f, blob.data(), X.data() + (size_t) t * E, L, ref.data() + (size_t) t * E);
            ds41_moe_cpu_expert(f, E, F, blob.data(), X.data(), cpuo.data(), pool);
            xe::XeJob j;
            j.blob = db; j.up_off = (uint32_t) f.up_off; j.down_off = (uint32_t) f.down_off;
            j.gu_type = (uint16_t) f.gu_type; j.d_type = (uint16_t) f.d_type;
            j.gu_row = (uint16_t) f.gu_row; j.d_row = (uint16_t) f.d_row;
            // decode form: 1 token, weight 1
            j.nt = 1; j.tok[0] = 0; j.w[0] = 1.f;
            q.memcpy(dj, &j, sizeof j).wait();
            xe::xe_experts(q, dj, 1, 1, dX, dH, dO, (int) E, (int) F, L, false).wait();
            q.memcpy(gpu.data(), dO, (size_t) E * 4).wait();
            const double r1 = rel(gpu.data(), ref.data(), E);
            // verify form: NT tokens in one job, token t at output row t, weight 1
            j.nt = NT;
            for (int t = 0; t < NT; ++t) { j.tok[t] = t; j.w[t] = 1.f; }
            q.memcpy(dj, &j, sizeof j).wait();
            xe::xe_experts(q, dj, 1, NT, dX, dH, dO, (int) E, (int) F, L, false).wait();
            q.memcpy(gpu.data(), dO, (size_t) NT * E * 4).wait();
            const double r3 = rel(gpu.data(), ref.data(), (int64_t) NT * E);
            // prompt form: Y + gather (token t's slot k=0 -> job 0 slot t, the other slots -1)
            std::vector<int32_t> S((size_t) NT * g.top_k, -1);
            std::vector<float> W((size_t) NT * g.top_k, 0.f);
            for (int t = 0; t < NT; ++t) { S[(size_t) t * g.top_k] = t; W[(size_t) t * g.top_k] = 1.f; }
            q.memcpy(dS, S.data(), S.size() * 4).wait();
            q.memcpy(dW, W.data(), W.size() * 4).wait();
            xe::xe_experts_y(q, dj, 1, dX, dH, dY, (int) E, (int) F, L);
            xe::xe_gather(q, dY, dS, dW, NT, (int) g.top_k, (int) E, dO, false).wait();
            q.memcpy(gpu.data(), dO, (size_t) NT * E * 4).wait();
            const double ry = rel(gpu.data(), ref.data(), (int64_t) NT * E);
            const double rc = rel(cpuo.data(), ref.data(), E);
            double rq = -1, rx = -1;
            {   // planar verify form: the dp4a kernels and the XMX kernels
                static xe::XeQ8 pXq{}, pHq{};
                static sycl::half *pxt = nullptr, *pht = nullptr;
                if (!pxt) {
                    pXq = {sycl::malloc_device<int8_t>((size_t) 8 * E, q), sycl::malloc_device<float>((size_t) 8 * E / 32, q),
                           sycl::malloc_device<int32_t>((size_t) 8 * E / 16, q)};
                    pHq = {sycl::malloc_device<int8_t>((size_t) 8 * F, q), sycl::malloc_device<float>((size_t) 8 * F / 32, q),
                           sycl::malloc_device<int32_t>((size_t) 8 * F / 16, q)};
                    pxt = sycl::malloc_device<sycl::half>((size_t) 8 * E, q);
                    pht = sycl::malloc_device<sycl::half>((size_t) F * 8, q);
                }
                uint8_t* pb = sycl::malloc_device<uint8_t>(f.bytes, q);
                xe::xe_repack_blob(q, db, pb, f.gu_type, f.d_type, f.up_off, f.down_off, (int) E, (int) F).wait();
                xe::XeJob pj = j;
                pj.blob = pb;
                q.memcpy(dj, &pj, sizeof pj).wait();
                xe::xe_quant_rows(q, dX, NT, (int) E, pXq);
                xe::xe_experts_q8(q, dj, 1, NT, pXq, pHq, dO, (int) E, (int) F, L, false, NT).wait();
                q.memcpy(gpu.data(), dO, (size_t) NT * E * 4).wait();
                rq = rel(gpu.data(), ref.data(), (int64_t) NT * E);
                xe::xe_xt(q, dX, NT, (int) E, pxt);
                xe::xe_experts_x(q, dj, 1, NT, pxt, pht, dO, (int) E, (int) F, L, false).wait();
                q.memcpy(gpu.data(), dO, (size_t) NT * E * 4).wait();
                rx = rel(gpu.data(), ref.data(), (int64_t) NT * E);
                if (rx > 0.05) {   // xt check: xe_xt's [K][8] layout; one-token jobs on tokens 1 and 2
                    std::vector<sycl::half> xth((size_t) 8 * E);
                    q.memcpy(xth.data(), pxt, xth.size() * 2).wait();
                    double mxd = 0;
                    for (int t = 0; t < NT; ++t)
                        for (int64_t k = 0; k < E; ++k) mxd = std::max(mxd, (double) std::fabs((float) xth[(size_t) k * 8 + t] - X[(size_t) t * E + k]));
                    std::printf("    xt check: max |xt - X| %.3e\n", mxd);
                    for (int t1 = 1; t1 < NT; ++t1) {
                        xe::XeJob oj = pj;
                        oj.nt = 1; oj.tok[0] = t1; oj.w[0] = 1.f;
                        q.memcpy(dj, &oj, sizeof oj).wait();
                        q.memset(dO, 0, (size_t) 8 * E * 4).wait();
                        xe::xe_experts_x(q, dj, 1, NT, pxt, pht, dO, (int) E, (int) F, L, false).wait();
                        std::vector<sycl::half> hh((size_t) F * 8);
                        q.memcpy(hh.data(), pht, hh.size() * 2).wait();
                        std::printf("    ht slots (row: s0 s1 s2):");
                        for (int r = 0; r < 4; ++r) std::printf("  %d: %.3f %.3f %.3f", r, (float) hh[r * 8], (float) hh[r * 8 + 1], (float) hh[r * 8 + 2]);
                        std::printf("\n");
                        std::vector<float> g1((size_t) NT * E);
                        q.memcpy(g1.data(), dO, g1.size() * 4).wait();
                        std::printf("    one-token job on token %d: row %d vs ref %.2e, row 0 vs ref row %d %.2e\n", t1, t1,
                                    rel(g1.data() + (size_t) t1 * E, ref.data() + (size_t) t1 * E, E), t1,
                                    rel(g1.data(), ref.data() + (size_t) t1 * E, E));
                    }
                    q.memcpy(dj, &pj, sizeof pj).wait();
                }
                if (rx > 0.05) {   // which tokens are off
                    std::printf("    xmx per token:");
                    for (int t = 0; t < NT; ++t) std::printf(" %.2e", rel(gpu.data() + (size_t) t * E, ref.data() + (size_t) t * E, E));
                    std::printf("\n");
                }
                sycl::free(pb, q);
            }
            const bool ok = r1 < 1e-4 && r3 < 1e-4 && ry < 1e-4;
            fails += !ok;
            ++checked;
            std::printf("L%-2d e%-3d class %d (gu %3d / d %3d): xe decode %.2e verify %.2e prompt %.2e | planar verify dp4a %.2e xmx %.2e | cpu native %.2e  %s\n",
                        l, e, c, f.gu_type, f.d_type, r1, r3, ry, rq, rx, rc, ok ? "ok" : "FAIL");
            sycl::free(db, q);
        }
    }
    std::printf("%d experts checked, %d failed\n", checked, fails);

    // bench: 6 experts of one call, one token (the decode shape), and 8 tokens (verify)
    if (bench > 0 && !bench_set.empty()) {
        std::vector<xe::XeJob> jobs;
        std::vector<uint8_t*> bufs;
        size_t bytes = 0;
        for (auto& le : bench_set) {
            const cpu::NativeFmt& f = fl[(size_t) g.cls(le.first, le.second)];
            std::vector<uint8_t> blob(f.bytes);
            src.read(le.first, le.second, blob.data());
            uint8_t* db = sycl::malloc_device<uint8_t>(f.bytes, q);
            q.memcpy(db, blob.data(), f.bytes).wait();
            bufs.push_back(db);
            bytes += f.bytes;
            xe::XeJob j;
            j.blob = db; j.up_off = (uint32_t) f.up_off; j.down_off = (uint32_t) f.down_off;
            j.gu_type = (uint16_t) f.gu_type; j.d_type = (uint16_t) f.d_type;
            j.gu_row = (uint16_t) f.gu_row; j.d_row = (uint16_t) f.d_row;
            j.nt = 1; j.tok[0] = 0; j.w[0] = 0.1f;
            jobs.push_back(j);
        }
        q.memcpy(dj, jobs.data(), jobs.size() * sizeof(xe::XeJob)).wait();
        for (int w = 0; w < 10; ++w) xe::xe_experts(q, dj, (int) jobs.size(), 1, dX, dH, dO, (int) E, (int) F, L, false);
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < bench; ++it) xe::xe_experts(q, dj, (int) jobs.size(), 1, dX, dH, dO, (int) E, (int) F, L, false);
        q.wait();
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / bench;
        std::printf("bench decode: %zu experts (%.1f MiB) in %.1f us/call = %.0f GB/s\n", jobs.size(), bytes / 1048576.0, us, bytes / us / 1e3);
        for (auto& j : jobs) { j.nt = 8; for (int t = 0; t < 8; ++t) { j.tok[t] = t; j.w[t] = 0.1f; } }
        q.memcpy(dj, jobs.data(), jobs.size() * sizeof(xe::XeJob)).wait();
        for (int w = 0; w < 5; ++w) xe::xe_experts(q, dj, (int) jobs.size(), 8, dX, dH, dO, (int) E, (int) F, L, false);
        q.wait();
        t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < bench; ++it) xe::xe_experts(q, dj, (int) jobs.size(), 8, dX, dH, dO, (int) E, (int) F, L, false);
        q.wait();
        us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / bench;
        std::printf("bench verify (8 tokens per expert): %.1f us/call = %.0f GB/s\n", us, bytes / us / 1e3);
        for (uint8_t* p : bufs) sycl::free(p, q);
    }
    if (bench_q8 > 0) {   // the planar int8 kernels at decode / verify shapes, experts rotated past the L2
        const int l = layers.empty() ? 20 : layers[0];
        constexpr int NE = 24;
        std::vector<uint8_t*> pl;
        std::vector<xe::XeJob> base;
        std::vector<size_t> nbytes;
        for (int e = 0; e < g.n_experts && (int) pl.size() < NE; ++e) {
            const cpu::NativeFmt& f = fl[(size_t) g.cls(l, e)];
            if (bench_q8_gu >= 0 && f.gu_type != bench_q8_gu) continue;
            std::vector<uint8_t> blob(f.bytes);
            src.read(l, e, blob.data());
            uint8_t* fb = sycl::malloc_device<uint8_t>(f.bytes, q);
            uint8_t* pb = sycl::malloc_device<uint8_t>(f.bytes, q);
            q.memcpy(fb, blob.data(), f.bytes).wait();
            xe::xe_repack_blob(q, fb, pb, f.gu_type, f.d_type, f.up_off, f.down_off, (int) E, (int) F).wait();
            sycl::free(fb, q);
            xe::XeJob j;
            j.blob = pb; j.up_off = (uint32_t) f.up_off; j.down_off = (uint32_t) f.down_off;
            j.gu_type = (uint16_t) f.gu_type; j.d_type = (uint16_t) f.d_type;
            j.gu_row = (uint16_t) f.gu_row; j.d_row = (uint16_t) f.d_row;
            pl.push_back(pb); base.push_back(j); nbytes.push_back(f.bytes);
        }
        xe::XeQ8 Xq{sycl::malloc_device<int8_t>((size_t) 8 * E, q), sycl::malloc_device<float>((size_t) 8 * E / 32, q),
                    sycl::malloc_device<int32_t>((size_t) 8 * E / 16, q)};
        xe::XeQ8 Hq{sycl::malloc_device<int8_t>((size_t) 64 * 8 * F, q), sycl::malloc_device<float>((size_t) 64 * 8 * F / 32, q),
                    sycl::malloc_device<int32_t>((size_t) 64 * 8 * F / 16, q)};
        xe::XeJob* djs = sycl::malloc_device<xe::XeJob>((size_t) NE * 64, q);
        sycl::half* dxt = sycl::malloc_device<sycl::half>((size_t) 8 * E, q);
        sycl::half* dht = sycl::malloc_device<sycl::half>((size_t) 64 * F * 8, q);
        float* dO2 = sycl::malloc_device<float>((size_t) 8 * E, q);
        {   // 8 activation rows of the distribution the engine sees (the call's X; the first 3 rows were set above)
            std::vector<float> X8((size_t) 8 * E);
            for (float& v : X8) v = nd(rng);
            q.memcpy(dX, X8.data(), X8.size() * 4).wait();
        }
        std::printf("bench-q8: layer %d, %d experts (%.1f MiB) rotated (gate/up type %d)\n", l, (int) pl.size(),
                    std::accumulate(nbytes.begin(), nbytes.end(), (size_t) 0) / 1048576.0, bench_q8_gu);
        // (nj experts a call, nt tokens a job, the call's tokens ntok): the job lists of NE / nj calls, staged once
        struct Cfg { int nj, nt, ntok, mnt; };
        for (const Cfg cf : {Cfg{3, 1, 1, 1}, Cfg{3, 1, 3, 1}, Cfg{3, 1, 3, 2}, Cfg{3, 1, 3, 4}, Cfg{3, 1, 5, 8},
                             Cfg{3, 2, 3, 2}, Cfg{3, 2, 3, 4}, Cfg{3, 3, 3, 4}, Cfg{3, 2, 5, 8}}) {
            const int ncalls = (int) pl.size() / cf.nj;
            std::vector<xe::XeJob> jl((size_t) ncalls * cf.nj);
            double bytes = 0;
            for (int c = 0; c < ncalls; ++c)
                for (int k = 0; k < cf.nj; ++k) {
                    xe::XeJob j = base[(size_t) (c * cf.nj + k)];
                    j.nt = (uint8_t) cf.nt;
                    for (int t = 0; t < cf.nt; ++t) { j.tok[t] = (k + t) % cf.ntok; j.w[t] = 0.1f; }
                    jl[(size_t) (c * cf.nj + k)] = j;
                    if (c == 0) bytes += (double) nbytes[(size_t) k];
                }
            q.memcpy(djs, jl.data(), jl.size() * sizeof(xe::XeJob)).wait();
            auto call = [&](int c) {
                xe::xe_quant_rows(q, dX, cf.ntok, (int) E, Xq);
                xe::xe_experts_q8(q, djs + (size_t) c * cf.nj, cf.nj, cf.ntok, Xq, Hq, dO, (int) E, (int) F, L, false, cf.mnt);
            };
            auto callx = [&](int c) {
                xe::xe_xt(q, dX, cf.ntok, (int) E, dxt);
                xe::xe_experts_x(q, djs + (size_t) c * cf.nj, cf.nj, cf.ntok, dxt, dht, dO2, (int) E, (int) F, L, false);
            };
            auto timed = [&](auto&& fn) {
                for (int c = 0; c < ncalls; ++c) fn(c);
                q.wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < bench_q8; ++it) fn(it % ncalls);
                q.wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / bench_q8;
            };
            const double us = timed(call), ux = timed(callx);
            // call 0 in the jobs' own form (max_nt = their token count): the wider form must give the same bits
            std::vector<float> aref((size_t) cf.ntok * E);
            xe::xe_quant_rows(q, dX, cf.ntok, (int) E, Xq);
            xe::xe_experts_q8(q, djs, cf.nj, cf.ntok, Xq, Hq, dO, (int) E, (int) F, L, false, cf.nt);
            q.wait();
            q.memcpy(aref.data(), dO, aref.size() * 4).wait();
            call(0); callx(0); q.wait();   // the same call both ways: the largest difference vs the largest value
            std::vector<float> a((size_t) cf.ntok * E), b2((size_t) cf.ntok * E);
            q.memcpy(a.data(), dO, a.size() * 4).wait();
            q.memcpy(b2.data(), dO2, b2.size() * 4).wait();
            double md = 0, mx = 0;
            size_t nbits = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                md = std::max(md, (double) std::fabs(a[i] - b2[i]));
                mx = std::max(mx, (double) std::fabs(a[i]));
                if (std::memcmp(&a[i], &aref[i], 4) != 0) ++nbits;
            }
            std::printf("  %d experts x %d tokens (call of %d, template %d): dp4a %6.1f us = %4.0f GB/s | xmx %6.1f us = %4.0f GB/s | max|diff| %.2e of %.2e | bits %s%zu\n",
                        cf.nj, cf.nt, cf.ntok, cf.mnt, us, bytes / us / 1e3, ux, bytes / ux / 1e3, md, mx, nbits ? "DIFFER " : "ok ", nbits);
        }
        for (uint8_t* p : pl) sycl::free(p, q);
    }
    if (gemm_bench) {   // the prompt path's GEMM form for one expert: dequant, gather, gate|up GEMM, SwiGLU, down GEMM
        namespace mkl = oneapi::mkl;
        const int l = layers.empty() ? 20 : layers[0];
        int shown = 0;
        for (int c = 0; c < (int) g.n_classes() && shown < 3; ++c) {
            int e = -1;
            for (int k = 0; k < g.n_experts; ++k) if (g.cls(l, k) == c) { e = k; break; }
            if (e < 0) continue;
            ++shown;
            const cpu::NativeFmt& f = fl[(size_t) c];
            std::vector<uint8_t> blob(f.bytes);
            src.read(l, e, blob.data());
            uint8_t* db = sycl::malloc_device<uint8_t>(f.bytes, q);
            q.memcpy(db, blob.data(), f.bytes).wait();
            const int MM = 1024;
            sycl::half* Xg = sycl::malloc_device<sycl::half>((size_t) MM * E, q);
            sycl::half* Wgu = sycl::malloc_device<sycl::half>((size_t) 2 * F * E, q);
            sycl::half* Wd = sycl::malloc_device<sycl::half>((size_t) E * F, q);
            float* G = sycl::malloc_device<float>((size_t) MM * 2 * F, q);
            sycl::half* Hh = sycl::malloc_device<sycl::half>((size_t) MM * F, q);
            float* Y = sycl::malloc_device<float>((size_t) MM * E, q);
            float* Xs = sycl::malloc_device<float>((size_t) MM * E, q);
            int32_t* tok = sycl::malloc_device<int32_t>(MM, q);
            std::vector<int32_t> ht(MM);
            for (int i = 0; i < MM; ++i) ht[(size_t) i] = i;
            q.memcpy(tok, ht.data(), MM * 4).wait();
            q.memset(Xs, 0, (size_t) MM * E * 4).wait();
            auto tm = [&](auto fn, int it) {
                fn(); q.wait();
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) fn();
                q.wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
            };
            const double tdq = tm([&] {
                xe::xe_dequant_rows_f16(q, f.gu_type, db, (int) F, (int) f.gu_row, (int) E, Wgu);
                xe::xe_dequant_rows_f16(q, f.gu_type, db + f.up_off, (int) F, (int) f.gu_row, (int) E, Wgu + (size_t) F * E);
                xe::xe_dequant_rows_f16(q, f.d_type, db + f.down_off, (int) E, (int) f.d_row, (int) F, Wd);
            }, 20);
            {   // the fp16 dequantisation against the CPU dequantiser (fp16 rounding only)
                std::vector<sycl::half> hg((size_t) 2 * F * E), hd((size_t) E * F);
                q.memcpy(hg.data(), Wgu, hg.size() * 2);
                q.memcpy(hd.data(), Wd, hd.size() * 2).wait();
                std::vector<float> ref((size_t) std::max(E, F));
                double worst = 0;
                for (int64_t rr = 0; rr < 2 * F; rr += 97) {
                    const uint8_t* src = blob.data() + (rr < F ? (size_t) rr * f.gu_row : f.up_off + (size_t) (rr - F) * f.gu_row);
                    dequant(f.gu_type, src, ref.data(), E);
                    for (int64_t i = 0; i < E; ++i) worst = std::max(worst, std::fabs((double) (float) hg[(size_t) (rr * E + i)] - ref[(size_t) i]) / (std::fabs(ref[(size_t) i]) + 1e-3));
                }
                for (int64_t rr = 0; rr < E; rr += 97) {
                    dequant(f.d_type, blob.data() + f.down_off + (size_t) rr * f.d_row, ref.data(), F);
                    for (int64_t i = 0; i < F; ++i) worst = std::max(worst, std::fabs((double) (float) hd[(size_t) (rr * F + i)] - ref[(size_t) i]) / (std::fabs(ref[(size_t) i]) + 1e-3));
                }
                std::printf("class %d: fp16 dequant vs CPU, worst relative error %.2e %s\n", c, worst, worst < 2e-3 ? "ok" : "FAIL");
                if (worst >= 2e-3) ++fails;
            }
            std::printf("class %d (gu %d / d %d): dequant gate+up+down %.0f us\n", c, f.gu_type, f.d_type, tdq);
            for (int m : {8, 16, 32, 64, 128, 256, 1024}) {
                const double tg = tm([&] { xe::xe_gather_rows_f16(q, Xs, tok, m, (int) E, Xg); }, 20);
                const double t1 = tm([&] { mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans, m, 2 * F, E, 1.0f, Xg, E, Wgu, E, 0.0f, G, 2 * F); }, 20);
                const double ts = tm([&] { xe::xe_swiglu_f16(q, G, m, (int) F, L, Hh); }, 20);
                const double t2 = tm([&] { mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans, m, E, F, 1.0f, Hh, F, Wd, F, 0.0f, Y, E); }, 20);
                const double fl = 2.0 * m * (2.0 * F * E + (double) E * F);
                std::printf("  m %4d: gather %.0f us, gemm1 %.0f us, swiglu %.0f us, gemm2 %.0f us -> %.1f TFLOPS on the GEMMs\n", m, tg, t1, ts, t2,
                            fl / (t1 + t2) / 1e6);
            }
            for (void* ptr : {(void*) db, (void*) Xg, (void*) Wgu, (void*) Wd, (void*) G, (void*) Hh, (void*) Y, (void*) Xs, (void*) tok}) sycl::free(ptr, q);
        }
    }
    return fails ? 1 : 0;
}
