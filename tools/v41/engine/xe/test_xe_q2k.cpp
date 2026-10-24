// tools/v41/engine/xe/test_xe_q2k.cpp - larry: the planar int8 expert kernels on Q2_K experts (engine1's DSpark draft:
// the sidecar's mtp.b.ffn.experts.w1 / w3 / w2) against two host references.
//
//   test_xe_q2k V41-MTP-Vision-1017.Q2_K.gguf [--block 0] [--experts 0,17,77,127] [--device 0] [--bench 200]
//
// Per expert and token count (1, 2, 3, 5 tokens in one job; then 5 tokens spread over 6 jobs):
//   q8  = the same int8 rounding as the kernels (x and H quantised per 32 values, the dp4a sums exact): only the float
//         summation order differs (and an H value at a rounding tie) - pass: rel < 1e-3;
//   f32 = dequantised F32 weights x F32 activations (what the ggml draft approximates too): reported, not gated.
#include "xe_kernels.hpp"

#include "ggml.h"
#include "strata/artifact/gguf_reader.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::v41;

namespace {

constexpr int E = 5120, F = 2304, QK = 256, BB = 84;   // the draft's n_embd, n_ff; Q2_K block

void deq(const uint8_t* src, float* dst, int64_t n) { ggml_get_type_traits(GGML_TYPE_Q2_K)->to_float(src, dst, n); }

/// int8 rounding per 32 values (xe_quant_rows / gate_up_q8's H): q = rint(v * 127 / amax), x' = q * amax / 127
void q8_round(const float* v, int64_t n, float* out) {
    for (int64_t g = 0; g < n; g += 32) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[g + i]));
        const float id = amax > 0.f ? 127.0f / amax : 0.f, d = amax / 127.0f;
        for (int i = 0; i < 32; ++i) out[g + i] = (float) (int) std::rint(v[g + i] * id) * d;
    }
}

/// y = Wd (silu(min(Wg x, L)) * clamp(Wu x, -L, L)); q8: x and h rounded as the kernels round them
void ref_expert(const uint8_t* blob, const float* x_in, float L, bool q8, float* y) {
    const size_t gr = (size_t) (E / QK) * BB, dr = (size_t) (F / QK) * BB;
    std::vector<float> x(x_in, x_in + E), row(E), h(F);
    if (q8) q8_round(x_in, E, x.data());
    for (int r = 0; r < F; ++r) {
        double g = 0, u = 0;
        deq(blob + (size_t) r * gr, row.data(), E);
        for (int i = 0; i < E; ++i) g += (double) row[(size_t) i] * x[(size_t) i];
        deq(blob + (size_t) F * gr + (size_t) r * gr, row.data(), E);
        for (int i = 0; i < E; ++i) u += (double) row[(size_t) i] * x[(size_t) i];
        const double gc = std::min(g, (double) L), uc = std::max(-(double) L, std::min(u, (double) L));
        h[(size_t) r] = (float) (gc / (1.0 + std::exp(-gc)) * uc);
    }
    if (q8) { std::vector<float> hq(F); q8_round(h.data(), F, hq.data()); h = hq; }
    for (int i = 0; i < E; ++i) {
        double s = 0;
        deq(blob + 2 * (size_t) F * gr + (size_t) i * dr, row.data(), F);
        for (int r = 0; r < F; ++r) s += (double) row[(size_t) r] * h[(size_t) r];
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
    if (argc < 2) { std::fprintf(stderr, "usage: test_xe_q2k SIDECAR.gguf [--block b] [--experts a,b] [--device d] [--bench n]\n"); return 2; }
    int block = 0, dev_i = 0, bench = 0;
    std::vector<int> experts = {0, 17, 77, 127};
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--block") block = std::atoi(argv[i + 1]);
        else if (k == "--experts") experts = csv(argv[i + 1]);
        else if (k == "--device") dev_i = std::atoi(argv[i + 1]);
        else if (k == "--bench") bench = std::atoi(argv[i + 1]);
    }
    strata::GgufModel m = strata::GgufModel::open(argv[1]);
    const std::string pre = "mtp." + std::to_string(block) + ".ffn.experts.";
    const strata::TensorInfo* tw[3];
    const uint8_t* td[3];
    const char* nm[3] = {"w1.weight", "w3.weight", "w2.weight"};   // gate, up, down
    for (int k = 0; k < 3; ++k) {
        size_t s = 0;
        tw[k] = m.find(pre + nm[k], &s);
        if (!tw[k] || tw[k]->type != GGML_TYPE_Q2_K) { std::fprintf(stderr, "no Q2_K tensor %s%s\n", pre.c_str(), nm[k]); return 1; }
        td[k] = m.shard(s).tensor_data(*tw[k]);
    }
    const int NE = (int) tw[0]->shape[2];
    const size_t gr = (size_t) (E / QK) * BB, dr = (size_t) (F / QK) * BB;
    const size_t gb = (size_t) F * gr, db = (size_t) E * dr, blob_b = 2 * gb + db;   // one matrix of gate rows = of up rows
    const float L = 10.0f;   // DeepSeek-V4.1's swiglu_limit

    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (dev_i >= (int) devs.size()) { std::fprintf(stderr, "no GPU %d\n", dev_i); return 1; }
    sycl::queue q(devs[(size_t) dev_i], sycl::property::queue::in_order());
    std::printf("device %s, block %d, %d experts of %.2f MB\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                block, NE, blob_b / 1e6);

    // the experts' blobs (gate | up | down, file layout) -> planar on the card
    const int ne = (int) experts.size();
    uint8_t* stage = sycl::malloc_device<uint8_t>(blob_b, q);
    uint8_t* blobs = sycl::malloc_device<uint8_t>(blob_b * ne, q);
    std::vector<std::vector<uint8_t>> host((size_t) ne);
    for (int i = 0; i < ne; ++i) {
        const int e = experts[(size_t) i];
        auto& h = host[(size_t) i];
        h.resize(blob_b);
        std::memcpy(h.data(), td[0] + (size_t) e * gb, gb);
        std::memcpy(h.data() + gb, td[1] + (size_t) e * gb, gb);
        std::memcpy(h.data() + 2 * gb, td[2] + (size_t) e * db, db);
        q.memcpy(stage, h.data(), blob_b).wait();
        xe::xe_repack_blob(q, stage, blobs + (size_t) i * blob_b, xe::kQ2K, xe::kQ2K, gb, 2 * gb, E, F).wait();
    }
    constexpr int NT = 8;
    float* dX = sycl::malloc_device<float>((size_t) NT * E, q);
    float* dO = sycl::malloc_device<float>((size_t) NT * E, q);
    xe::XeQ8 Xq{sycl::malloc_device<int8_t>((size_t) NT * E, q), sycl::malloc_device<float>((size_t) NT * E / 32, q),
                sycl::malloc_device<int32_t>((size_t) NT * E / 16, q)};
    const int maxj = 16;
    const size_t hr = (size_t) maxj * xe::kMaxJobTok;
    xe::XeQ8 Hq{sycl::malloc_device<int8_t>(hr * F, q), sycl::malloc_device<float>(hr * F / 32, q), sycl::malloc_device<int32_t>(hr * F / 16, q)};
    xe::XeJob* dj = sycl::malloc_device<xe::XeJob>(maxj, q);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> X((size_t) NT * E);
    for (auto& v : X) v = nd(rng);
    q.memcpy(dX, X.data(), X.size() * 4).wait();
    xe::xe_quant_rows(q, dX, NT, E, Xq).wait();

    auto job = [&](int i, const std::vector<int>& toks, const std::vector<float>& w) {
        xe::XeJob j;
        j.blob = blobs + (size_t) i * blob_b;
        j.up_off = (uint32_t) gb; j.down_off = (uint32_t) (2 * gb);
        j.gu_type = j.d_type = (uint16_t) xe::kQ2K;
        j.gu_row = (uint16_t) gr; j.d_row = (uint16_t) dr;
        j.nt = (uint8_t) toks.size();
        for (size_t t = 0; t < toks.size(); ++t) { j.tok[t] = toks[t]; j.w[t] = w[t]; }
        return j;
    };
    int bad = 0;
    std::vector<float> out((size_t) NT * E), r8(E), r32(E);
    // (a) one job of nt tokens (tokens 0..nt-1, weight 1)
    for (int i = 0; i < ne; ++i) {
        for (int nt : {1, 2, 3, 5}) {
            const xe::XeJob j = job(i, [&] { std::vector<int> v; for (int t = 0; t < nt; ++t) v.push_back(t); return v; }(),
                                    std::vector<float>((size_t) nt, 1.0f));
            q.memcpy(dj, &j, sizeof j).wait();
            xe::xe_experts_q8(q, dj, 1, nt, Xq, Hq, dO, E, F, L, false, nt, nullptr, 0, false, true).wait();
            q.memcpy(out.data(), dO, (size_t) nt * E * 4).wait();
            double w8 = 0, w32 = 0;
            for (int t = 0; t < nt; ++t) {
                ref_expert(host[(size_t) i].data(), X.data() + (size_t) t * E, L, true, r8.data());
                ref_expert(host[(size_t) i].data(), X.data() + (size_t) t * E, L, false, r32.data());
                w8 = std::max(w8, rel(out.data() + (size_t) t * E, r8.data(), E));
                w32 = std::max(w32, rel(out.data() + (size_t) t * E, r32.data(), E));
            }
            const bool ok = w8 < 1e-3;
            bad += !ok;
            std::printf("e%-3d nt %d: rel vs q8 ref %.2e, vs f32 ref %.2e  %s\n", experts[(size_t) i], nt, w8, w32, ok ? "ok" : "FAIL");
        }
    }
    // (b) 5 tokens over every expert: each job a subset, weights summed per token (the down kernel's accumulation)
    {
        std::vector<xe::XeJob> js;
        std::vector<std::vector<float>> want((size_t) 5, std::vector<float>(E, 0.f));
        for (int i = 0; i < ne; ++i) {
            std::vector<int> toks;
            std::vector<float> w;
            for (int t = 0; t < 5; ++t) if ((t + i) % 2 == 0 || i == 0) { toks.push_back(t); w.push_back(0.25f + 0.1f * (float) (t + i)); }
            js.push_back(job(i, toks, w));
            for (size_t k = 0; k < toks.size(); ++k) {
                ref_expert(host[(size_t) i].data(), X.data() + (size_t) toks[k] * E, L, true, r8.data());
                for (int c = 0; c < E; ++c) want[(size_t) toks[k]][(size_t) c] += w[k] * r8[(size_t) c];
            }
        }
        q.memcpy(dj, js.data(), js.size() * sizeof(xe::XeJob)).wait();
        q.memset(dO, 0, (size_t) 5 * E * 4).wait();
        xe::xe_experts_q8(q, dj, (int) js.size(), 5, Xq, Hq, dO, E, F, L, true, 5, nullptr, 0, false, true).wait();
        q.memcpy(out.data(), dO, (size_t) 5 * E * 4).wait();
        double w8 = 0;
        for (int t = 0; t < 5; ++t) w8 = std::max(w8, rel(out.data() + (size_t) t * E, want[(size_t) t].data(), E));
        const bool ok = w8 < 1e-3;
        bad += !ok;
        std::printf("%zu jobs, 5 tokens, accumulated: rel vs q8 ref %.2e  %s\n", js.size(), w8, ok ? "ok" : "FAIL");
    }
    {   // the coalesced repack (form 1) against the first kernel (form 0): byte-identical, every type; their times
        struct T { int type, qk, bb; const char * name; };
        const T ts[] = { { xe::kQ3K, 256, 110, "Q3_K" }, { xe::kQ20, 64, 18, "Q2_0" }, { xe::kPtq1, 128, 28, "PTQ1_0" }, { xe::kQ2K, 256, 84, "Q2_K" } };
        for (const T & t : ts) {
            const size_t g1 = (size_t) F * (E / t.qk) * t.bb, d1 = (size_t) E * (F / t.qk) * t.bb, bytes = 2 * g1 + d1;
            std::vector<uint8_t> hb(bytes), o0(bytes), o1(bytes);
            for (auto & b : hb) b = (uint8_t) rng();
            uint8_t * s = sycl::malloc_device<uint8_t>(bytes, q), * a = sycl::malloc_device<uint8_t>(bytes, q), * b = sycl::malloc_device<uint8_t>(bytes, q);
            q.memcpy(s, hb.data(), bytes).wait();
            double us[2];
            for (int f = 0; f < 2; ++f) {
                uint8_t * d = f ? b : a;
                xe::xe_repack_blob(q, s, d, t.type, t.type, g1, 2 * g1, E, F, f).wait();   // (warm)
                const auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < 5; ++it) xe::xe_repack_blob(q, s, d, t.type, t.type, g1, 2 * g1, E, F, f);
                q.wait();
                us[f] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 5;
            }
            q.memcpy(o0.data(), a, bytes).wait();
            q.memcpy(o1.data(), b, bytes).wait();
            const bool same = o0 == o1;
            bad += !same;
            std::printf("repack %-6s (%.2f MB): first form %.0f us, coalesced %.0f us, %s\n", t.name, bytes / 1e6, us[0], us[1],
                        same ? "byte-identical" : "DIFFER");
            sycl::free(s, q); sycl::free(a, q); sycl::free(b, q);
        }
    }
    if (bench > 0) {   // the draft's shapes on one card: ~4 jobs, 1-3 tokens each, 5 tokens in the call
        std::vector<xe::XeJob> js;
        for (int i = 0; i < std::min(ne, 4); ++i) js.push_back(job(i, i == 0 ? std::vector<int>{0, 1, 2} : std::vector<int>{i}, {1.f, 1.f, 1.f}));
        q.memcpy(dj, js.data(), js.size() * sizeof(xe::XeJob)).wait();
        for (int w = 0; w < 10; ++w) xe::xe_experts_q8(q, dj, (int) js.size(), 5, Xq, Hq, dO, E, F, L, false, 3, nullptr, 0, false, true);
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < bench; ++it) xe::xe_experts_q8(q, dj, (int) js.size(), 5, Xq, Hq, dO, E, F, L, false, 3, nullptr, 0, false, true);
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / bench;
        std::printf("bench: %zu jobs (%zu experts, %.1f MB), 5 tokens: %.1f us a call (%.0f GB/s, L2-warm)\n", js.size(), js.size(),
                    js.size() * blob_b / 1e6, us, js.size() * blob_b / us / 1e3);
    }
    std::printf("%s\n", bad ? "FAILED" : "PASS");
    return bad ? 1 : 0;
}
