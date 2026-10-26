// xe2_bench.cpp - the planar int8 expert kernels (xe_planar.cpp) against the F32 file-layout kernels (xe_kernels.cpp)
// and a double-precision CPU reference, on synthetic experts at DeepSeek-V4.1's shapes (n_embd 5120, n_ff 2304).
//
//   xe2_bench [iters]
#include "xe_kernels.hpp"
#include "xe_planar.cpp"   // its kernels directly: gate/up and down timed apart

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::v41::xe;
namespace xp = strata::v41::xe;

static const int E = 5120, F = 2304;
static const float L = 10.f;

static float hf16(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>((sycl::half) f); }
static int qk(int t) { return t == kQ3K ? 256 : t == kQ20 ? 64 : 128; }
static int bbytes(int t) { return t == kQ3K ? 110 : t == kQ20 ? 18 : 28; }

// the file layout's dequantisation (xe_kernels.cpp dequant_one)
static float deq(int type, const uint8_t* row, int v) {
    auto ld32 = [](const uint8_t* p) { return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24); };
    auto hf = [](const uint8_t* p) { return hf16((uint16_t) (p[0] | (p[1] << 8))); };
    if (type == kQ20) {
        const uint8_t* blk = row + (size_t) (v / 64) * 18;
        const int j = v % 64;
        return (float) ((int) ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * hf(blk);
    }
    if (type == kPtq1) {
        const uint8_t* blk = row + (size_t) (v / 128) * 28;
        const int j = v % 128;
        int byte, dig;
        if (j < 80) { byte = j % 16; dig = j / 16; }
        else if (j < 120) { byte = 16 + (j - 80) % 8; dig = (j - 80) / 8; }
        else { byte = 24 + (j - 120) % 2; dig = (j - 120) / 2; }
        const uint32_t p = dig == 0 ? 1u : dig == 1 ? 3u : dig == 2 ? 9u : dig == 3 ? 27u : 81u;
        const int t = (int) ((((blk[byte] * p) & 0xFFu) * 3u) >> 8);
        return (float) (t - 1) * hf(blk + 26);
    }
    const uint8_t* blk = row + (size_t) (v / 256) * 110;
    const int j256 = v % 256;
    const int h = j256 / 128, jj = (j256 % 128) / 32, l = j256 % 32;
    const uint8_t* sc = blk + 96;
    const uint32_t a0 = ld32(sc), a1 = ld32(sc + 4), tmp = ld32(sc + 8);
    const uint32_t s4[4] = {(a0 & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4), (a1 & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4),
                            ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4), ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4)};
    const int is = h * 8 + 2 * jj + (l >= 16 ? 1 : 0);
    const float scl = (float) ((int) ((s4[is >> 2] >> (8 * (is & 3))) & 0xffu) - 32);
    const int qv = (int) ((blk[32 + h * 32 + l] >> (2 * jj)) & 3u) - (((blk[l] >> (h * 4 + jj)) & 1u) ? 0 : 4);
    return hf(blk + 108) * scl * (float) qv;
}

struct Expert {
    int gu, dt;
    size_t gu_row, d_row, up_off, down_off, bytes;
    std::vector<uint8_t> blob;
};

static Expert make_expert(int gu, int dt, std::mt19937& rng) {
    Expert x;
    x.gu = gu; x.dt = dt;
    x.gu_row = (size_t) (E / qk(gu)) * bbytes(gu);
    x.d_row = (size_t) (F / qk(dt)) * bbytes(dt);
    x.up_off = (size_t) F * x.gu_row;
    x.down_off = 2 * x.up_off;
    x.bytes = x.down_off + (size_t) E * x.d_row;
    x.blob.resize(x.bytes);
    std::uniform_int_distribution<int> ub(0, 255);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    auto fill = [&](uint8_t* p, size_t nblk, int t, float dscale) {
        for (size_t i = 0; i < nblk; ++i) {
            uint8_t* b = p + i * bbytes(t);
            for (int k = 0; k < bbytes(t); ++k) b[k] = (uint8_t) ub(rng);
            const uint16_t d = f2h(dscale * ud(rng));
            const int off = t == kQ3K ? 108 : t == kQ20 ? 0 : 26;
            b[off] = (uint8_t) d; b[off + 1] = (uint8_t) (d >> 8);
        }
    };
    // scales chosen so that gate/up outputs are O(1) and the SwiGLU clamp rarely binds
    const float gs = gu == kQ3K ? 0.0006f : 0.012f, ds = dt == kQ3K ? 0.001f : 0.02f;
    fill(x.blob.data(), (size_t) 2 * F * (E / qk(gu)), gu, gs);
    fill(x.blob.data() + x.down_off, (size_t) E * (F / qk(dt)), dt, ds);
    return x;
}

// y += w * expert(x), double precision
static void ref_expert(const Expert& ex, const float* x, float w, double* y) {
    std::vector<float> row((size_t) std::max(E, F));
    std::vector<double> h((size_t) F);
    for (int r = 0; r < F; ++r) {
        double g = 0, u = 0;
        for (int i = 0; i < E; ++i) row[(size_t) i] = deq(ex.gu, ex.blob.data() + (size_t) r * ex.gu_row, i);
        for (int i = 0; i < E; ++i) g += (double) row[(size_t) i] * x[i];
        for (int i = 0; i < E; ++i) row[(size_t) i] = deq(ex.gu, ex.blob.data() + ex.up_off + (size_t) r * ex.gu_row, i);
        for (int i = 0; i < E; ++i) u += (double) row[(size_t) i] * x[i];
        const double gc = std::min(g, (double) L), uc = std::max(-(double) L, std::min(u, (double) L));
        h[(size_t) r] = gc / (1.0 + std::exp(-gc)) * uc;
    }
    for (int i = 0; i < E; ++i) {
        double s = 0;
        const uint8_t* dr = ex.blob.data() + ex.down_off + (size_t) i * ex.d_row;
        for (int r = 0; r < F; ++r) s += (double) deq(ex.dt, dr, r) * h[(size_t) r];
        y[i] += w * s;
    }
}

static double rel(const float* a, const double* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
    return std::sqrt(num / std::max(den, 1e-30));
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(42);
    const int classes[][2] = {{kQ3K, kQ3K}, {kQ3K, kQ20}, {kQ20, kQ20}, {kPtq1, kPtq1}, {kQ20, kPtq1}, {kPtq1, kQ3K}};
    const int NX = 6;
    std::vector<Expert> ex;
    for (const auto& c : classes) ex.push_back(make_expert(c[0], c[1], rng));
    const int NCOPY = 12;   // timing rotates over this many copies of the experts (1.5 GB: not L2 resident)
    std::vector<uint8_t*> draw(NX * NCOPY), dpl(NX * NCOPY);
    size_t tot = 0;
    uint8_t* stage = sycl::malloc_device<uint8_t>(ex[0].bytes + (1 << 20), q);
    for (int c = 0; c < NCOPY; ++c)
        for (int k = 0; k < NX; ++k) {
            const int i = c * NX + k;
            draw[i] = sycl::malloc_device<uint8_t>(ex[k].bytes, q);
            dpl[i] = sycl::malloc_device<uint8_t>(ex[k].bytes, q);
            q.memcpy(draw[i], ex[k].blob.data(), ex[k].bytes).wait();
            xe_repack_blob(q, draw[i], dpl[i], ex[k].gu, ex[k].dt, ex[k].up_off, ex[k].down_off, E, F).wait();
            tot += ex[k].bytes;
        }
    (void) stage;
    const int MT = 8;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> X((size_t) MT * E);
    for (float& v : X) v = nd(rng);
    float* dX = sycl::malloc_device<float>((size_t) MT * E, q);
    q.memcpy(dX, X.data(), X.size() * 4).wait();
    XeQ8 xq, hq;
    xq.q = sycl::malloc_device<int8_t>((size_t) MT * E, q);
    xq.d = sycl::malloc_device<float>((size_t) MT * E / 32, q);
    xq.s = sycl::malloc_device<int32_t>((size_t) MT * E / 16, q);
    const int MJ = 64;
    hq.q = sycl::malloc_device<int8_t>((size_t) MJ * 8 * F, q);
    hq.d = sycl::malloc_device<float>((size_t) MJ * 8 * F / 32, q);
    hq.s = sycl::malloc_device<int32_t>((size_t) MJ * 8 * F / 16, q);
    float* dH = sycl::malloc_device<float>((size_t) MJ * 8 * F, q);
    float* dO = sycl::malloc_device<float>((size_t) MT * E, q);
    XeJob* djr = sycl::malloc_device<XeJob>(MJ * NCOPY, q);
    XeJob* djp = sycl::malloc_device<XeJob>(MJ * NCOPY, q);
    auto job = [&](int k, bool planar, int copy = 0) {
        XeJob j;
        j.blob = planar ? dpl[copy * NX + k] : draw[copy * NX + k];
        j.up_off = (uint32_t) ex[k].up_off; j.down_off = (uint32_t) ex[k].down_off;
        j.gu_type = (uint16_t) ex[k].gu; j.d_type = (uint16_t) ex[k].dt;
        j.gu_row = (uint16_t) ex[k].gu_row; j.d_row = (uint16_t) ex[k].d_row;
        return j;
    };
    // ---- correctness: (a) each expert alone, 1 token; (b) all six experts, 3 tokens each with different weights
    std::printf("per-expert decode (1 token), rel vs f64 reference: f32 kernels | int8 planar\n");
    std::vector<float> got((size_t) MT * E);
    for (int k = 0; k < NX; ++k) {
        std::vector<double> ref((size_t) E, 0.0);
        ref_expert(ex[k], X.data(), 1.f, ref.data());
        XeJob jr = job(k, false), jp = job(k, true);
        jr.nt = jp.nt = 1; jr.tok[0] = jp.tok[0] = 0; jr.w[0] = jp.w[0] = 1.f;
        q.memcpy(djr, &jr, sizeof jr); q.memcpy(djp, &jp, sizeof jp).wait();
        xe_experts(q, djr, 1, 1, dX, dH, dO, E, F, L, false).wait();
        q.memcpy(got.data(), dO, (size_t) E * 4).wait();
        const double r0 = rel(got.data(), ref.data(), E);
        xe_quant_rows(q, dX, 1, E, xq);
        xe_experts_q8(q, djp, 1, 1, xq, hq, dO, E, F, L, false).wait();
        q.memcpy(got.data(), dO, (size_t) E * 4).wait();
        const double r1 = rel(got.data(), ref.data(), E);
        std::printf("  class gu %3d / d %3d: %.2e | %.2e %s\n", ex[k].gu, ex[k].dt, r0, r1, r1 < 2e-2 ? "ok" : "FAIL");
    }
    {
        const int NT = 3;
        std::vector<double> ref((size_t) NT * E, 0.0);
        std::vector<XeJob> jr, jp;
        for (int k = 0; k < NX; ++k) {
            XeJob a = job(k, false), b = job(k, true);
            a.nt = b.nt = NT;
            for (int s = 0; s < NT; ++s) {
                const int t = (s + k) % NT;
                const float w = 0.1f + 0.05f * (float) (k + s);
                a.tok[s] = b.tok[s] = t; a.w[s] = b.w[s] = w;
                ref_expert(ex[k], X.data() + (size_t) t * E, w, ref.data() + (size_t) t * E);
            }
            jr.push_back(a); jp.push_back(b);
        }
        q.memcpy(djr, jr.data(), jr.size() * sizeof(XeJob)); q.memcpy(djp, jp.data(), jp.size() * sizeof(XeJob)).wait();
        xe_experts(q, djr, NX, NT, dX, dH, dO, E, F, L, false).wait();
        q.memcpy(got.data(), dO, (size_t) NT * E * 4).wait();
        const double r0 = rel(got.data(), ref.data(), (size_t) NT * E);
        xe_quant_rows(q, dX, NT, E, xq);
        xe_experts_q8(q, djp, NX, NT, xq, hq, dO, E, F, L, false).wait();
        q.memcpy(got.data(), dO, (size_t) NT * E * 4).wait();
        const double r1 = rel(got.data(), ref.data(), (size_t) NT * E);
        std::printf("verify form (6 experts x 3 tokens, weighted): %.2e | %.2e %s\n", r0, r1, r1 < 2e-2 ? "ok" : "FAIL");
    }
    {   // planar fp16 dequantisation == the file layout's, bit for bit; prompt form (experts_y + gather)
        sycl::half* a = sycl::malloc_device<sycl::half>((size_t) F * E, q);
        sycl::half* b = sycl::malloc_device<sycl::half>((size_t) F * E, q);
        std::vector<uint16_t> ha((size_t) F * E), hb((size_t) F * E);
        int bad = 0;
        for (int k = 0; k < NX; ++k) {
            const int mats[3][4] = {{ex[k].gu, 0, F, E}, {ex[k].gu, (int) ex[k].up_off, F, E}, {ex[k].dt, (int) ex[k].down_off, E, F}};
            for (const auto& m : mats) {
                const size_t row_b = (size_t) (m[3] / qk(m[0])) * bbytes(m[0]);
                xe_dequant_rows_f16(q, m[0], draw[k] + m[1], m[2], (int) row_b, m[3], a);
                xe_dequant_planar_f16(q, m[0], dpl[k] + m[1], m[2], m[3], b).wait();
                q.memcpy(ha.data(), a, (size_t) m[2] * m[3] * 2);
                q.memcpy(hb.data(), b, (size_t) m[2] * m[3] * 2).wait();
                for (size_t i = 0; i < (size_t) m[2] * m[3]; ++i) bad += ha[i] != hb[i];
            }
        }
        std::printf("planar fp16 dequant vs file layout: %d mismatches %s\n", bad, bad ? "FAIL" : "ok");
        sycl::free(a, q); sycl::free(b, q);
        const int NT = 3;
        std::vector<double> ref((size_t) NT * E, 0.0);
        std::vector<XeJob> jp;
        std::vector<int32_t> S((size_t) NT * 6, -1);
        std::vector<float> W((size_t) NT * 6, 0.f);
        for (int k = 0; k < NX; ++k) {   // job k = expert k with tokens (k + s) % 3; token t's slot k
            XeJob j = job(k, true);
            j.nt = NT;
            for (int s2 = 0; s2 < NT; ++s2) {
                const int t = (s2 + k) % NT;
                j.tok[s2] = t; j.w[s2] = 1.f;
                const float w = 0.1f + 0.05f * (float) (k + s2);
                S[(size_t) t * 6 + k] = k * kMaxJobTok + s2;
                W[(size_t) t * 6 + k] = w;
                ref_expert(ex[k], X.data() + (size_t) t * E, w, ref.data() + (size_t) t * E);
            }
            jp.push_back(j);
        }
        int32_t* dS = sycl::malloc_device<int32_t>(S.size(), q);
        float* dW = sycl::malloc_device<float>(W.size(), q);
        float* dY = sycl::malloc_device<float>((size_t) NX * kMaxJobTok * E, q);
        q.memcpy(dS, S.data(), S.size() * 4); q.memcpy(dW, W.data(), W.size() * 4);
        q.memcpy(djp, jp.data(), jp.size() * sizeof(XeJob)).wait();
        xe_quant_rows(q, dX, NT, E, xq);
        xe_experts_y_q8(q, djp, NX, xq, hq, dY, E, F, L, NT);
        xe_gather(q, dY, dS, dW, NT, 6, E, dO, false).wait();
        q.memcpy(got.data(), dO, (size_t) NT * E * 4).wait();
        const double r1 = rel(got.data(), ref.data(), (size_t) NT * E);
        std::printf("prompt form (experts_y_q8 + gather, 6 experts x 3 tokens): %.2e %s\n", r1, r1 < 2e-2 ? "ok" : "FAIL");
        sycl::free(dS, q); sycl::free(dW, q); sycl::free(dY, q);
    }
    {   // run-to-run: the same inputs, every token-count variant, bit-identical outputs?
        std::vector<int> nts = {1, 2, 3, 5, 8};
        for (int ntv : nts) {
            std::vector<XeJob> jp;
            for (int k = 0; k < NX; ++k) {
                XeJob j = job(k, true);
                j.nt = (uint8_t) ntv;
                for (int t = 0; t < ntv; ++t) { j.tok[t] = (t + k) % 8; j.w[t] = 0.1f * (1 + t); }
                jp.push_back(j);
            }
            q.memcpy(djp, jp.data(), jp.size() * sizeof(XeJob)).wait();
            xe_quant_rows(q, dX, 8, E, xq).wait();
            float* dY = sycl::malloc_device<float>((size_t) NX * kMaxJobTok * E, q);
            std::vector<float> ref((size_t) NX * kMaxJobTok * E), cur(ref.size()), o1((size_t) 8 * E), o2(o1.size());
            int bad_y = 0, bad_o = 0;
            for (int rep = 0; rep < 20; ++rep) {
                q.memset(dY, 0, ref.size() * 4).wait();
                xe_experts_y_q8(q, djp, NX, xq, hq, dY, E, F, L, ntv);
                q.wait();
                q.memcpy(rep ? cur.data() : ref.data(), dY, ref.size() * 4).wait();
                if (rep) for (size_t i = 0; i < ref.size(); ++i) bad_y += cur[i] != ref[i];
                xe_experts_q8(q, djp, NX, 8, xq, hq, dO, E, F, L, false, ntv).wait();
                q.memcpy(rep ? o2.data() : o1.data(), dO, o1.size() * 4).wait();
                if (rep) for (size_t i = 0; i < o1.size(); ++i) bad_o += o2[i] != o1[i];
            }
            std::printf("determinism nt %d: experts_y_q8 %d / experts_q8 %d values differ over 19 reruns %s\n", ntv, bad_y, bad_o,
                        bad_y || bad_o ? "FAIL" : "ok");
            sycl::free(dY, q);
        }
    }
    // ---- speed
    int cur = 0;   // the copy the timed call uses
    auto tm = [&](auto fn) {
        for (int i = 0; i < 5; ++i) { cur = i % NCOPY; fn(); }
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) { cur = i % NCOPY; fn(); }
        q.wait();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    };
    {   // the card's streaming read bandwidth: every lane sums 16-byte loads, consecutive lanes consecutive chunks
        const size_t nb = (size_t) 1 << 30;
        sycl::uint4* big = sycl::malloc_device<sycl::uint4>(nb / 16, q);
        uint32_t* sink = sycl::malloc_device<uint32_t>(1 << 20, q);
        q.parallel_for(sycl::range<1>(nb / 16), [=](sycl::id<1> i) {   // incompressible
            uint32_t x = (uint32_t) i[0] * 2654435761u;
            x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
            big[i] = sycl::uint4(x, x * 3u + 1u, x ^ 0xdeadbeefu, x * 7u);
        }).wait();
        const size_t nwi = (size_t) 1 << 18;
        auto rd = [&] {
            q.parallel_for(sycl::nd_range<1>(nwi, 256), [=](sycl::nd_item<1> it) {
                const size_t g = it.get_global_id(0);
                uint32_t a = 0;
                for (size_t i = g; i < nb / 16; i += nwi) { const sycl::uint4 v = big[i]; a ^= v.x() + v.y() + v.z() + v.w(); }
                if (a == 0x12345678u) sink[g & 1023] = a;
            });
        };
        for (int i = 0; i < 3; ++i) rd();
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) rd();
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
        std::printf("streaming read: %.0f GB/s\n", nb / us / 1e3);
        sycl::free(big, q); sycl::free(sink, q);
    }
    // per class, one token: gate/up and down alone (4 copies of the one expert in a call: enough work to fill the card)
    for (int k = 0; k < NX; ++k) {
        std::vector<XeJob> jp;
        for (int cp = 0; cp < NCOPY; ++cp)
            for (int r = 0; r < 4; ++r) {
                XeJob b = job(k, true, (cp + r * 3) % NCOPY);
                b.nt = 1; b.tok[0] = 0; b.w[0] = 0.1f;
                jp.push_back(b);
            }
        q.memcpy(djp, jp.data(), jp.size() * sizeof(XeJob)).wait();
        auto JP = [&] { return djp + (size_t) cur * 4; };
        xe_quant_rows(q, dX, 1, E, xq).wait();
        const double tg = tm([&] { xp::gate_up_q8<1>(q, JP(), 4, xq, hq, E, F, L); });
        const double td = tm([&] { xp::down_q8<1, 1>(q, JP(), 4, 1, hq, dO, E, F, false); });
        const double gb = 4.0 * ex[k].down_off, dbb = 4.0 * (ex[k].bytes - ex[k].down_off);
        std::printf("class gu %3d / d %3d, 4 experts: gate_up %6.1f us (%3.0f GB/s), down %6.1f us (%3.0f GB/s)\n", ex[k].gu, ex[k].dt,
                    tg, gb / tg / 1e3, td, dbb / td / 1e3);
    }
    for (int NTk : {2, 4}) {   // per class, NTk tokens per job
        for (int k = 0; k < NX; ++k) {
            std::vector<XeJob> jp;
            for (int cp = 0; cp < NCOPY; ++cp)
                for (int r = 0; r < 4; ++r) {
                    XeJob b = job(k, true, (cp + r * 3) % NCOPY);
                    b.nt = (uint8_t) NTk;
                    for (int t = 0; t < NTk; ++t) { b.tok[t] = t; b.w[t] = 0.1f; }
                    jp.push_back(b);
                }
            q.memcpy(djp, jp.data(), jp.size() * sizeof(XeJob)).wait();
            auto JP = [&] { return djp + (size_t) cur * 4; };
            xe_quant_rows(q, dX, NTk, E, xq).wait();
            double tg, td;
            if (NTk == 2) {
                tg = tm([&] { xp::gate_up_q8<2>(q, JP(), 4, xq, hq, E, F, L); });
                td = tm([&] { xp::down_q8<2, 8>(q, JP(), 4, NTk, hq, dO, E, F, false); });
            } else {
                tg = tm([&] { xp::gate_up_q8<4>(q, JP(), 4, xq, hq, E, F, L); });
                td = tm([&] { xp::down_q8<4, 8>(q, JP(), 4, NTk, hq, dO, E, F, false); });
            }
            const double gb = 4.0 * ex[k].down_off, dbb = 4.0 * (ex[k].bytes - ex[k].down_off);
            std::printf("class gu %3d / d %3d, 4 experts x %d tok: gate_up %6.1f us (%3.0f GB/s), down %6.1f us (%3.0f GB/s)\n", ex[k].gu, ex[k].dt,
                        NTk, tg, gb / tg / 1e3, td, dbb / td / 1e3);
        }
    }
    struct Case { const char* name; int nexp; int ntok; int nt[6]; };
    const Case cases[] = {{"decode, 1 expert", 1, 1, {1}}, {"decode, 2 experts (one card)", 2, 1, {1, 1}},
                          {"decode, 6 experts", 6, 1, {1, 1, 1, 1, 1, 1}},
                          {"verify 5 tok, 6 experts x 1", 6, 5, {1, 1, 1, 1, 1, 1}},
                          {"verify 5 tok, nt 1,1,2,1,3,2", 6, 5, {1, 1, 2, 1, 3, 2}},
                          {"verify, 6 experts x 2 tokens", 6, 2, {2, 2, 2, 2, 2, 2}}, {"verify, 6 experts x 5 tokens", 6, 5, {5, 5, 5, 5, 5, 5}},
                          {"verify, 6 experts x 8 tokens", 6, 8, {8, 8, 8, 8, 8, 8}}};
    for (const Case& c : cases) {
        std::vector<XeJob> jr, jp;
        size_t bytes = 0;
        int max_nt = 0;
        for (int cp = 0; cp < NCOPY; ++cp)
            for (int k = 0; k < c.nexp; ++k) {
                XeJob a = job(k, false, cp), b = job(k, true, cp);
                a.nt = b.nt = (uint8_t) c.nt[k];
                max_nt = std::max(max_nt, c.nt[k]);
                for (int s = 0; s < c.nt[k]; ++s) { a.tok[s] = b.tok[s] = (s + k) % c.ntok; a.w[s] = b.w[s] = 0.1f; }
                jr.push_back(a); jp.push_back(b);
                if (cp == 0) bytes += ex[k].bytes;
            }
        q.memcpy(djr, jr.data(), jr.size() * sizeof(XeJob)); q.memcpy(djp, jp.data(), jp.size() * sizeof(XeJob)).wait();
        auto JR = [&] { return djr + (size_t) cur * c.nexp; };
        auto JP = [&] { return djp + (size_t) cur * c.nexp; };
        const double t0 = tm([&] { xe_experts(q, JR(), c.nexp, c.ntok, dX, dH, dO, E, F, L, false); });
        const double t1 = tm([&] { xe_quant_rows(q, dX, c.ntok, E, xq); xe_experts_q8(q, JP(), c.nexp, c.ntok, xq, hq, dO, E, F, L, false, max_nt); });
        const double tq = tm([&] { xe_quant_rows(q, dX, c.ntok, E, xq); });
        double tg, td;
        if (max_nt <= 1) {
            tg = tm([&] { xp::gate_up_q8<1>(q, JP(), c.nexp, xq, hq, E, F, L); });
            td = c.ntok <= 1 ? tm([&] { xp::down_q8<1, 1>(q, JP(), c.nexp, c.ntok, hq, dO, E, F, false); })
                             : tm([&] { xp::down_q8<1, 8>(q, JP(), c.nexp, c.ntok, hq, dO, E, F, false); });
        } else if (max_nt <= 2) {
            tg = tm([&] { xp::gate_up_q8<2>(q, JP(), c.nexp, xq, hq, E, F, L); });
            td = tm([&] { xp::down_q8<2, 8>(q, JP(), c.nexp, c.ntok, hq, dO, E, F, false); });
        } else if (max_nt <= 4) {
            tg = tm([&] { xp::gate_up_q8<4>(q, JP(), c.nexp, xq, hq, E, F, L); });
            td = tm([&] { xp::down_q8<4, 8>(q, JP(), c.nexp, c.ntok, hq, dO, E, F, false); });
        } else {
            tg = tm([&] { xp::gate_up_q8<8>(q, JP(), c.nexp, xq, hq, E, F, L); });
            td = tm([&] { xp::down_q8<8, 8>(q, JP(), c.nexp, c.ntok, hq, dO, E, F, false); });
        }
        size_t gub = 0, db = 0;
        for (int k = 0; k < c.nexp; ++k) { gub += ex[k].down_off; db += ex[k].bytes - ex[k].down_off; }
        std::printf("%-30s %5.1f MiB: f32 %7.1f us (%4.0f GB/s) | int8 %7.1f us (%4.0f GB/s) x%.2f  [quant %.1f, gate_up %.1f (%3.0f GB/s), down %.1f (%3.0f GB/s)]\n",
                    c.name, bytes / 1048576.0, t0, bytes / t0 / 1e3, t1, bytes / t1 / 1e3, t0 / t1, tq, tg, gub / tg / 1e3, td, db / td / 1e3);
    }
    return 0;
}
