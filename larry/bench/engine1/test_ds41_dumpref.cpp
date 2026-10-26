// test_ds41_dumpref.cpp - larry, engine1 debug: a STRATA_DUMP_COL dump (per layer of one prompt column: ids[6],
// wts[6], the FFN input fn[E], the routed sum the tier gave) against the F32 reference experts from the GGUF.
//
//   dumpref V41-GSQ.gguf DUMP [--threads N] [--search L]
//
// Per layer: rel = ||routed - sum_j w_j E_j(fn)|| / ||sum||.  --search L: for the routed sum of layer L, which
// expert's output (any expert of layers L-1..L+2, same class as the slot it would sit in) explains the difference
// best if it had stood in for one of the six (a slot holding another expert's weights).
#include "ds41_moe.hpp"

#include "ggml.h"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/ptq1_0.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace strata::v41;
namespace cpu = strata::kernels::cpu;

namespace {

constexpr int kPtq1 = 143;

void dequant(int type, const uint8_t* src, float* dst, int64_t n) {
    if (type == kPtq1) { cpu::ptq1_0_dequant(src, dst, n); return; }
    ggml_get_type_traits((ggml_type) type)->to_float(src, dst, n);
}

/// F32 reference (as test_ds41_xe): y = Wd (silu(min(Wg x, L)) * clamp(Wu x, -L, L))
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

double nrm(const std::vector<double>& a) { double s = 0; for (double v : a) s += v * v; return std::sqrt(s); }

struct Rec {
    int layer = -1;
    int ids[6] = {};
    float wts[6] = {};
    std::vector<float> fn, ro;
};

template <class Fn> void parallel(int n, int threads, Fn fn) {
    std::atomic<int> next{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([&]() { for (int i; (i = next.fetch_add(1)) < n;) fn(i); });
    for (auto& t : ts) t.join();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: dumpref GGUF DUMP [--threads N] [--search L]\n"); return 2; }
    int threads = 8, search = -1;
    for (int i = 3; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (k == "--search" && i + 1 < argc) search = std::atoi(argv[++i]);
    }
    Ds41FileBlobs src(argv[1]);
    std::string err;
    if (!src.open(err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom& g = src.geom();
    const int64_t E = g.n_embd;
    const float L = g.swiglu_limit;
    std::vector<cpu::NativeFmt> fl((size_t) g.n_classes());
    for (int64_t c = 0; c < g.n_classes(); ++c)
        if (!ds41_moe_class_layout(g, c, fl[(size_t) c], err)) { std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1; }

    std::vector<Rec> recs;
    {
        FILE* fp = std::fopen(argv[2], "rb");
        if (!fp) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        for (;;) {
            Rec r;
            r.fn.resize((size_t) E); r.ro.resize((size_t) E);
            if (std::fread(&r.layer, 4, 1, fp) != 1) break;
            if (std::fread(r.ids, 4, 6, fp) != 6 || std::fread(r.wts, 4, 6, fp) != 6 ||
                std::fread(r.fn.data(), 4, (size_t) E, fp) != (size_t) E || std::fread(r.ro.data(), 4, (size_t) E, fp) != (size_t) E) break;
            recs.push_back(std::move(r));
        }
        std::fclose(fp);
    }
    std::printf("%zu records, n_embd %lld, swiglu limit %g\n", recs.size(), (long long) E, (double) L);

    auto expert_out = [&](int l, int e, const float* x, std::vector<float>& y) {
        const cpu::NativeFmt& f = fl[(size_t) g.cls(l, e)];
        std::vector<uint8_t> blob(f.bytes);
        src.read(l, e, blob.data());
        y.assign((size_t) E, 0.f);
        ref_expert(f, blob.data(), x, L, y.data());
    };

    // every record: the six experts' reference outputs
    std::vector<std::vector<float>> outs(recs.size() * 6);
    parallel((int) outs.size(), threads, [&](int i) {
        const Rec& r = recs[(size_t) i / 6];
        expert_out(r.layer, r.ids[i % 6], r.fn.data(), outs[(size_t) i]);
    });
    for (size_t ri = 0; ri < recs.size(); ++ri) {
        const Rec& r = recs[ri];
        std::vector<double> s((size_t) E, 0.0), d((size_t) E);
        for (int j = 0; j < 6; ++j)
            for (int64_t i = 0; i < E; ++i) s[(size_t) i] += (double) r.wts[j] * outs[ri * 6 + j][(size_t) i];
        for (int64_t i = 0; i < E; ++i) d[(size_t) i] = r.ro[(size_t) i] - s[(size_t) i];
        std::vector<double> ro(r.ro.begin(), r.ro.end());
        const double rel = nrm(d) / std::max(nrm(s), 1e-30);
        std::printf("layer %2d  rel %.3e  |routed| %9.3f  |ref| %9.3f  classes", r.layer, rel, nrm(ro), nrm(s));
        for (int j = 0; j < 6; ++j) std::printf(" %d", g.cls(r.layer, r.ids[j]));
        std::printf("%s\n", rel > 0.1 ? "   <== BAD" : "");
        if (rel > 0.1) {
            // per expert: |w_j E_j|, cos(d, E_j)
            for (int j = 0; j < 6; ++j) {
                std::vector<double> e(outs[ri * 6 + j].begin(), outs[ri * 6 + j].end());
                double dot = 0;
                for (int64_t i = 0; i < E; ++i) dot += d[(size_t) i] * e[(size_t) i];
                std::printf("    expert %3d (class %d) w %.4f  |E| %8.3f  |wE| %7.3f  cos(d, E) %+.3f\n", r.ids[j],
                            g.cls(r.layer, r.ids[j]), r.wts[j], nrm(e), r.wts[j] * nrm(e), dot / (nrm(d) * nrm(e) + 1e-30));
            }
        }
    }

    if (search >= 0) {
        size_t ri = recs.size();
        for (size_t i = 0; i < recs.size(); ++i) if (recs[i].layer == search) ri = i;
        if (ri == recs.size()) { std::fprintf(stderr, "no record for layer %d\n", search); return 1; }
        const Rec& r = recs[ri];
        // Y_k = what expert k's slot gave, if it alone was wrong: (routed - sum_{j != k} w_j E_j) / w_k
        std::vector<std::vector<double>> Y(6, std::vector<double>((size_t) E));
        for (int k = 0; k < 6; ++k)
            for (int64_t i = 0; i < E; ++i) {
                double s = r.ro[(size_t) i];
                for (int j = 0; j < 6; ++j) if (j != k) s -= (double) r.wts[j] * outs[ri * 6 + j][(size_t) i];
                Y[(size_t) k][(size_t) i] = s / r.wts[k];
            }
        for (int k = 0; k < 6; ++k) {
            std::vector<double> e(outs[ri * 6 + k].begin(), outs[ri * 6 + k].end());
            std::printf("slot of expert %3d: |Y| %9.3f (its own |E| %8.3f)\n", r.ids[k], nrm(Y[(size_t) k]), nrm(e));
        }
        struct Cand { int l, e; };
        std::vector<Cand> cands;
        for (int l = search - 1; l <= search + 2; ++l)
            if (l >= 0 && l < g.n_layers)
                for (int e = 0; e < g.n_experts; ++e) cands.push_back({l, e});
        std::vector<std::vector<float>> co(cands.size());
        std::printf("search: %zu candidate experts (layers %d..%d) on the dumped fn ...\n", cands.size(), search - 1, search + 2);
        std::fflush(stdout);
        parallel((int) cands.size(), threads, [&](int i) { expert_out(cands[(size_t) i].l, cands[(size_t) i].e, r.fn.data(), co[(size_t) i]); });
        for (int k = 0; k < 6; ++k) {
            double best = 1e30; int bi = -1;
            std::vector<std::pair<double, int>> top;
            for (size_t ci = 0; ci < cands.size(); ++ci) {
                double num = 0, den = 0;
                for (int64_t i = 0; i < E; ++i) {
                    const double dv = Y[(size_t) k][(size_t) i] - co[ci][(size_t) i];
                    num += dv * dv; den += Y[(size_t) k][(size_t) i] * Y[(size_t) k][(size_t) i];
                }
                const double rr = std::sqrt(num / std::max(den, 1e-30));
                top.push_back({rr, (int) ci});
                if (rr < best) { best = rr; bi = (int) ci; }
            }
            std::sort(top.begin(), top.end());
            std::printf("slot of expert %3d (class %d): best stand-ins", r.ids[k], g.cls(r.layer, r.ids[k]));
            for (int t = 0; t < 3; ++t)
                std::printf("  L%d e%d (class %d) rel %.3f", cands[(size_t) top[(size_t) t].second].l, cands[(size_t) top[(size_t) t].second].e,
                            g.cls(cands[(size_t) top[(size_t) t].second].l, cands[(size_t) top[(size_t) t].second].e), top[(size_t) t].first);
            std::printf("\n");
            (void) bi;
        }
    }
    return 0;
}
