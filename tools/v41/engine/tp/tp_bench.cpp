// tools/v41/engine/tp/tp_bench.cpp - larry: engine1 phase A - the TP decode core on the real dense weights, synthetic
// persistent state, N cards; times passes (eager or graph replay) and compares a pass's logits with a reference run.
//
//   tp_bench -m GGUF [-N cards] [-L layers] [--ctx C] [--pos P] [-n tokens] [--iters I] [--graph 0|1]
//            [--cache DIR] [--dump FILE] [--cmp FILE] [--seed S]
//
// --dump writes the FIRST pass's logits (after fill_state); --cmp compares them with a dump (N = 1 vs N = 4: only the
// split sums' association differs).  GPU clocks pinned, service stopped (one tenant a GPU).
#include "tp_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::v41;

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    Ds41TpConfig cfg;
    int64_t pos = 8192;
    int n = 1, iters = 50;
    bool graph = true;
    std::string dump, cmp;
    uint32_t seed = 1234;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> const char * { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "-m") cfg.gguf = nx();
        else if (a == "-N") cfg.n_cards = std::atoi(nx());
        else if (a == "-L") cfg.n_layer = std::atoi(nx());
        else if (a == "--ctx") cfg.ctx_max = std::atoll(nx());
        else if (a == "--pos") pos = std::atoll(nx());
        else if (a == "-n") n = std::atoi(nx());
        else if (a == "--iters") iters = std::atoi(nx());
        else if (a == "--graph") graph = std::atoi(nx()) != 0;
        else if (a == "--cache") cfg.cache_dir = nx();
        else if (a == "--dump") dump = nx();
        else if (a == "--cmp") cmp = nx();
        else if (a == "--seed") seed = (uint32_t) std::atoll(nx());
        else if (a == "--no-shexp") cfg.shexp = false;
        else if (a == "--experts") cfg.bench_experts = std::atoi(nx());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (cfg.gguf.empty()) { std::fprintf(stderr, "usage: tp_bench -m GGUF [-N cards] ...\n"); return 2; }
    Ds41Tp tp;
    std::string err;
    if (!tp.init(cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const int64_t V = tp.vocab();
    if (std::getenv("TP_WARM")) {   // diagnostic: one throw-away pass first (its waits may give up while modules load)
        std::vector<int> w(1, 1000);
        if (!tp.pass(w.data(), 1, (int) pos, nullptr, false)) std::printf("warm-up pass: %s\n", tp.last_error().c_str());
    }
    tp.fill_state(pos, seed);
    std::vector<int> tids;
    for (int t = 0; t < n; ++t) tids.push_back(1000 + 37 * t);
    std::vector<float> lg((size_t) n * V);
    // the first pass right after fill_state: the compared one (eager - no recording in the way)
    if (!tp.pass(tids.data(), n, (int) pos, lg.data(), false)) { std::fprintf(stderr, "pass: %s\n", tp.last_error().c_str()); return 1; }
    {
        double s = 0, s2 = 0;
        int nan = 0;
        for (float v : lg) { if (!std::isfinite(v)) ++nan; else { s += v; s2 += (double) v * v; } }
        int am = 0;
        for (int64_t i = 1; i < V; ++i) if (lg[(size_t) i] > lg[(size_t) am]) am = (int) i;
        std::printf("first pass: logits mean %.5f rms %.5f, non-finite %d, argmax(row 0) %d (%.4f)\n", s / lg.size(),
                    std::sqrt(s2 / lg.size()), nan, am, lg[(size_t) am]);
    }
    if (const char * dd = std::getenv("TP_DUMPDIR")) tp.dump(dd, n);
    if (!dump.empty()) {
        FILE * f = std::fopen(dump.c_str(), "wb");
        if (!f || std::fwrite(lg.data(), 4, lg.size(), f) != lg.size()) { std::fprintf(stderr, "cannot write %s\n", dump.c_str()); return 1; }
        std::fclose(f);
    }
    if (!cmp.empty()) {
        std::vector<float> ref(lg.size());
        FILE * f = std::fopen(cmp.c_str(), "rb");
        if (!f || std::fread(ref.data(), 4, ref.size(), f) != ref.size()) { std::fprintf(stderr, "cannot read %s\n", cmp.c_str()); return 1; }
        std::fclose(f);
        for (int t = 0; t < n; ++t) {
            double md = 0, mr = 0;
            int a0 = 0, a1 = 0;
            const float * x = lg.data() + (size_t) t * V, * y = ref.data() + (size_t) t * V;
            for (int64_t i = 0; i < V; ++i) {
                md = std::max(md, (double) std::fabs(x[i] - y[i]));
                mr = std::max(mr, (double) std::fabs(y[i]));
                if (x[i] > x[a0]) a0 = (int) i;
                if (y[i] > y[a1]) a1 = (int) i;
            }
            std::printf("row %d: max |diff| %.3g (max |ref| %.3g), argmax %d vs %d\n", t, md, mr, a0, a1);
        }
    }
    // timing
    for (int w = 0; w < 3; ++w)
        if (!tp.pass(tids.data(), n, (int) pos, nullptr, graph)) { std::fprintf(stderr, "pass: %s\n", tp.last_error().c_str()); return 1; }
    std::vector<double> us;
    for (int it = 0; it < iters; ++it) {
        const double t0 = now_us();
        if (!tp.pass(tids.data(), n, (int) pos, nullptr, graph)) { std::fprintf(stderr, "pass: %s\n", tp.last_error().c_str()); return 1; }
        us.push_back(now_us() - t0);
    }
    std::sort(us.begin(), us.end());
    const double med = us[us.size() / 2];
    std::printf("N %d, %d layers, n %d, pos %lld, %s: pass median %.1f us (min %.1f, max %.1f) = %.1f us a layer, %.1f tok/s\n",
                cfg.n_cards, tp.n_layer(), n, (long long) pos, graph ? "graph" : "eager", med, us.front(), us.back(),
                med / tp.n_layer(), 1e6 / med * n);
    return 0;
}
