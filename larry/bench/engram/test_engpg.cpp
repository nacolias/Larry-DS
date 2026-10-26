// test_engpg.cpp GGUF SEQ.i32 DIRS_A DIRS_B - larry: the Engram reader over two table sets (e.g. the safetensors shards
// vs a paged engram-L1.epg first) must give the same rows, byte for byte: rows() (the batch reader), start_into() +
// wait_slot() per slot, prefetch() + rows() (the row cache), and rows_reference(), at several (pos0, n).
#include "ds41_engram.hpp"
#include "ds41_geometry.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::v41;

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> v;
    for (size_t a = 0, b; a <= s.size(); a = b + 1) { b = s.find(':', a); if (b == std::string::npos) b = s.size(); if (b > a) v.push_back(s.substr(a, b - a)); }
    return v;
}

static int cmp(const char * what, const std::vector<std::vector<float>> & x, const std::vector<std::vector<float>> & y) {
    int bad = 0;
    for (size_t s = 0; s < x.size(); ++s)
        if (s >= y.size() || x[s].size() != y[s].size() || std::memcmp(x[s].data(), y[s].data(), x[s].size() * sizeof(float)) != 0) ++bad;
    std::printf("  %-34s %s\n", what, bad ? "MISMATCH" : "identical");
    return bad;
}

int main(int argc, char ** argv) {
    if (argc < 5) { std::fprintf(stderr, "test_engpg GGUF SEQ.i32 DIRS_A DIRS_B\n"); return 2; }
    GgufModel m = GgufModel::open(argv[1]);
    Ds41Geometry G;
    std::string e = ds41_read_geometry(m, G);
    if (!e.empty()) { std::fprintf(stderr, "geometry: %s\n", e.c_str()); return 1; }
    std::vector<int32_t> seq;
    { std::ifstream f(argv[2], std::ios::binary); int32_t x; while (f.read((char *) &x, 4)) seq.push_back(x); }
    Ds41Engram A, B;
    if (!A.open(G, split(argv[3]), 64, e) || !B.open(G, split(argv[4]), 64, e)) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    for (size_t p = 0; p < seq.size(); ++p) { A.set_token((int64_t) p, seq[p]); B.set_token((int64_t) p, seq[p]); }
    int bad = 0;
    const int cases[][2] = { { 0, 1 }, { 0, 7 }, { 0, 64 }, { 100, 513 }, { 0, 4096 }, { 2000, 16000 } };
    for (const auto & c : cases) {
        const int pos0 = c[0], n = c[1];
        if ((size_t) (pos0 + n) > seq.size()) continue;
        std::printf("pos0 %d n %d\n", pos0, n);
        std::vector<std::vector<float>> a, b, r, s;
        if (!A.rows(pos0, n, a, e) || !B.rows(pos0, n, b, e)) { std::fprintf(stderr, "rows: %s\n", e.c_str()); return 1; }
        bad += cmp("rows A vs B", a, b);
        // start_into + wait_slot, slot by slot (the dense half's prompt path)
        const size_t per = (size_t) n * (size_t) (B.n_cols() * B.row_dim());
        std::vector<std::vector<float>> o(2, std::vector<float>(per, -1.0f));
        float * op[2] = { o[0].data(), o[1].data() };
        auto bh = B.start_into(pos0, n, op, e);
        if (!bh || !B.wait_slot(bh, 0, e) || !B.wait_slot(bh, 1, e) || !B.wait(bh, e)) { std::fprintf(stderr, "start_into: %s\n", e.c_str()); return 1; }
        bad += cmp("start_into+wait_slot B vs rows A", a, o);
        if (n <= 4096) {
            if (!B.rows_reference(pos0, n, r, e)) { std::fprintf(stderr, "reference: %s\n", e.c_str()); return 1; }
            bad += cmp("rows_reference B vs rows A", a, r);
        }
    }
    // the row cache: a prefetch fills it, rows() then reads it
    {
        const int pos0 = 20, n = 3000;
        auto pf = B.prefetch(pos0, n, e);
        if (pf && !B.wait(pf, e)) { std::fprintf(stderr, "prefetch: %s\n", e.c_str()); return 1; }
        std::vector<std::vector<float>> a, b;
        if (!A.rows(pos0, n, a, e) || !B.rows(pos0, n, b, e)) return 1;
        std::printf("after a prefetch (cache hits %llu)\n", (unsigned long long) B.cache_hits());
        bad += cmp("rows A vs B (cached)", a, b);
    }
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
