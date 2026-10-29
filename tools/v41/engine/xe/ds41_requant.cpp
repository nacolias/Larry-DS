// tools/v41/engine/xe/ds41_requant.cpp - larry, 2026-10-29 (optimization #4: fewer bytes per cold expert): the coldest
// experts' Q3_K parts requantized to Q2_0 (or every part to PTQ1_0) into a sidecar file the engine reads in their place
// (STRATA_BLOB_RQ=<file>, patch-blobrq.py: Ds41FileBlobs gives each such expert its new class and reads its blob there).
//
//   ds41_requant MODEL.gguf RANK.bin OUT [--from-rank R] [--to q2_0|ptq1] [--threads N] [--max N]
//
// Experts at rank position >= R (rank.bin: int32 (layer, expert) pairs, hottest first - the placement's order) are the
// ones no card keeps in VRAM; the CPU pool runs them (and the prompts stream them), so its DRAM traffic is their bytes.
// --to q2_0 (default): gate / up / down parts in Q3_K become Q2_0 (3.4375 -> 2.25 bits a weight); Q2_0 and PTQ1_0 parts
// are copied.  --to ptq1: every Q3_K or Q2_0 part becomes PTQ1_0 (1.75).  An expert with nothing to change is left out.
//
// The quantizers search each block for the least squared error - not ggml's reference Q2_0 (its scale is the block's
// max |w|, so its fourth level, +2 d, is never used) nor Strata's reference PTQ1_0 (amax scale):
//   Q2_0 (64 values, levels {-1, 0, 1, 2} * d, d of either sign): from d = +-f * (the side's largest |w|) for a few f,
//        alternate round-to-level and the least-squares d twice; the best of all, d to fp16 and the levels again.
//   PTQ1_0 (128 values, {-1, 0, 1} * d): the k largest |w| non-zero for the k with the least error (d = their mean).
// The source is the expert as the GGUF has it (Q3_K dequantised): the error adds to the GSQ quant's own.
//
// File: a 4096-byte-aligned header {"S41RQNT1", u64 model file size, u64 n, u64 data start} then n x {i32 layer,
// i32 expert, i32 gate/up type, i32 down type, u64 offset}; the blobs [gate | up | down] each 4096-aligned.
#include "ds41_moe.hpp"
#include "strata/kernels/cpu/ptq1_0.hpp"
#include "ggml.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

using namespace strata::v41;
namespace cpu = strata::kernels::cpu;

namespace {

constexpr int kQ3K = 11, kQ20 = 42, kPtq1 = 143;   // GGML_TYPE_Q3_K, GGML_TYPE_Q2_0, Strata's PTQ1_0
constexpr int kQ20Block = 64, kQ20Bytes = 18, kPtqBlock = 128, kPtqBytes = 28;

uint64_t row_bytes(int t, int64_t n) {
    if (t == kQ20) return (uint64_t) (n / kQ20Block) * kQ20Bytes;
    if (t == kPtq1) return (uint64_t) (n / kPtqBlock) * kPtqBytes;
    return (uint64_t) ggml_row_size((ggml_type) t, n);
}

void dequant_row(int t, const uint8_t* src, float* y, int64_t n) {
    if (t == kPtq1) { cpu::ptq1_0_dequant(src, y, n); return; }
    ggml_get_type_traits((ggml_type) t)->to_float(src, y, n);
}

uint16_t f2h(float f) { return ggml_fp32_to_fp16(f); }
float h2f(uint16_t h) { return ggml_fp16_to_fp32(h); }

/// One Q2_0 block: levels {-1, 0, 1, 2} * d, the least squared error over a few starts (both signs of d).
void q20_block(const float* w, uint8_t* dst) {
    float best_err = INFINITY, best_d = 0.f;
    float mx = 0.f, mn = 0.f;
    for (int j = 0; j < kQ20Block; ++j) { mx = std::max(mx, w[j]); mn = std::min(mn, w[j]); }
    static const float fs[] = {0.40f, 0.45f, 0.50f, 0.55f, 0.62f, 0.75f};
    for (int sgn = 0; sgn < 2; ++sgn) {
        const float side = sgn == 0 ? mx : -mn;   // the extreme the "2" level reaches
        if (side <= 0.f) continue;
        for (float f : fs) {
            float d = (sgn == 0 ? 1.f : -1.f) * f * side;
            for (int it = 0; it < 3; ++it) {
                double sq = 0, sw = 0;
                for (int j = 0; j < kQ20Block; ++j) {
                    const int q = std::clamp((int) std::lround(w[j] / d), -1, 2);
                    sw += (double) w[j] * q;
                    sq += (double) q * q;
                }
                if (sq <= 0) break;
                const float nd = (float) (sw / sq);
                if (nd == 0.f || (nd > 0) != (d > 0)) break;
                d = nd;
            }
            double err = 0;
            for (int j = 0; j < kQ20Block; ++j) {
                const int q = std::clamp((int) std::lround(w[j] / d), -1, 2);
                const double r = w[j] - (double) q * d;
                err += r * r;
            }
            if (err < best_err) { best_err = (float) err; best_d = d; }
        }
    }
    const uint16_t hd = f2h(best_d);
    const float d = h2f(hd);
    std::memcpy(dst, &hd, 2);
    uint8_t* qs = dst + 2;
    std::memset(qs, 0, kQ20Block / 4);
    for (int j = 0; j < kQ20Block; ++j) {
        const int q = d != 0.f ? std::clamp((int) std::lround(w[j] / d), -1, 2) : 0;
        qs[j / 4] |= (uint8_t) ((q + 1) << ((j % 4) * 2));
    }
}

/// One PTQ1_0 block: the k largest |w| become +-d (d their mean), the rest 0, for the k with the least error.
void ptq1_block(const float* w, uint8_t* dst) {
    float a[kPtqBlock];
    int idx[kPtqBlock];
    for (int j = 0; j < kPtqBlock; ++j) { a[j] = std::fabs(w[j]); idx[j] = j; }
    std::sort(idx, idx + kPtqBlock, [&](int x, int y) { return a[x] > a[y]; });
    double tot2 = 0;
    for (int j = 0; j < kPtqBlock; ++j) tot2 += (double) a[j] * a[j];
    double s = 0, s2 = 0, best = tot2;
    int bk = 0;
    for (int k = 1; k <= kPtqBlock; ++k) {
        const double v = a[idx[k - 1]];
        s += v;
        s2 += v * v;
        const double d = s / k;   // the least-squares d for these k
        const double err = (s2 - 2 * d * s + k * d * d) + (tot2 - s2);
        if (err < best) { best = err; bk = k; }
    }
    double sum = 0;
    for (int k = 0; k < bk; ++k) sum += a[idx[k]];
    const float d = bk ? (float) (sum / bk) : 0.f;
    int8_t t[kPtqBlock] = {0};
    for (int k = 0; k < bk; ++k) t[idx[k]] = w[idx[k]] >= 0 ? 1 : -1;
    const float dh = h2f(f2h(d));
    cpu::ptq1_0_pack(t, &dh, dst, kPtqBlock);
}

void quant_row(int t, const float* x, uint8_t* dst, int64_t n) {
    if (t == kQ20) {
        for (int64_t b = 0; b < n / kQ20Block; ++b) q20_block(x + b * kQ20Block, dst + b * kQ20Bytes);
    } else {
        for (int64_t b = 0; b < n / kPtqBlock; ++b) ptq1_block(x + b * kPtqBlock, dst + b * kPtqBytes);
    }
}

struct Ent { int32_t l, e, gu, d; uint64_t off; };

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: ds41_requant MODEL.gguf RANK.bin OUT [--from-rank R] [--to q2_0|ptq1] [--threads N] [--max N]\n");
        return 2;
    }
    const std::string model = argv[1], rank = argv[2], out = argv[3];
    int64_t from = 9600, maxn = -1;
    int threads = 8;
    std::string to = "q2_0";
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--from-rank") from = std::atoll(argv[i + 1]);
        else if (k == "--to") to = argv[i + 1];
        else if (k == "--threads") threads = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--max") maxn = std::atoll(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    if (to != "q2_0" && to != "ptq1") { std::fprintf(stderr, "--to q2_0 or ptq1\n"); return 2; }
    Ds41FileBlobs src(model);
    std::string err;
    if (!src.open(err)) { std::fprintf(stderr, "open %s: %s\n", model.c_str(), err.c_str()); return 1; }
    const Ds41MoeGeom& g = src.geom();
    const auto ranked = ds41_read_ranked(rank, err);
    if (ranked.empty()) { std::fprintf(stderr, "rank: %s\n", err.c_str()); return 1; }
    const int64_t D = g.n_embd, FF = g.n_ff;
    auto map_t = [&](int t) {
        if (to == "q2_0") return t == kQ3K ? kQ20 : t;
        return (t == kQ3K || t == kQ20) ? kPtq1 : t;
    };
    std::vector<Ent> ents;
    uint64_t in_bytes = 0;
    for (int64_t r = from; r < (int64_t) ranked.size(); ++r) {
        const int32_t l = ranked[(size_t) r].first, e = ranked[(size_t) r].second;
        if (l < 0 || l >= g.n_layers || e < 0 || e >= g.n_experts) continue;
        const int c = g.cls(l, e), gu = g.cls_gu[(size_t) c], dt = g.cls_d[(size_t) c];
        const int ngu = map_t(gu), nd = map_t(dt);
        if (ngu == gu && nd == dt) continue;
        ents.push_back({l, e, ngu, nd, 0});
        in_bytes += src.expert_bytes(l, e);
        if (maxn > 0 && (int64_t) ents.size() >= maxn) break;
    }
    auto blob_new = [&](const Ent& x) { return 2 * row_bytes(x.gu, D) * (uint64_t) FF + row_bytes(x.d, FF) * (uint64_t) D; };
    const uint64_t head = 32 + (uint64_t) ents.size() * sizeof(Ent);
    uint64_t at = (head + 4095) & ~(uint64_t) 4095;
    const uint64_t data = at;
    uint64_t out_bytes = 0;
    for (Ent& x : ents) {
        x.off = at;
        const uint64_t b = blob_new(x);
        out_bytes += b;
        at += (b + 4095) & ~(uint64_t) 4095;
    }
    std::fprintf(stderr, "ds41_requant: %zu experts from rank %lld (of %zu) to %s: %.2f GiB -> %.2f GiB (file %.2f GiB)\n",
                 ents.size(), (long long) from, ranked.size(), to.c_str(), in_bytes / 1073741824.0, out_bytes / 1073741824.0,
                 at / 1073741824.0);
    struct stat st{};
    if (::stat(model.c_str(), &st) != 0) { std::fprintf(stderr, "stat %s\n", model.c_str()); return 1; }
    const std::string tmp = out + ".part";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { std::fprintf(stderr, "cannot write %s\n", tmp.c_str()); return 1; }
    std::vector<uint8_t> hdr((size_t) data, 0);
    std::memcpy(hdr.data(), "S41RQNT1", 8);
    const uint64_t hv[3] = {(uint64_t) st.st_size, (uint64_t) ents.size(), data};
    std::memcpy(hdr.data() + 8, hv, sizeof hv);
    std::memcpy(hdr.data() + 32, ents.data(), ents.size() * sizeof(Ent));
    if (::pwrite(fd, hdr.data(), hdr.size(), 0) != (ssize_t) hdr.size()) { std::fprintf(stderr, "header write\n"); return 1; }
    std::atomic<size_t> next{0}, done{0};
    std::atomic<bool> bad{false};
    // the requantization's error: sum (w - w')^2 and sum w^2 per part kind (0 gate/up, 1 down), over every row changed
    std::mutex emu;
    double e2[2] = {0, 0}, w2[2] = {0, 0};
    auto worker = [&] {
        std::vector<uint8_t> in(src.blob_bytes()), outb;
        std::vector<float> row((size_t) std::max(D, FF)), back((size_t) std::max(D, FF));
        double le2[2] = {0, 0}, lw2[2] = {0, 0};
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= ents.size() || bad) {
                std::lock_guard<std::mutex> lk(emu);
                for (int k = 0; k < 2; ++k) { e2[k] += le2[k]; w2[k] += lw2[k]; }
                return;
            }
            const Ent& x = ents[i];
            const int c = g.cls(x.l, x.e), gu = g.cls_gu[(size_t) c], dt = g.cls_d[(size_t) c];
            src.read(x.l, x.e, in.data());
            outb.assign((size_t) blob_new(x), 0);
            // the parts: gate (FF rows of D), up (FF rows of D), down (D rows of FF)
            const uint64_t igu = row_bytes(gu, D), id = row_bytes(dt, FF), ogu = row_bytes(x.gu, D), od = row_bytes(x.d, FF);
            uint64_t ip = 0, op = 0;
            for (int part = 0; part < 3; ++part) {
                const int ti = part < 2 ? gu : dt, to2 = part < 2 ? x.gu : x.d;
                const int64_t rows = part < 2 ? FF : D, cols = part < 2 ? D : FF;
                const uint64_t irb = part < 2 ? igu : id, orb = part < 2 ? ogu : od;
                for (int64_t r = 0; r < rows; ++r) {
                    if (ti == to2) {
                        std::memcpy(outb.data() + op + (uint64_t) r * orb, in.data() + ip + (uint64_t) r * irb, (size_t) irb);
                    } else {
                        dequant_row(ti, in.data() + ip + (uint64_t) r * irb, row.data(), cols);
                        quant_row(to2, row.data(), outb.data() + op + (uint64_t) r * orb, cols);
                        dequant_row(to2, outb.data() + op + (uint64_t) r * orb, back.data(), cols);
                        const int kk = part < 2 ? 0 : 1;
                        for (int64_t j = 0; j < cols; ++j) {
                            const double dv = (double) row[(size_t) j] - back[(size_t) j];
                            le2[kk] += dv * dv;
                            lw2[kk] += (double) row[(size_t) j] * row[(size_t) j];
                        }
                    }
                }
                ip += irb * (uint64_t) rows;
                op += orb * (uint64_t) rows;
            }
            if (::pwrite(fd, outb.data(), outb.size(), (off_t) x.off) != (ssize_t) outb.size()) { bad = true; return; }
            const size_t k = done.fetch_add(1) + 1;
            if (k % 250 == 0) std::fprintf(stderr, "ds41_requant: %zu / %zu\n", k, ents.size());
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t) th.emplace_back(worker);
    for (auto& t : th) t.join();
    if (bad) { std::fprintf(stderr, "write failed\n"); ::close(fd); return 1; }
    for (int k = 0; k < 2; ++k)
        if (w2[k] > 0)
            std::fprintf(stderr, "ds41_requant: %s parts changed: relative RMS error %.4f\n", k ? "down" : "gate/up",
                         std::sqrt(e2[k] / w2[k]));
    if (::ftruncate(fd, (off_t) at) != 0 || ::fsync(fd) != 0) { std::fprintf(stderr, "fsync\n"); return 1; }
    ::close(fd);
    if (::rename(tmp.c_str(), out.c_str()) != 0) { std::fprintf(stderr, "rename\n"); return 1; }
    std::fprintf(stderr, "ds41_requant: wrote %s\n", out.c_str());
    return 0;
}
