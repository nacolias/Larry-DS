// xmx_gu_test.cpp - larry: gate_up_x (xe_xmx.cpp) alone on synthetic planar Q2_0 gate / up matrices (F = 16 rows,
// E = 256, d = 1) and 8 token rows: its H against the CPU (silu(min(g, L)) * clamp(u, -L, L)) for a 3-token job
#include "../../xolotl/tools/v41/engine/xe/xe_xmx.cpp"
#include <cstdio>
#include <random>
#include <vector>
using namespace strata::v41::xe;
int main(int argc, char** argv) {
    const int ks_mode = argc > 1 ? std::atoi(argv[1]) : 4;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    const int E = 1024, F = 16, NB = F * (E / 64);
    std::mt19937 rng(3);
    // planar Q2_0 matrix: [NB][16] code bytes, then [NB] fp16 d (= 1)
    auto make = [&](std::vector<int>& codes) {
        std::vector<uint8_t> m((size_t) NB * 18);
        codes.assign((size_t) F * E, 0);
        for (int r = 0; r < F; ++r)
            for (int b = 0; b < E / 64; ++b) {
                const int blk = r * (E / 64) + b;
                for (int u = 0; u < 4; ++u) {
                    uint32_t w = 0;
                    for (int jp = 0; jp < 16; ++jp) {
                        const int c = (int) (rng() % 4);
                        codes[(size_t) r * E + b * 64 + 16 * u + jp] = c - 1;
                        w |= (uint32_t) c << (8 * (jp % 4) + 2 * (jp / 4));
                    }
                    std::memcpy(&m[(size_t) blk * 16 + 4 * u], &w, 4);
                }
                const sycl::half one = (sycl::half) 1.0f;
                std::memcpy(&m[(size_t) 16 * NB + blk * 2], &one, 2);
            }
        return m;
    };
    std::vector<int> cg, cu;
    std::vector<uint8_t> mg = make(cg), mu = make(cu);
    std::vector<uint8_t> blob(mg.size() + mu.size());
    std::memcpy(blob.data(), mg.data(), mg.size());
    std::memcpy(blob.data() + mg.size(), mu.data(), mu.size());
    uint8_t* dblob = sycl::malloc_device<uint8_t>(blob.size(), q);
    q.memcpy(dblob, blob.data(), blob.size()).wait();
    std::vector<float> X((size_t) 8 * E);
    std::normal_distribution<float> nd(0.f, 0.05f);
    for (float& v : X) v = nd(rng);
    float* dX = sycl::malloc_device<float>(X.size(), q);
    q.memcpy(dX, X.data(), X.size() * 4).wait();
    sycl::half* xt = sycl::malloc_device<sycl::half>((size_t) 8 * E, q);
    sycl::half* ht = sycl::malloc_device<sycl::half>((size_t) F * 8, q);
    q.memset(ht, 0, (size_t) F * 8 * 2).wait();
    xe_xt(q, dX, 8, E, xt).wait();
    XeJob j;
    j.blob = dblob; j.up_off = (uint32_t) mg.size(); j.gu_type = kQ20; j.d_type = kQ20; j.nt = 3;
    for (int t = 0; t < 3; ++t) { j.tok[t] = 2 * t + 1; j.w[t] = 1.f; }   // tokens 1, 3, 5
    XeJob* dj = sycl::malloc_device<XeJob>(1, q);
    q.memcpy(dj, &j, sizeof j).wait();
    const float L = 10.0f;
    if (ks_mode == 1) gate_up_x<1>(q, dj, 1, xt, ht, E, F, L).wait();
    else gate_up_x<4>(q, dj, 1, xt, ht, E, F, L).wait();
    std::vector<sycl::half> H((size_t) F * 8);
    q.memcpy(H.data(), ht, H.size() * 2).wait();
    double md = 0;
    for (int r = 0; r < F; ++r) {
        for (int s = 0; s < 8; ++s) {
            float ref = 0.f;
            if (s < 3) {
                const int t = j.tok[s];
                double g = 0, u = 0;
                for (int k = 0; k < E; ++k) {
                    const double x = (double) (float) (sycl::half) X[(size_t) t * E + k];
                    g += cg[(size_t) r * E + k] * x; u += cu[(size_t) r * E + k] * x;
                }
                const float gg = std::fmin((float) g, L), uu = std::fmin(std::fmax((float) u, -L), L);
                ref = gg / (1.0f + std::exp(-gg)) * uu;
            }
            md = std::max(md, (double) std::fabs((float) H[(size_t) r * 8 + s] - ref));
            if (r < 2) std::printf("row %d slot %d: xmx %8.4f ref %8.4f\n", r, s, (float) H[(size_t) r * 8 + s], ref);
        }
    }
    std::printf("KS %d: max |H - ref| %.3e\n", ks_mode, md);
    return 0;
}
