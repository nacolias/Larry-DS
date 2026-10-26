// bench_attn.cpp - larry: the decode attention kernels in isolation (GPU event times, median of 100): ggml-sycl's
// dsv4dec::launch_xmx on keys gathered beforehand (+ the gather it needs, as kgather2 does it) vs dsv41dec::attn_dec
// (gather in-kernel, RoPE by table).  Shapes of a decode layer: 64 heads x 512, window 128 + K compressed keys.
//   icpx -fsycl -O3 -std=c++17 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" -I. bench_attn.cpp
#include <sycl/sycl.hpp>
#include "fattn-dsv4-dec.hpp"
#include "dsv41-dec.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

static double ev_us(const sycl::event & e) {
    return (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
            e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-3;
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main() {
    sycl::queue q { sycl::gpu_selector_v, sycl::property_list { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() } };
    std::printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int NH = 64, D = 512, SWA = 128, RAW = 144, C = 4096, ND = 64;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto dev = [&](size_t nf, float sc) {
        std::vector<float> h(nf);
        for (auto & x : h) x = sc * nd(rng);
        float * d = sycl::malloc_device<float>(nf, q);
        q.memcpy(d, h.data(), nf * 4).wait();
        return d;
    };
    for (int K : { 0, 512 }) {
        for (int n : { 1, 5 }) {
            const int M = SWA + K;
            float * qd = dev((size_t) n * NH * D, 1.0f);
            float * ring = dev((size_t) RAW * D, 1.0f);
            float * rows = dev((size_t) n * D, 1.0f);
            float * k32 = dev((size_t) n * M * D, 1.0f);
            float * msk = dev((size_t) n * M, 0.0f);
            float * tab = dev((size_t) n * ND, 0.6f);
            float * sinks = dev(NH, 1.0f);
            float * comp = dev((size_t) C * D / 2, 0.5f);   // bf16 rows: C x D halves
            std::vector<int32_t> hw((size_t) n * SWA), hs((size_t) n * std::max(K, 1));
            for (int t = 0; t < n; ++t) {
                for (int j = 0; j < SWA; ++j) hw[(size_t) t * SWA + j] = j < SWA - n ? (j * 7) % RAW : RAW + (j - (SWA - n));
                for (int i = 0; i < K; ++i) hs[(size_t) t * K + i] = (i * 13 + t) % C;
            }
            int32_t * widx = sycl::malloc_device<int32_t>(hw.size(), q);
            int32_t * sel = sycl::malloc_device<int32_t>(hs.size(), q);
            q.memcpy(widx, hw.data(), hw.size() * 4).wait();
            q.memcpy(sel, hs.data(), hs.size() * 4).wait();
            float * wm = dev((size_t) n * SWA, 0.0f);
            float * sm = dev((size_t) n * std::max(K, 1), 0.0f);
            float * o1 = sycl::malloc_device<float>((size_t) n * NH * D, q);
            float * o2 = sycl::malloc_device<float>((size_t) n * NH * D, q);
            float * part = sycl::malloc_device<float>(dsv41dec::scratch_floats(NH, M, n), q);
            std::vector<double> t_old1, t_old2, t_new1, t_new2;
            for (int it = 0; it < 120; ++it) {
                dsv4dec::Args a;
                a.q = qd; a.q_h = D; a.q_b = (size_t) NH * D;
                a.k = k32; a.k_f16 = false; a.k_b = (size_t) M * D;
                a.mask = msk; a.m_f16 = false; a.m_b = M;
                a.sinks = sinks; a.o = o1; a.D = D; a.NH = NH; a.M = M; a.n = n; a.scale = 0.044f; a.part = part;
                sycl::event f1;
                sycl::event e1 = dsv4dec::launch_xmx(q, a, &f1);
                dsv41dec::AttnArgs b;
                b.q = qd; b.q_h = D; b.q_b = (size_t) NH * D; b.tab = tab; b.n_dims = ND;
                b.ring = ring; b.ring_cs = D; b.RAW = RAW; b.rows = rows; b.rows_cs = D;
                b.widx = widx; b.wi_b = SWA; b.wmask = wm; b.wm_b = SWA;
                if (K) { b.comp = comp; b.comp_t = 1; b.comp_rb = (size_t) D * 2; b.sel = sel; b.sel_b = K; b.selm = sm; b.sm_b = K; }
                b.swa = SWA; b.K = K; b.sinks = sinks; b.o = o2; b.o_h = D; b.o_b = (size_t) NH * D; b.NH = NH; b.n = n;
                b.scale = 0.044f; b.part = part;
                sycl::event f2;
                sycl::event e2 = dsv41dec::attn_dec(q, b, &f2);
                q.wait();
                if (it >= 20) { t_old1.push_back(ev_us(f1)); t_old2.push_back(ev_us(e1)); t_new1.push_back(ev_us(f2)); t_new2.push_back(ev_us(e2)); }
            }
            std::printf("n=%d M=%4d  old main %6.1f combine %5.1f us | new main %6.1f combine %5.1f us\n", n, M,
                        med(t_old1), med(t_old2), med(t_new1), med(t_new2));
            for (void * p : { (void *) qd, (void *) ring, (void *) rows, (void *) k32, (void *) msk, (void *) tab, (void *) sinks,
                              (void *) comp, (void *) widx, (void *) sel, (void *) wm, (void *) sm, (void *) o1, (void *) o2, (void *) part })
                sycl::free(p, q);
        }
    }
    return 0;
}
