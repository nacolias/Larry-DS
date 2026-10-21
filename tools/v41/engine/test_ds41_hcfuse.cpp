// test_ds41_hcfuse [--gpu] - larry: the fused hyper-connection ops (ggml_dsv41_hc_post2 / _hc_fa / _hc_fb) vs the
// stock-op composite ds41_dense.cpp builds without them (norm_mv, hc_mixes, hc_comb, hc_pre, rms_norm * w for the
// attention input; hc_post first for the FFN input; add + hc_post for the finish) on the same backend, n = 1, 3, 8,
// fn F16 and F32.  The fused ops sum in another order, so: max |fused - composite| / max |composite| per output.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
    ggml_backend_t be = gpu ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr) : nullptr;
    if (!be) be = ggml_backend_cpu_init();
    const int64_t D = 5120, HC = 4, NR = (2 + HC) * HC;   // the model's n_embd / hc
    const float eps = 1e-6f, hc_eps = 1e-6f;
    const int iters = 20;
    std::mt19937 rng(20261021);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    auto randv = [&](size_t n, float sc, float off = 0.0f) {
        std::vector<float> v(n);
        for (auto & x : v) x = off + sc * nd(rng);
        return v;
    };
    auto unif = [&](size_t n, float lo, float hi) {
        std::vector<float> v(n);
        for (auto & x : v) x = lo + (hi - lo) * ud(rng);
        return v;
    };
    double worst = 0;
    for (int64_t n : { (int64_t) 1, (int64_t) 3, (int64_t) 8 }) {
        for (int w16 = 0; w16 < 2; ++w16) {
            ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
            ggml_context * c = ggml_init(ip);
            auto in = [&](ggml_tensor * t) { ggml_set_input(t); return t; };
            ggml_tensor * x = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, n));
            ggml_tensor * fn = in(ggml_new_tensor_2d(c, w16 ? GGML_TYPE_F16 : GGML_TYPE_F32, HC * D, NR));
            ggml_tensor * sc = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, 3));
            ggml_tensor * bs = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, NR));
            ggml_tensor * w = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, D));
            ggml_tensor * pre_in = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, n));
            ggml_tensor * y = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n));
            ggml_tensor * post_in = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, n));
            ggml_tensor * comb_in = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, n));
            ggml_tensor * y2 = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n));
            ggml_tensor * xdst = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, n));

            std::vector<std::pair<std::string, std::pair<ggml_tensor *, ggml_tensor *>>> cmp;   // composite, fused
            // ---- the composite (ds41_dense.cpp's B::hc_mixes with STRATA_DEC_FUSE, hc_pre, rms_w)
            auto composite = [&](ggml_tensor * h, ggml_tensor * pre, ggml_tensor ** o_xn, ggml_tensor ** o_pre,
                                 ggml_tensor ** o_post, ggml_tensor ** o_comb) {
                ggml_tensor * mixes = ggml_dsv41_norm_mv(c, fn, ggml_reshape_2d(c, h, HC * D, n), eps);
                ggml_tensor * mix = ggml_dsv41_hc_mixes(c, mixes, ggml_view_1d(c, sc, 2, 0), ggml_view_1d(c, bs, 2 * HC, 0), hc_eps);
                *o_pre = ggml_cont(c, ggml_view_2d(c, mix, HC, n, mix->nb[1], 0));
                *o_post = ggml_cont(c, ggml_view_2d(c, mix, HC, n, mix->nb[1], (size_t) n * mix->nb[1]));
                *o_comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, hc_eps, iters);
                *o_xn = ggml_mul(c, ggml_rms_norm(c, ggml_dsv4_hc_pre(c, h, pre), eps), w);
            };
            auto views = [&](ggml_tensor * fb, ggml_tensor ** o_xn, ggml_tensor ** o_pre, ggml_tensor ** o_post, ggml_tensor ** o_comb) {
                size_t o = 0;
                *o_xn = ggml_view_2d(c, fb, D, n, (size_t) D * 4, o);
                o += (size_t) (D * n) * 4;
                *o_pre = ggml_view_2d(c, fb, HC, n, (size_t) HC * 4, o);
                o += (size_t) (HC * n) * 4;
                *o_post = ggml_view_2d(c, fb, HC, n, (size_t) HC * 4, o);
                o += (size_t) (HC * n) * 4;
                *o_comb = ggml_view_3d(c, fb, HC, HC, n, (size_t) HC * 4, (size_t) (HC * HC) * 4, o);
            };
            // attention input (no post): mixes of x, z = hc_pre(x, pre_in)
            ggml_tensor * c1[4], * f1[4];
            composite(x, pre_in, &c1[0], &c1[1], &c1[2], &c1[3]);
            ggml_tensor * fa1 = ggml_dsv41_hc_fa(c, x, fn, pre_in, nullptr, nullptr, nullptr);
            ggml_tensor * fb1 = ggml_dsv41_hc_fb(c, fa1, sc, bs, w, eps, hc_eps, iters);
            views(fb1, &f1[0], &f1[1], &f1[2], &f1[3]);
            // FFN input: h = hc_post(y, x, post_in, comb_in), its mixes, z = hc_pre(h, pre_in)
            ggml_tensor * h = ggml_dsv4_hc_post(c, y, x, post_in, comb_in);
            ggml_tensor * c2[4], * f2[4];
            composite(h, pre_in, &c2[0], &c2[1], &c2[2], &c2[3]);
            ggml_tensor * fa2 = ggml_dsv41_hc_fa(c, x, fn, pre_in, y, post_in, comb_in);
            ggml_tensor * fh = ggml_view_3d(c, fa2, D, HC, n, (size_t) D * 4, (size_t) (HC * D) * 4, 0);
            ggml_tensor * fb2 = ggml_dsv41_hc_fb(c, fa2, sc, bs, w, eps, hc_eps, iters);
            views(fb2, &f2[0], &f2[1], &f2[2], &f2[3]);
            // finish: x_state = hc_post(y + y2, x, post_in, comb_in)
            ggml_tensor * c3 = ggml_dsv4_hc_post(c, ggml_add(c, y, y2), x, post_in, comb_in);
            ggml_tensor * f3 = ggml_dsv41_hc_post2(c, y, y2, x, post_in, comb_in, xdst);
            const char * nm[4] = { "xn", "pre", "post", "comb" };
            for (int k = 0; k < 4; ++k) {
                cmp.push_back({ std::string("attn ") + nm[k], { c1[k], f1[k] } });
                cmp.push_back({ std::string("ffn  ") + nm[k], { c2[k], f2[k] } });
            }
            cmp.push_back({ "ffn  h", { h, fh } });
            cmp.push_back({ "finish", { c3, f3 } });

            ggml_cgraph * gf = ggml_new_graph(c);
            for (ggml_tensor * t : { fa1, fb1, fa2, fb2 }) ggml_set_output(t);
            for (auto & e : cmp) {
                ggml_set_output(e.second.first);
                ggml_set_output(e.second.second);
                ggml_build_forward_expand(gf, e.second.first);
                ggml_build_forward_expand(gf, e.second.second);
            }
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (!ggml_gallocr_alloc_graph(al, gf)) { std::printf("alloc failed\n"); return 2; }
            auto set = [&](ggml_tensor * t, const std::vector<float> & v) {
                if (t->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> h16(v.size());
                    ggml_fp32_to_fp16_row(v.data(), h16.data(), (int64_t) v.size());
                    ggml_backend_tensor_set(t, h16.data(), 0, h16.size() * 2);
                } else {
                    ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
                }
            };
            set(x, randv((size_t) (D * HC * n), 1.0f));
            set(fn, randv((size_t) (HC * D * NR), 3.0f / std::sqrt((float) (HC * D))));
            set(sc, { 0.7f, 0.4f, 1.3f });
            set(bs, randv((size_t) NR, 0.5f));
            set(w, randv((size_t) D, 0.1f, 1.0f));
            set(pre_in, unif((size_t) (HC * n), 0.05f, 1.0f));
            set(y, randv((size_t) (D * n), 2.0f));
            set(post_in, unif((size_t) (HC * n), 0.0f, 2.0f));
            set(comb_in, unif((size_t) (HC * HC * n), 0.0f, 0.5f));
            set(y2, randv((size_t) (D * n), 1.0f));
            set(xdst, std::vector<float>((size_t) (D * HC * n), NAN));
            ggml_backend_graph_compute(be, gf);
            for (auto & e : cmp) {
                ggml_tensor * a = e.second.first, * b = e.second.second;
                const size_t ne = (size_t) ggml_nelements(a);
                std::vector<float> va(ne), vb(ne);
                // composite results are contiguous; the fused views are contiguous too (region-major layouts)
                ggml_backend_tensor_get(a, va.data(), 0, ne * 4);
                ggml_backend_tensor_get(b, vb.data(), 0, ne * 4);
                double md = 0, mr = 0;
                size_t bad = 0;
                for (size_t i = 0; i < ne; ++i) {
                    if (!std::isfinite(vb[i])) ++bad;
                    md = std::max(md, (double) std::fabs(va[i] - vb[i]));
                    mr = std::max(mr, (double) std::fabs(va[i]));
                }
                const double rel = bad ? 1e30 : md / std::max(mr, 1e-30);
                worst = std::max(worst, rel);
                std::printf("n=%lld fn %s %-10s max|ref| %.4g  max|diff| %.3g  rel %.3g%s\n", (long long) n, w16 ? "F16" : "F32",
                            e.first.c_str(), mr, md, rel, bad ? "  NON-FINITE" : (rel > 1e-4 ? "  <-- BAD" : ""));
            }
            ggml_gallocr_free(al);
            ggml_free(c);
        }
    }
    ggml_backend_free(be);
    std::printf("%s (worst rel %.3g)\n", worst <= 1e-4 ? "fused ~= composite" : "DIFFERENCES", worst);
    return worst <= 1e-4 ? 0 : 1;
}
