// test_ds41_attnfuse - larry: the fused decode attention core (ggml_dsv41_kv_fin + ggml_dsv41_attn_dec) vs the op chain
// ds41_dense.cpp's build_attn builds without them (rms_norm * w, rope, fake_quant, concat, kgather2, concat + cast of the
// mask, flash_attn_ext + sinks, rope back, set_rows into the ring) on the GPU backend: n = 1, 3, 5, 7 queries, the
// window alone (ratio 0, no YaRN) and window + 512 compressed rows (ratio 1, YaRN), a full window and one reaching
// before position 0.  The fused kernels compute the same values in other kernels: the window rows may differ by the
// odd FP8 step where a 1-ulp RoPE difference crosses a rounding boundary, so the check is on the attention output.
// Part 2 (STRATA_Q8_FUSE): every producer's q8 (rms_q8, swiglu_q8, hc_fb_q8, attn_dec_q8) through its Q6_K product
// (ggml_dsv41_mvq / _oproj_q8) vs the plain op + ggml_mul_mat / ggml_dsv41_oproj, which quantise the same f32 values:
// the products must match bit for bit.
// Part 3 (STRATA_LI_FUSE; run with GGML_SYCL_LI_XMX=1 so the chain's indexer is the same XMX kernel): ggml_dsv41_li_dec
// + ggml_dsv41_sel_fin vs build_attn's indexer chain (rope, fake_quant, scale, the visibility ramp, lightning_indexer,
// the candidate mask, pool + pin + top_k for the candidate blocks, top_k + argsort + get_rows for sel / selm).
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

int main() {
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) { std::printf("no GPU backend\n"); return 2; }
    const int64_t DH = 512, NH = 64, SWA = 128, RAW = SWA + 16, C = 700, ND = 64;
    const float eps = 1e-6f, scale = 1.0f / std::sqrt((float) DH);
    std::mt19937 rng(20261022);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto randv = [&](size_t n, float sc, float off = 0.0f) {
        std::vector<float> v(n);
        for (auto & x : v) x = off + sc * nd(rng);
        return v;
    };
    double worst = 0;
    int fails = 0;
    const char * part_env = std::getenv("DS41_TEST_PART");
    const int only_part = part_env ? std::atoi(part_env) : 0;
    const char * q8_env = std::getenv("DS41_TEST_Q8");
    const std::string q8_only = q8_env ? q8_env : "rshapqmf";
    for (int ratio : { 0, 1 }) {
        if (only_part && only_part != 1) break;
        for (int64_t pos0 : { (int64_t) 1000, (int64_t) 40 }) {
            for (int64_t n : { (int64_t) 1, (int64_t) 3, (int64_t) 5, (int64_t) 7 }) {
                const int64_t K = ratio ? 512 : 0, M = SWA + K;
                ggml_dsv41_rope rp;
                const bool yarn = ratio != 0;
                rp.n_dims = (int) ND;
                rp.n_ctx_orig = yarn ? 65536 : 0;
                rp.freq_base = yarn ? 160000.0f : 10000.0f;
                rp.freq_scale = yarn ? 1.0f / 16.0f : 1.0f;
                rp.ext_factor = yarn ? 1.0f : 0.0f;
                rp.attn_factor = yarn ? 1.0f / (1.0f + 0.1f * std::log(1.0f / rp.freq_scale)) : 1.0f;
                rp.beta_fast = yarn ? 32.0f : 0.0f;
                rp.beta_slow = yarn ? 1.0f : 0.0f;
                auto rope = [&](ggml_context * c, ggml_tensor * a, ggml_tensor * pos, bool back) {
                    ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, rp.n_dims, GGML_ROPE_TYPE_NORMAL, rp.n_ctx_orig, rp.freq_base,
                                                                rp.freq_scale, rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow)
                                           : ggml_rope_ext(c, a, pos, nullptr, rp.n_dims, GGML_ROPE_TYPE_NORMAL, rp.n_ctx_orig, rp.freq_base,
                                                           rp.freq_scale, rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
                    return ggml_rope_set_offset(r, (int) (a->ne[0] - rp.n_dims));
                };
                ggml_init_params ip = { ggml_tensor_overhead() * 128 + ggml_graph_overhead(), nullptr, true };
                ggml_context * c = ggml_init(ip);
                auto in = [&](ggml_tensor * t) { ggml_set_input(t); return t; };
                ggml_tensor * q = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, DH, NH, n));
                ggml_tensor * kvy = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, DH + 256, n));   // the fused q_a|kv product's kv rows
                ggml_tensor * w = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, DH));
                ggml_tensor * pos = in(ggml_new_tensor_1d(c, GGML_TYPE_I32, n));
                ggml_tensor * ring_c = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, DH, RAW));
                ggml_tensor * ring_f = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, DH, RAW));
                ggml_tensor * slot = in(ggml_new_tensor_1d(c, GGML_TYPE_I32, n));
                ggml_tensor * widx = in(ggml_new_tensor_1d(c, GGML_TYPE_I32, SWA * n));
                ggml_tensor * wmask = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, SWA, n));
                ggml_tensor * comp = K ? in(ggml_new_tensor_2d(c, GGML_TYPE_BF16, DH, C)) : nullptr;
                ggml_tensor * sel = K ? in(ggml_new_tensor_2d(c, GGML_TYPE_I32, K, n)) : nullptr;
                ggml_tensor * selm = K ? in(ggml_new_tensor_2d(c, GGML_TYPE_F32, K, n)) : nullptr;
                ggml_tensor * sinks = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, NH));
                ggml_tensor * kv_y = ggml_view_2d(c, kvy, DH, n, kvy->nb[1], 256 * 4);

                // ---- the composite (build_attn without STRATA_ATTN_FUSE)
                ggml_tensor * kv = ggml_mul(c, ggml_rms_norm(c, kv_y, eps), w);
                kv = rope(c, ggml_reshape_3d(c, kv, DH, 1, n), pos, false);
                ggml_tensor * kv2 = ggml_dsv41_fake_quant(c, ggml_reshape_2d(c, kv, DH, n), 0);
                ggml_tensor * wsrc = ggml_concat(c, ring_c, kv2, 1);
                ggml_tensor * k4 = ggml_dsv41_kgather2(c, wsrc, widx, comp, sel, (int) SWA, GGML_TYPE_F32);
                ggml_tensor * Mall = K ? ggml_concat(c, wmask, selm, 0) : wmask;
                ggml_tensor * m4 = ggml_cast(c, ggml_reshape_4d(c, Mall, M, 1, 1, n), GGML_TYPE_F16);
                ggml_tensor * qr = rope(c, q, pos, false);
                ggml_tensor * q4 = ggml_permute(c, ggml_reshape_4d(c, qr, DH, NH, 1, n), 0, 2, 1, 3);
                ggml_tensor * oc = ggml_flash_attn_ext(c, q4, k4, k4, m4, scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_add_sinks(oc, sinks);
                oc = rope(c, ggml_reshape_3d(c, oc, DH, NH, n), pos, true);
                ggml_tensor * rset = ggml_set_rows(c, ring_c, kv2, slot);
                // ---- the fused ops
                ggml_tensor * kf = ggml_dsv41_kv_fin(c, kv_y, w, pos, ring_f, slot, eps, rp, true);
                ggml_tensor * kvn = ggml_view_2d(c, kf, DH, n, DH * 4, 0);
                ggml_tensor * tab = ggml_view_1d(c, kf, ND * n, (size_t) (DH * n) * 4);
                ggml_tensor * of = ggml_dsv41_attn_dec(c, q, ring_f, kvn, widx, wmask, comp, sel, selm, sinks, tab, (int) SWA, scale, rp.n_dims);

                ggml_cgraph * gf = ggml_new_graph(c);
                for (ggml_tensor * t : { kv2, oc, kf, of, ring_c, ring_f }) ggml_set_output(t);
                ggml_build_forward_expand(gf, oc);
                ggml_build_forward_expand(gf, rset);   // after the composite's reads of ring_c
                ggml_build_forward_expand(gf, kf);
                ggml_build_forward_expand(gf, of);
                ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                if (!ggml_gallocr_alloc_graph(al, gf)) { std::printf("alloc failed\n"); return 2; }
                auto setf = [&](ggml_tensor * t, const std::vector<float> & v) { ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4); };
                auto seti = [&](ggml_tensor * t, const std::vector<int32_t> & v) { ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4); };
                setf(q, randv((size_t) (DH * NH * n), 1.0f));
                setf(kvy, randv((size_t) ((DH + 256) * n), 1.5f));
                setf(w, randv((size_t) DH, 0.1f, 1.0f));
                std::vector<int32_t> vpos(n), vslot(n), vwidx((size_t) (SWA * n));
                std::vector<float> vwm((size_t) (SWA * n));
                for (int64_t t = 0; t < n; ++t) {
                    const int64_t p = pos0 + t;
                    vpos[t] = (int32_t) p;
                    vslot[t] = (int32_t) (p % RAW);
                    for (int64_t j = 0; j < SWA; ++j) {   // ds41_dense.cpp upload_inputs' window layout
                        const int64_t qq = p - SWA + 1 + j;
                        int32_t row = 0;
                        float m = 0.0f;
                        if (qq < 0) m = -INFINITY;
                        else if (qq >= pos0) row = (int32_t) (RAW + (qq - pos0));
                        else row = (int32_t) (qq % RAW);
                        vwidx[(size_t) (t * SWA + j)] = row;
                        vwm[(size_t) (t * SWA + j)] = m;
                    }
                }
                seti(pos, vpos);
                seti(slot, vslot);
                seti(widx, vwidx);
                setf(wmask, vwm);
                const std::vector<float> vring = randv((size_t) (DH * RAW), 1.0f);
                setf(ring_c, vring);
                setf(ring_f, vring);
                if (K) {
                    std::vector<float> vc = randv((size_t) (DH * C), 0.7f);
                    std::vector<ggml_bf16_t> b16(vc.size());
                    ggml_fp32_to_bf16_row(vc.data(), b16.data(), (int64_t) vc.size());
                    ggml_backend_tensor_set(comp, b16.data(), 0, b16.size() * 2);
                    std::vector<int32_t> vsel((size_t) (K * n));
                    std::vector<float> vsm((size_t) (K * n));
                    std::vector<int32_t> perm(C);
                    for (int64_t t = 0; t < n; ++t) {
                        std::iota(perm.begin(), perm.end(), 0);
                        std::shuffle(perm.begin(), perm.end(), rng);
                        std::sort(perm.begin(), perm.begin() + K);
                        for (int64_t i = 0; i < K; ++i) {
                            vsel[(size_t) (t * K + i)] = perm[(size_t) i];
                            vsm[(size_t) (t * K + i)] = i < K - 40 - 7 * t ? 0.0f : -INFINITY;   // the newest rows: not visible yet
                        }
                    }
                    seti(sel, vsel);
                    setf(selm, vsm);
                }
                setf(sinks, randv((size_t) NH, 1.0f));
                ggml_backend_graph_compute(be, gf);
                auto get = [&](ggml_tensor * t) {
                    std::vector<float> v((size_t) ggml_nelements(t));
                    ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
                    return v;
                };
                auto cmp = [&](const char * what, const std::vector<float> & a, const std::vector<float> & b, double tol) {
                    double md = 0, mr = 0;
                    size_t bad = 0, ndiff = 0;
                    for (size_t i = 0; i < a.size(); ++i) {
                        if (!std::isfinite(b[i])) ++bad;
                        if (a[i] != b[i]) ++ndiff;
                        md = std::max(md, (double) std::fabs(a[i] - b[i]));
                        mr = std::max(mr, (double) std::fabs(a[i]));
                    }
                    const double rel = bad ? 1e30 : md / std::max(mr, 1e-30);
                    const bool ok = rel <= tol;
                    if (!ok) ++fails;
                    std::printf("ratio %d pos0 %4lld n=%lld %-5s max|ref| %.4g  max|diff| %.3g  rel %.3g  (%zu of %zu differ)%s\n", ratio,
                                (long long) pos0, (long long) n, what, mr, md, rel, ndiff, a.size(), bad ? "  NON-FINITE" : ok ? "" : "  <-- BAD");
                    return rel;
                };
                cmp("kv", get(kv2), get(kvn), 0.07);              // an FP8 step at most
                cmp("ring", get(ring_c), get(ring_f), 0.07);
                worst = std::max(worst, cmp("o", get(oc), get(of), 2e-3));
                ggml_gallocr_free(al);
                ggml_free(c);
            }
        }
    }
    // ---------------------------------------------------------------- part 2: the q8 producers
    auto q6k = [&](int64_t K, int64_t N, float sc) {   // a random Q6_K weight's bytes
        std::vector<float> f = randv((size_t) (K * N), sc);
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q6_K, K) * (size_t) N);
        ggml_quantize_chunk(GGML_TYPE_Q6_K, f.data(), q.data(), 0, N, K, nullptr);
        return q;
    };
    for (int64_t n : { (int64_t) 1, (int64_t) 3, (int64_t) 5, (int64_t) 7 }) {
        if (only_part && only_part != 2) break;
        const int64_t RQ = 1280, NQB = 4096, D = 5120, HC = 4, NR = 24, NF = 2304, OGD = 4096, OL = 1024, OG = 8;
        ggml_init_params ip = { ggml_tensor_overhead() * 160 + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        auto in = [&](ggml_tensor * t) { ggml_set_input(t); return t; };
        // rms_q8 -> q_b
        ggml_tensor * qa = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, RQ + 512, n));
        ggml_tensor * qaw = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, RQ));
        ggml_tensor * wqb = in(ggml_new_tensor_2d(c, GGML_TYPE_Q6_K, RQ, NQB));
        ggml_tensor * qa_y = ggml_view_2d(c, qa, RQ, n, qa->nb[1], 0);
        ggml_tensor * r_c = ggml_mul_mat(c, wqb, ggml_mul(c, ggml_rms_norm(c, qa_y, eps), qaw));
        ggml_tensor * rq = ggml_dsv41_rms_q8(c, qa_y, qaw, eps);
        ggml_tensor * r_f = ggml_dsv41_mvq(c, wqb, ggml_view_2d(c, rq, RQ, n, RQ * 4, 0), ggml_view_1d(c, rq, ggml_dsv41_q8_floats(RQ, n), RQ * n * 4));
        // swiglu_q8 -> down
        ggml_tensor * gu = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * NF, n));
        ggml_tensor * wdn = in(ggml_new_tensor_2d(c, GGML_TYPE_Q6_K, NF, D));
        ggml_tensor * s_c = ggml_mul_mat(c, wdn, ggml_dsv41_swiglu_lim(c, gu, 10.0f));
        ggml_tensor * sw = ggml_dsv41_swiglu_q8(c, gu, 10.0f);
        ggml_tensor * s_f = ggml_dsv41_mvq(c, wdn, ggml_view_2d(c, sw, NF, n, NF * 4, 0), ggml_view_1d(c, sw, ggml_dsv41_q8_floats(NF, n), NF * n * 4));
        // hc_fb_q8 -> q_a|kv
        ggml_tensor * hx = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, n));
        ggml_tensor * hfn = in(ggml_new_tensor_2d(c, GGML_TYPE_F16, HC * D, NR));
        ggml_tensor * hsc = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, 3));
        ggml_tensor * hbs = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, NR));
        ggml_tensor * hw = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, D));
        ggml_tensor * hpre = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, n));
        ggml_tensor * wqkv = in(ggml_new_tensor_2d(c, GGML_TYPE_Q6_K, D, RQ + 512));
        ggml_tensor * fa = ggml_dsv41_hc_fa(c, hx, hfn, hpre, nullptr, nullptr, nullptr);
        ggml_tensor * fb_c = ggml_dsv41_hc_fb(c, fa, hsc, hbs, hw, eps, 1e-6f, 20);
        ggml_tensor * fb_f = ggml_dsv41_hc_fb_q8(c, fa, hsc, hbs, hw, eps, 1e-6f, 20);
        ggml_tensor * h_c = ggml_mul_mat(c, wqkv, ggml_view_2d(c, fb_c, D, n, D * 4, 0));
        ggml_tensor * h_f = ggml_dsv41_mvq(c, wqkv, ggml_view_2d(c, fb_f, D, n, D * 4, 0),
                                           ggml_view_1d(c, fb_f, ggml_dsv41_q8_floats(D, n), (size_t) (n * (D + 2 * HC + HC * HC)) * 4));
        // attn_dec_q8 -> oproj_q8 (window only)
        ggml_tensor * aq = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, DH, NH, n));
        ggml_tensor * ar = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, DH, RAW));
        ggml_tensor * arows = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, DH, n));
        ggml_tensor * awi = in(ggml_new_tensor_1d(c, GGML_TYPE_I32, SWA * n));
        ggml_tensor * awm = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, SWA, n));
        ggml_tensor * asi = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, NH));
        ggml_tensor * atab = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, ND * n));
        ggml_tensor * woa2 = in(ggml_new_tensor_2d(c, GGML_TYPE_Q6_K, OGD, OL * OG));
        ggml_tensor * wob = in(ggml_new_tensor_2d(c, GGML_TYPE_Q6_K, OL * OG, D));
        ggml_tensor * woa = ggml_reshape_3d(c, woa2, OGD, OL, OG);
        ggml_tensor * ao_c = ggml_dsv41_attn_dec(c, aq, ar, arows, awi, awm, nullptr, nullptr, nullptr, asi, atab, (int) SWA, scale, (int) ND);
        ggml_tensor * ao_f = ggml_dsv41_attn_dec_q8(c, aq, ar, arows, awi, awm, nullptr, nullptr, nullptr, asi, atab, (int) SWA, scale, (int) ND, (int) (OGD / DH));
        ggml_tensor * of_v = ggml_view_3d(c, ao_f, DH, NH, n, DH * 4, DH * NH * 4, 0);
        ggml_tensor * p_c = ggml_dsv41_oproj(c, ao_c, woa, wob);
        ggml_tensor * p_f = ggml_dsv41_oproj_q8(c, of_v, ggml_view_1d(c, ao_f, ggml_dsv41_q8_floats(OGD, n * OG), (size_t) (DH * NH * n) * 4), woa, wob);

        ggml_cgraph * gf = ggml_new_graph(c);
        auto want = [&](char k) { return q8_only.find(k) != std::string::npos; };
        std::vector<ggml_tensor *> outs2;
        if (want('r')) { outs2.push_back(r_c); outs2.push_back(r_f); }
        if (want('s')) { outs2.push_back(s_c); outs2.push_back(s_f); }
        if (want('h')) { outs2.push_back(fb_c); outs2.push_back(fb_f); outs2.push_back(h_c); outs2.push_back(h_f); }
        // hc_fa + hc_fb_q8 as one kernel (ggml_dsv41_hc_merge), without and with the hc_post input
        ggml_tensor * fam = ggml_dsv41_hc_fa(c, hx, hfn, hpre, nullptr, nullptr, nullptr);
        ggml_tensor * fbm = ggml_dsv41_hc_merge(ggml_dsv41_hc_fb_q8(c, fam, hsc, hbs, hw, eps, 1e-6f, 20));
        ggml_tensor * hy = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n));
        ggml_tensor * hpo = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, n));
        ggml_tensor * hcb = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, n));
        ggml_tensor * fa2 = ggml_dsv41_hc_fa(c, hx, hfn, hpre, hy, hpo, hcb);
        ggml_tensor * fb2 = ggml_dsv41_hc_fb_q8(c, fa2, hsc, hbs, hw, eps, 1e-6f, 20);
        ggml_tensor * fa2m = ggml_dsv41_hc_fa(c, hx, hfn, hpre, hy, hpo, hcb);
        ggml_tensor * fb2m = ggml_dsv41_hc_merge(ggml_dsv41_hc_fb_q8(c, fa2m, hsc, hbs, hw, eps, 1e-6f, 20));
        if (want('m')) for (ggml_tensor * t : { fb_f, fam, fbm, fa2, fb2, fa2m, fb2m }) outs2.push_back(t);
        // the finish folded into the next hc (STRATA_FIN_DEFER): hc_post2 -> x_state, then hc_fa / hc_fb on it, vs
        // hc_fa2(hap, .., shexp, routed, ..) writing x_state, merged with its hc_fb
        ggml_tensor * fsh = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n));
        ggml_tensor * frt = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n));
        ggml_tensor * fhap = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, n));
        ggml_tensor * xsA = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, n));
        ggml_tensor * xsB = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, n));
        ggml_tensor * p2 = ggml_dsv41_hc_post2(c, fsh, frt, fhap, hpo, hcb, xsA);
        ggml_tensor * fbA = ggml_dsv41_hc_fb_q8(c, ggml_dsv41_hc_fa(c, p2, hfn, hpre, nullptr, nullptr, nullptr), hsc, hbs, hw, eps, 1e-6f, 20);
        ggml_tensor * faB = ggml_dsv41_write_to(ggml_dsv41_hc_fa2(c, fhap, hfn, hpre, fsh, frt, hpo, hcb), xsB, nullptr, nullptr, nullptr, nullptr);
        ggml_tensor * fbB = ggml_dsv41_hc_merge(ggml_dsv41_hc_fb_q8(c, faB, hsc, hbs, hw, eps, 1e-6f, 20));
        if (want('f')) { for (ggml_tensor * t : { fbA, fbB }) outs2.push_back(t); ggml_set_output(xsA); ggml_set_output(xsB); }
        if (want('a')) { outs2.push_back(ao_c); outs2.push_back(ao_f); }   // the attention's q8 alone
        if (want('p')) outs2.push_back(p_c);                                // the oproj op as before (quant_x inside)
        if (want('q')) outs2.push_back(p_f);                                // the oproj op on the attention's q8
        for (ggml_tensor * t : outs2) { ggml_set_output(t); ggml_build_forward_expand(gf, t); }
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(al, gf)) { std::printf("alloc failed\n"); return 2; }
        auto setf = [&](ggml_tensor * t, const std::vector<float> & v) { if (t->buffer) ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4); };
        auto setb = [&](ggml_tensor * t, const std::vector<uint8_t> & v) { if (t->buffer) ggml_backend_tensor_set(t, v.data(), 0, v.size()); };
        setf(qa, randv((size_t) ((RQ + 512) * n), 2.0f));
        setf(qaw, randv((size_t) RQ, 0.1f, 1.0f));
        setb(wqb, q6k(RQ, NQB, 0.02f));
        setf(gu, randv((size_t) (2 * NF * n), 4.0f));
        setb(wdn, q6k(NF, D, 0.02f));
        setf(hx, randv((size_t) (D * HC * n), 1.0f));
        {
            std::vector<float> f = randv((size_t) (HC * D * NR), 3.0f / std::sqrt((float) (HC * D)));
            std::vector<ggml_fp16_t> h16(f.size());
            ggml_fp32_to_fp16_row(f.data(), h16.data(), (int64_t) f.size());
            if (hfn->buffer) ggml_backend_tensor_set(hfn, h16.data(), 0, h16.size() * 2);
        }
        setf(hsc, { 0.7f, 0.4f, 1.3f });
        setf(hbs, randv((size_t) NR, 0.5f));
        setf(hw, randv((size_t) D, 0.1f, 1.0f));
        setf(hpre, randv((size_t) (HC * n), 0.2f, 0.5f));
        setf(hy, randv((size_t) (D * n), 2.0f));
        setf(fsh, randv((size_t) (D * n), 1.0f));
        setf(frt, randv((size_t) (D * n), 1.0f));
        setf(fhap, randv((size_t) (D * HC * n), 1.0f));
        setf(hpo, randv((size_t) (HC * n), 0.5f, 1.0f));
        setf(hcb, randv((size_t) (HC * HC * n), 0.1f, 0.25f));
        setb(wqkv, q6k(D, RQ + 512, 0.02f));
        setf(aq, randv((size_t) (DH * NH * n), 1.0f));
        setf(ar, randv((size_t) (DH * RAW), 1.0f));
        setf(arows, randv((size_t) (DH * n), 1.0f));
        {
            std::vector<int32_t> vw((size_t) (SWA * n));
            std::vector<float> vm((size_t) (SWA * n), 0.0f);
            for (int64_t t = 0; t < n; ++t) {
                for (int64_t j = 0; j < SWA; ++j) {
                    const int64_t qq = 500 + t - SWA + 1 + j;
                    vw[(size_t) (t * SWA + j)] = (int32_t) (qq >= 500 ? RAW + (qq - 500) : qq % RAW);
                }
            }
            if (awi->buffer) ggml_backend_tensor_set(awi, vw.data(), 0, vw.size() * 4);
            setf(awm, vm);
            setf(atab, randv((size_t) (ND * n), 0.6f));
        }
        setf(asi, randv((size_t) NH, 1.0f));
        setb(woa2, q6k(OGD, OL * OG, 0.02f));
        setb(wob, q6k(OL * OG, D, 0.02f));
        ggml_backend_graph_compute(be, gf);
        auto get = [&](ggml_tensor * t, size_t ne) {
            std::vector<float> v(ne);
            ggml_backend_tensor_get(t, v.data(), 0, ne * 4);
            return v;
        };
        auto exact = [&](const char * what, ggml_tensor * a, ggml_tensor * b2, size_t ne) {
            const std::vector<float> va = get(a, ne), vb = get(b2, ne);
            size_t nd = 0;
            double md = 0, mr = 0;
            for (size_t i = 0; i < ne; ++i) {   // bit patterns: the q8 regions hold int8 bytes (NaN-looking floats)
                if (std::memcmp(&va[i], &vb[i], 4)) ++nd;
                if (std::isfinite(va[i]) && std::isfinite(vb[i])) {
                    md = std::max(md, (double) std::fabs(va[i] - vb[i]));
                    mr = std::max(mr, (double) std::fabs(va[i]));
                }
            }
            if (nd) ++fails;
            std::printf("q8 n=%lld %-22s max|ref| %.4g  %zu of %zu differ (max %.3g)%s\n", (long long) n, what, mr, nd, ne, md, nd ? "  <-- BAD" : "");
        };
        if (want('r')) exact("rms_q8 -> q_b", r_c, r_f, (size_t) (NQB * n));
        if (want('s')) exact("swiglu_q8 -> down", s_c, s_f, (size_t) (D * n));
        if (want('h')) { exact("hc_fb_q8 xn", fb_c, fb_f, (size_t) ((D + 2 * HC + HC * HC) * n)); exact("hc_fb_q8 -> q_a|kv", h_c, h_f, (size_t) ((RQ + 512) * n)); }
        if (want('f')) {
            exact("finish folded: x_state", xsA, xsB, (size_t) (D * HC * n));
            exact("finish folded: next hc", fbA, fbB, (size_t) ggml_nelements(fbA));
        }
        if (want('m')) {
            const size_t nfb = (size_t) ggml_nelements(fb_f);
            exact("hc merged (no post)", fb_f, fbm, nfb);
            exact("hc merged (post) fb", fb2, fb2m, nfb);
            exact("hc merged (post) h,z", fa2, fa2m, (size_t) ((HC * D + D) * n));
        }
        if (want('a')) exact("attn_dec_q8 o", ao_c, ao_f, (size_t) (DH * NH * n));
        if (want('p') && want('q')) exact("attn_dec_q8 -> oproj", p_c, p_f, (size_t) (D * n));
        ggml_gallocr_free(al);
        ggml_free(c);
    }
    // ---------------------------------------------------------------- part 3: the indexer
    for (int mode = 0; mode < 3; ++mode) {   // 0 plain, 1 uses_cand, 2 cand_src
        if (only_part && only_part != 3) break;
        for (int64_t n : { (int64_t) 1, (int64_t) 5 }) {
            const int64_t ID = 128, IH = 32, CAP = 4096, K = 512, NBLK = CAP / 8, NCB = 200;
            const float wscale = 1.0f / std::sqrt((float) (ID * IH));
            ggml_dsv41_rope rp = { (int) ND, 65536, 160000.0f, 1.0f / 16.0f, 1.0f, 1.0f / (1.0f + 0.1f * std::log(16.0f)), 32.0f, 1.0f };
            auto rope = [&](ggml_context * c, ggml_tensor * a, ggml_tensor * pos) {
                ggml_tensor * r = ggml_rope_ext(c, a, pos, nullptr, rp.n_dims, GGML_ROPE_TYPE_NORMAL, rp.n_ctx_orig, rp.freq_base,
                                                rp.freq_scale, rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
                return ggml_rope_set_offset(r, (int) (a->ne[0] - rp.n_dims));
            };
            ggml_init_params ip = { ggml_tensor_overhead() * 160 + ggml_graph_overhead(), nullptr, true };
            ggml_context * c = ggml_init(ip);
            auto in = [&](ggml_tensor * t) { ggml_set_input(t); return t; };
            ggml_tensor * iqr = in(ggml_new_tensor_3d(c, GGML_TYPE_F32, ID, IH, n));
            ggml_tensor * iwr = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, IH, n));
            ggml_tensor * idxk = in(ggml_new_tensor_2d(c, GGML_TYPE_BF16, ID, CAP + 8));
            ggml_tensor * nvis = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, n));
            ggml_tensor * pos = in(ggml_new_tensor_1d(c, GGML_TYPE_I32, n));
            ggml_tensor * ramp = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, CAP, 1));
            ggml_tensor * bramp = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, NBLK, 1));
            ggml_tensor * cand = in(ggml_new_tensor_2d(c, GGML_TYPE_I32, NCB, n));
            ggml_tensor * lastb = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, n));
            ggml_tensor * seld = in(ggml_new_tensor_2d(c, GGML_TYPE_I32, K, n));
            ggml_tensor * selmd = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, K, n));
            ggml_tensor * kvx = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, 512, n));   // a kv_fin for the pass's RoPE table
            ggml_tensor * wx = in(ggml_new_tensor_1d(c, GGML_TYPE_F32, 512));
            ggml_tensor * ringx = in(ggml_new_tensor_2d(c, GGML_TYPE_F32, 512, 16));
            ggml_tensor * slotx = in(ggml_new_tensor_1d(c, GGML_TYPE_I32, n));
            // the chain
            ggml_tensor * iq = ggml_dsv41_fake_quant(c, rope(c, iqr, pos), 2);
            ggml_tensor * iw = ggml_scale(c, iwr, wscale);
            ggml_tensor * ksrc = ggml_view_3d(c, idxk, ID, 1, CAP, idxk->nb[1], idxk->nb[1], 0);
            ggml_tensor * nvs = ggml_reshape_2d(c, nvis, 1, n);
            ggml_tensor * vis = ggml_log(c, ggml_step(c, ggml_add(c, ggml_repeat(c, ramp, ggml_new_tensor_2d(c, GGML_TYPE_F32, CAP, n)), nvs)));
            ggml_tensor * sc = ggml_reshape_2d(c, ggml_lightning_indexer(c, iq, ksrc, iw, ggml_cast(c, vis, GGML_TYPE_F16)), CAP, n);
            if (mode == 1) {
                ggml_tensor * m = ggml_fill(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, 8, NBLK, n), -INFINITY);
                ggml_tensor * z = ggml_fill(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, 8, NCB, n), 0.0f);
                m = ggml_set_rows(c, m, z, ggml_cont(c, cand));
                sc = ggml_add(c, sc, ggml_view_2d(c, m, CAP, n, m->nb[2], 0));
            }
            ggml_tensor * bm = nullptr;
            if (mode == 2) {
                bm = ggml_pool_2d(c, sc, GGML_OP_POOL_MAX, 8, 1, 8, 1, 0, 0);
                ggml_tensor * eq = ggml_step(c, ggml_scale_bias(c, ggml_abs(c, ggml_sub(c, ggml_repeat(c, bramp, bm), ggml_reshape_2d(c, lastb, 1, n))), -1.0f, 0.5f));
                bm = ggml_add(c, bm, ggml_scale(c, eq, 1e30f));
            }
            ggml_tensor * sel = ggml_cont(c, ggml_top_k(c, sc, (int) K));
            sel = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, sel, 1, K, n), ggml_argsort(c, ggml_cast(c, sel, GGML_TYPE_F32), GGML_SORT_ORDER_ASC)), K, n);
            ggml_tensor * selm = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, vis, 1, CAP, n), sel), K, n);
            // the fused ops
            ggml_tensor * kfx = ggml_dsv41_kv_fin(c, kvx, wx, pos, ringx, slotx, eps, rp, true);
            ggml_tensor * tabx = ggml_view_1d(c, kfx, ND * n, (size_t) (512 * n) * 4);
            ggml_tensor * li = ggml_dsv41_li_dec(c, iqr, ggml_view_2d(c, idxk, ID, CAP, idxk->nb[1], 0), iwr, nvis, tabx,
                                                 mode == 1 ? cand : nullptr, mode == 2 ? lastb : nullptr, wscale, rp.n_dims, true);
            ggml_tensor * scf = mode == 2 ? ggml_view_2d(c, li, CAP, n, CAP * 4, 0) : li;
            ggml_tensor * bmf = mode == 2 ? ggml_view_2d(c, li, NBLK, n, NBLK * 4, (size_t) (CAP * n) * 4) : nullptr;
            ggml_tensor * self = ggml_dsv41_sel_fin(c, ggml_top_k(c, scf, (int) K), nvis, seld, selmd);
            ggml_cgraph * gf = ggml_new_graph(c);
            for (ggml_tensor * t : { sc, sel, selm, li, self }) { ggml_set_output(t); ggml_build_forward_expand(gf, t); }
            // written in place by sel_fin: outputs, or the allocator hands their memory to later nodes
            ggml_set_output(seld);
            ggml_set_output(selmd);
            if (bm) { ggml_set_output(bm); ggml_build_forward_expand(gf, bm); }
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (!ggml_gallocr_alloc_graph(al, gf)) { std::printf("alloc failed\n"); return 2; }
            auto setf = [&](ggml_tensor * t, const std::vector<float> & v) { if (t->buffer) ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4); };
            setf(iqr, randv((size_t) (ID * IH * n), 1.0f));
            setf(iwr, randv((size_t) (IH * n), 1.0f));
            {   // keys: FP4-grid values (what the kv source writes), bf16
                std::vector<float> kv = randv((size_t) (ID * (CAP + 8)), 2.0f);
                for (auto & x : kv) x = std::round(x * 2.0f) / 2.0f;
                std::vector<ggml_bf16_t> kb(kv.size());
                ggml_fp32_to_bf16_row(kv.data(), kb.data(), (int64_t) kv.size());
                ggml_backend_tensor_set(idxk, kb.data(), 0, kb.size() * 2);
            }
            std::vector<float> vn(n), vlb(n), vr(CAP), vbr(NBLK);
            std::vector<int32_t> vp(n), vc((size_t) (NCB * n));
            for (int64_t t = 0; t < n; ++t) {
                vn[t] = (float) (CAP - 700 + 37 * t);          // the newest rows not visible
                vlb[t] = (float) (((int64_t) vn[t] - 1) / 8);
                vp[t] = (int32_t) (9000 + t);
                std::vector<int32_t> perm(NBLK);
                std::iota(perm.begin(), perm.end(), 0);
                std::shuffle(perm.begin(), perm.end(), rng);
                for (int64_t i = 0; i < NCB; ++i) vc[(size_t) (t * NCB + i)] = perm[(size_t) i];
            }
            for (int64_t i = 0; i < CAP; ++i) vr[(size_t) i] = -(float) i;
            for (int64_t i = 0; i < NBLK; ++i) vbr[(size_t) i] = (float) i;
            setf(nvis, vn); setf(lastb, vlb); setf(ramp, vr); setf(bramp, vbr);
            ggml_backend_tensor_set(pos, vp.data(), 0, vp.size() * 4);
            if (cand->buffer) ggml_backend_tensor_set(cand, vc.data(), 0, vc.size() * 4);
            setf(kvx, randv((size_t) (512 * n), 1.0f));
            setf(wx, randv(512, 0.1f, 1.0f));
            setf(ringx, randv((size_t) (512 * 16), 1.0f));
            {
                std::vector<int32_t> vs(n);
                for (int64_t t = 0; t < n; ++t) vs[t] = (int32_t) t;
                ggml_backend_tensor_set(slotx, vs.data(), 0, vs.size() * 4);
            }
            ggml_backend_graph_compute(be, gf);
            auto getb = [&](ggml_tensor * t, size_t off, size_t nbytes) {
                std::vector<uint8_t> v(nbytes);
                ggml_backend_tensor_get(t, v.data(), off, nbytes);
                return v;
            };
            auto same = [&](const char * what, const std::vector<uint8_t> & a, const std::vector<uint8_t> & b2) {
                size_t nd = 0;
                for (size_t i = 0; i < a.size(); i += 4) if (std::memcmp(&a[i], &b2[i], 4)) ++nd;
                if (nd) ++fails;
                std::printf("li mode %d n=%lld %-10s %zu of %zu words differ%s\n", mode, (long long) n, what, nd, a.size() / 4, nd ? "  <-- BAD" : "");
            };
            same("scores", getb(sc, 0, (size_t) (CAP * n) * 4), getb(li, 0, (size_t) (CAP * n) * 4));
            same("sel", getb(sel, 0, (size_t) (K * n) * 4), getb(seld, 0, (size_t) (K * n) * 4));
            same("selm", getb(selm, 0, (size_t) (K * n) * 4), getb(selmd, 0, (size_t) (K * n) * 4));
            if (std::getenv("DS41_TEST_DUMP")) {
                std::vector<float> a(8), b2(8);
                std::vector<int32_t> si(8);
                ggml_backend_tensor_get(selm, a.data(), 0, 32);
                ggml_backend_tensor_get(selmd, b2.data(), 0, 32);
                ggml_backend_tensor_get(sel, si.data(), 0, 32);
                for (int i = 0; i < 8; ++i) std::printf("  sel %d  selm chain %g  fused %g\n", si[i], a[i], b2[i]);
            }
            if (bm) same("blockmax", getb(bm, 0, (size_t) (NBLK * n) * 4), getb(li, (size_t) (CAP * n) * 4, (size_t) (NBLK * n) * 4));
            ggml_gallocr_free(al);
            ggml_free(c);
        }
    }
    ggml_backend_free(be);
    std::printf("%s (worst o rel %.3g)\n", fails == 0 ? "fused ~= composite" : "DIFFERENCES", worst);
    return fails == 0 ? 0 : 1;
}
