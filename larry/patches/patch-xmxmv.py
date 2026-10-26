#!/usr/bin/env python3
# patch-xmxmv.py - larry, GGML_SYCL_LARRY_XMXMV (default 1): DSpark verify passes' (2..8 columns) Q6_K mat-vecs on the
# XMX engines for the shapes where the dp4a row-blocked kernel is latency-bound (bench/dqb/xmx_mv.cpp, 5 columns:
# shexp down 2304x5120 40 -> 26 us, q_a 5120x1280 27 -> 18, q_b 1280x32768 84 -> 77; the big square ones and wo_a's
# 8 groups stay on dp4a).  A sub-group owns 16 weight rows; lane n decodes row n's 16-value scale groups into the B
# fragment as fp16 integers (q - 32) * sc; A = the columns' int8 activations (quant_x's rounding) transposed to
# xT[k][8] fp16 integers, so a 32-value block's two 8x16x16 mads are an exact integer dot product, scaled by
# xd[token] * d[row].  Idempotent.
import sys
G = "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/"


def rep(path, old, new, cnt=1):
    s = open(path).read()
    if new.strip() in s:
        print(f"{path}: already"); return
    if s.count(old) != cnt: sys.exit(f"{path}: anchor x{s.count(old)}: {old[:90]!r}")
    open(path, "w").write(s.replace(old, new))
    print(f"{path}: patched")


H = G + "fastmv.hpp"
rep(H, """// launch; returns false when the shape is not covered (caller falls back)
bool launch(sycl::queue & q, WType t, const Args & a);""", """// launch; returns false when the shape is not covered (caller falls back)
bool launch(sycl::queue & q, WType t, const Args & a);

// larry: the XMX path for 2..8 columns of a Q6K_R weight (planar).  quant_xT: G batches x ncols columns (as quant_x)
// -> xT[g][k][8] (fp16 integers, the int8 rounding of quant_x; columns >= ncols zero) and xdT[g][k / 32][8] scales.
sycl::event quant_xT(sycl::queue & q, const float * x, int64_t x_col_stride, int64_t x_batch_stride, int ncols, int64_t G,
                     int64_t K, sycl::half * xT, float * xdT);
// whether launch_xmx takes this shape (the latency-bound ones), and the launch (a.xq / a.xd unused)
bool xmx_wants(const Args & a);
bool launch_xmx(sycl::queue & q, const Args & a, const sycl::half * xT, const float * xdT);""")

C = G + "fastmv.cpp"
rep(C, """sycl::event quant_x(sycl::queue & q, const float * x, int64_t x_col_stride, int64_t x_batch_stride, int ncols, int64_t G,
                    int64_t K, int8_t * xq, float * xd) {""", r'''// ---- larry: the XMX multi-column path (see fastmv.hpp)
namespace xmxd {
namespace mx = sycl::ext::oneapi::experimental::matrix;
using u4 = sycl::vec<uint32_t, 4>;
using h2 = sycl::vec<sycl::half, 2>;

template <int KS>
void run(sycl::queue & q, const Args & a, const sycl::half * xT, const float * xdT) {
    const int64_t N = a.N, K = a.K, G = a.G, g_div = a.g_div, nb_tot = a.n_blocks_total;
    const size_t wbb = a.w_batch_bytes;
    const int64_t bpr = K / 256, nrg = (N + 15) / 16;
    const int ncols = a.ncols;
    const void * w = a.w;
    float * y = a.y;
    const int64_t ycs = a.y_col_stride, ybs = a.y_batch_stride;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(KS > 1 ? (size_t) KS * 16 * 8 : 1), h);
        auto kern = [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t g = it.get_global_id(0);
            const int64_t rg = it.get_group(1);
            const sycl::sub_group sg = it.get_sub_group();
            const int ks = (int) sg.get_group_linear_id();
            const int ln = (int) sg.get_local_linear_id();
            const int64_t row = sycl::min(rg * 16 + ln, N - 1);   // a tail lane decodes the last row (not stored)
            const uint8_t * base = (const uint8_t *) w + (g / g_div) * wbb;
            const uint8_t * ql_all = base;
            const uint8_t * qh_all = base + 128 * nb_tot;
            const int8_t * sc_all = (const int8_t *) (base + 192 * nb_tot);
            const uint16_t * d_all = (const uint16_t *) (base + 208 * nb_tot);
            const sycl::half * xg = xT + g * K * 8;
            const float * xdg = xdT + g * (K / 32) * 8;
            const int64_t per = (bpr + KS - 1) / KS;
            const int64_t b0 = ks * per, b1 = sycl::min(bpr, b0 + per);
            const h2 off((sycl::half) 1056.0f, (sycl::half) 1056.0f);
            float c[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int64_t b = b0; b < b1; ++b) {
                const int64_t blk = row * bpr + b;
                const u4 * qlp = reinterpret_cast<const u4 *>(ql_all + blk * 128);
                const u4 * qhp = reinterpret_cast<const u4 *>(qh_all + blk * 64);
                const u4 SC = *reinterpret_cast<const u4 *>(sc_all + blk * 16);
                const float d = (float) sycl::bit_cast<sycl::half>(d_all[blk]);
                const sycl::half * xb = xg + b * 256 * 8;
#pragma unroll
                for (int ipx = 0; ipx < 2; ++ipx) {   // half ipx: its ql (64 B) and qh (32 B) in registers
                    u4 QL[4], QH[2];
#pragma unroll
                    for (int i = 0; i < 4; ++i) QL[i] = qlp[4 * ipx + i];
#pragma unroll
                    for (int i = 0; i < 2; ++i) QH[i] = qhp[2 * ipx + i];
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {   // 32-value block j: scale groups 2 j, 2 j + 1, activation scale j
                        const int j = 4 * ipx + jj;
                        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> acc;
                        mx::joint_matrix_fill(sg, acc, 0.0f);
#pragma unroll
                        for (int mm = 0; mm < 2; ++mm) {   // values 16 m .. 16 m + 15: scale group m
                            const int m = 2 * j + mm;
                            const int k = (m % 8) / 2, hh = m % 2;
                            const u4 qlw = QL[2 * (k & 1) + hh];
                            const u4 qhw = QH[hh];
                            const int8_t scm = (int8_t) ((SC[m / 4] >> (8 * (m % 4))) & 0xFF);
                            const sycl::half sh = (sycl::half) (float) scm;
                            const h2 s2(sh, sh);
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> mb;
                            auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
#pragma unroll
                            for (int wd = 0; wd < 4; ++wd) {   // 4 values a word
                                const uint32_t nib = (k & 2) ? ((qlw[wd] >> 4) & 0x0F0F0F0Fu) : (qlw[wd] & 0x0F0F0F0Fu);
                                const uint32_t q6 = nib | (((qhw[wd] >> (2 * k)) & 0x03030303u) << 4);
                                const h2 lo = (sycl::bit_cast<h2>((q6 & 0x00FF00FFu) | 0x64006400u) - off) * s2;          // v, v + 2
                                const h2 hi = (sycl::bit_cast<h2>(((q6 >> 8) & 0x00FF00FFu) | 0x64006400u) - off) * s2;   // v + 1, v + 3
                                wb[4 * wd + 0] = lo[0]; wb[4 * wd + 1] = hi[0]; wb[4 * wd + 2] = lo[1]; wb[4 * wd + 3] = hi[1];
                            }
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16, mx::layout::row_major> ma;
                            const sycl::vec<sycl::half, 8> a8 = *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(xb + (16 * m + ln) * 8);
                            auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
#pragma unroll
                            for (int r = 0; r < 8; ++r) wa[r] = a8[r];
                            mx::joint_matrix_mad(sg, acc, ma, mb, acc);
                        }
                        // the block's exact integer sums, times xd[token] * d[row]
                        const sycl::float4 * xdp = reinterpret_cast<const sycl::float4 *>(xdg + (b * 8 + j) * 8);
                        const sycl::float4 x0 = xdp[0], x1 = xdp[1];
                        const float xs[8] = {x0.x(), x0.y(), x0.z(), x0.w(), x1.x(), x1.y(), x1.z(), x1.w()};
                        auto wc = sycl::ext::oneapi::detail::get_wi_data(sg, acc);
#pragma unroll
                        for (int i = 0; i < 8; ++i) c[i] += (float) wc[i] * (xs[i] * d);
                    }
                }
            }
            if constexpr (KS > 1) {
#pragma unroll
                for (int i = 0; i < 8; ++i) part[((size_t) ks * 16 + ln) * 8 + i] = c[i];
                it.barrier(sycl::access::fence_space::local_space);
                if (ks != 0) return;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float s = 0.0f;
#pragma unroll
                    for (int k2 = 0; k2 < KS; ++k2) s += part[((size_t) k2 * 16 + ln) * 8 + i];
                    c[i] = s;
                }
            }
            const int64_t orow = rg * 16 + ln;
            if (orow < N) {
#pragma unroll
                for (int t = 0; t < 8; ++t)
                    if (t < ncols) y[g * ybs + t * ycs + orow] = c[t];
            }
        };
        const sycl::nd_range<2> nr(sycl::range<2>((size_t) G, (size_t) (nrg * KS * 16)), sycl::range<2>(1, (size_t) KS * 16));
        namespace sxo = sycl::ext::oneapi::experimental;
        h.parallel_for(nr, sxo::properties{sycl::ext::intel::experimental::grf_size<256>}, kern);
    });
}
}  // namespace xmxd

bool xmx_wants(const Args & a) {
    static const int on = [] { const char * e = std::getenv("GGML_SYCL_LARRY_XMXMV"); return e ? std::atoi(e) : 1; }();
    if (!on || a.ncols < 2 || a.ncols > 8 || a.K % 256 || a.G != 1) return false;
    return a.K <= 2304 || a.N <= 1536;   // latency-bound for dp4a (bench/dqb/xmx_mv.cpp)
}

bool launch_xmx(sycl::queue & q, const Args & a, const sycl::half * xT, const float * xdT) {
    if (!xmx_wants(a)) return false;
    const int64_t bpr = a.K / 256, sg = (a.N + 15) / 16 * a.G;
    int ks = 1;
    while (ks < 16 && sg * ks < 512 && ks * 2 <= bpr) ks *= 2;
    switch (ks) {
        case 1: xmxd::run<1>(q, a, xT, xdT); break;
        case 2: xmxd::run<2>(q, a, xT, xdT); break;
        case 4: xmxd::run<4>(q, a, xT, xdT); break;
        case 8: xmxd::run<8>(q, a, xT, xdT); break;
        default: xmxd::run<16>(q, a, xT, xdT); break;
    }
    return true;
}

// one sub-group a 32-value block: lane l holds values 2 l, 2 l + 1 of every column
sycl::event quant_xT(sycl::queue & q, const float * x, int64_t x_col_stride, int64_t x_batch_stride, int ncols, int64_t G,
                     int64_t K, sycl::half * xT, float * xdT) {
    const int64_t nj = K / 32;
    const size_t gsz = (size_t) ((G * nj * 16 + 63) / 64 * 64);   // whole work-groups; extra sub-groups leave
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsz), sycl::range<1>(64)),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t sgid = (int64_t) it.get_global_id(0) / 16;
        if (sgid >= G * nj) return;
        const int64_t g = sgid / nj, j = sgid % nj;
        const sycl::sub_group sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        float v0[8], v1[8];
        float sc[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < ncols) {
                const sycl::float2 p = *reinterpret_cast<const sycl::float2 *>(x + g * x_batch_stride + t * x_col_stride + 32 * j + 2 * l);
                v0[t] = p.x(); v1[t] = p.y();
            } else { v0[t] = 0.f; v1[t] = 0.f; }
            const float m = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0[t]), sycl::fabs(v1[t])), sycl::maximum<float>());
            sc[t] = m > 0.f ? 127.0f / m : 0.f;
            if (l == t) xdT[(g * nj + j) * 8 + t] = m / 127.0f;
        }
        sycl::vec<sycl::half, 8> o0, o1;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o0[t] = (sycl::half) sycl::rint(v0[t] * sc[t]); o1[t] = (sycl::half) sycl::rint(v1[t] * sc[t]); }
        sycl::half * op = xT + (g * K + 32 * j + 2 * l) * 8;
        *reinterpret_cast<sycl::vec<sycl::half, 8> *>(op) = o0;
        *reinterpret_cast<sycl::vec<sycl::half, 8> *>(op + 8) = o1;
    });
}

sycl::event quant_x(sycl::queue & q, const float * x, int64_t x_col_stride, int64_t x_batch_stride, int ncols, int64_t G,
                    int64_t K, int8_t * xq, float * xd) {''')

S = G + "ggml-sycl.cpp"
rep(S, """    a.ncols = (int) ncols;
    static const int qreuse = ggml_sycl_get_env("GGML_SYCL_LARRY_QREUSE", 1);""", """    a.ncols = (int) ncols;
    if (t == fastmv::WType::Q6K_R8 && fastmv::xmx_wants(a)) {
        // larry, GGML_SYCL_LARRY_XMXMV: XMX for the latency-bound verify shapes; the transposed activations are kept
        // per graph like the int8 ones (same input = q_a / kv's attn_norm, quantised once)
        struct xt_cache { const ggml_tensor * t = nullptr; const void * data = nullptr; int64_t K = 0, nc = 0, G = 0, xc = 0, xb = 0;
                          uint64_t epoch = 0; sycl::half * xT = nullptr; float * xd = nullptr; size_t cap = 0; };
        static std::unordered_map<const void *, std::array<xt_cache, 2>> cache;
        static std::unordered_map<const void *, int> next;
        auto & cs = cache[&ctx];
        xt_cache * hit = nullptr;
        for (auto & e : cs)
            if (e.epoch == g_larry_graph_epoch && e.t == src1 && e.data == src1->data && e.K == K && e.nc == ncols && e.G == G &&
                e.xc == a.x_col_stride && e.xb == a.x_batch_stride) { hit = &e; break; }
        if (!hit) {
            int & nx = next[&ctx];
            hit = &cs[(size_t) nx];
            nx = (nx + 1) % 2;
            const size_t need = (size_t) (G * K * 8);
            if (need > hit->cap) {   // grows a few times early on: queued kernels may still read the old buffer
                ctx.stream()->wait();
                if (hit->xT) { sycl::free(hit->xT, *ctx.stream()); sycl::free(hit->xd, *ctx.stream()); }
                hit->cap = std::max<size_t>(need, (size_t) 8 * 8192);
                hit->xT = sycl::malloc_device<sycl::half>(hit->cap, *ctx.stream());
                hit->xd = sycl::malloc_device<float>(hit->cap / 32, *ctx.stream());
                if (!hit->xT || !hit->xd) GGML_ABORT("larry_fastmv: xT cache allocation failed");
            }
            fastmv::quant_xT(*ctx.stream(), a.x, a.x_col_stride, a.x_batch_stride, (int) ncols, G, K, hit->xT, hit->xd);
            hit->t = src1; hit->data = src1->data; hit->K = K; hit->nc = ncols; hit->G = G;
            hit->xc = a.x_col_stride; hit->xb = a.x_batch_stride; hit->epoch = g_larry_graph_epoch;
        }
        static const int xchk = ggml_sycl_get_env("GGML_SYCL_LARRY_XMXMV_CHECK", 0);   // diagnostic: vs the dp4a path
        if (xchk) {
            const size_t ny = (size_t) (N * ncols * G);
            ggml_sycl_pool_alloc<float> ty(ctx.pool(), ny);
            fastmv::Args b = a;
            b.y = ty.get(); b.y_col_stride = N; b.y_batch_stride = N * ncols;
            if (!fastmv::launch_xmx(*ctx.stream(), b, hit->xT, hit->xd)) GGML_ABORT("xmx check: launch");
            ggml_sycl_pool_alloc<int8_t> rq(ctx.pool(), (size_t) (G * ncols * K));
            ggml_sycl_pool_alloc<float> rd(ctx.pool(), (size_t) (G * ncols * K / 32));
            fastmv::quant_x(*ctx.stream(), a.x, a.x_col_stride, a.x_batch_stride, (int) ncols, G, K, rq.get(), rd.get());
            fastmv::Args r = a;
            r.xq = rq.get(); r.xd = rd.get();   // into dst
            fastmv::launch(*ctx.stream(), t, r);
            static int nchk = 0;
            if (nchk < 40) {
                ++nchk;
                std::vector<float> va(ny), vb(ny);
                ctx.stream()->wait();
                for (int64_t g2 = 0; g2 < G; ++g2)
                    for (int64_t c2 = 0; c2 < ncols; ++c2)
                        ctx.stream()->memcpy(va.data() + (g2 * ncols + c2) * N, a.y + g2 * a.y_batch_stride + c2 * a.y_col_stride, N * 4);
                ctx.stream()->memcpy(vb.data(), ty.get(), ny * 4).wait();
                double md = 0, mx = 0;
                for (size_t i = 0; i < ny; ++i) { md = std::max(md, (double) std::fabs(va[i] - vb[i])); mx = std::max(mx, (double) std::fabs(va[i])); }
                GGML_LOG_INFO("larry xmx check %d: %s %lld x %lld x %d cols: max |diff| %.3e, max |dp4a| %.3e\n", nchk, dst->name,
                              (long long) K, (long long) N, (int) ncols, md, mx);
            }
            return true;
        }
        if (fastmv::launch_xmx(*ctx.stream(), a, hit->xT, hit->xd)) return true;
    }
    static const int qreuse = ggml_sycl_get_env("GGML_SYCL_LARRY_QREUSE", 1);""")
