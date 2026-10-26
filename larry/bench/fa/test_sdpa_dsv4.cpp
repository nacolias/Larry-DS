// test_sdpa_dsv4.cpp - larry: DeepSeek-V4.1's prompt attention (a 512-query span: 64 MQA heads x 512, M keys a query,
// a sink logit a head) as ONE oneDNN graph SDPA - the fused micro-kernel that ggml-sycl's fattn-onednn.cpp gets for
// GQA: batch = the queries, one KV head, a group of 64 = the heads, a query length of 1.  The head's sink: an extra
// key [0 .. 0, 1, 0 ..] scored by an extra Q dimension holding sink / scale - its score is the sink, and as a value row
// it touches only dimension 512 (dropped).  Checked against the naive reference with the same roundings as
// test_mqa_pf.cpp; timed against the oneMKL path (Q -> f16, gemm_batch QK, softmax, gemm_batch PV).
//   test_sdpa_dsv4 [n queries, 512] [extra dims, 16] [M keys, 640]
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace dnnl::graph;

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// the oneMKL path of fattn-dsv4.cpp (a plain softmax kernel): timed for the comparison
static void mkl_path(sycl::queue & q, const float * Q, const sycl::half * K, const sycl::half * mask, const float * sinks,
                     float * O, int n, int NH, int D, int M, float scale, sycl::half * qh, float * s, sycl::half * p) {
    namespace mkl = oneapi::mkl;
    const size_t total = (size_t) n * NH * D;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) { qh[id[0]] = (sycl::half) (Q[id[0]] * scale); });
    mkl::blas::row_major::gemm_batch(q, mkl::transpose::nontrans, mkl::transpose::trans, NH, M, D, 1.0f, qh, D,
                                     (int64_t) NH * D, K, D, (int64_t) M * D, 0.0f, s, M, (int64_t) NH * M, n);
    const int WGS = 256;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n * NH * WGS), sycl::range<1>(WGS)), [=](sycl::nd_item<1> it) {
        const int64_t row = (int64_t) it.get_group(0);
        const int64_t b = row / NH, h = row % NH;
        const int lid = (int) it.get_local_id(0);
        const float * sr = s + row * M;
        const sycl::half * mr = mask + b * M;
        float mx = sinks[h];
        for (int64_t m = lid; m < M; m += WGS) mx = sycl::fmax(mx, sr[m] + (float) mr[m]);
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float sum = 0.0f;
        for (int64_t m = lid; m < M; m += WGS) sum += sycl::exp(sr[m] + (float) mr[m] - mx);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>()) + sycl::exp(sinks[h] - mx);
        const float inv = 1.0f / sum;
        sycl::half * pr = p + row * M;
        for (int64_t m = lid; m < M; m += WGS) pr[m] = (sycl::half) (sycl::exp(sr[m] + (float) mr[m] - mx) * inv);
    });
    mkl::blas::row_major::gemm_batch(q, mkl::transpose::nontrans, mkl::transpose::nontrans, NH, D, M, 1.0f, p, M,
                                     (int64_t) NH * M, K, D, (int64_t) M * D, 0.0f, O, D, (int64_t) NH * D, n);
    q.wait();
}

int main(int argc, char ** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 512, XD = argc > 2 ? std::atoi(argv[2]) : 16, M = argc > 3 ? std::atoi(argv[3]) : 640;
    const int NH = 64, D = 512, Dp = D + XD, Mp = (M + 1 + 15) / 16 * 16;
    const float scale = 1.0f / std::sqrt((float) D);
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::queue q(devs[devs.size() > 1 ? 1 : 0], sycl::property::queue::in_order());
    std::printf("%s: n %d, %d heads x %d, M %d keys (+ sink, padded %d), D' %d\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), n, NH, D, M, Mp, Dp);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::vector<float> hq((size_t) n * NH * D), hs(NH);
    std::vector<sycl::half> hk((size_t) n * M * D), hm((size_t) n * M);
    for (auto & x : hq) x = 2.0f * nd(rng);
    for (auto & x : hk) x = (sycl::half) nd(rng);
    for (auto & x : hs) x = 2.0f * nd(rng);
    for (int b = 0; b < n; ++b)
        for (int j = 0; j < M; ++j) {
            const bool off = b == 0 || ud(rng) < 0.1f || (j < 128 && j > 64 + b);
            hm[(size_t) b * M + j] = off ? (sycl::half) -INFINITY : (sycl::half) 0.0f;
        }
    // the reference (double sums, the oneMKL path's roundings: Q scaled to f16, P f16)
    std::vector<float> ref((size_t) n * NH * D);
    {
        std::vector<double> sc(M);
        for (int b = 0; b < n; b += 16)   // every 16th query (query 0: fully masked - only its sink)
            for (int h = 0; h < NH; ++h) {
                const float * qr = &hq[((size_t) b * NH + h) * D];
                double mx = hs[h];
                for (int j = 0; j < M; ++j) {
                    double a = 0;
                    for (int d = 0; d < D; ++d) a += (double) (float) (sycl::half) (qr[d] * scale) * (double) (float) hk[((size_t) b * M + j) * D + d];
                    sc[j] = a + (double) (float) hm[(size_t) b * M + j];
                    mx = std::max(mx, sc[j]);
                }
                double sum = std::exp(hs[h] - mx);
                for (int j = 0; j < M; ++j) sum += std::isfinite(sc[j]) ? std::exp(sc[j] - mx) : 0.0;
                float * o = &ref[((size_t) b * NH + h) * D];
                for (int d = 0; d < D; ++d) o[d] = 0.0f;
                for (int j = 0; j < M; ++j) {
                    if (!std::isfinite(sc[j])) continue;
                    const float pj = (float) (sycl::half) (float) (std::exp(sc[j] - mx) / sum);
                    for (int d = 0; d < D; ++d) o[d] += pj * (float) hk[((size_t) b * M + j) * D + d];
                }
            }
    }
    float * Q = sycl::malloc_device<float>(hq.size(), q);
    sycl::half * K = sycl::malloc_device<sycl::half>(hk.size(), q);
    sycl::half * Mk = sycl::malloc_device<sycl::half>(hm.size(), q);
    float * S = sycl::malloc_device<float>(NH, q);
    float * O1 = sycl::malloc_device<float>((size_t) n * NH * D, q);
    sycl::half * qh = sycl::malloc_device<sycl::half>((size_t) n * NH * D, q);
    float * s = sycl::malloc_device<float>((size_t) n * NH * M, q);
    sycl::half * p = sycl::malloc_device<sycl::half>((size_t) n * NH * M, q);
    q.memcpy(Q, hq.data(), hq.size() * 4); q.memcpy(K, hk.data(), hk.size() * 2);
    q.memcpy(Mk, hm.data(), hm.size() * 2); q.memcpy(S, hs.data(), NH * 4).wait();
    std::vector<double> tm;
    for (int r = 0; r < 11; ++r) { const double t0 = now_ms(); mkl_path(q, Q, K, Mk, S, O1, n, NH, D, M, scale, qh, s, p); tm.push_back(now_ms() - t0); }
    // ---- the SDPA inputs: Q' [n][64][D'] f16 (dim D = sink / scale), K' = V' [n][Mp][D'], mask' [n][Mp]
    sycl::half * Qp = sycl::malloc_device<sycl::half>((size_t) n * NH * Dp, q);
    sycl::half * Kp = sycl::malloc_device<sycl::half>((size_t) n * Mp * Dp, q);
    sycl::half * Mp_ = sycl::malloc_device<sycl::half>((size_t) n * Mp, q);
    sycl::half * Op = sycl::malloc_device<sycl::half>((size_t) n * NH * Dp, q);
    float * scl = sycl::malloc_device<float>(1, q);
    {
        const float isc = 1.0f / scale;
        q.parallel_for(sycl::range<1>((size_t) n * NH * Dp), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int d = (int) (i % Dp);
            const size_t bh = i / Dp;
            const int h = (int) (bh % NH);
            Qp[i] = d < D ? (sycl::half) (Q[bh * D + d] * scale) : d == D ? (sycl::half) (S[h] * isc * scale) : (sycl::half) 0.0f;
        });
        q.parallel_for(sycl::range<1>((size_t) n * Mp * Dp), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int d = (int) (i % Dp);
            const size_t bj = i / Dp;
            const int j = (int) (bj % Mp);
            const size_t b = bj / Mp;
            Kp[i] = j < M ? (d < D ? K[(b * M + j) * D + d] : (sycl::half) 0.0f) : (j == M && d == D ? (sycl::half) 1.0f : (sycl::half) 0.0f);
        });
        q.parallel_for(sycl::range<1>((size_t) n * Mp), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int j = (int) (i % Mp);
            const size_t b = i / Mp;
            Mp_[i] = j < M ? Mk[b * M + j] : j == M ? (sycl::half) 0.0f : (sycl::half) -INFINITY;
        });
        const float one = 1.0f;   // Q' carries the scale already (as the oneMKL path's Qh): the graph's Divide by 1
        q.memcpy(scl, &one, 4).wait();
    }
    // ---- the graph (ggml-sycl fattn-onednn.cpp's GQA form; mask [n, 1, 1, 1, Mp])
    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);
    using ltype = logical_tensor::layout_type;
    using dt = logical_tensor::data_type;
    const logical_tensor::dims q_sz = {n, 1, NH, 1, Dp}, kv_sz = {n, 1, 1, Mp, Dp}, s_sz = {n, 1, NH, 1, Mp},
                               sc = {1, 1, 1, 1, 1}, msk = {n, 1, 1, 1, Mp}, o_sz = {n, 1, NH, 1, Dp};
    int64_t id = 0;
    auto query = logical_tensor(id++, dt::f16, q_sz, ltype::strided);
    auto key = logical_tensor(id++, dt::f16, kv_sz, ltype::strided);
    auto score = logical_tensor(id++, dt::f32, s_sz, ltype::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});
    auto scale_t = logical_tensor(id++, dt::f32, sc, ltype::strided);
    auto scaled = logical_tensor(id++, dt::f32, s_sz, ltype::strided);
    auto sdiv = op(id++, op::kind::Divide, "scale_div");
    sdiv.add_inputs({score, scale_t}); sdiv.add_outputs({scaled});
    auto mask = logical_tensor(id++, dt::f16, msk, ltype::strided);
    auto masked = logical_tensor(id++, dt::f32, s_sz, ltype::strided);
    auto madd = op(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask}); madd.add_outputs({masked});
    auto probs = logical_tensor(id++, dt::f16, s_sz, ltype::strided);
    auto smax = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked}); smax.add_outputs({probs});
    auto value = logical_tensor(id++, dt::f16, kv_sz, ltype::strided);
    auto output = logical_tensor(id++, dt::f16, o_sz, ltype::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});
    dnnl::graph::graph g(eng.get_kind());
    g.add_op(bmm1); g.add_op(sdiv); g.add_op(madd); g.add_op(smax); g.add_op(bmm2);
    g.finalize();
    auto parts = g.get_partitions();
    std::printf("partitions: %zu, supported %d\n", parts.size(), parts.empty() ? 0 : (int) parts[0].is_supported());
    if (parts.size() != 1 || !parts[0].is_supported()) { std::printf("NOT FUSED\n"); return 1; }
    auto ins = parts[0].get_input_ports();
    auto outp = parts[0].get_output_ports()[0];
    const double tc = now_ms();
    compiled_partition cp = parts[0].compile(ins, {outp}, eng);
    std::printf("compiled in %.0f ms\n", now_ms() - tc);
    std::vector<tensor> tin;
    for (auto & lt : ins) {
        void * ptr = lt.get_id() == query.get_id() ? (void *) Qp : lt.get_id() == key.get_id() ? (void *) Kp :
                     lt.get_id() == value.get_id() ? (void *) Kp : lt.get_id() == scale_t.get_id() ? (void *) scl : (void *) Mp_;
        tin.emplace_back(lt, eng, ptr);
    }
    tensor tout(cp.query_logical_tensor(outp.get_id()), eng, Op);
    std::vector<double> ts;
    for (int r = 0; r < 11; ++r) {
        const double t0 = now_ms();
        cp.execute(strm, tin, {tout});
        strm.wait();
        ts.push_back(now_ms() - t0);
    }
    // compare (dims 0..511) against the reference and the oneMKL path
    std::vector<sycl::half> ho((size_t) n * NH * Dp);
    std::vector<float> o1((size_t) n * NH * D);
    q.memcpy(ho.data(), Op, ho.size() * 2);
    q.memcpy(o1.data(), O1, o1.size() * 4).wait();
    double mx = 0, dS = 0, dM = 0;
    size_t bad = 0;
    for (size_t bh = 0; bh < (size_t) n * NH; ++bh)
        for (int d = 0; d < D && (bh / NH) % 16 == 0; ++d) {
            const double r = ref[bh * D + d], a = (float) ho[bh * Dp + d], c = o1[bh * D + d];
            mx = std::max(mx, std::fabs(r));
            dS = std::max(dS, std::fabs(a - r)); dM = std::max(dM, std::fabs(c - r));
            if (!std::isfinite(a)) ++bad;
        }
    std::printf("max|O| %.3f: SDPA vs ref %.2e (rel %.2e), oneMKL vs ref %.2e; non-finite %zu\n", mx, dS, dS / mx, dM, bad);
    std::printf("time (wall, median of 11): oneMKL path %.3f ms, SDPA %.3f ms (%.2fx)\n", med(tm), med(ts), med(tm) / med(ts));
    return 0;
}
