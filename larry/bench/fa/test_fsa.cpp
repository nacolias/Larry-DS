// test_fsa.cpp - larry (#3): fsa_kernel.inc (the prompt attention as one kernel that reads each query's keys where
// they lie) against the service's path on the same inputs - DSV41_KGATHER (pad form: [n, 656, 576] f16) + the Q' /
// mask kernels + oneDNN's graph SDPA + the output copy - and a double-precision reference (every 16th query).
//   test_fsa [n 512] [p0: the span's first position, 30000] [compressed rows visible, 16000] [nslot 64] [device 1]
// Inputs like a layer-2..19 span: 64 heads x 512 dims; 128 window rows a query (F32, from a [144 ring + n] row set,
// masked before position 0) + 512 selected BF16 compressed rows (sorted ids, masked past the visible count).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#define FSA_TIMING 1
#include "fsa_kernel.inc"

using namespace dnnl::graph;
using half = sycl::half;

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t) (u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; }

int main(int argc, char ** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 512;
    const int p0 = argc > 2 ? std::atoi(argv[2]) : 30000;
    const int NCV = argc > 3 ? std::atoi(argv[3]) : 16000;   // compressed rows visible to the span's last query
    const int nslot = argc > 4 ? std::atoi(argv[4]) : 64;
    const int dev = argc > 5 ? std::atoi(argv[5]) : 1;
    const int skip = argc > 6 ? std::atoi(argv[6]) : 0;
    const int NH = 64, D = 512, SWA = 128, K = 512, M = SWA + K, RAW = 144, NW = RAW + n, DP = 576, MP = 656;
    const float scale = 1.0f / std::sqrt((float) D);
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::queue q(devs[(size_t) dev], sycl::property::queue::in_order());
    std::printf("%s: n %d at position %d, %d compressed rows visible, %d slots\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                n, p0, NCV, nslot);
    // ---- inputs
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> hq((size_t) n * NH * D), hwin((size_t) NW * D), hsink(NH), hmask((size_t) n * M);
    std::vector<int32_t> hwidx((size_t) n * SWA), hsel((size_t) n * K);
    const int NC = NCV + 64;
    std::vector<uint16_t> hcomp((size_t) NC * D);
    for (auto & x : hq) x = 2.0f * nd(rng);
    for (auto & x : hwin) x = nd(rng);
    for (auto & x : hcomp) x = f2bf(nd(rng));
    for (auto & x : hsink) x = 1.5f * nd(rng);
    for (int t = 0; t < n; ++t) {
        const int pos = p0 + t;
        for (int m = 0; m < SWA; ++m) {   // window row m = position pos - 127 + m
            const int pm = pos - (SWA - 1) + m;
            const bool ok = pm >= 0;
            hwidx[(size_t) t * SWA + m] = !ok ? 0 : pm >= p0 ? RAW + (pm - p0) : (pm % RAW);
            hmask[(size_t) t * M + m] = ok ? 0.0f : -INFINITY;
        }
        const int nvis = std::min(NCV, (pos + 1) / 2);
        std::vector<int32_t> ids;
        if (nvis <= K) { for (int i = 0; i < nvis; ++i) ids.push_back(i); }
        else {
            std::set<int32_t> st;
            while ((int) st.size() < K) st.insert((int32_t) (rng() % (uint32_t) nvis));
            ids.assign(st.begin(), st.end());
        }
        for (int j = 0; j < K; ++j) {
            const bool ok = j < (int) ids.size();
            hsel[(size_t) t * K + j] = ok ? ids[(size_t) j] : 0;
            hmask[(size_t) t * M + SWA + j] = ok ? 0.0f : -INFINITY;
        }
    }
    float * Q = sycl::malloc_device<float>(hq.size(), q);
    float * W = sycl::malloc_device<float>(hwin.size(), q);
    uint16_t * Cm = sycl::malloc_device<uint16_t>(hcomp.size(), q);
    int32_t * WI = sycl::malloc_device<int32_t>(hwidx.size(), q);
    int32_t * SL = sycl::malloc_device<int32_t>(hsel.size(), q);
    float * MK = sycl::malloc_device<float>(hmask.size(), q);
    float * SK = sycl::malloc_device<float>(NH, q);
    q.memcpy(Q, hq.data(), hq.size() * 4); q.memcpy(W, hwin.data(), hwin.size() * 4); q.memcpy(Cm, hcomp.data(), hcomp.size() * 2);
    q.memcpy(WI, hwidx.data(), hwidx.size() * 4); q.memcpy(SL, hsel.data(), hsel.size() * 4); q.memcpy(MK, hmask.data(), hmask.size() * 4);
    q.memcpy(SK, hsink.data(), NH * 4).wait();
    // ---- the fused kernel
    half * O1 = sycl::malloc_device<half>((size_t) n * NH * D, q);
    half * scr = sycl::malloc_device<half>((size_t) nslot * fsa::SLOT, q);
    half * wcp = sycl::malloc_device<half>((size_t) (n + SWA) * D, q);
    fsa::Args fa;
    fa.q = Q; fa.win = (const char *) W; fa.wnb = (size_t) D * 4; fa.widx = WI; fa.swa = SWA;
    fa.comp = (const char *) Cm; fa.cnb = (size_t) D * 2; fa.sel = SL; fa.ldsel = K; fa.mask = MK; fa.ldm = M; fa.M = M;
    fa.sinks = SK; fa.scale = scale; fa.out = O1; fa.n = n; fa.scratch = scr; fa.nslot = nslot; fa.skip = skip; fa.wcopy = wcp;
    fsa::run(q, fa).wait();
    std::vector<double> tf;
    for (int r = 0; r < 11; ++r) { const double t0 = now_ms(); fsa::run(q, fa).wait(); tf.push_back(now_ms() - t0); }
    // ---- the service's path: KGATHER (pad) + Q' + mask + oneDNN SDPA + the output copy
    half * KP = sycl::malloc_device<half>((size_t) n * MP * DP, q);
    half * QP = sycl::malloc_device<half>((size_t) n * NH * DP, q);
    half * MS = sycl::malloc_device<half>((size_t) n * MP, q);
    half * OP = sycl::malloc_device<half>((size_t) n * NH * DP, q);
    half * O2 = sycl::malloc_device<half>((size_t) n * NH * D, q);
    float * one = sycl::malloc_device<float>(1, q);
    { const float v = 1.0f; q.memcpy(one, &v, 4).wait(); }
    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);
    using ltype = logical_tensor::layout_type;
    using dt = logical_tensor::data_type;
    const logical_tensor::dims q_sz = {n, 1, NH, 1, DP}, kv_sz = {n, 1, 1, MP, DP}, s_sz = {n, 1, NH, 1, MP},
                               sc = {1, 1, 1, 1, 1}, msk = {n, 1, 1, 1, MP}, o_sz = {n, 1, NH, 1, DP};
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
    if (parts.size() != 1 || !parts[0].is_supported()) { std::printf("oneDNN: NOT FUSED\n"); return 1; }
    auto ins = parts[0].get_input_ports();
    auto outp = parts[0].get_output_ports()[0];
    compiled_partition cp = parts[0].compile(ins, {outp}, eng);
    std::vector<tensor> tin;
    for (auto & lt : ins) {
        void * ptr = lt.get_id() == query.get_id() ? (void *) QP : lt.get_id() == key.get_id() ? (void *) KP :
                     lt.get_id() == value.get_id() ? (void *) KP : lt.get_id() == scale_t.get_id() ? (void *) one : (void *) MS;
        tin.emplace_back(lt, eng, ptr);
    }
    tensor tout(cp.query_logical_tensor(outp.get_id()), eng, OP);
    auto service_path = [&]() {
        const float * pw = W;
        const int32_t * pwi = WI;
        const uint16_t * pc = Cm;
        const int32_t * ps = SL;
        half * po = KP;
        const int VS = D / 8, VP = DP / 8;
        q.parallel_for(sycl::range<1>((size_t) n * MP * VP), [=](sycl::id<1> idx) {   // dsv41-fused.cpp's pad form
            const size_t i = idx[0];
            const int v = (int) (i % VP), row = (int) (i / VP), m = row % MP, t = row / MP;
            sycl::vec<half, 8> o(0.0f);
            if (m < M && v < VS) {
                float f[8];
                if (m < SWA) {
                    const float * s = pw + (size_t) pwi[t * SWA + m] * D + v * 8;
                    for (int j = 0; j < 8; ++j) f[j] = s[j];
                } else {
                    const int32_t r = ps[(size_t) t * K + (m - SWA)];
                    const uint16_t * s = pc + (size_t) r * D + v * 8;
                    for (int j = 0; j < 8; ++j) f[j] = sycl::bit_cast<float>((uint32_t) s[j] << 16);
                }
                for (int j = 0; j < 8; ++j) o[j] = (half) f[j];
            } else if (m == M && v == VS) {
                o[0] = (half) 1.0f;
            }
            *(sycl::vec<half, 8> *) (po + (size_t) row * DP + v * 8) = o;
        });
        const float * qq = Q;
        const float * sk = SK;
        half * qp = QP;
        q.parallel_for(sycl::range<1>((size_t) n * NH * (DP / 4)), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            const int vv = (int) (i % (DP / 4)), bh = (int) (i / (DP / 4)), d = vv * 4;
            sycl::vec<half, 4> o(0.0f);
            if (d < D) { for (int j = 0; j < 4; ++j) o[j] = (half) (qq[(size_t) bh * D + d + j] * scale); }
            else if (d == D) o[0] = (half) sk[bh % NH];
            *(sycl::vec<half, 4> *) (qp + (size_t) bh * DP + d) = o;
        });
        const float * mk = MK;
        half * ms = MS;
        q.parallel_for(sycl::range<1>((size_t) n * MP), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            const int j = (int) (i % MP), b = (int) (i / MP);
            float x = -INFINITY;
            if (j < M) x = mk[(size_t) b * M + j];
            else if (j == M) x = 0.0f;
            ms[i] = (half) x;
        });
        cp.execute(strm, tin, {tout});
        const half * op = OP;
        half * o2 = O2;
        q.parallel_for(sycl::range<1>((size_t) n * NH * (D / 4)), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            const int vv = (int) (i % (D / 4)), bh = (int) (i / (D / 4));
            *(sycl::vec<half, 4> *) (o2 + (size_t) bh * D + vv * 4) = *(const sycl::vec<half, 4> *) (op + (size_t) bh * DP + vv * 4);
        });
    };
    service_path();
    q.wait();
    std::vector<double> ts;
    for (int r = 0; r < 11; ++r) { const double t0 = now_ms(); service_path(); q.wait(); ts.push_back(now_ms() - t0); }
    // ---- compare: both against the reference (every 16th query), and each other (all)
    std::vector<half> h1((size_t) n * NH * D), h2((size_t) n * NH * D);
    q.memcpy(h1.data(), O1, h1.size() * 2);
    q.memcpy(h2.data(), O2, h2.size() * 2).wait();
    double mx = 0, e1 = 0, e2 = 0, e12 = 0, s1 = 0, s2 = 0;
    size_t nb1 = 0, cnt = 0;
    std::vector<double> sco(M), kf((size_t) M * D);
    for (int t = 0; t < n; t += 16) {
        for (int m = 0; m < M; ++m) {   // the keys as the paths round them (f16)
            const float * src = nullptr;
            for (int d = 0; d < D; ++d) {
                float v;
                if (m < SWA) v = hwin[(size_t) hwidx[(size_t) t * SWA + m] * D + d];
                else v = bf2f(hcomp[(size_t) hsel[(size_t) t * K + (m - SWA)] * D + d]);
                kf[(size_t) m * D + d] = (double) (float) (half) v;
            }
            (void) src;
        }
        for (int hd = 0; hd < NH; ++hd) {
            const float * qr = &hq[((size_t) t * NH + hd) * D];
            double mxs = hsink[hd];
            for (int m = 0; m < M; ++m) {
                if (hmask[(size_t) t * M + m] != 0.0f) { sco[m] = -INFINITY; continue; }
                double a = 0;
                for (int d = 0; d < D; ++d) a += (double) (float) (half) (qr[d] * scale) * kf[(size_t) m * D + d];
                sco[m] = a;
                mxs = std::max(mxs, a);
            }
            double den = std::exp(hsink[hd] - mxs);
            for (int m = 0; m < M; ++m) if (std::isfinite(sco[m])) den += std::exp(sco[m] - mxs);
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int m = 0; m < M; ++m) if (std::isfinite(sco[m])) o += std::exp(sco[m] - mxs) / den * kf[(size_t) m * D + d];
                const size_t ix = ((size_t) t * NH + hd) * D + d;
                const double a1 = (float) h1[ix], a2 = (float) h2[ix];
                mx = std::max(mx, std::fabs(o));
                e1 = std::max(e1, std::fabs(a1 - o)); e2 = std::max(e2, std::fabs(a2 - o));
                s1 += (a1 - o) * (a1 - o); s2 += (a2 - o) * (a2 - o);
                ++cnt;
                if (!std::isfinite(a1)) ++nb1;
            }
        }
    }
    for (size_t i = 0; i < h1.size(); ++i) e12 = std::max(e12, std::fabs((double) (float) h1[i] - (double) (float) h2[i]));
    std::printf("vs the reference (%zu values, max |O| %.3f): fused max %.2e rms %.2e, service max %.2e rms %.2e; fused vs service max %.2e; non-finite %zu\n",
                cnt, mx, e1, std::sqrt(s1 / cnt), e2, std::sqrt(s2 / cnt), e12, nb1);
    const double flop = 4.0 * n * NH * (double) M * D;
    std::printf("skip %d: time (median of 11): fused %.3f ms (%.1f TFLOPS), service path %.3f ms (%.2fx)\n", skip, med(tf), flop / med(tf) * 1e-9, med(ts),
                med(ts) / med(tf));
    return 0;
}
