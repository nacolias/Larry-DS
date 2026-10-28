// test_sdpa_blk.cpp - larry (#3): a 512-query span's prompt attention as oneDNN graph SDPAs over BLOCKS of Bq
// consecutive queries that share one key set (the union of their selected rows + their window rows + the sink row,
// Kb keys, each query masked to its own): batch = the 512 / Bq blocks, one KV head, a group of 64 = the heads, a query
// length of Bq, a mask [blocks, 1, 1, Bq, Kb].  Times the SDPA alone (the gather / union / mask kernels are cheap next
// to it) at a few (Bq, Kb) - what a block-union attention would cost against today's per-query SDPA (Bq 1, Kb 656,
// ~1.1 ms) + its key gather (~0.78 ms).  Random data; each query masks ~(1 - 640 / Kb) of the block's keys off.
//   test_sdpa_blk [D' (576)] then pairs "Bq Kb" on stdin or the built-in list
#include <sycl/sycl.hpp>
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

static void run(sycl::queue & q, dnnl::engine & eng, dnnl::stream & strm, int Bq, int Kb, int Dp) {
    const int NQ = 512, NH = 64, nb = NQ / Bq;
    using ltype = logical_tensor::layout_type;
    using dt = logical_tensor::data_type;
    const logical_tensor::dims q_sz = {nb, 1, NH, Bq, Dp}, kv_sz = {nb, 1, 1, Kb, Dp}, s_sz = {nb, 1, NH, Bq, Kb},
                               sc = {1, 1, 1, 1, 1}, msk = {nb, 1, 1, Bq, Kb}, o_sz = {nb, 1, NH, Bq, Dp};
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
    if (parts.size() != 1 || !parts[0].is_supported()) { std::printf("Bq %3d Kb %5d: NOT FUSED (%zu partitions)\n", Bq, Kb, parts.size()); return; }
    auto ins = parts[0].get_input_ports();
    auto outp = parts[0].get_output_ports()[0];
    const double tc = now_ms();
    compiled_partition cp = parts[0].compile(ins, {outp}, eng);
    const double tcomp = now_ms() - tc;
    const size_t nq = (size_t) nb * NH * Bq * Dp, nk = (size_t) nb * Kb * Dp, nm = (size_t) nb * Bq * Kb;
    sycl::half * Qp = sycl::malloc_device<sycl::half>(nq, q);
    sycl::half * Kp = sycl::malloc_device<sycl::half>(nk, q);
    sycl::half * Mk = sycl::malloc_device<sycl::half>(nm, q);
    sycl::half * Op = sycl::malloc_device<sycl::half>(nq, q);
    float * scl = sycl::malloc_device<float>(1, q);
    {
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        std::vector<sycl::half> hq(nq), hk(nk), hm(nm);
        for (auto & x : hq) x = (sycl::half) (0.09f * nd(rng));
        for (auto & x : hk) x = (sycl::half) nd(rng);
        const float keep = std::min(1.0f, 641.0f / (float) Kb);
        for (auto & x : hm) x = ud(rng) < keep ? (sycl::half) 0.0f : (sycl::half) -INFINITY;
        const float one = 1.0f;
        q.memcpy(Qp, hq.data(), nq * 2); q.memcpy(Kp, hk.data(), nk * 2); q.memcpy(Mk, hm.data(), nm * 2);
        q.memcpy(scl, &one, 4).wait();
    }
    std::vector<tensor> tin;
    for (auto & lt : ins) {
        void * ptr = lt.get_id() == query.get_id() ? (void *) Qp : lt.get_id() == key.get_id() ? (void *) Kp :
                     lt.get_id() == value.get_id() ? (void *) Kp : lt.get_id() == scale_t.get_id() ? (void *) scl : (void *) Mk;
        tin.emplace_back(lt, eng, ptr);
    }
    tensor tout(cp.query_logical_tensor(outp.get_id()), eng, Op);
    std::vector<double> ts;
    for (int r = 0; r < 15; ++r) {
        const double t0 = now_ms();
        cp.execute(strm, tin, {tout});
        strm.wait();
        if (r >= 2) ts.push_back(now_ms() - t0);
    }
    const double flop = 4.0 * NQ * NH * (double) Kb * Dp, tm = med(ts);
    std::printf("Bq %3d Kb %5d (Dp %d): SDPA %.3f ms (%.1f TFLOPS on the block's keys; %.2f keys a query vs 656), compiled in %.0f ms; "
                "a span's gather %.1f MB\n", Bq, Kb, Dp, tm, flop / tm * 1e-9, (double) Kb / 656.0, tcomp, (double) nb * Kb * Dp * 2 / 1e6);
    sycl::free(Qp, q); sycl::free(Kp, q); sycl::free(Mk, q); sycl::free(Op, q); sycl::free(scl, q);
}

int main(int argc, char ** argv) {
    const int Dp = argc > 1 ? std::atoi(argv[1]) : 576;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::queue q(devs[devs.size() > 1 ? 1 : 0], sycl::property::queue::in_order());
    std::printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);
    const int list[][2] = {{1, 656}, {2, 768}, {2, 1024}, {4, 896}, {4, 1280}, {8, 1152}, {8, 1536}, {8, 2048}, {16, 1280},
                           {16, 1792}, {16, 2560}, {32, 2048}, {32, 3072}, {64, 3072}, {64, 4608}};
    for (auto & p : list) run(q, eng, strm, p[0], p[1], Dp);
    return 0;
}
