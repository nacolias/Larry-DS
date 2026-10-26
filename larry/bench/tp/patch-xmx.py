#!/usr/bin/env python3
# patch-xmx.py - larry, STRATA_XE_XMX (default 0 for the A/B): decode / verify routed experts on the XMX engines
# (xe_xmx.cpp, fp16 DPAS: the weights stream once whatever the tokens a job has) instead of the planar dp4a kernels.
# Adds xe_xmx.cpp to the tier, its two entry points to xe_kernels.hpp, each card's fp16 X / H scratch, the switch in
# card_experts, and test_ds41_xe --bench-q8 runs both paths (time + max difference).  Idempotent.
import shutil, sys

XE = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe"
FILES = {"hpp": XE + "/xe_kernels.hpp", "cmake": XE + "/xe.cmake", "tier": XE + "/ds41_xe_tier.cpp", "test": XE + "/test_ds41_xe.cpp"}
src = {k: open(p).read() for k, p in FILES.items()}
if "xe_experts_x" in src["hpp"]:
    print("already"); sys.exit(0)


def rep(k, old, new, cnt=1):
    s = src[k]
    if s.count(old) != cnt:
        sys.exit(f"{k}: anchor x{s.count(old)}: {old[:100]!r}")
    src[k] = s.replace(old, new)


rep("hpp", """/// xe_experts_y on PLANAR experts (Y: [nj][8][n_embd]; H scratch as above).""",
    """/// larry (xe_xmx.cpp): the same on the XMX engines (fp16 DPAS) - xt = xe_xt of the call's rows ([n_embd][8] fp16,
/// rows >= ntok zero), ht = scratch [nj][n_ff][8] fp16.  The weights stream once whatever a job's tokens.
sycl::event xe_xt(sycl::queue& q, const float* x, int ntok, int n, sycl::half* xt);
sycl::event xe_experts_x(sycl::queue& q, const XeJob* jobs, int nj, int ntok, const sycl::half* xt, sycl::half* ht,
                         float* OUT, int n_embd, int n_ff, float swiglu_limit, bool accumulate);

/// xe_experts_y on PLANAR experts (Y: [nj][8][n_embd]; H scratch as above).""")
rep("cmake", "${_xe_dir}/xe_planar.cpp ${_xe_dir}/xe_dqgemm.cpp", "${_xe_dir}/xe_planar.cpp ${_xe_dir}/xe_xmx.cpp ${_xe_dir}/xe_dqgemm.cpp")

# the tier: per card fp16 scratch, the switch
rep("tier", """    xe::XeQ8 xq, hq;                ///< decode / verify: X [8][n_embd], H [kMaxDecodeJobs][8][n_ff]
""", """    xe::XeQ8 xq, hq;                ///< decode / verify: X [8][n_embd], H [kMaxDecodeJobs][8][n_ff]
    sycl::half* xt = nullptr;       ///< STRATA_XE_XMX: X as [n_embd][8] fp16, H as [kMaxDecodeJobs][n_ff][8] fp16
    sycl::half* ht = nullptr;
""")
rep("tier", """        if (!c.rstage || !alloc_q8(c.xq, kMaxJobTok, (size_t) H, c.q) || !alloc_q8(c.hq, (size_t) kMaxDecodeJobs * kMaxJobTok, (size_t) F, c.q)) {
            err = "xe tier: int8 scratch allocation failed";
            return false;
        }
""", """        if (!c.rstage || !alloc_q8(c.xq, kMaxJobTok, (size_t) H, c.q) || !alloc_q8(c.hq, (size_t) kMaxDecodeJobs * kMaxJobTok, (size_t) F, c.q)) {
            err = "xe tier: int8 scratch allocation failed";
            return false;
        }
        c.xt = sycl::malloc_device<sycl::half>((size_t) 8 * H, c.q);
        c.ht = sycl::malloc_device<sycl::half>((size_t) kMaxDecodeJobs * F * 8, c.q);
        if (!c.xt || !c.ht) { err = "xe tier: fp16 scratch allocation failed"; return false; }
""")
rep("tier", """    xe::xe_quant_rows(c.q, X, nt, H, c.xq);
    return xe::xe_experts_q8(c.q, jobs, n, nt, c.xq, c.hq, dst, H, F, im.L, false, max_nt);""",
    """    static const bool xmx = env_i("STRATA_XE_XMX", 0) != 0;   // larry: the XMX kernels (xe_xmx.cpp)
    if (xmx) {
        xe::xe_xt(c.q, X, nt, H, c.xt);
        return xe::xe_experts_x(c.q, jobs, n, nt, c.xt, c.ht, dst, H, F, im.L, false);
    }
    xe::xe_quant_rows(c.q, X, nt, H, c.xq);
    return xe::xe_experts_q8(c.q, jobs, n, nt, c.xq, c.hq, dst, H, F, im.L, false, max_nt);""")
rep("tier", """        fr(c.gscratch); fr(c.cX2); fr(c.bounce); fr(c.dslots); fr(c.rstage);
""", """        fr(c.gscratch); fr(c.cX2); fr(c.bounce); fr(c.dslots); fr(c.rstage); fr(c.xt); fr(c.ht);
""")

# the bench: both paths, time and the largest difference of their outputs
rep("test", """        xe::XeJob* djs = sycl::malloc_device<xe::XeJob>((size_t) NE * 64, q);""",
    """        xe::XeJob* djs = sycl::malloc_device<xe::XeJob>((size_t) NE * 64, q);
        sycl::half* dxt = sycl::malloc_device<sycl::half>((size_t) 8 * E, q);
        sycl::half* dht = sycl::malloc_device<sycl::half>((size_t) 64 * F * 8, q);
        float* dO2 = sycl::malloc_device<float>((size_t) 8 * E, q);
        {   // 8 activation rows of the distribution the engine sees (the call's X; the first 3 rows were set above)
            std::vector<float> X8((size_t) 8 * E);
            for (float& v : X8) v = nd(rng);
            q.memcpy(dX, X8.data(), X8.size() * 4).wait();
        }""")
rep("test", """            auto call = [&](int c) {
                xe::xe_quant_rows(q, dX, cf.ntok, (int) E, Xq);
                xe::xe_experts_q8(q, djs + (size_t) c * cf.nj, cf.nj, cf.ntok, Xq, Hq, dO, (int) E, (int) F, L, false, cf.nt);
            };
            for (int c = 0; c < ncalls; ++c) call(c);
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < bench_q8; ++it) call(it % ncalls);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / bench_q8;
            std::printf("  %d experts x %d tokens (call of %d): %6.1f us/call = %4.0f GB/s\\n", cf.nj, cf.nt, cf.ntok, us, bytes / us / 1e3);""",
    """            auto call = [&](int c) {
                xe::xe_quant_rows(q, dX, cf.ntok, (int) E, Xq);
                xe::xe_experts_q8(q, djs + (size_t) c * cf.nj, cf.nj, cf.ntok, Xq, Hq, dO, (int) E, (int) F, L, false, cf.nt);
            };
            auto callx = [&](int c) {
                xe::xe_xt(q, dX, cf.ntok, (int) E, dxt);
                xe::xe_experts_x(q, djs + (size_t) c * cf.nj, cf.nj, cf.ntok, dxt, dht, dO2, (int) E, (int) F, L, false);
            };
            auto timed = [&](auto&& fn) {
                for (int c = 0; c < ncalls; ++c) fn(c);
                q.wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < bench_q8; ++it) fn(it % ncalls);
                q.wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / bench_q8;
            };
            const double us = timed(call), ux = timed(callx);
            call(0); callx(0); q.wait();   // the same call both ways: the largest difference vs the largest value
            std::vector<float> a((size_t) cf.ntok * E), b2((size_t) cf.ntok * E);
            q.memcpy(a.data(), dO, a.size() * 4).wait();
            q.memcpy(b2.data(), dO2, b2.size() * 4).wait();
            double md = 0, mx = 0;
            for (size_t i = 0; i < a.size(); ++i) { md = std::max(md, (double) std::fabs(a[i] - b2[i])); mx = std::max(mx, (double) std::fabs(a[i])); }
            std::printf("  %d experts x %d tokens (call of %d): dp4a %6.1f us = %4.0f GB/s | xmx %6.1f us = %4.0f GB/s | max|diff| %.2e of %.2e\\n",
                        cf.nj, cf.nt, cf.ntok, us, bytes / us / 1e3, ux, bytes / ux / 1e3, md, mx);""")

for k, p in FILES.items():
    shutil.copy(p, p + ".pre-xmx")
    open(p, "w").write(src[k])
print("patched")
