#!/usr/bin/env python3
# patch-benchq8.py - test_ds41_xe --bench-q8 N: the PLANAR int8 expert kernels (what decode / verify run on the cards:
# xe_quant_rows + xe_experts_q8) at decode and verify shapes - nj experts a call, nt tokens a job - rotating over 24
# experts of one layer so the card's L2 does not hold them between calls.  Prints us a call and GB/s.  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/test_ds41_xe.cpp"
s = open(P).read()
if "bench_q8_gu" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


if "bench_q8" in s:   # the first version: add the class filter only
    s = s.replace("    int bench_q8 = 0;\n", "    int bench_q8 = 0, bench_q8_gu = -1;\n")
    s = s.replace("        else if (k == \"--bench-q8\" && i + 1 < argc) bench_q8 = std::atoi(argv[++i]);\n",
                  "        else if (k == \"--bench-q8\" && i + 1 < argc) bench_q8 = std::atoi(argv[++i]);\n        else if (k == \"--bench-q8-gu\" && i + 1 < argc) bench_q8_gu = std::atoi(argv[++i]);\n")
    s = s.replace("            const cpu::NativeFmt& f = fl[(size_t) g.cls(l, e)];\n            std::vector<uint8_t> blob(f.bytes);\n            src.read(l, e, blob.data());\n            uint8_t* fb",
                  "            const cpu::NativeFmt& f = fl[(size_t) g.cls(l, e)];\n            if (bench_q8_gu >= 0 && f.gu_type != bench_q8_gu) continue;\n            std::vector<uint8_t> blob(f.bytes);\n            src.read(l, e, blob.data());\n            uint8_t* fb")
    open(P, "w").write(s)
    print("class filter added"); sys.exit(0)
rep("""    bool gemm_bench = false;
""", """    bool gemm_bench = false;
    int bench_q8 = 0, bench_q8_gu = -1;
""")
rep("""        else if (k == "--gemm-bench") gemm_bench = true;
""", """        else if (k == "--gemm-bench") gemm_bench = true;
        else if (k == "--bench-q8" && i + 1 < argc) bench_q8 = std::atoi(argv[++i]);
        else if (k == "--bench-q8-gu" && i + 1 < argc) bench_q8_gu = std::atoi(argv[++i]);
""")
rep("""    if (gemm_bench) {   // the prompt path's GEMM form for one expert: dequant, gather, gate|up GEMM, SwiGLU, down GEMM""",
    """    if (bench_q8 > 0) {   // the planar int8 kernels at decode / verify shapes, experts rotated past the L2
        const int l = layers.empty() ? 20 : layers[0];
        constexpr int NE = 24;
        std::vector<uint8_t*> pl;
        std::vector<xe::XeJob> base;
        std::vector<size_t> nbytes;
        for (int e = 0; e < g.n_experts && (int) pl.size() < NE; ++e) {
            const cpu::NativeFmt& f = fl[(size_t) g.cls(l, e)];
            if (bench_q8_gu >= 0 && f.gu_type != bench_q8_gu) continue;
            std::vector<uint8_t> blob(f.bytes);
            src.read(l, e, blob.data());
            uint8_t* fb = sycl::malloc_device<uint8_t>(f.bytes, q);
            uint8_t* pb = sycl::malloc_device<uint8_t>(f.bytes, q);
            q.memcpy(fb, blob.data(), f.bytes).wait();
            xe::xe_repack_blob(q, fb, pb, f.gu_type, f.d_type, f.up_off, f.down_off, (int) E, (int) F).wait();
            sycl::free(fb, q);
            xe::XeJob j;
            j.blob = pb; j.up_off = (uint32_t) f.up_off; j.down_off = (uint32_t) f.down_off;
            j.gu_type = (uint16_t) f.gu_type; j.d_type = (uint16_t) f.d_type;
            j.gu_row = (uint16_t) f.gu_row; j.d_row = (uint16_t) f.d_row;
            pl.push_back(pb); base.push_back(j); nbytes.push_back(f.bytes);
        }
        xe::XeQ8 Xq{sycl::malloc_device<int8_t>((size_t) 8 * E, q), sycl::malloc_device<float>((size_t) 8 * E / 32, q),
                    sycl::malloc_device<int32_t>((size_t) 8 * E / 16, q)};
        xe::XeQ8 Hq{sycl::malloc_device<int8_t>((size_t) 64 * 8 * F, q), sycl::malloc_device<float>((size_t) 64 * 8 * F / 32, q),
                    sycl::malloc_device<int32_t>((size_t) 64 * 8 * F / 16, q)};
        xe::XeJob* djs = sycl::malloc_device<xe::XeJob>((size_t) NE * 64, q);
        std::printf("bench-q8: layer %d, %d experts (%.1f MiB) rotated, gate/up type %d\\n", l, (int) pl.size(), bench_q8_gu,
                    std::accumulate(nbytes.begin(), nbytes.end(), (size_t) 0) / 1048576.0);
        // (nj experts a call, nt tokens a job, the call's tokens ntok): the job lists of NE / nj calls, staged once
        struct Cfg { int nj, nt, ntok; };
        for (const Cfg cf : {Cfg{2, 1, 1}, Cfg{3, 1, 1}, Cfg{2, 1, 3}, Cfg{3, 1, 3}, Cfg{3, 2, 3}, Cfg{3, 3, 3},
                             Cfg{3, 2, 5}, Cfg{4, 2, 5}, Cfg{6, 2, 5}, Cfg{3, 5, 5}, Cfg{6, 1, 1}}) {
            const int ncalls = (int) pl.size() / cf.nj;
            std::vector<xe::XeJob> jl((size_t) ncalls * cf.nj);
            double bytes = 0;
            for (int c = 0; c < ncalls; ++c)
                for (int k = 0; k < cf.nj; ++k) {
                    xe::XeJob j = base[(size_t) (c * cf.nj + k)];
                    j.nt = (uint8_t) cf.nt;
                    for (int t = 0; t < cf.nt; ++t) { j.tok[t] = (k + t) % cf.ntok; j.w[t] = 0.1f; }
                    jl[(size_t) (c * cf.nj + k)] = j;
                    if (c == 0) bytes += (double) nbytes[(size_t) k];
                }
            q.memcpy(djs, jl.data(), jl.size() * sizeof(xe::XeJob)).wait();
            auto call = [&](int c) {
                xe::xe_quant_rows(q, dX, cf.ntok, (int) E, Xq);
                xe::xe_experts_q8(q, djs + (size_t) c * cf.nj, cf.nj, cf.ntok, Xq, Hq, dO, (int) E, (int) F, L, false, cf.nt);
            };
            for (int c = 0; c < ncalls; ++c) call(c);
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < bench_q8; ++it) call(it % ncalls);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / bench_q8;
            std::printf("  %d experts x %d tokens (call of %d): %6.1f us/call = %4.0f GB/s\\n", cf.nj, cf.nt, cf.ntok, us, bytes / us / 1e3);
        }
        for (uint8_t* p : pl) sycl::free(p, q);
    }
    if (gemm_bench) {   // the prompt path's GEMM form for one expert: dequant, gather, gate|up GEMM, SwiGLU, down GEMM""")
if "#include <numeric>" not in s:
    s = s.replace("#include <chrono>", "#include <chrono>\n#include <numeric>", 1)
open(P, "w").write(s)
print("patched")
