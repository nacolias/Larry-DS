#!/usr/bin/env python3
# patch-jobnt-bench.py - larry: test_ds41_xe --bench-q8 also checks each config's output against the same call in the
# jobs' own form (max_nt = their token count): a wider form (the armed verify sequences' pass width, STRATA_XE_JOBNT
# or not) must give the same bits.  Prints "bits ok" / "bits N differ".  Usage: patch-jobnt-bench.py [test_ds41_xe.cpp]
# Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/test_ds41_xe.cpp"
s = open(P).read()
if "own form" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


rep("""            const double us = timed(call), ux = timed(callx);
            call(0); callx(0); q.wait();   // the same call both ways: the largest difference vs the largest value
            std::vector<float> a((size_t) cf.ntok * E), b2((size_t) cf.ntok * E);
            q.memcpy(a.data(), dO, a.size() * 4).wait();
            q.memcpy(b2.data(), dO2, b2.size() * 4).wait();
            double md = 0, mx = 0;
            for (size_t i = 0; i < a.size(); ++i) { md = std::max(md, (double) std::fabs(a[i] - b2[i])); mx = std::max(mx, (double) std::fabs(a[i])); }
            std::printf("  %d experts x %d tokens (call of %d, template %d): dp4a %6.1f us = %4.0f GB/s | xmx %6.1f us = %4.0f GB/s | max|diff| %.2e of %.2e\\n",
                        cf.nj, cf.nt, cf.ntok, cf.mnt, us, bytes / us / 1e3, ux, bytes / ux / 1e3, md, mx);""",
    """            const double us = timed(call), ux = timed(callx);
            // call 0 in the jobs' own form (max_nt = their token count): the wider form must give the same bits
            std::vector<float> aref((size_t) cf.ntok * E);
            xe::xe_quant_rows(q, dX, cf.ntok, (int) E, Xq);
            xe::xe_experts_q8(q, djs, cf.nj, cf.ntok, Xq, Hq, dO, (int) E, (int) F, L, false, cf.nt);
            q.wait();
            q.memcpy(aref.data(), dO, aref.size() * 4).wait();
            call(0); callx(0); q.wait();   // the same call both ways: the largest difference vs the largest value
            std::vector<float> a((size_t) cf.ntok * E), b2((size_t) cf.ntok * E);
            q.memcpy(a.data(), dO, a.size() * 4).wait();
            q.memcpy(b2.data(), dO2, b2.size() * 4).wait();
            double md = 0, mx = 0;
            size_t nbits = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                md = std::max(md, (double) std::fabs(a[i] - b2[i]));
                mx = std::max(mx, (double) std::fabs(a[i]));
                if (std::memcmp(&a[i], &aref[i], 4) != 0) ++nbits;
            }
            std::printf("  %d experts x %d tokens (call of %d, template %d): dp4a %6.1f us = %4.0f GB/s | xmx %6.1f us = %4.0f GB/s | max|diff| %.2e of %.2e | bits %s%zu\\n",
                        cf.nj, cf.nt, cf.ntok, cf.mnt, us, bytes / us / 1e3, ux, bytes / ux / 1e3, md, mx, nbits ? "DIFFER " : "ok ", nbits);""")
# (an old argument order: the MiB and the type swapped - printed right only by the x86-64 varargs registers)
rep("""        std::printf("bench-q8: layer %d, %d experts (%.1f MiB) rotated (gate/up type %d)\\n", l, (int) pl.size(), bench_q8_gu,
                    std::accumulate(nbytes.begin(), nbytes.end(), (size_t) 0) / 1048576.0);""",
    """        std::printf("bench-q8: layer %d, %d experts (%.1f MiB) rotated (gate/up type %d)\\n", l, (int) pl.size(),
                    std::accumulate(nbytes.begin(), nbytes.end(), (size_t) 0) / 1048576.0, bench_q8_gu);""")
if "<cstring>" not in s:
    rep("#include <cstdio>\n", "#include <cstdio>\n#include <cstring>\n")
open(P, "w").write(s)
print("patched")
