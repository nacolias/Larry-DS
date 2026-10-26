#!/usr/bin/env python3
# patch-xmxdbg.py - test_ds41_xe parity, XMX debug: xe_xt's layout against X, and one-token jobs on tokens 1 / 2
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/test_ds41_xe.cpp"
s = open(P).read()
if "xt check" in s:
    print("already"); sys.exit(0)
old = """                if (rx > 0.05) {   // which tokens are off"""
new = """                if (rx > 0.05) {   // xt check: xe_xt's [K][8] layout; one-token jobs on tokens 1 and 2
                    std::vector<sycl::half> xth((size_t) 8 * E);
                    q.memcpy(xth.data(), pxt, xth.size() * 2).wait();
                    double mxd = 0;
                    for (int t = 0; t < NT; ++t)
                        for (int64_t k = 0; k < E; ++k) mxd = std::max(mxd, (double) std::fabs((float) xth[(size_t) k * 8 + t] - X[(size_t) t * E + k]));
                    std::printf("    xt check: max |xt - X| %.3e\\n", mxd);
                    for (int t1 = 1; t1 < NT; ++t1) {
                        xe::XeJob oj = pj;
                        oj.nt = 1; oj.tok[0] = t1; oj.w[0] = 1.f;
                        q.memcpy(dj, &oj, sizeof oj).wait();
                        q.memset(dO, 0, (size_t) 8 * E * 4).wait();
                        xe::xe_experts_x(q, dj, 1, NT, pxt, pht, dO, (int) E, (int) F, L, false).wait();
                        std::vector<float> g1((size_t) NT * E);
                        q.memcpy(g1.data(), dO, g1.size() * 4).wait();
                        std::printf("    one-token job on token %d: row %d vs ref %.2e, row 0 vs ref row %d %.2e\\n", t1, t1,
                                    rel(g1.data() + (size_t) t1 * E, ref.data() + (size_t) t1 * E, E), t1,
                                    rel(g1.data(), ref.data() + (size_t) t1 * E, E));
                    }
                    q.memcpy(dj, &pj, sizeof pj).wait();
                }
                if (rx > 0.05) {   // which tokens are off"""
if s.count(old) != 1: sys.exit("anchor")
s = s.replace(old, new)
open(P, "w").write(s)
print("patched")
