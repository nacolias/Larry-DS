#!/usr/bin/env python3
# patch-mvsplit.py - larry: fastmv's split-K choice as knobs, for tuning the latency-bound mat-vecs of the decode /
# verify layer on the GPU (unitrace: the shared expert's down 9.7 MB at 297-353 GB/s, q_a 278-321, kv 196-287, the F16
# router 265-336 GB/s vs ~480 for the big ones).  Both kernels double K's split while the launch has fewer than
# GGML_SYCL_FASTMV_SG sub-groups (4096) and each lane would keep >= GGML_SYCL_FASTMV_UPL units of 32 (2) - e.g. the
# shared expert's down (K 2304: 72 units a row) splits 4 ways today, 2-3 units a lane.  Defaults = today's.
# Usage: patch-mvsplit.py [fastmv.cpp]   Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/fastmv.cpp"
s = open(P).read()
if "GGML_SYCL_FASTMV_UPL" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


rep("""template <int NT, int R, bool BIG>
void pick_r8m(sycl::queue & q, const Args & a) {""", """// larry: the split-K knobs (GGML_SYCL_FASTMV_SG: the sub-groups a launch should reach, 4096 ~ two waves of the B70's
// hardware threads; GGML_SYCL_FASTMV_UPL: the units of 32 each lane keeps at least when K is split further)
inline int64_t mv_want_sg() {
    static const int64_t v = [] { const char * e = std::getenv("GGML_SYCL_FASTMV_SG"); return e ? std::atoll(e) : 4096; }();
    return v;
}
inline int64_t mv_min_upl() {
    static const int64_t v = [] { const char * e = std::getenv("GGML_SYCL_FASTMV_UPL"); return e ? std::max<int64_t>(1, std::atoll(e)) : 2; }();
    return v;
}

template <int NT, int R, bool BIG>
void pick_r8m(sycl::queue & q, const Args & a) {""")
rep("""    const int64_t rows = (a.N + R - 1) / R * a.G;                    // row groups
    const int64_t want_sg = 4096;
    int ks = 1;
    while (ks < 64 && rows * ks * lpr / SG < want_sg && upr / (ks * 2) >= lpr * 2) ks *= 2;""",
    """    const int64_t rows = (a.N + R - 1) / R * a.G;                    // row groups
    const int64_t want_sg = mv_want_sg();
    int ks = 1;
    while (ks < 64 && rows * ks * lpr / SG < want_sg && upr / (ks * 2) >= lpr * mv_min_upl()) ks *= 2;""")
rep("""    const int64_t rows = a.N * a.G;
    const int64_t want_sg = 4096;                                  // ~2 waves of B70 hardware threads
    int ks = 1;
    while (ks < 64 && rows * ks * lpr / SG < want_sg && upr / (ks * 2) >= lpr * 2) ks *= 2;""",
    """    const int64_t rows = a.N * a.G;
    const int64_t want_sg = mv_want_sg();                           // ~2 waves of B70 hardware threads
    int ks = 1;
    while (ks < 64 && rows * ks * lpr / SG < want_sg && upr / (ks * 2) >= lpr * mv_min_upl()) ks *= 2;""")
if "#include <cstdlib>" not in s:
    rep('#include "fastmv.hpp"\n', '#include "fastmv.hpp"\n\n#include <algorithm>\n#include <cstdlib>\n')
open(P, "w").write(s)
print("patched")
