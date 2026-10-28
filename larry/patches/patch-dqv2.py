#!/usr/bin/env python3
# patch-dqv2.py [XOLOTL_DIR] - larry, 2026-10-28 (#3, big prompts: the expert cards' side): xe_dqgemm v2 behind
# STRATA_XE_DQ2=1 (default 0).  The same tiles (128 output rows x 128 tokens), but each chunk's 64 weight values a row are
# decoded once into local memory VNNI-packed (B by joint_matrix_load ext_intel_packed - 85 TFLOPS of operand rate on the
# B70 against 66 for v1's A fragments by hand), and a sub-group computes 32 tokens x 64 output rows (4 x 4 accumulators)
# with its token rows by 2D block loads from global memory: half v1's local-memory bytes a multiply.  The same decode and
# the same k order: the same bits (bench/dqb/bench_dqgemm_v2.cpp: 0 values differ).  96 experts x 128 tokens planar:
# Q3_K 44 -> 63 TFLOPS, Q2_0 51 -> 89, PTQ1_0 48 -> 87; the file layout 37 / 43 / 47 -> 50 / 62 / 77; 64 tokens 34 / 42 / 33
# -> 40 / 56 / 53 (abwT163).  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
K = os.path.join(X, "tools/v41/engine/xe/xe_dqgemm.cpp")
INC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqv2_kernel.inc")
k = open(K).read()
if "dqgemm2_t" in k:
    print("already patched")
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d" % (what, s.count(old)))
    return s.replace(old, new)


kernel = open(INC).read()
k = rep(k, """}  // namespace

sycl::event xe_dqgemm(sycl::queue& q, int type, bool planar, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    if (ntiles <= 0) return {};""", """// ---- larry, STRATA_XE_DQ2 (patch-dqv2.py): v2 - the chunk's weights decoded once into local memory VNNI-packed, a
// sub-group 32 tokens x 64 output rows with its token rows by 2D block loads (half v1's local-memory bytes a multiply;
// the same decode and k order: the same bits)
""" + kernel + """
}  // namespace

sycl::event xe_dqgemm(sycl::queue& q, int type, bool planar, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    if (ntiles <= 0) return {};
    static const bool v2 = [] { const char* e = std::getenv("STRATA_XE_DQ2"); return e && *e && *e != '0'; }();   // (patch-dqv2.py)
    if (v2) {
        if (type == kQ20) return planar ? dqgemm2_t<kQ20, true>(q, jobs, tiles, ntiles, K) : dqgemm2_t<kQ20, false>(q, jobs, tiles, ntiles, K);
        if (type == kPtq1) return planar ? dqgemm2_t<kPtq1, true>(q, jobs, tiles, ntiles, K) : dqgemm2_t<kPtq1, false>(q, jobs, tiles, ntiles, K);
        return planar ? dqgemm2_t<kQ3K, true>(q, jobs, tiles, ntiles, K) : dqgemm2_t<kQ3K, false>(q, jobs, tiles, ntiles, K);
    }""", "dispatch")
if "#include <cstdlib>" not in k:
    k = rep(k, "#include <type_traits>\n", "#include <cstdlib>\n#include <type_traits>\n", "include")
open(K, "w").write(k)
print("patched", K)
