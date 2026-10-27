#!/usr/bin/env python3
# patch-batch-kern.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, stage 2: batched decode): attn_dec can run
# on a run of a pass's rows - rows row0 .. row0 + n - 1 of a pass of n_rows (the TP core's pass of several
# conversations: each conversation's rows attend over its own state, one launch a run).  Only the grouped q8 output
# (oq / od and its XMX layout oT / odT) is indexed by the pass's rows; the other row-major inputs and outputs take
# offset pointers.  Defaults (row0 0, n_rows 0 = n): the kernel as before.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "third_party/llama.cpp/ggml/src/ggml-sycl/dsv41-dec.hpp")
s = open(P).read()
if "n_rows" in s:
    print("already patched", P)
    sys.exit(0)
pairs = [
("""    sycl::half * oT = nullptr;
    float * odT = nullptr;
};
""",
"""    sycl::half * oT = nullptr;
    float * odT = nullptr;
    // larry (batched decode, patch-batch-kern.py): this launch is rows row0 .. row0 + n - 1 of a pass of n_rows (0: n) -
    // the q8 outputs above are indexed by the pass's rows (the other row-major pointers point at the run's first row)
    int row0 = 0, n_rows = 0;
};
"""),
("""        if (A.hpg) {   // q8: a 32-value block is 4 lanes x 8 dims (lanes 4 j .. 4 j + 3 of one g)
            const int K8 = A.hpg * D;
            const size_t r8 = (size_t) (hh / A.hpg) * n + b;
""",
"""        if (A.hpg) {   // q8: a 32-value block is 4 lanes x 8 dims (lanes 4 j .. 4 j + 3 of one g)
            const int K8 = A.hpg * D;
            const int NR = A.n_rows > 0 ? A.n_rows : n, B8 = b + A.row0;   // (larry: a run's rows in the pass's q8)
            const size_t r8 = (size_t) (hh / A.hpg) * NR + B8;
"""),
("""                    for (int e = 0; e < 8; ++e) A.oT[(gb + k + e) * 8 + b] = (sycl::half) sycl::rint(acc[8 * g + e] * id);
                    if ((lane & 3) == 0) A.odT[(gb + k) / 32 * 8 + b] = am / 127.0f;
""",
"""                    for (int e = 0; e < 8; ++e) A.oT[(gb + k + e) * 8 + B8] = (sycl::half) sycl::rint(acc[8 * g + e] * id);
                    if ((lane & 3) == 0) A.odT[(gb + k) / 32 * 8 + B8] = am / 127.0f;
"""),
]
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:90]))
    s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
