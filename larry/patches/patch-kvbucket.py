#!/usr/bin/env python3
# patch-kvbucket.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrency: the slot enabler, optimization #2): card 0's KV state
# mapped on demand (STRATA_DENSE_LAZY_KV=1) by the readers' own rule.  The first try (2026-10-24, patch-lazykv.py) mapped
# positions + 32K and died with DEVICE_LOST in the 16K batch starting at 524,288, twice: every reader scans whole
# power-of-two "cap buckets" of compressed rows (the prompt path's span_pass_caps: next_pow2(max(visible rows, CBS)); the
# TP core's passes: next_pow2(max(visible, 8, 1024)); its replica growth and state sync: next_pow2(max(1024, rows + 1))
# rows copied FROM card 0), and at 2^19 positions both ratios' buckets become the whole tensor - rows nobody had mapped.
#  - kv_bytes_for(p): rows = min(cmax, next_pow2(max(1024, p / ratio + 2)) + 4) - at least every bucket a reader at a
#    position < p uses (the +4 covers the sync's + 1 rows past a bucket; the map rounds to its 2 MiB pages anyway).
#  - unpark(): the written range is the parked positions + 4 (it said 2 * (positions + 1), so the TP state sync pulled
#    rows past what card 0 maps; with the whole tensor allocated that only cost copies).
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_dense.cpp")
s = open(P).read()
if "patch-kvbucket.py" in s:
    print("already patched", P)
    sys.exit(0)
pairs = [
("""    for (const auto & u : im.kv_units) {   // row r of a ratio-k source holds a position >= r * k: rows [0, p / k] + a margin
        const int64_t k = u.second == 2 ? 2 : 1;
        const int64_t rows = std::min<int64_t>(u.first->ne[1] - 1, positions / k + 2);
        bytes.push_back((size_t) rows * u.first->nb[1]);
    }
""",
"""    for (const auto & u : im.kv_units) {   // row r of a ratio-k source holds a position >= r * k: rows [0, p / k] + a margin
        const int64_t k = u.second == 2 ? 2 : 1;
        // larry, patch-kvbucket.py: the readers' cap bucket, not the positions - every scan reads next_pow2 rows
        // (span_pass_caps, the TP core's caps and its replica growth / state sync from card 0); positions + 32K died at
        // 524,288 where both ratios' buckets became the whole tensor (DEVICE_LOST, 2026-10-24)
        const int64_t rows = std::min<int64_t>(u.first->ne[1] - 1, next_pow2(std::max<int64_t>(1024, positions / k + 2)) + 4);
        bytes.push_back((size_t) rows * u.first->nb[1]);
    }
"""),
("""    im.wrote(0, 2 * (S.n + 1));   // the parked rows (rows [0, positions + 1] of either ratio)
""",
"""    im.wrote(0, S.n + 4);   // the parked positions (patch-kvbucket.py: was 2 * (positions + 1) - past card 0's lazy map)
"""),
]
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:80]))
    s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
