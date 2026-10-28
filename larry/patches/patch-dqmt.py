#!/usr/bin/env python3
# patch-dqmt.py [XOLOTL_DIR] [TILE_M] - larry, 2026-10-28 (#3, big prompts: the expert cards' side): xe_dqgemm's tiles
# TILE_M (default 192) token rows tall instead of 128.  A tile decodes its 128 weight rows' values once for all of its
# tokens; a phase-A call gives an expert ~130-160 rows a card, which at 128 a tile took a second, mostly empty tile (the
# whole decode again for ~32 rows).  kDqTileM (xe_kernels.hpp) is the height: the kernel's accumulators (MT = kDqTileM / 8,
# 24 at 192: 192 GRFs of 256), its activation staging (a second row a work-item past 128) and the tier's tiles / tables.
# bench/dqb/dqmt.sh (abwT158): MT 16 / 20 / 24 at 96 experts x 96 .. 256 tokens.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
TM = int(sys.argv[2]) if len(sys.argv) > 2 else 192
assert TM % 8 == 0 and 128 <= TM <= 256
E = os.path.join(X, "tools/v41/engine/xe")
H = os.path.join(E, "xe_kernels.hpp")
K = os.path.join(E, "xe_dqgemm.cpp")
T = os.path.join(E, "ds41_xe_tier.cpp")
h, k, t = open(H).read(), open(K).read(), open(T).read()
if "kDqTileM" in h:
    print("already patched")
    sys.exit(0)


def rep(s, old, new, what, count=1):
    if s.count(old) != count:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:90]))
    return s.replace(old, new)


h = rep(h, """struct DqTile { int32_t job = 0, m0 = 0, n0 = 0, pad = 0; };""",
"""struct DqTile { int32_t job = 0, m0 = 0, n0 = 0, pad = 0; };
/// larry (patch-dqmt.py): token rows a tile (a multiple of 8, 128 .. 256): the tier's tiles step m0 by it
inline constexpr int kDqTileM = %d;""" % TM, "hpp")

k = rep(k, """constexpr int TM = 8, TN = 16, TK = 16, NW = 128, KC = 64, MT = 16;   // tile: 128 output rows x 128 token rows""",
"""constexpr int TM = 8, TN = 16, TK = 16, NW = 128, KC = 64, MT = kDqTileM / TM;   // tile: 128 output rows x kDqTileM token rows
static_assert(MT * TM <= 2 * NW, "a work-item stages at most two token rows");""", "MT")
k = rep(k, """            const sycl::half* arow = jb.A + (size_t) (tl.m0 + lid) * jb.lda;   // the token row this lane stages
            const bool astage = lid < live * TM;""",
"""            const sycl::half* arow = jb.A + (size_t) (tl.m0 + lid) * jb.lda;   // the token row this lane stages
            const bool astage = lid < live * TM;
            const sycl::half* arow2 = jb.A + (size_t) (tl.m0 + lid + NW) * jb.lda;   // (patch-dqmt.py: rows past 128)
            const bool astage2 = MT * TM > NW && lid + NW < live * TM;""", "astage")
k = rep(k, """                        for (int j = 0; j < 8; ++j) As[(size_t) (v + j) * MPS + lid] = x[j];
                    }
                }""",
"""                        for (int j = 0; j < 8; ++j) As[(size_t) (v + j) * MPS + lid] = x[j];
                    }
                }
                if (astage2) {
#pragma unroll
                    for (int v = 0; v < KC; v += 8) {
                        const sycl::vec<sycl::half, 8> x = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(arow2 + k0 + v);
#pragma unroll
                        for (int j = 0; j < 8; ++j) As[(size_t) (v + j) * MPS + lid + NW] = x[j];
                    }
                }""", "stage2")
t = rep(t, "nt_all += (m + 127) / 128 * (2 * F / 128 + H / 128); }",
        "nt_all += (m + xe::kDqTileM - 1) / xe::kDqTileM * (2 * F / 128 + H / 128); }", "nt_all")
t = rep(t, "const int64_t mt = ((int64_t) by_e[(size_t) ge[i]].size() + 127) / 128;",
        "const int64_t mt = ((int64_t) by_e[(size_t) ge[i]].size() + xe::kDqTileM - 1) / xe::kDqTileM;   // (patch-dqmt.py)", "mt")
t = rep(t, "for (int64_t m0 = 0; m0 < m; m0 += 128)", "for (int64_t m0 = 0; m0 < m; m0 += xe::kDqTileM)", "m0", count=2)
open(H, "w").write(h)
open(K, "w").write(k)
open(T, "w").write(t)
print("patched", H, K, T, "(tiles of %d token rows)" % TM)
