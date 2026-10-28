#!/usr/bin/env python3
# patch-dumpsel.py [XOLOTL_DIR] - larry, 2026-10-28 (#3, debug; default off): STRATA_DUMP_SEL=1 appends, after each
# pipelined prompt batch, every index source's (layers 2 / 8 / 14) top-k rows and their mask over the batch's columns to
# logs/dump_sel.<pid>.bin: per layer {int32 layer, int32 B, int32 ITOP, int64 the batch's first position} + int32
# sel[B][ITOP] + f32 mask[B][ITOP] (0 / -inf).  For measuring how much consecutive queries' selections overlap
# (bench/fa/selunion.py).  Ds41Dense::read_sel_cols reads the columns.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
H = os.path.join(X, "tools/v41/engine/ds41_dense.hpp")
D = os.path.join(X, "tools/v41/engine/ds41_dense.cpp")
G = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
h, d, g = open(H).read(), open(D).read(), open(G).read()
if "read_sel_cols" in h:
    print("already patched")
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d" % (what, s.count(old)))
    return s.replace(old, new)


h = rep(h, """    bool read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K);
""", """    bool read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K);
    /// larry, STRATA_DUMP_SEL (patch-dumpsel.py, debug): an index source's top-k rows and their mask over state columns
    /// [c0, c0 + n) of the batch (ITOP ints / floats a column, column-major as the layer keeps them)
    bool read_sel_cols(int il, int c0, int n, std::vector<int32_t> & sel, std::vector<float> & mask, int * itop);
""", "hpp")
d = rep(d, """bool Ds41Dense::read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
""", """bool Ds41Dense::read_sel_cols(int il, int c0, int n, std::vector<int32_t> & sel, std::vector<float> & mask, int * itop) {
    Impl & im = *p_;
    *itop = 0;
    if (il < 0 || il >= (int) im.ly.size()) return false;
    const auto & L = im.ly[(size_t) il];
    if (!L.idx_src || !L.sel || !L.selm || c0 < 0 || n < 1 || c0 + n > im.NC) return false;
    if (!fin_flush(im)) return false;
    ggml_backend_synchronize(im.backend);
    sel.resize((size_t) im.ITOP * n);
    mask.resize((size_t) im.ITOP * n);
    ggml_backend_tensor_get(L.sel, sel.data(), (size_t) c0 * im.ITOP * 4, sel.size() * 4);
    ggml_backend_tensor_get(L.selm, mask.data(), (size_t) c0 * im.ITOP * 4, mask.size() * 4);
    *itop = (int) im.ITOP;
    return true;
}

bool Ds41Dense::read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
""", "cpp")
g = rep(g, """                    btr("pipeline");
""", """                    btr("pipeline");
                    // debug, STRATA_DUMP_SEL=1 (patch-dumpsel.py): every index source's (before the candidate layer) top-k
                    // rows and mask over the batch's columns, appended to logs/dump_sel.<pid>.bin: per layer {int32 layer,
                    // int32 B, int32 ITOP, int64 the batch's first position} + int32 sel[B][ITOP] + f32 mask[B][ITOP]
                    static const bool dump_sel = std::getenv("STRATA_DUMP_SEL") != nullptr;
                    if (okp && dump_sel) {
                        const std::string dn = "logs/dump_sel." + std::to_string((long) getpid()) + ".bin";
                        if (std::FILE * df = std::fopen(dn.c_str(), "ab")) {
                            for (int l = 0; l < n_layer; ++l) {
                                if (!G.is_index_source(l) || (G.cand_layer >= 0 && l >= G.cand_layer)) continue;
                                std::vector<int32_t> ss;
                                std::vector<float> sm;
                                int itop = 0;
                                if (!dense.read_sel_cols(l, 0, B, ss, sm, &itop) || itop <= 0) continue;
                                const int32_t hd[3] = {(int32_t) l, (int32_t) B, (int32_t) itop};
                                const int64_t p0 = (int64_t) pos0 + c0;
                                std::fwrite(hd, 4, 3, df);
                                std::fwrite(&p0, 8, 1, df);
                                std::fwrite(ss.data(), 4, ss.size(), df);
                                std::fwrite(sm.data(), 4, sm.size(), df);
                            }
                            std::fclose(df);
                            std::fprintf(stderr, "STRATA_DUMP_SEL: batch at %lld (%d columns) -> %s\\n", (long long) pos0 + c0, B, dn.c_str());
                        }
                    }
""", "gen")
open(H, "w").write(h)
open(D, "w").write(d)
open(G, "w").write(g)
print("patched", H, D, G)
