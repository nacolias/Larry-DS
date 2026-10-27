#!/usr/bin/env python3
# patch-batch-dense.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, stage 2: batched decode): the Engram rows of
# conversation s's tokens while another conversation is bound - Ds41Dense::engram_rows_slot_seq (the TP core's pass of
# several conversations reads each run's rows from its own n-gram history: hooks.engram_slot_seq).  The Engram is bound
# to s for the tokens' record and the hashing (both at the batch's start), then back.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
H = os.path.join(X, "tools/v41/engine/ds41_dense.hpp")
C = os.path.join(X, "tools/v41/engine/ds41_dense.cpp")
h = open(H).read()
c = open(C).read()
if "engram_rows_slot_seq" in h:
    print("already patched", H)
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


h = rep(h, """    bool engram_rows_slot(const int * tids, int n, int pos0, int slot, float * out);
""", """    bool engram_rows_slot(const int * tids, int n, int pos0, int slot, float * out);
    /// larry (batched decode, patch-batch-dense.py): the same for conversation s's history (s need not be bound)
    bool engram_rows_slot_seq(int s, const int * tids, int n, int pos0, int slot, float * out);
""", "hpp")
c = rep(c, """    float * outp[2] = { slot == 0 ? out : nullptr, slot == 1 ? out : nullptr };
    auto b = im.engram.start_into(pos0, n, outp, im.err);
    return b && im.engram.wait(b, im.err);
}
""", """    float * outp[2] = { slot == 0 ? out : nullptr, slot == 1 ? out : nullptr };
    auto b = im.engram.start_into(pos0, n, outp, im.err);
    return b && im.engram.wait(b, im.err);
}

bool Ds41Dense::engram_rows_slot_seq(int s, const int * tids, int n, int pos0, int slot, float * out) {
    Impl & im = *p_;
    if (s == im.seq) return engram_rows_slot(tids, n, pos0, slot, out);
    if (!im.engram.enabled() || slot < 0 || slot >= im.engram.n_slots()) return true;
    if (s < 0 || (size_t) s >= im.seqs.size()) { im.err = "Engram rows: no conversation slot " + std::to_string(s); return false; }
    if (slot == 0) {
        if (!engram_ready_all(im)) return false;
        if (im.eng_pf) {
            std::string e2;
            im.engram.wait(im.eng_pf, e2);
            im.eng_pf.reset();
        }
    }
    im.engram.bind_seq(s);   // its history: the tokens' record and the hashes (start_into hashes before it returns)
    if (slot == 0)
        for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);
    float * outp[2] = { slot == 0 ? out : nullptr, slot == 1 ? out : nullptr };
    auto b = im.engram.start_into(pos0, n, outp, im.err);
    im.engram.bind_seq(im.seq);
    return b && im.engram.wait(b, im.err);
}
""", "cpp")
open(H, "w").write(h)
open(C, "w").write(c)
print("patched", H, C)
