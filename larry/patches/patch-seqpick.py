#!/usr/bin/env python3
# patch-seqpick.py [XOLOTL_DIR] - larry, 2026-10-27: which conversation slot a request goes to.  patch-seqs-gen.py took the
# slot with the longest common prefix only when it was >= STRATA_SERVE_PARK_MIN (2048) tokens - a short chat's follow-up
# (the tool test's 300-400 tokens) went to the other slot and read its prompt again (tsvc2-slots: its turn 2 1.9 s, 0.7
# before).  Now: the slot whose conversation the request CONTINUES - it extends what the slot holds, or reaches past its
# snapshot (the reuse paths), or diverges within the last 16 positions of its last prompt (the rewind) - the longest such
# match, at any length; else an empty slot, else the one used longest ago.  A long shared prefix that is NOT a
# continuation (two agents with the same system prompt) no longer picks a busy conversation's slot.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
s = open(P).read()
if "patch-seqpick.py" in s:
    print("already patched", P)
    sys.exit(0)
old = r'''                for (int q = 0; q < n_seq; ++q) {
                    if (phys_busy(q)) continue;   // (patch-slots-serve.py: a decoding slot's)
                    const std::vector<int> & h = seq_hist(q);
                    size_t l = 0;
                    while (l < h.size() && l < ids.size() && h[l] == ids[l]) ++l;
                    l = img_cap(l, seq_img(q), req_img);
                    if (l > bl) { bl = l; best = q; }
                }
                if (best < 0 || bl < (size_t) park_min) {   // continues none: an empty slot, else the one used longest ago'''
new = r'''                for (int q = 0; q < n_seq; ++q) {
                    if (phys_busy(q)) continue;   // (patch-slots-serve.py: a decoding slot's)
                    const std::vector<int> & h = seq_hist(q);
                    size_t l = 0;
                    while (l < h.size() && l < ids.size() && h[l] == ids[l]) ++l;
                    l = img_cap(l, seq_img(q), req_img);
                    // larry (patch-seqpick.py): only a continuation counts - it extends the slot's conversation, reaches
                    // past its snapshot, or diverges within the last 16 positions of its last prompt (the rewind)
                    const int sl = q == cur_seq ? snap_len : seq_stash[(size_t) q].snap_len;
                    const size_t ev = l & ~(size_t) 1;
                    const bool cont = l > 0 && ((l == h.size() && l < ids.size()) || (sl > 0 && l >= (size_t) sl) ||
                                                (sl > 0 && ev >= 64 && (size_t) sl - ev <= 16));
                    if (cont && l > bl) { bl = l; best = q; }
                }
                if (best < 0) {   // continues none: an empty slot, else the one used longest ago'''
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
