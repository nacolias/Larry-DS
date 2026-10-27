#!/usr/bin/env python3
# patch-tpsynclog.py [XOLOTL_DIR] - larry, 2026-10-27 (diagnostic, default off; the lazy card-0 KV map's garbled 1M decode):
# STRATA_TP_SYNC_LOG=1 - every Ds41Tp::sync_state on stderr: the conversation, lo / hi, all or not, the replicas' mapped
# rows, the rows copied a layer and the time.  STRATA_TP_SYNC_CHECK=1 - after the copies, rows lo / the middle / hi - 1
# (and the spare row) of every compressed-KV and indexer-key source compared byte for byte between card 0 and each
# replica, and the window rings (a mismatch: "tp sync check: MISMATCH ...").  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
C = os.path.join(X, "tools/v41/engine/tp/tp_core.cpp")
c = open(C).read()
if "STRATA_TP_SYNC_LOG" in c:
    print("already patched", C)
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


c = rep(c, """bool Ds41Tp::sync_state(int64_t lo, int64_t hi, bool all) {
    Impl & im = *p_;
    if (im.N < 2) return true;
    const bool every = all || hi > im.cmax[1] || lo < 0;
""", """bool Ds41Tp::sync_state(int64_t lo, int64_t hi, bool all) {
    Impl & im = *p_;
    if (im.N < 2) return true;
    const bool every = all || hi > im.cmax[1] || lo < 0;
    // larry (patch-tpsynclog.py, diagnostic)
    static const bool sync_log = std::getenv("STRATA_TP_SYNC_LOG") != nullptr, sync_check = std::getenv("STRATA_TP_SYNC_CHECK") != nullptr;
    const double ts0 = now_us();
""", "head")
c = rep(c, """    try {
        for (int k = 1; k < im.N; ++k) im.cards[(size_t) k]->q.wait_and_throw();
    } catch (const std::exception & e) {
        im.err = std::string("tp: state sync: ") + e.what();
        return false;
    }
    return true;
}
""", """    try {
        for (int k = 1; k < im.N; ++k) im.cards[(size_t) k]->q.wait_and_throw();
    } catch (const std::exception & e) {
        im.err = std::string("tp: state sync: ") + e.what();
        return false;
    }
    if (sync_log)   // (patch-tpsynclog.py)
        std::fprintf(stderr, "tp sync: conversation %d, lo %lld hi %lld%s, replicas mapped %lld / %lld rows, ratio-1 rows %lld..%lld copied, %.1f ms\\n",
                     im.seq, (long long) lo, (long long) hi, every ? " (every row)" : "", (long long) im.rows_done[1], (long long) im.rows_done[2],
                     (long long) (every ? 0 : std::max<int64_t>(0, lo)), (long long) (every ? (im.lazy ? im.rows_done[1] : im.cmax[1]) : std::min<int64_t>(im.lazy ? im.rows_done[1] : im.cmax[1], hi + 1)),
                     (now_us() - ts0) / 1e3);
    if (sync_check) {   // (patch-tpsynclog.py) card 0's rows against each replica's, byte for byte
        Card & k0 = *im.cards[0];
        long bad = 0, checked = 0;
        std::vector<uint8_t> h0, h1;
        auto cmp = [&](const void * a, const void * b, size_t bytes, int k, int l, const char * what, int64_t row) {
            h0.resize(bytes); h1.resize(bytes);
            k0.q.memcpy(h0.data(), a, bytes).wait();
            im.cards[(size_t) k]->q.memcpy(h1.data(), b, bytes).wait();
            ++checked;
            if (std::memcmp(h0.data(), h1.data(), bytes) != 0) {
                if (++bad <= 12) std::fprintf(stderr, "tp sync check: MISMATCH card %d layer %d %s row %lld (conversation %d)\\n", k, l, what, (long long) row, im.seq);
            }
        };
        for (int k = 1; k < im.N; ++k) {
            Card & c = *im.cards[(size_t) k];
            for (int l = 0; l < im.NL; ++l) {
                const Impl::LInfo & L = im.li[(size_t) l];
                const LS & a = k0.ls[(size_t) l];
                const LS & b = c.ls[(size_t) l];
                cmp(a.raw, b.raw, (size_t) (im.RAW * im.DH) * 4, k, l, "window ring", -1);
                if (!L.kv_src) continue;
                const int64_t r = L.ratio, rows = im.cmax[r] + 1;
                const int64_t top = im.lazy ? im.rows_done[r] : rows - 1;
                const int64_t r0 = every ? 0 : std::max<int64_t>(0, lo / r), r1 = every ? top : std::min<int64_t>(top, (hi + r - 1) / r + 1);
                const int64_t pick[4] = { r0, (r0 + r1) / 2, r1 - 1, rows - 1 };
                for (int64_t row : pick) {
                    if (row < 0 || (row >= top && row != rows - 1)) continue;
                    cmp(a.comp + row * im.DH, b.comp + row * im.DH, (size_t) im.DH * 2, k, l, "comp", row);
                    cmp(a.idxk + row * im.ID, b.idxk + row * im.ID, (size_t) im.ID * 2, k, l, "idxk", row);
                }
            }
        }
        std::fprintf(stderr, "tp sync check: conversation %d, %ld regions compared, %ld differ\\n", im.seq, checked, bad);
    }
    return true;
}
""", "tail")
open(C, "w").write(c)
print("patched", C)
