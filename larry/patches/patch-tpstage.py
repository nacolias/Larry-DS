#!/usr/bin/env python3
# patch-tpstage.py [XOLOTL_DIR] - larry, 2026-10-27 (card 0's KV mapped on demand, the fix): with STRATA_DENSE_LAZY_KV the
# TP core's replicas on the other cards came out different from card 0 exactly in the rows card 0 holds in a physical
# allocation mapped after the first one (patch-tpsynclog.py's check: 24 of 216 sampled regions at a 131K conversation's
# first turn, 48 after a 984K prompt -> a garbled, slow decode and then faults); the replica's own rows all lay in one
# allocation, so the other cards' reads of card 0's lazily mapped pages were wrong (in isolation the same mapping order
# reads right: bench/conc/vmm_span_probe4.cpp S1).  Now the other cards never read those pages: hooks.kv_stage (the
# engine sets it when card 0's KV is mapped lazily) - card 0 copies the compressed KV / indexer-key rows (and the spare
# rows) into a plain device buffer of its own (STRATA_TP_STAGE_MIB, default 64) and every other card pulls them from
# there, a chunk at a time; in sync_state and in the replicas' growth (ensure_rows).  The window rings and the ratio-2
# states (plain tensors) are copied as before.  Needs patch-tpsynclog.py (anchors after its check).  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
H = os.path.join(X, "tools/v41/engine/tp/tp_core.hpp")
C = os.path.join(X, "tools/v41/engine/tp/tp_core.cpp")
G = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
h = open(H).read()
c = open(C).read()
g = open(G).read()
if "kv_stage" in h:
    print("already patched", H, C, G)
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


h = rep(h, """    std::function<int(int l, int n, const int32_t * ids, const float * w, const float * x, float * out)> cpu_part;
};
""", """    std::function<int(int l, int n, const int32_t * ids, const float * w, const float * x, float * out)> cpu_part;
    /// larry (patch-tpstage.py): card 0's compressed KV / indexer keys live in virtual memory mapped as the context grows
    /// (STRATA_DENSE_LAZY_KV) - the other cards pull those rows through a plain staging buffer on card 0, never from
    /// the mapped pages themselves (their reads of pages mapped after the first allocation came back other data)
    bool kv_stage = false;
};
""", "hpp hook")

# the staging buffer and the pull
c = rep(c, """    // larry (batched decode): conversation s's state on card c (the bound one's is c.ls; the others' wait in ls_seq)
""", """    // larry (patch-tpstage.py): bytes from card 0's src to dst[k] on every other card k through a staging buffer on card 0
    // (a plain device allocation): card 0 copies a chunk in, the other cards pull it out, chunk after chunk
    uint8_t * kv_stage_buf = nullptr;
    size_t kv_stage_cap = 0;
    bool pull_staged(const void * src, const std::vector<void *> & dst, size_t bytes) {
        if (!bytes) return true;
        Card & k0 = *cards[0];
        if (!kv_stage_buf) {
            const char * e = std::getenv("STRATA_TP_STAGE_MIB");
            kv_stage_cap = (size_t) std::max(1, e ? std::atoi(e) : 64) << 20;
            kv_stage_buf = sycl::malloc_device<uint8_t>(kv_stage_cap, k0.q);
            if (!kv_stage_buf) { err = "tp: no staging buffer on card 0"; return false; }
        }
        try {
            for (size_t off = 0; off < bytes; off += kv_stage_cap) {
                const size_t n = std::min(kv_stage_cap, bytes - off);
                k0.q.memcpy(kv_stage_buf, (const uint8_t *) src + off, n).wait_and_throw();
                for (int k = 1; k < N; ++k) cards[(size_t) k]->q.memcpy((uint8_t *) dst[(size_t) k] + off, kv_stage_buf, n);
                for (int k = 1; k < N; ++k) cards[(size_t) k]->q.wait_and_throw();
            }
        } catch (const std::exception & ex) {
            err = std::string("tp: staged KV rows: ") + ex.what();
            return false;
        }
        return true;
    }
    // layer l's compressed KV + indexer-key rows [r0, r1) (row counts of its ratio) to every other card's replica, staged
    bool pull_rows_staged(int l, int64_t r0, int64_t r1) {
        if (r1 <= r0) return true;
        const LS & a = cards[0]->ls[(size_t) l];
        std::vector<void *> dc((size_t) N, nullptr), di((size_t) N, nullptr);
        for (int k = 1; k < N; ++k) {
            const LS & b = cards[(size_t) k]->ls[(size_t) l];
            dc[(size_t) k] = b.comp + r0 * DH;
            di[(size_t) k] = b.idxk + r0 * ID;
        }
        return pull_staged(a.comp + r0 * DH, dc, (size_t) ((r1 - r0) * DH) * 2) && pull_staged(a.idxk + r0 * ID, di, (size_t) ((r1 - r0) * ID) * 2);
    }
    // larry (batched decode): conversation s's state on card c (the bound one's is c.ls; the others' wait in ls_seq)
""", "pull_staged")
c = rep(c, """            c->graphs_m.clear();
            if (c->st_keep) sycl::free(c->st_keep, c->q);
""", """            c->graphs_m.clear();
            if (c->st_keep) sycl::free(c->st_keep, c->q);
            if (c->rank == 0 && kv_stage_buf) { sycl::free(kv_stage_buf, c->q); kv_stage_buf = nullptr; }   // (patch-tpstage.py)
""", "dtor")

# ensure_rows: the new replica rows staged
c = rep(c, """                const LS & a = c0.ls[(size_t) l];
                c.q.memcpy(s.comp + r0 * DH, a.comp + r0 * DH, (size_t) ((r1 - r0) * DH) * 2);
                c.q.memcpy(s.idxk + r0 * ID, a.idxk + r0 * ID, (size_t) ((r1 - r0) * ID) * 2);
            }
        }
""", """                if (hooks.kv_stage) continue;   // (patch-tpstage.py: pulled staged below, once every card grew)
                const LS & a = c0.ls[(size_t) l];
                c.q.memcpy(s.comp + r0 * DH, a.comp + r0 * DH, (size_t) ((r1 - r0) * DH) * 2);
                c.q.memcpy(s.idxk + r0 * ID, a.idxk + r0 * ID, (size_t) ((r1 - r0) * ID) * 2);
            }
        }
""", "ensure_rows copies")
c = rep(c, """        std::fprintf(stderr, "tp: KV replicas now %lld / %lld rows (ratio 1 / 2), +%.0f MiB on the other cards\\n",
""", """        if (hooks.kv_stage)   // (patch-tpstage.py) the new rows of every replica from card 0, staged
            for (int l = 0; l < NL; ++l) {
                const LInfo & L = li[(size_t) l];
                if (L.kv_src && !pull_rows_staged(l, rows_done[L.ratio], want[L.ratio])) return false;
            }
        std::fprintf(stderr, "tp: KV replicas now %lld / %lld rows (ratio 1 / 2), +%.0f MiB on the other cards\\n",
""", "ensure_rows staged")

# sync_state: the rows staged
c = rep(c, """            if (r1 > r0) {
                c.q.memcpy(b.comp + r0 * im.DH, a.comp + r0 * im.DH, (size_t) ((r1 - r0) * im.DH) * 2);
                c.q.memcpy(b.idxk + r0 * im.ID, a.idxk + r0 * im.ID, (size_t) ((r1 - r0) * im.ID) * 2);
            }
            // the spare row (an open group's)
            c.q.memcpy(b.comp + (rows - 1) * im.DH, a.comp + (rows - 1) * im.DH, (size_t) im.DH * 2);
            c.q.memcpy(b.idxk + (rows - 1) * im.ID, a.idxk + (rows - 1) * im.ID, (size_t) im.ID * 2);
""", """            if (!im.hooks.kv_stage) {   // (patch-tpstage.py: else staged below)
            if (r1 > r0) {
                c.q.memcpy(b.comp + r0 * im.DH, a.comp + r0 * im.DH, (size_t) ((r1 - r0) * im.DH) * 2);
                c.q.memcpy(b.idxk + r0 * im.ID, a.idxk + r0 * im.ID, (size_t) ((r1 - r0) * im.ID) * 2);
            }
            // the spare row (an open group's)
            c.q.memcpy(b.comp + (rows - 1) * im.DH, a.comp + (rows - 1) * im.DH, (size_t) im.DH * 2);
            c.q.memcpy(b.idxk + (rows - 1) * im.ID, a.idxk + (rows - 1) * im.ID, (size_t) im.ID * 2);
            }
""", "sync copies")
c = rep(c, """        im.err = std::string("tp: state sync: ") + e.what();
        return false;
    }
    if (sync_log)   // (patch-tpsynclog.py)
""", """        im.err = std::string("tp: state sync: ") + e.what();
        return false;
    }
    if (im.hooks.kv_stage)   // (patch-tpstage.py) the compressed rows and the spare rows, through card 0's staging buffer
        for (int l = 0; l < im.NL; ++l) {
            const Impl::LInfo & L = im.li[(size_t) l];
            if (!L.kv_src) continue;
            const int64_t r = L.ratio, rows = im.cmax[r] + 1;
            const int64_t top = im.lazy ? im.rows_done[r] : rows - 1;
            const int64_t r0 = every ? 0 : std::max<int64_t>(0, lo / r), r1 = every ? top : std::min<int64_t>(top, (hi + r - 1) / r + 1);
            if (!im.pull_rows_staged(l, r0, r1) || !im.pull_rows_staged(l, rows - 1, rows)) return false;
        }
    if (sync_log)   // (patch-tpsynclog.py)
""", "sync staged")

# the engine: on when card 0's KV is mapped lazily
g = rep(g, """        Ds41TpHooks th;
""", """        Ds41TpHooks th;
        th.kv_stage = kvmap && dense.kv_lazy();   // larry (patch-tpstage.py): peers pull card 0's lazily mapped KV staged
""", "engine hook")
open(H, "w").write(h)
open(C, "w").write(c)
open(G, "w").write(g)
print("patched", H, C, G)
