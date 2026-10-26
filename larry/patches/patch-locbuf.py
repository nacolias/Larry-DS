#!/usr/bin/env python3
# patch-locbuf.py [TIER_CPP] - larry: an expert card's chunk-call X buffer (cX: written by its own queue's
# card-to-card copy from the dense card, read by its own kernels and its own push) and its GEMM scratch in card-local
# memory (local_alloc: no UR peer residency) - grown under the tier's submit lock in run_big, a SYCL allocation there
# held the lock while UR made it resident on every peer (the first big prompts after a load: 2.5-7.8 s of the dense
# half's lock waits).  cOUT stays SYCL memory: the copy-after combine (STRATA_XE_COMBINE=0 or no slots) reads it from
# the dense card.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "grow_loc(" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:90]))
    s = s.replace(old, new)


rep("""/// larry: the card's stream buffer, card-local (local_alloc) and grown by half its size at least (to `cap`; whole 32 MiB)
""", """/// larry: grow_dev in card-local memory (local_alloc / local_free): buffers no other card's queue touches
template <class T>
bool grow_loc(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); local_free(q, p); }
    p = reinterpret_cast<T*>(local_alloc(q, need * sizeof(T)));
    cap = p ? need : 0;
    return p != nullptr;
}
/// larry: the card's stream buffer, card-local (local_alloc) and grown by half its size at least (to `cap`; whole 32 MiB)
""")
rep("""        if (!grow_dev(c.cX, c.cX_cap, (size_t) n * H, c.q) || !grow_dev(c.cOUT, c.cOUT_cap, (size_t) n * H, c.q)) {
""", """        if (!grow_loc(c.cX, c.cX_cap, (size_t) n * H, c.q) || !grow_dev(c.cOUT, c.cOUT_cap, (size_t) n * H, c.q)) {
""")
rep("""        if (!grow_stream(c, sb, sb) || !grow_dev(c.cX, c.cX_cap, (size_t) nc * H, c.q) ||
""", """        if (!grow_stream(c, sb, sb) || !grow_loc(c.cX, c.cX_cap, (size_t) nc * H, c.q) ||
""")
rep("""                    grown = grow_dev(c.gscratch, c.gscratch_cap, c.gscratch_cap < need ? need + need / 4 : need, c.q);   // headroom: few regrowths
""", """                    grown = grow_loc(c.gscratch, c.gscratch_cap, c.gscratch_cap < need ? need + need / 4 : need, c.q);   // headroom: few regrowths
""")
rep("""                    grown = grow_dev(c.gscratch, c.gscratch_cap, need, c.q);
""", """                    grown = grow_loc(c.gscratch, c.gscratch_cap, need, c.q);
""")
rep("""        fr(c.cX); fr(c.cOUT); fr(c.cOUT_host);""", """        local_free(c.q, c.cX); fr(c.cOUT); fr(c.cOUT_host);""")
rep("""        fr(c.gscratch); fr(c.cX2);""", """        local_free(c.q, c.gscratch); fr(c.cX2);""")
if "grow_dev(c.cX," in s or "grow_dev(c.gscratch" in s or "fr(c.cX)" in s or "fr(c.gscratch)" in s:
    sys.exit("a SYCL alloc / free of cX / gscratch left")
open(P, "w").write(s)
print("patched", P)
