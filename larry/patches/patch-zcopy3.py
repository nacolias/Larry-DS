#!/usr/bin/env python3
# patch-zcopy3.py [TIER_CPP] - larry, the fix for STRATA_XE_ZCOPY=2's GPU fault (2026-10-25 12:38, card 1, CCS reads at
# 0x65000-0x86000): a call's GEMM block runs once a ROUND (overlap mode: the resident experts in round -1, the streamed
# ones in round 0) and each round wrote its xe_dqgemm job / tile tables from offset 0 of the card's one pinned table
# buffer (c.dq_host).  The old path's host wait at the block's start (after the row-list upload) meant round -1's table
# copies had run; without it round 0 overwrote tables they had not read yet - round -1's GEMMs took half-written jobs
# (null weight pointers).  Under ZCOPY the tables now come from the per-call staging (fresh memory each round), sized
# for every GEMM expert's jobs and tiles of the call.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "patch-zcopy3" in s:
    print("already patched"); sys.exit(0)


def rep(old, new):
    global s
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:100]))
    s = s.replace(old, new)


rep("""            int64_t rg = 0, njp = 0, njr = 0;
            for (int32_t e : cl[(size_t) c.idx]) {
                const int64_t m = (int64_t) by_e[(size_t) e].size();
                if (m >= gemm_min) rg += (m + 7) & ~(int64_t) 7;""", """            int64_t rg = 0, njp = 0, njr = 0, ng_all = 0, nt_all = 0;   // larry, patch-zcopy3: GEMM experts, their tiles
            for (int32_t e : cl[(size_t) c.idx]) {
                const int64_t m = (int64_t) by_e[(size_t) e].size();
                if (m >= gemm_min) { rg += (m + 7) & ~(int64_t) 7; ++ng_all; nt_all += (m + 127) / 128 * (2 * F / 128 + H / 128); }""")
rep("""            const size_t need = 4 * 256 + (size_t) rg * 4 + zc_rounds * ((size_t) n * K * 4 + 512) +
                                ((size_t) nbat + zc_rounds) * ((size_t) BJ * sizeof(XeJob) + (size_t) n * K * 4 + 512);""",
    """            const size_t need = 4 * 256 + (size_t) rg * 4 + zc_rounds * ((size_t) n * K * 4 + 512) +
                                ((size_t) nbat + zc_rounds) * ((size_t) BJ * sizeof(XeJob) + (size_t) n * K * 4 + 512) +
                                (zc ? (size_t) 3 * ng_all * sizeof(xe::DqJob) + (size_t) nt_all * sizeof(xe::DqTile) + zc_rounds * 1024 : 0);""")
rep("""                xe::DqJob* hjobs = (xe::DqJob*) c.dq_host;
                xe::DqTile* htiles = (xe::DqTile*) (c.dq_host + jbytes);""", """                // larry, patch-zcopy3: under ZCOPY each round's tables in fresh per-call staging (round 0 must not
                // overwrite what round -1's table copies have yet to read: no host wait between them any more)
                xe::DqJob* hjobs = zc ? (xe::DqJob*) nw_take(jbytes) : (xe::DqJob*) c.dq_host;
                xe::DqTile* htiles = zc ? (xe::DqTile*) nw_take(tbytes) : (xe::DqTile*) (c.dq_host + jbytes);""")
open(P, "w").write(s)
print("patched", P)
