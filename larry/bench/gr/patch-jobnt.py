#!/usr/bin/env python3
# patch-jobnt.py - larry, STRATA_XE_JOBNT (default 0 for the A/B): in a decode / verify call of several tokens, a job's
# OWN token count picks its kernel form.  Today the call's widest job (armed verify sequences: the pass width) sets the
# form of every job: test_ds41_xe --bench-q8 (3 experts, layer 20) 1-token jobs 67.1 us in the 1-token form, 74.6 /
# 76.5 / 82.9 us in the 2 / 4 / 8 forms (+11 / +14 / +24 %); 2-token jobs 87.3 / 91.0 / 96.5 in the 2 / 4 / 8 forms.
#   1: gate/up and down pick the form per job inside the kernel (a 1-token job: the 1-token dot code; down: 2-token
#      jobs the 2-token code too) - one launch each, the wide form's registers / GRF mode stay.
#   2: as 1, gate/up in two launches - the 1-token jobs in the 1-token kernel (its registers, 128-GRF mode), the wider
#      jobs in the wide kernel (a launch whose jobs are all another's leaves at once).
# Usage: patch-jobnt.py [xe_planar.cpp]   Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/xe_planar.cpp"
s = open(P).read()
if "STRATA_XE_JOBNT" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


# ---- gate/up: the job filter (lo..hi tokens) and the per-job form (jn)
rep("""template <int NT, int LPR = 4>   // LPR 16: one row per sub-group, 4x the sub-groups (decode: few jobs fill the card)
sycl::event gate_up_q8(sycl::queue& q, const XeJob* jobs, int nj, XeQ8 X, XeQ8 Hq, int n_embd, int n_ff, float L) {
""", """/// gu_rows for a job of any gate/up type
template <int NT, int LPR>
inline void gu_any(const XeJob& jb, const XSet& xs, const int* rows, int row, int bl, int n_ff, int nt, float (&acc)[2][NT]) {
    const int type = jb.gu_type;
    if (type == kQ3K) gu_rows<kQ3K, NT, LPR>(jb, xs, rows, row, bl, n_ff, nt, acc);
    else if (type == kQ20) gu_rows<kQ20, NT, LPR>(jb, xs, rows, row, bl, n_ff, nt, acc);
    else gu_rows<kPtq1, NT, LPR>(jb, xs, rows, row, bl, n_ff, nt, acc);
}

// larry (STRATA_XE_JOBNT, see xe_experts_q8): only the jobs of lo..hi tokens run here (the others are another launch's),
// and jn: a 1-token job runs the 1-token form
template <int NT, int LPR = 4>   // LPR 16: one row per sub-group, 4x the sub-groups (decode: few jobs fill the card)
sycl::event gate_up_q8(sycl::queue& q, const XeJob* jobs, int nj, XeQ8 X, XeQ8 Hq, int n_embd, int n_ff, float L,
                       int lo = 1, int hi = kMaxJobTok, bool jn = false) {
""")
rep("""            if (nt == 0) return;   // a null job (a replayed graph's padding): the whole work-group leaves together
""", """            if (nt < lo || nt > hi) return;   // a null job (a graph's padding) or another launch's: the work-group leaves together
""")
rep("""                for (int t = 0; t < NT; ++t) acc[i][t] = 0.f;
            const int type = jb.gu_type;
            if (type == kQ3K) gu_rows<kQ3K, NT, LPR>(jb, xs, rows, row, bl, n_ff, nt, acc);
            else if (type == kQ20) gu_rows<kQ20, NT, LPR>(jb, xs, rows, row, bl, n_ff, nt, acc);
            else gu_rows<kPtq1, NT, LPR>(jb, xs, rows, row, bl, n_ff, nt, acc);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    float v = acc[i][t];
""", """                for (int t = 0; t < NT; ++t) acc[i][t] = 0.f;
            if constexpr (NT > 1) {
                if (jn && nt == 1) {
                    float a1[2][1] = {{0.f}, {0.f}};
                    gu_any<1, LPR>(jb, xs, rows, row, bl, n_ff, 1, a1);
                    acc[0][0] = a1[0][0];
                    acc[1][0] = a1[1][0];
                } else {
                    gu_any<NT, LPR>(jb, xs, rows, row, bl, n_ff, nt, acc);
                }
            } else {
                gu_any<NT, LPR>(jb, xs, rows, row, bl, n_ff, nt, acc);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t >= nt) continue;   // the job's tokens only (nt is the sub-group's: one job a work-group)
                    float v = acc[i][t];
""")

# ---- down: the per-job form
rep("""template <int NT, int NTOK, int LPR = 4>   // NT: tokens per job (bound), NTOK: tokens of the call (bound)
sycl::event down_q8(sycl::queue& q, const XeJob* jobs, int nj, int ntok, XeQ8 Hq, float* OUT, int n_embd, int n_ff,
                    bool accumulate) {
""", """/// down_job for a job of any down type (its H rows: j * kMaxJobTok + slot)
template <int NT, int NTOK, int LPR>
inline void down_any(const XeJob& jb, const XSet& hs, int j, int row, int bl, int n_embd, int nt, const sycl::sub_group& sg,
                     float (&o)[NTOK]) {
    int rows[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) rows[t] = j * kMaxJobTok + (t < nt ? t : 0);
    const int type = jb.d_type;
    if (type == kQ3K) down_job<kQ3K, NT, NTOK, LPR>(jb, hs, rows, row, bl, n_embd, nt, sg, o);
    else if (type == kPtq1) down_job<kPtq1, NT, NTOK, LPR>(jb, hs, rows, row, bl, n_embd, nt, sg, o);
    else down_job<kQ20, NT, NTOK, LPR>(jb, hs, rows, row, bl, n_embd, nt, sg, o);
}

// jn (larry, STRATA_XE_JOBNT): a 1- or 2-token job of a wider form runs the 1- / 2-token form
template <int NT, int NTOK, int LPR = 4>   // NT: tokens per job (bound), NTOK: tokens of the call (bound)
sycl::event down_q8(sycl::queue& q, const XeJob* jobs, int nj, int ntok, XeQ8 Hq, float* OUT, int n_embd, int n_ff,
                    bool accumulate, bool jn = false) {
""")
rep("""            if (nt == 0) continue;   // a null job (padding)
            int rows[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) rows[t] = j * kMaxJobTok + (t < nt ? t : 0);
            const int type = jb.d_type;
            if (type == kQ3K) down_job<kQ3K, NT, NTOK, LPR>(jb, hs, rows, row, bl, n_embd, nt, sg, o);
            else if (type == kPtq1) down_job<kPtq1, NT, NTOK, LPR>(jb, hs, rows, row, bl, n_embd, nt, sg, o);
            else down_job<kQ20, NT, NTOK, LPR>(jb, hs, rows, row, bl, n_embd, nt, sg, o);
        }
""", """            if (nt == 0) continue;   // a null job (padding)
            if constexpr (NT > 1) {
                if (jn && nt == 1) { down_any<1, NTOK, LPR>(jb, hs, j, row, bl, n_embd, 1, sg, o); continue; }
                if constexpr (NT > 2) {
                    if (jn && nt == 2) { down_any<2, NTOK, LPR>(jb, hs, j, row, bl, n_embd, 2, sg, o); continue; }
                }
            }
            down_any<NT, NTOK, LPR>(jb, hs, j, row, bl, n_embd, nt, sg, o);
        }
""")

# ---- the dispatch
rep("""    if (max_nt <= 1) {
        gate_up_q8<1>(q, jobs, nj, X, H, n_embd, n_ff, Lm);
        return down_q8<1, kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
    }
    if (max_nt <= 2) {""", """    if (max_nt <= 1) {
        gate_up_q8<1>(q, jobs, nj, X, H, n_embd, n_ff, Lm);
        return down_q8<1, kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
    }
    // larry, STRATA_XE_JOBNT (default 0): a job's own token count picks its form (the widest job used to pick everyone's:
    // armed verify sequences use the pass width) - 1: per job inside the kernels, 2: gate/up in two launches as well
    // (1-token jobs / the rest), so the 1-token jobs keep the 1-token kernel's registers and GRF mode
    static const int jobnt = [] { const char* e = std::getenv("STRATA_XE_JOBNT"); return e ? std::atoi(e) : 0; }();
    if (jobnt > 0) {
        const bool two = jobnt >= 2;
        if (two) gate_up_q8<1>(q, jobs, nj, X, H, n_embd, n_ff, Lm, 1, 1);
        const int lo = two ? 2 : 1;
        if (max_nt <= 2) {
            gate_up_q8<2>(q, jobs, nj, X, H, n_embd, n_ff, Lm, lo, kMaxJobTok, true);
            return down_q8<2, kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate, true);
        }
        if (max_nt <= 4) {
            gate_up_q8<4>(q, jobs, nj, X, H, n_embd, n_ff, Lm, lo, kMaxJobTok, true);
            return down_q8<4, kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate, true);
        }
        gate_up_q8<kMaxJobTok>(q, jobs, nj, X, H, n_embd, n_ff, Lm, lo, kMaxJobTok, true);
        return down_q8<kMaxJobTok, kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate, true);
    }
    if (max_nt <= 2) {""")
open(P, "w").write(s)
print("patched")
