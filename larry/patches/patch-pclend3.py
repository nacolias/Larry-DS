#!/usr/bin/env python3
# patch-pclend3.py [XOLOTL_DIR] - larry, 2026-10-29 (after patch-pclend.py / patch-pclend2.py): a prompt call keeps the lent
# promotion slots when it cannot grow a buffer.  Part 1 gave the lent slots back before every prompt call over 8 tokens,
# so the follow-up prompts of an agent session never used them (their expert calls wait on the CPU pool: 129-512 tokens
# 20.0 ms a call, the CPU-light part 24.6 experts in 18.2 ms, last to end 57 % of the time - bench/loop nm1) and each
# reply started with them empty.  The prompt path's device buffers only grow (and a growth that fails fails the call),
# so: the slots go back only when this call might grow one on an expert card - X / OUT, the token map, the job
# buffers, the int8 X / H, or the GEMM scratch against a bound with every row on one card (the stream buffer sizes
# itself to the free VRAM: STRATA_XE_STREAM_FIT); else the call runs with the lent experts resident.  The prefetch, the
# KV rows' room and close give them back as before.  STRATA_XE_PC_LEND_KEEP_CALLS=0: every call gives them back.
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
T = os.path.join(X, "tools/v41/engine/xe/ds41_xe_tier.cpp")
s = open(T).read()
if "patch-pclend3.py" in s:
    print("already patched")
    sys.exit(0)
if "patch-pclend2.py" not in s:
    sys.exit("patch-pclend.py and patch-pclend2.py first")


def rep(old, new, what):
    global s
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:90]))
    s = s.replace(old, new)


FUNC = """/// larry (patch-pclend3.py): might a prompt call of n tokens grow a device buffer on an expert card?  (Only then do the
/// lent slots go back before it: the buffers only grow, and a failed growth fails the call.)  True when nothing is lent.
bool pc_call_grows(const Ds41XeImpl& im, int n) {
    static const bool keep_calls = env_i("STRATA_XE_PC_LEND_KEEP_CALLS", 1) != 0;
    bool lent = false;
    for (const auto& cp : im.cards) lent = lent || cp->pc_mem2;
    if (!lent) return false;   // nothing to give back
    if (!keep_calls) return true;
    const size_t H = (size_t) im.g.n_embd, F = (size_t) im.g.n_ff, K = (size_t) im.g.top_k, N = (size_t) n;
    const size_t BJ = (size_t) env_i("STRATA_XE_CHUNK_JOBS", 512);
    // the GEMM scratch's two needs (run_big's formulas) with every row of the call on this card and its largest expert
    // taking all n tokens: a bound, never below what the call asks for
    const size_t rb = std::max<size_t>((size_t) std::max(256, env_i("STRATA_XE_GEMM_RB", 1024)), (N + 7) & ~(size_t) 7);
    const size_t R = N * K, NG = std::min<size_t>((size_t) im.g.n_experts, R);
    const size_t ntile = (R / 128 + NG) * (2 * F / 128 + H / 128);
    const size_t g1 = rb * H * 2 + rb * 2 * F * 4 + rb * F * 2 + R * H * 4 + R * 4 + 3 * NG * sizeof(xe::DqJob) +
                      ntile * sizeof(xe::DqTile) + 8 * 256;
    const size_t g2 = N * H * 2 + 2 * F * H * 2 + H * F * 2 + N * 2 * F * 4 + N * F * 2 + R * H * 4 + R * 4 + 8192;
    const size_t gb = std::max(g1, g2);
    for (const auto& cp : im.cards) {
        const XeCard& c = *cp;
        if (c.dense) continue;
        if (c.cX_cap < N * H || c.cOUT_cap < N * H || c.cS_cap < N * K || c.cjobs_cap < (int) BJ) return true;
        if (c.gscratch_cap < gb) return true;
        if (c.cXh && c.cXh_cap < N * H) return true;
        if (c.cxq_rows < N || c.chq_jobs < (int) BJ) return true;
    }
    return false;
}

/// larry (patch-pclend.py, STRATA_XE_PC_LEND): each expert card's free VRAM less STRATA_XE_PC_LEND_KEEP_GIB becomes more"""
rep("""/// larry (patch-pclend.py, STRATA_XE_PC_LEND): each expert card's free VRAM less STRATA_XE_PC_LEND_KEEP_GIB becomes more""",
    FUNC, "function")
rep("""        if (n > kMaxJobTok) pc_unlend(*im_);   // larry (patch-pclend.py)
        if (n > kMaxJobTok) disarm(*im_);
        if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, out, nullptr);""",
    """        if (n > kMaxJobTok && pc_call_grows(*im_, n)) pc_unlend(*im_);   // larry (patch-pclend.py; patch-pclend3.py)
        if (n > kMaxJobTok) disarm(*im_);
        if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, out, nullptr);""", "run_chunk")
rep("""        if (n > kMaxJobTok) pc_unlend(*im_);   // larry (patch-pclend.py)
        if (n > kMaxJobTok) disarm(*im_);
        if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, nullptr, out_dev);""",
    """        if (n > kMaxJobTok && pc_call_grows(*im_, n)) pc_unlend(*im_);   // larry (patch-pclend.py; patch-pclend3.py)
        if (n > kMaxJobTok) disarm(*im_);
        if (n <= kMaxJobTok) return run_small(*im_, l, n, ids, w, x, nullptr, out_dev);""", "run_chunk_dev")
rep("""    if (n > kMaxJobTok) pc_unlend(im);   // larry (patch-pclend.py): here, before the call's thread""",
    """    if (n > kMaxJobTok && pc_call_grows(im, n)) pc_unlend(im);   // larry (patch-pclend.py; patch-pclend3.py): before the call's thread""",
    "run_chunk_dev_async")
# the bound's check: a device buffer growing on an expert card while a call runs with lent slots is logged (it must not)
rep("""template <class T>
bool grow_dev(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
""", """/// larry (patch-pclend3.py): set while run_big runs with lent promotion slots kept - a buffer that grows then is logged
std::atomic<int> g_pc_kept{0};
std::atomic<int64_t> g_pc_kept_grew{0};
inline void pc_kept_note(const char* what, size_t bytes) {
    if (!g_pc_kept.load(std::memory_order_relaxed)) return;
    if (g_pc_kept_grew.fetch_add(1) < 20)
        std::fprintf(stderr, "xe tier: %s grew to %.1f MiB in a prompt call that kept the lent slots (patch-pclend3.py's "
                     "bound missed it)\\n", what, bytes / 1048576.0);
}
template <class T>
bool grow_dev(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    pc_kept_note("a device buffer", need * sizeof(T));
""", "grow_dev")
rep("""template <class T>
bool grow_loc(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
""", """template <class T>
bool grow_loc(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    pc_kept_note("a card-local buffer", need * sizeof(T));
""", "grow_loc")
rep("""             void* out_dev, bool last_call = true, const float* x_dev = nullptr) {
    namespace mkl = oneapi::mkl;
    const int64_t H = im.g.n_embd, F = im.g.n_ff;""", """             void* out_dev, bool last_call = true, const float* x_dev = nullptr) {
    // larry (patch-pclend3.py): the call runs with lent slots kept (pc_call_grows said no growth) - growths are logged
    struct Kept {
        bool on = false;
        explicit Kept(const Ds41XeImpl& im) { for (const auto& cp : im.cards) on = on || cp->pc_mem2; if (on) g_pc_kept.store(1); }
        ~Kept() { if (on) g_pc_kept.store(0); }
    } pc_kept_guard(im);
    namespace mkl = oneapi::mkl;
    const int64_t H = im.g.n_embd, F = im.g.n_ff;""", "run_big")
open(T, "w").write(s)
print("patched", T)
