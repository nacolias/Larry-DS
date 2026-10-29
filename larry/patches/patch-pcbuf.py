#!/usr/bin/env python3
# patch-pcbuf.py [XOLOTL_DIR] - larry, 2026-10-29 (decode at depth, part 2; after patch-pclend.py): STRATA_XE_PC_LEND_BUFS
# (default 1 with STRATA_XE_PC_LEND) - the prompt path's own buffers on the expert cards lent too.  run_big's buffers grow
# to the largest call seen and stay (cX, cXh, cOUT, cS / cW, the job buffers cjobs / cH / cY, gscratch: ~1.2-1.7 GiB a card
# once a 16K batch ran), so after long prompts the free VRAM part 1 lends was a few slots (tsvc2-pclinst: replies +8 %
# before the 8K-31K needles, even after them).  Now pc_lend frees them first (their sizes remembered; decode's small calls
# never touch them) and pc_unlend - before a prompt call over 8 tokens, the prefetch, the TP core's KV rows - allocates them
# again at those sizes, so a prompt, the KV rows' room and the prefetch's sizing see the card as they did before.
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
T = os.path.join(X, "tools/v41/engine/xe/ds41_xe_tier.cpp")
s = open(T).read()
if "patch-pcbuf.py" in s:
    print("already patched")
    sys.exit(0)
if "patch-pclend.py" not in s:
    sys.exit("patch-pclend.py first")


def rep(old, new, what, count=1):
    global s
    if s.count(old) != count:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:90]))
    s = s.replace(old, new)


rep("""    uint8_t* pc_mem2 = nullptr;
    size_t pc_cap2 = 0;
    int pc_n1 = 0;
""", """    uint8_t* pc_mem2 = nullptr;
    size_t pc_cap2 = 0;
    int pc_n1 = 0;
    /// larry (patch-pcbuf.py, STRATA_XE_PC_LEND_BUFS): the prompt buffers freed for a lend - their sizes (elements /
    /// bytes as their caps count) until pc_unlend allocates them again
    bool pb_released = false;
    size_t pb_cX = 0, pb_cXh = 0, pb_cOUT = 0, pb_cS = 0, pb_gs = 0;
    bool pb_jobs = false;
""", "card fields")

FUNCS = """/// larry (patch-pcbuf.py): card c's prompt buffers freed (between TP passes: nothing in flight uses them; decode's small
/// calls never do), their sizes kept for pc_restore_bufs.  The bytes freed.
size_t pc_release_bufs(Ds41XeImpl& im, XeCard& c) {
    if (c.dense || c.pb_released) return 0;
    c.q.wait();
    size_t b = 0;
    c.pb_cX = c.cX ? c.cX_cap : 0;
    if (c.cX) { local_free(c.q, c.cX); b += c.cX_cap * sizeof(float); c.cX = nullptr; c.cX_cap = 0; }
    c.pb_cXh = c.cXh ? c.cXh_cap : 0;
    if (c.cXh) { local_free(c.q, c.cXh); b += c.cXh_cap * sizeof(sycl::half); c.cXh = nullptr; c.cXh_cap = 0; }
    c.pb_cOUT = c.cOUT ? c.cOUT_cap : 0;
    if (c.cOUT) { sycl::free(c.cOUT, c.q); b += c.cOUT_cap * sizeof(float); c.cOUT = nullptr; c.cOUT_cap = 0; }
    c.pb_cS = c.cS ? c.cS_cap : 0;
    if (c.cS) {
        sycl::free(c.cS, c.q);
        sycl::free(c.cW, c.q);
        b += c.cS_cap * (sizeof(int32_t) + sizeof(float));
        c.cS = nullptr; c.cW = nullptr; c.cS_cap = 0;
    }
    c.pb_jobs = c.cjobs_cap > 0 && c.cjobs && c.cH && c.cY;
    if (c.pb_jobs) {   // (the pinned host copy cjobs_host stays)
        const size_t BJ = (size_t) c.cjobs_cap;
        sycl::free(c.cjobs, c.q);
        sycl::free(c.cH, c.q);
        sycl::free(c.cY, c.q);
        c.cjobs = nullptr; c.cH = nullptr; c.cY = nullptr; c.cjobs_cap = 0;
        b += BJ * (sizeof(XeJob) + (size_t) kMaxJobTok * (size_t) (im.g.n_ff + im.g.n_embd) * sizeof(float));
    }
    c.pb_gs = c.gscratch ? c.gscratch_cap : 0;
    if (c.gscratch) { local_free(c.q, c.gscratch); b += c.gscratch_cap; c.gscratch = nullptr; c.gscratch_cap = 0; }
    c.pb_released = true;
    return b;
}

/// larry (patch-pcbuf.py): the prompt buffers back at their sizes (a failed allocation leaves that one empty: the prompt
/// call's own growth then tries again and reports as before)
void pc_restore_bufs(Ds41XeImpl& im) {
    const int64_t H = im.g.n_embd, F = im.g.n_ff;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.pb_released) continue;
        if (c.pb_cX) { c.cX = reinterpret_cast<float*>(local_alloc(c.q, c.pb_cX * sizeof(float))); c.cX_cap = c.cX ? c.pb_cX : 0; }
        if (c.pb_cXh) { c.cXh = reinterpret_cast<sycl::half*>(local_alloc(c.q, c.pb_cXh * sizeof(sycl::half))); c.cXh_cap = c.cXh ? c.pb_cXh : 0; }
        if (c.pb_cOUT) { c.cOUT = sycl::malloc_device<float>(c.pb_cOUT, c.q); c.cOUT_cap = c.cOUT ? c.pb_cOUT : 0; }
        if (c.pb_cS) {
            c.cS = sycl::malloc_device<int32_t>(c.pb_cS, c.q);
            c.cW = sycl::malloc_device<float>(c.pb_cS, c.q);
            if (c.cS && c.cW) c.cS_cap = c.pb_cS;
            else { if (c.cS) sycl::free(c.cS, c.q); if (c.cW) sycl::free(c.cW, c.q); c.cS = nullptr; c.cW = nullptr; c.cS_cap = 0; }
        }
        if (c.pb_jobs) {
            const int BJ = env_i("STRATA_XE_CHUNK_JOBS", 512);
            c.cjobs = sycl::malloc_device<XeJob>((size_t) BJ, c.q);
            c.cH = sycl::malloc_device<float>((size_t) BJ * kMaxJobTok * F, c.q);
            c.cY = sycl::malloc_device<float>((size_t) BJ * kMaxJobTok * H, c.q);
            if (c.cjobs && c.cH && c.cY) c.cjobs_cap = BJ;
            else {
                if (c.cjobs) sycl::free(c.cjobs, c.q);
                if (c.cH) sycl::free(c.cH, c.q);
                if (c.cY) sycl::free(c.cY, c.q);
                c.cjobs = nullptr; c.cH = nullptr; c.cY = nullptr; c.cjobs_cap = 0;
            }
        }
        if (c.pb_gs) { c.gscratch = local_alloc(c.q, c.pb_gs); c.gscratch_cap = c.gscratch ? c.pb_gs : 0; }
        c.pb_released = false;
        c.pb_cX = c.pb_cXh = c.pb_cOUT = c.pb_cS = c.pb_gs = 0;
        c.pb_jobs = false;
    }
}

/// larry (patch-pclend.py, STRATA_XE_PC_LEND): each expert card's free VRAM less STRATA_XE_PC_LEND_KEEP_GIB becomes more"""
rep("""/// larry (patch-pclend.py, STRATA_XE_PC_LEND): each expert card's free VRAM less STRATA_XE_PC_LEND_KEEP_GIB becomes more""",
    FUNCS, "functions")
# pc_lend: the prompt buffers out first (and the driver's free VRAM waited for)
rep("""    const double t0 = now_ms();
    int added = 0;
    std::string per;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.pc_mem || c.pc_mem2 || c.pbuf_cap) continue;
        const int64_t fr = c.free_vram();""", """    const double t0 = now_ms();
    int added = 0;
    std::string per;
    // larry (patch-pcbuf.py, STRATA_XE_PC_LEND_BUFS): the prompt path's buffers are lent too
    static const bool bufs = env_i("STRATA_XE_PC_LEND_BUFS", 1) != 0;
    if (bufs) {
        for (auto& cp : im.cards) {
            XeCard& c = *cp;
            if (c.dense || !c.pc_mem || c.pc_mem2 || c.pbuf_cap) continue;
            const int64_t fr0 = c.free_vram();
            const size_t freed = pc_release_bufs(im, c);
            if (im.pb_bytes < freed) im.pb_bytes = freed;
            // the xe driver shows freed VRAM a little later
            for (int i = 0, same = 0; freed && fr0 >= 0 && i < 50 && c.free_vram() < fr0 + (int64_t) freed * 9 / 10 && same < 5; ++i) {
                const int64_t f0 = c.free_vram();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                same = c.free_vram() == f0 ? same + 1 : 0;
            }
        }
    }
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.pc_mem || c.pc_mem2 || c.pbuf_cap) continue;
        const int64_t fr = c.free_vram();""", "pc_lend release")
# pc_unlend: also when only the buffers are out; they come back at the end
rep("""void pc_unlend(Ds41XeImpl& im) {
    bool any = false;
    for (auto& cp : im.cards) any = any || cp->pc_mem2;
    if (!any) return;""", """void pc_unlend(Ds41XeImpl& im) {
    bool any = false, bufs = false;
    for (auto& cp : im.cards) { any = any || cp->pc_mem2; bufs = bufs || cp->pb_released; }
    if (!any && !bufs) return;   // (larry, patch-pcbuf.py: the prompt buffers may be out with nothing lent)
    if (!any) {
        const double tb = now_ms();
        pc_restore_bufs(im);
        im.pc_unlend_ms += now_ms() - tb;
        im.pc_unlend_pass = im.tp_pass;
        return;
    }""", "pc_unlend head")
rep("""    ++im.pc_unlends;
    im.pc_unlend_ms += now_ms() - t0;
    im.pc_unlend_pass = im.tp_pass;
}
""", """    pc_restore_bufs(im);   // larry (patch-pcbuf.py): the prompt buffers back at their sizes
    ++im.pc_unlends;
    im.pc_unlend_ms += now_ms() - t0;
    im.pc_unlend_pass = im.tp_pass;
}
""", "pc_unlend tail")
# pc_drop gives everything back the same way first (the prefetch sizes its buffers from the free VRAM after it)
rep("""void pc_drop(Ds41XeImpl& im) {
    im.pc_calls = 0;
    im.pc_tried = false;""", """void pc_unlend(Ds41XeImpl& im);
void pc_drop(Ds41XeImpl& im) {
    pc_unlend(im);   // larry (patch-pcbuf.py): the lent region and the prompt buffers back first
    im.pc_calls = 0;
    im.pc_tried = false;""", "pc_drop")
rep("""    int64_t pc_unlend_pass = 0, pc_lend_next = 0, pc_lends = 0, pc_unlends = 0, pc_lent_max = 0;
""", """    int64_t pc_unlend_pass = 0, pc_lend_next = 0, pc_lends = 0, pc_unlends = 0, pc_lent_max = 0;
    size_t pb_bytes = 0;   ///< larry (patch-pcbuf.py): the most prompt-buffer bytes one card lent
""", "impl field")
rep("""                     "times (%.1f ms each)\\n", (long long) im.pc_lends, (long long) im.pc_lent_max,
                     im.pc_lend_ms / std::max<int64_t>(1, im.pc_lends), (long long) im.pc_unlends,
                     im.pc_unlend_ms / std::max<int64_t>(1, im.pc_unlends));""",
    """                     "times (%.1f ms each); prompt buffers lent up to %.2f GiB a card\\n", (long long) im.pc_lends,
                     (long long) im.pc_lent_max, im.pc_lend_ms / std::max<int64_t>(1, im.pc_lends), (long long) im.pc_unlends,
                     im.pc_unlend_ms / std::max<int64_t>(1, im.pc_unlends), im.pb_bytes / kGiB);""", "close stats")
rep("""    static int logged = 0;
    if (logged++ < 3 || env_i("STRATA_XE_PROF", 0))
        std::fprintf(stderr, "xe tier: promotion cache: +%d slots (%.2f GiB) of the expert cards' free VRAM lent (%s; %.1f ms)\\n",""",
    """    static const bool lend_log = env_i("STRATA_XE_PC_LEND_LOG", 1) != 0;   // (patch-pcbuf.py: every lend - one a turn)
    if (lend_log)
        std::fprintf(stderr, "xe tier: promotion cache: +%d slots (%.2f GiB) of the expert cards' free VRAM lent (%s; %.1f ms)\\n",""",
    "lend log")
open(T, "w").write(s)
print("patched", T)
