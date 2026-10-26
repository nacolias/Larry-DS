#!/usr/bin/env python3
# patch-calltl.py [TIER_CPP] - larry, STRATA_XE_CALL_TL=1 (diagnostic, default 0): a prompt chunk call's timeline per card,
# read at the waits the card threads make anyway (no extra synchronisation): thread start, the overflow stream's
# copies submitted (experts / MB / pinned ones), the GPU done with X + the prefetch barrier + the stream (the wait
# before the GEMM experts are submitted), the GEMMs submitted, the GPU done with them (the first job batch's wait),
# the card's end (its part pushed), and the call's CPU-light part; one "call tl:" line a call, ms after its start.
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_CALL_TL" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


# the per-card record, before card_work
rep("""    auto card_work = [&](XeCard& c) -> bool {
        std::vector<int32_t> S((size_t) n * K);
""", """    // larry, STRATA_XE_CALL_TL: per card, ms after the call's start (0 = not reached)
    static const bool call_tl = env_i("STRATA_XE_CALL_TL", 0) != 0;
    struct CallTl { double s = 0, st = 0, a = 0, g = 0, b = 0, e = 0; int64_t nst = 0, npin = 0, stb = 0, ng = 0, rows = 0, nj = 0; };
    std::vector<CallTl> ctl(NC);
    double tl_light = 0;
    auto card_work = [&](XeCard& c) -> bool {
        CallTl& tl = ctl[(size_t) c.idx];
        if (call_tl) tl.s = now_ms() - t0;
        std::vector<int32_t> S((size_t) n * K);
""")
# streamed experts: counts / bytes (pinned ones DMA straight)
rep("""                    if (pinned) {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.stream + soff[(size_t) e], hb, bytes);
                    } else if (hot_on()""", """                    if (call_tl) { ++tl.nst; tl.stb += (int64_t) bytes; tl.npin += pinned ? 1 : 0; }
                    if (pinned) {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.stream + soff[(size_t) e], hb, bytes);
                    } else if (hot_on()""")
rep("""                flush();
                lap(g_bp.stream_us);
""", """                flush();
                if (call_tl) tl.st = now_ms() - t0;
                lap(g_bp.stream_us);
""")
# the GPU done with X, the prefetch barrier and the stream (both GEMM paths wait here before their submissions)
rep("""                c.q.wait();   // S and ptok are reused below
""", """                c.q.wait();   // S and ptok are reused below
                if (call_tl && tl.a == 0) tl.a = now_ms() - t0;
""", 2)
rep("""                g_bp.gemm_e += (int64_t) ge.size();
                lap(g_bp.gemm_us);
""", """                g_bp.gemm_e += (int64_t) ge.size();
                if (call_tl) { tl.g = now_ms() - t0; tl.ng += (int64_t) ge.size(); tl.rows += gp; }
                lap(g_bp.gemm_us);
""", 2)
rep("""                    c.q.wait();   // cjobs / cjobs_host and cS are read by the previous batch / the GEMM gather
""", """                    c.q.wait();   // cjobs / cjobs_host and cS are read by the previous batch / the GEMM gather
                    if (call_tl) { if (tl.b == 0) tl.b = now_ms() - t0; tl.nj += (int64_t) nb; }
""")
rep("""        c.q.wait();
        if (prof) {   // the lowest free VRAM at the end of a call (every buffer of the call allocated)
""", """        c.q.wait();
        if (call_tl) tl.e = now_ms() - t0;
        if (prof) {   // the lowest free VRAM at the end of a call (every buffer of the call allocated)
""")
rep("""        const double tw = now_ms();
        for (auto& t : th) t.join();
""", """        const double tw = now_ms();
        if (call_tl && !light.empty()) tl_light = tw - t0;
        for (auto& t : th) t.join();
""")
# the line, after the call trace
rep("""                         t_cards - t_prep, now_ms() - t_cards);
    }
""", """                         t_cards - t_prep, now_ms() - t_cards);
    }
    if (call_tl) {   // larry, STRATA_XE_CALL_TL
        std::string ln;
        char b[320];
        std::snprintf(b, sizeof b, "call tl: layer %lld n %d: prep %.1f cards %.1f end %.1f light %zu %.1f |", (long long) l, n,
                      t_prep - t0, t_cards - t0, now_ms() - t0, light.size(), tl_light);
        ln += b;
        for (size_t k = 0; k < NC; ++k) {
            const CallTl& q = ctl[k];
            if (q.s == 0 && q.e == 0) continue;
            std::snprintf(b, sizeof b, " c%zu[s %.1f st %.1f (%lld e %lld pin %.0f MB) A %.1f G %.1f B %.1f E %.1f; g %lld r %lld j %lld]", k,
                          q.s, q.st, (long long) q.nst, (long long) q.npin, q.stb / 1e6, q.a, q.g, q.b, q.e, (long long) q.ng,
                          (long long) q.rows, (long long) q.nj);
            ln += b;
        }
        std::fprintf(stderr, "%s\\n", ln.c_str());
    }
""")
open(P, "w").write(s)
print("patched", P)
