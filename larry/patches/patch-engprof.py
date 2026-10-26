#!/usr/bin/env python3
# patch-engprof.py - STRATA_ENG_PROF also times a prompt batch's Engram: start() (hash / sort / cached-row dequant on
# the caller), the host wait for the reads, and the rows' upload.  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "eng_t_start" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


rep("""static bool engram_wait_host(Ds41Dense::Impl & im) {
    if (!im.eng_batch || im.eng_waited) return true;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_waited = true;
    return true;
}
""", """// STRATA_ENG_PROF: a prompt batch's Engram phases (start on the caller, the host wait, the rows' upload), per batch
static const bool g_eng_prof = std::getenv("STRATA_ENG_PROF") != nullptr;
static double eng_t_start = 0, eng_t_wait = 0;
static int eng_t_n = 0;
static double eng_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static bool engram_wait_host(Ds41Dense::Impl & im) {
    if (!im.eng_batch || im.eng_waited) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    if (g_eng_prof) eng_t_wait += eng_now() - t0;
    im.eng_waited = true;
    return true;
}
""")
rep("""static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;
    if (!im.eng_waited && !im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_waited = false;
    im.eng_batch.reset();
    for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e) im.up_set(im.i_eng[e], im.eng_rows[e].data(), im.eng_rows[e].size() * 4);
""", """static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.eng_waited && !im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    const double t1 = g_eng_prof ? eng_now() : 0;
    im.eng_waited = false;
    im.eng_batch.reset();
    for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e) im.up_set(im.i_eng[e], im.eng_rows[e].data(), im.eng_rows[e].size() * 4);
    if (g_eng_prof && eng_t_n > 64) {
        std::fprintf(stderr, "engram prompt batch: %d tokens, start %.1f ms, host wait %.1f ms, upload %.1f ms\\n", eng_t_n,
                     eng_t_start, eng_t_wait + (t1 - t0), eng_now() - t1);
        eng_t_start = eng_t_wait = 0;
        eng_t_n = 0;
    }
""")
rep("""        } else {
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            im.eng_waited = false;
            if (!im.eng_batch) return false;
        }
    }
    if (im.up_async) ggml_backend_event_record(im.up_ev, im.backend);   // the staging buffer's reuse waits for it""",
    """        } else {
            const double t0 = g_eng_prof ? eng_now() : 0;
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            if (g_eng_prof) { eng_t_start = eng_now() - t0; eng_t_wait = 0; eng_t_n = n; }
            im.eng_waited = false;
            if (!im.eng_batch) return false;
        }
    }
    if (im.up_async) ggml_backend_event_record(im.up_ev, im.backend);   // the staging buffer's reuse waits for it""")
open(P, "w").write(s)
print("patched")
