#!/usr/bin/env python3
# patch-engslot.py [DENSE_CPP] - larry, after patch-engpg.py: the dense half waits for and uploads each Engram slot's
# rows at its own layer (slot 0 at layer 1, slot 1 at layer 14) instead of the whole pass's at layer 1 - the second
# layer's reads (the other table, queued behind the first's) go on while layers 1-13 run.  A 16K-token prompt waited
# 2.4-3.0 s at layer 1 (both tables' ~1.2M reads).  STRATA_ENG_PROF prints both slots' waits.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "engram_ready_all" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:90]))
    s = s.replace(old, new)


rep("""    bool eng_waited = false;                  // larry: engram_wait_host ran for eng_batch
""", """    bool eng_waited[2] = {};                  // larry: engram_wait_host ran for eng_batch's slot
    bool eng_up[2] = {};                      // larry: eng_batch's slot uploaded (the batch goes once both are)
""")
rep("""static double eng_t_start = 0, eng_t_wait = 0;
""", """static double eng_t_start = 0, eng_t_wait = 0, eng_t_wait1 = 0;   // larry: eng_t_wait1 = slot 1's wait
""")
rep("""// larry: the Engram batch's host-side wait alone (no device work: a pipelined span runs it before the submit lock)
static bool engram_wait_host(Ds41Dense::Impl & im) {
    if (!im.eng_batch || im.eng_waited) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    if (g_eng_prof) eng_t_wait += eng_now() - t0;
    im.eng_waited = true;
    eng_pf_launch(im);   // outside the submit lock
    return true;
}

static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.eng_waited && !im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    const double t1 = g_eng_prof ? eng_now() : 0;
    im.eng_waited = false;
    im.eng_batch.reset();
    eng_pf_launch(im);
    if (im.eng_pin_n) {   // larry: pinned rows - async copies in queue order (before this layer's graph)
        for (int e = 0; e < im.engram.n_slots() && e < 2; ++e)
            ggml_backend_tensor_set_async(im.backend, im.i_eng[e], im.eng_pin[e], 0, im.eng_pin_n * sizeof(float));
        ggml_backend_event_record(im.eng_pin_ev, im.backend);
        im.eng_pin_n = 0;
    } else {
        for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e) im.up_set(im.i_eng[e], im.eng_rows[e].data(), im.eng_rows[e].size() * 4);
    }
    if (g_eng_prof && eng_t_n > 64) {
        std::fprintf(stderr, "engram prompt batch: %d tokens, start %.1f ms, host wait %.1f ms, upload %.1f ms\\n", eng_t_n,
                     eng_t_start, eng_t_wait + (t1 - t0), eng_now() - t1);
        eng_t_start = eng_t_wait = 0;
        eng_t_n = 0;
    }
    if (im.up_async) ggml_backend_event_record(im.up_ev, im.backend);   // the staging buffer's reuse waits for these too
    return true;
}
""", """// larry: the Engram batch's host-side wait alone for one slot (no device work: a pipelined span runs it before the
// submit lock)
static bool engram_wait_host(Ds41Dense::Impl & im, int slot) {
    if (!im.eng_batch || slot < 0 || slot > 1 || im.eng_waited[slot] || im.eng_up[slot]) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.engram.wait_slot(im.eng_batch, slot, im.err)) { im.eng_batch.reset(); return false; }
    if (g_eng_prof) (slot ? eng_t_wait1 : eng_t_wait) += eng_now() - t0;
    im.eng_waited[slot] = true;
    return true;
}

// Slot `slot`'s rows up (after its reads): pinned rows async in queue order (before this layer's graph); the batch is
// done - and the next batch's prefetch launched - once every slot is up
static bool engram_ready(Ds41Dense::Impl & im, int slot) {
    if (!im.eng_batch) return true;
    const int ns = std::min(im.engram.n_slots(), 2);
    if (slot < 0 || slot >= ns || im.eng_up[slot]) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.eng_waited[slot] && !im.engram.wait_slot(im.eng_batch, slot, im.err)) { im.eng_batch.reset(); return false; }
    const double t1 = g_eng_prof ? eng_now() : 0;
    if (g_eng_prof) (slot ? eng_t_wait1 : eng_t_wait) += t1 - t0;
    im.eng_up[slot] = true;
    if (im.eng_pin_n) {
        ggml_backend_tensor_set_async(im.backend, im.i_eng[slot], im.eng_pin[slot], 0, im.eng_pin_n * sizeof(float));
        ggml_backend_event_record(im.eng_pin_ev, im.backend);
    } else if ((size_t) slot < im.eng_rows.size()) {
        im.up_set(im.i_eng[slot], im.eng_rows[(size_t) slot].data(), im.eng_rows[(size_t) slot].size() * 4);
    }
    bool all = true;
    for (int e = 0; e < ns; ++e) all = all && im.eng_up[e];
    if (all) {
        im.engram.wait(im.eng_batch, im.err);   // done already: the batch's read stats
        im.eng_batch.reset();
        for (int e = 0; e < 2; ++e) im.eng_waited[e] = im.eng_up[e] = false;
        im.eng_pin_n = 0;
        eng_pf_launch(im);
        if (g_eng_prof && eng_t_n > 64) {
            std::fprintf(stderr, "engram prompt batch: %d tokens, start %.1f ms, host wait %.1f ms (layer 1) + %.1f ms (layer 14), "
                         "upload %.1f ms\\n", eng_t_n, eng_t_start, eng_t_wait, eng_t_wait1, eng_now() - t1);
            eng_t_start = eng_t_wait = eng_t_wait1 = 0;
            eng_t_n = 0;
        }
    }
    if (im.up_async) ggml_backend_event_record(im.up_ev, im.backend);   // the staging buffer's reuse waits for these too
    return true;
}

// every slot of a still-pending batch up (a new pass / an engine1 read flushes it first)
static bool engram_ready_all(Ds41Dense::Impl & im) {
    for (int e = 0; e < 2; ++e) if (!engram_ready(im, e)) return false;
    return true;
}
""")
rep("""        if (!engram_ready(im)) return false;    // flush a still-pending batch from a previous pass first
""", """        if (!engram_ready_all(im)) return false;    // flush a still-pending batch from a previous pass first
""")
rep("""            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            im.eng_waited = false;
            if (!im.eng_batch) return false;
        } else if (n <= 64) {""", """            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            for (int e = 0; e < 2; ++e) im.eng_waited[e] = im.eng_up[e] = false;
            if (!im.eng_batch) return false;
        } else if (n <= 64) {""")
rep("""            if (g_eng_prof) { eng_t_start = eng_now() - t0; eng_t_wait = 0; eng_t_n = n; }
            im.eng_waited = false;
            if (!im.eng_batch) return false;
""", """            if (g_eng_prof) { eng_t_start = eng_now() - t0; eng_t_wait = eng_t_wait1 = 0; eng_t_n = n; }
            for (int e = 0; e < 2; ++e) im.eng_waited[e] = im.eng_up[e] = false;
            if (!im.eng_batch) return false;
""")
# the layer calls: the layer's own slot
rep("""    if (im.sub_mu && im.ly[(size_t) il].eslot >= 0 && !engram_wait_host(im)) return false;
""", """    if (im.sub_mu && im.ly[(size_t) il].eslot >= 0 && !engram_wait_host(im, (int) im.ly[(size_t) il].eslot)) return false;
""", 2)
rep("""    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // the Engram add feeds this layer's K/V
""", """    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;   // the Engram add feeds this layer's K/V
""")
rep("""    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // wait the pass's Engram reads here
""", """    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;   // wait the slot's Engram reads here
""")
rep("""    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;
""", """    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;
""")
rep("""    if (!engram_ready(im)) return false;   // a dense pass's batch still pending: flushed first
""", """    if (!engram_ready_all(im)) return false;   // a dense pass's batch still pending: flushed first
""")
rep("""    if (slot == 0) {
        if (!engram_ready(im)) return false;
""", """    if (slot == 0) {
        if (!engram_ready_all(im)) return false;
""")
if "engram_ready(im)" in s or "engram_wait_host(im)" in s:
    sys.exit("an engram_ready(im) / engram_wait_host(im) call left")
open(P, "w").write(s)
print("patched", P)
