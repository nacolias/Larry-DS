#!/usr/bin/env python3
# patch-engpin2.py - the rest of patch-engpin.py (two anchors had moved): start_into's leftover vector code, the
# pinned upload in engram_ready, and upload_inputs' start_into call.  Idempotent.
import sys
EN = "/opt/ai/dsv41/xolotl/tools/v41/engine/"
C = EN + "ds41_engram.cpp"
s = open(C).read()
a = s.find("std::shared_ptr<Ds41Engram::Batch> Ds41Engram::start_into(")
z = s.find("\n}\n", a)
body = s[a:z]
old1 = """    out.resize((size_t) nel);
    b->uniq.resize((size_t) nel);"""
old2 = """        out[(size_t) e].assign((size_t) (m * dim), 0.0f);   // reuses the caller's capacity - no per-pass allocation
"""
if old1 in body or old2 in body:
    if body.count(old1) != 1 or body.count(old2) != 1: sys.exit("start_into leftovers")
    body = body.replace(old1, "    b->uniq.resize((size_t) nel);").replace(old2, "")
    s = s[:a] + body + s[z:]
    open(C, "w").write(s)
    print(C + ": start_into cleaned")
else:
    print(C + ": already")

D = EN + "ds41_dense.cpp"
s = open(D).read()
old = """    im.eng_waited = false;
    im.eng_batch.reset();
    eng_pf_launch(im);
    for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e) im.up_set(im.i_eng[e], im.eng_rows[e].data(), im.eng_rows[e].size() * 4);
"""
new = """    im.eng_waited = false;
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
"""
if new in s: print(D + ": upload already")
else:
    if s.count(old) != 1: sys.exit("engram_ready anchor")
    s = s.replace(old, new); print(D + ": upload patched")
old = """            const double t0 = g_eng_prof ? eng_now() : 0;
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            if (g_eng_prof) { eng_t_start = eng_now() - t0; eng_t_wait = 0; eng_t_n = n; }"""
new = """            const double t0 = g_eng_prof ? eng_now() : 0;
            const size_t per = (size_t) n * (size_t) (im.EC * im.ED);
            if (eng_pin_ok(im, per)) {
                im.eng_batch = im.engram.start_into(pos0, n, im.eng_pin, im.err);
                im.eng_pin_n = im.eng_batch ? per : 0;
            } else {
                im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            }
            if (g_eng_prof) { eng_t_start = eng_now() - t0; eng_t_wait = 0; eng_t_n = n; }"""
if new in s: print(D + ": start already")
else:
    if s.count(old) != 1: sys.exit("upload_inputs anchor")
    s = s.replace(old, new); print(D + ": start patched")
open(D, "w").write(s)
