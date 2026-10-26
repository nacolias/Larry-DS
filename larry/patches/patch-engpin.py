#!/usr/bin/env python3
# patch-engpin.py - larry: a prompt batch's Engram rows land in PINNED memory (the I/O pool's dequant writes them
# there; nothing is zero-filled first) and go up with async copies in queue order: 0.8 GB a 16K batch went up from
# pageable memory in ~0.45 s under the submit lock, plus a ~0.2 s memset in start().  STRATA_ENG_PIN (default 1).
# Idempotent.
import sys
EN = "/opt/ai/dsv41/xolotl/tools/v41/engine/"


def rep(path, old, new, cnt=1):
    s = open(path).read()
    if new.strip() in s:
        print(f"{path}: already"); return
    if s.count(old) != cnt: sys.exit(f"{path}: anchor x{s.count(old)}: {old[:90]!r}")
    open(path, "w").write(s.replace(old, new))
    print(f"{path}: patched")


H = EN + "ds41_engram.hpp"
rep(H, """    bool enabled() const { return !tables_.empty(); }
""", """    bool enabled() const { return !tables_.empty(); }
    int n_slots() const { return (int) tables_.size(); }        ///< Engram layers (2)
""")
rep(H, """    std::shared_ptr<Batch> start(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err);
""", """    std::shared_ptr<Batch> start(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err);
    /// larry: start() into caller storage: outp[e] holds n * cols * dim floats for slot e (e.g. pinned memory); it is
    /// not cleared first (every slot of a good batch is written) and must stay untouched until wait() returns.
    std::shared_ptr<Batch> start_into(int64_t pos0, int n, float * const * outp, std::string & err);
""")

C = EN + "ds41_engram.cpp"
rep(C, """    std::vector<std::vector<float>> * out = nullptr;  // [layer][n * cols * dim], the caller's storage
""", """    std::vector<float *> outp;                        // [layer] n * cols * dim floats, the caller's storage
""")
rep(C, """std::shared_ptr<Ds41Engram::Batch> Ds41Engram::start(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    if (nel == 0) { err = "engram: no tables"; return nullptr; }
    if (n <= 0) { err = "engram: empty batch"; return nullptr; }
    auto b = std::make_shared<Batch>();
    b->n = n;
    b->out = &out;
    const int64_t cols = cols_, dim = dim_, srsz = dim / 32, m = n * cols;""",
    """std::shared_ptr<Ds41Engram::Batch> Ds41Engram::start(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    if (nel == 0) { err = "engram: no tables"; return nullptr; }
    if (n <= 0) { err = "engram: empty batch"; return nullptr; }
    // every slot of a good batch is written (a cached or a read row): sized, not cleared - the caller's capacity is
    // reused with no per-pass allocation (and no 0.8 GB memset for a 16K batch)
    out.resize((size_t) nel);
    std::vector<float *> p((size_t) nel);
    for (int64_t e = 0; e < nel; ++e) {
        out[(size_t) e].resize((size_t) (n * cols_ * dim_));
        p[(size_t) e] = out[(size_t) e].data();
    }
    return start_into(pos0, n, p.data(), err);
}

std::shared_ptr<Ds41Engram::Batch> Ds41Engram::start_into(int64_t pos0, int n, float * const * outp, std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    if (nel == 0) { err = "engram: no tables"; return nullptr; }
    if (n <= 0) { err = "engram: empty batch"; return nullptr; }
    auto b = std::make_shared<Batch>();
    b->n = n;
    b->outp.assign(outp, outp + nel);
    const int64_t cols = cols_, dim = dim_, srsz = dim / 32, m = n * cols;""")
rep(C, """    out.resize((size_t) nel);
    b->uniq.resize((size_t) nel); b->packed.resize((size_t) nel); b->run_begin.resize((size_t) nel);
    for (int64_t e = 0; e < nel; ++e) {
        const Table & T = tables_[(size_t) e];
        out[(size_t) e].assign((size_t) (m * dim), 0.0f);   // reuses the caller's capacity - no per-pass allocation
""", """    b->uniq.resize((size_t) nel); b->packed.resize((size_t) nel); b->run_begin.resize((size_t) nel);
    for (int64_t e = 0; e < nel; ++e) {
        const Table & T = tables_[(size_t) e];
""")
rep(C, """            std::lock_guard<std::mutex> lk(cache_mu_);   // larry: a prefetch's reads may still be filling it
            float * oe = out[(size_t) e].data();""", """            std::lock_guard<std::mutex> lk(cache_mu_);   // larry: a prefetch's reads may still be filling it
            float * oe = b->outp[(size_t) e];""")
rep(C, """            uint8_t bw[4096 + 512], bs[1024];
            auto & outv = *bp->out;
            for (int64_t g = a; g < c; ++g) {
                if (g >= ng) {""", """            uint8_t bw[4096 + 512], bs[1024];
            for (int64_t g = a; g < c; ++g) {
                if (g >= ng) {""")
rep(C, """                        float * oe = outv[cr.e].data();""", """                        float * oe = bp->outp[cr.e];""")
rep(C, """                float * oe = outv[(size_t) G.e].data();""", """                float * oe = bp->outp[(size_t) G.e];""")

D = EN + "ds41_dense.cpp"
rep(D, """    std::shared_ptr<Ds41Engram::Batch> eng_pf;   // larry: the next batch's rows being read into the row cache
""", """    std::shared_ptr<Ds41Engram::Batch> eng_pf;   // larry: the next batch's rows being read into the row cache
    // larry, STRATA_ENG_PIN: a prompt batch's rows in pinned memory, uploaded async (eng_pin_n floats a slot; 0 = the
    // batch used eng_rows), the next batch's writes waiting on eng_pin_ev
    ggml_backend_buffer_t eng_pin_buf = nullptr;
    float * eng_pin[2] = {};
    size_t eng_pin_n = 0;
    ggml_backend_event_t eng_pin_ev = nullptr;
""")
rep(D, """        if (rb_ev) ggml_backend_event_free(rb_ev);
        if (up_ev) ggml_backend_event_free(up_ev);
""", """        if (rb_ev) ggml_backend_event_free(rb_ev);
        if (up_ev) ggml_backend_event_free(up_ev);
        if (eng_pin_ev) ggml_backend_event_free(eng_pin_ev);
        if (eng_pin_buf) ggml_backend_buffer_free(eng_pin_buf);
""")
rep(D, """static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;""", """// larry: the pinned rows of a prompt batch of `per` floats a slot (sized once for the widest batch); false = the
// pageable eng_rows path (off, not SYCL, or the allocation failed)
static bool eng_pin_ok(Ds41Dense::Impl & im, size_t per) {
    static const bool on = env_on("STRATA_ENG_PIN", true);
    const int ns = im.engram.n_slots();
    if (!on || !im.sycl || ns < 1 || ns > 2) return false;
    if (im.eng_pin_ev) ggml_backend_event_synchronize(im.eng_pin_ev);   // the previous batch's rows are up
    const size_t per_max = std::max(per, (size_t) im.cfg.prompt_batch * (size_t) (im.EC * im.ED));
    if (!im.eng_pin_buf || ggml_backend_buffer_get_size(im.eng_pin_buf) < (size_t) ns * per * sizeof(float)) {
        if (im.eng_pin_buf) { ggml_backend_buffer_free(im.eng_pin_buf); im.eng_pin_buf = nullptr; }
        ggml_backend_buffer_type_t ht = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(im.backend));
        if (!ht) return false;
        im.eng_pin_buf = ggml_backend_buft_alloc_buffer(ht, (size_t) ns * per_max * sizeof(float));
        if (!im.eng_pin_buf) { std::fprintf(stderr, "dense: Engram pinned rows (%.0f MB) not allocated - pageable\\n", ns * per_max * 4 / 1e6); return false; }
        if (!im.eng_pin_ev) im.eng_pin_ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
        if (!im.eng_pin_ev) { ggml_backend_buffer_free(im.eng_pin_buf); im.eng_pin_buf = nullptr; return false; }
    }
    float * base = (float *) ggml_backend_buffer_get_base(im.eng_pin_buf);
    const size_t stride = ggml_backend_buffer_get_size(im.eng_pin_buf) / sizeof(float) / (size_t) ns;
    for (int e = 0; e < ns; ++e) im.eng_pin[e] = base + (size_t) e * stride;
    return true;
}

static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;""")
rep(D, """    im.eng_waited = false;
    im.eng_batch.reset();
    for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e) im.up_set(im.i_eng[e], im.eng_rows[e].data(), im.eng_rows[e].size() * 4);
""", """    im.eng_waited = false;
    im.eng_batch.reset();
    if (im.eng_pin_n) {   // larry: pinned rows - async copies in queue order (before this layer's graph)
        for (int e = 0; e < im.engram.n_slots() && e < 2; ++e)
            ggml_backend_tensor_set_async(im.backend, im.i_eng[e], im.eng_pin[e], 0, im.eng_pin_n * sizeof(float));
        ggml_backend_event_record(im.eng_pin_ev, im.backend);
        im.eng_pin_n = 0;
    } else {
        for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e) im.up_set(im.i_eng[e], im.eng_rows[e].data(), im.eng_rows[e].size() * 4);
    }
""")
rep(D, """            const double t0 = g_eng_prof ? eng_now() : 0;
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            if (g_eng_prof) { eng_t_start = eng_now() - t0; eng_t_wait = 0; eng_t_n = n; }""",
    """            const double t0 = g_eng_prof ? eng_now() : 0;
            const size_t per = (size_t) n * (size_t) (im.EC * im.ED);
            if (eng_pin_ok(im, per)) {
                im.eng_batch = im.engram.start_into(pos0, n, im.eng_pin, im.err);
                im.eng_pin_n = im.eng_batch ? per : 0;
            } else {
                im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            }
            if (g_eng_prof) { eng_t_start = eng_now() - t0; eng_t_wait = 0; eng_t_n = n; }""")
