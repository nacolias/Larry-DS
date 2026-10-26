# patch-pipe2.py - shorter dense submit-lock holds in the pipelined batch: the Engram host wait before the lock, the
# span's index maps through a pinned staging buffer (async copies; each span ends with a full wait, so it is reusable);
# tier: lock-hold timing for the GEMM / job / prep sites (STRATA_XE_PROF)
root = '/opt/ai/dsv41/xolotl/tools/v41/engine/'
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:90])
        s = s.replace(old, new)
    open(path, 'w').write(s)

dc = root + 'ds41_dense.cpp'
patch(dc, [
    ('''    HostRows<uint8_t> up_stage;''', '''    HostRows<uint8_t> up_stage;
    HostRows<uint8_t> span_stage;             // larry: upload_span's index maps (pinned; async copies)
    bool eng_waited = false;                  // larry: engram_wait_host ran for eng_batch'''),
    ('''static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_batch.reset();''', '''// larry: the Engram batch's host-side wait alone (no device work: a pipelined span runs it before the submit lock)
static bool engram_wait_host(Ds41Dense::Impl & im) {
    if (!im.eng_batch || im.eng_waited) return true;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_waited = true;
    return true;
}

static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;
    if (!im.eng_waited && !im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_waited = false;
    im.eng_batch.reset();'''),
    ('''    im.span_memo_pos0 = (int64_t) im.batch_pos0;
    im.span_memo_col0 = col0;
    im.span_memo_n = n;''', '''    im.span_memo_pos0 = (int64_t) im.batch_pos0;
    im.span_memo_col0 = col0;
    im.span_memo_n = n;
    // larry: with a submit lock (pipelined batch) the maps go up through a pinned buffer, async - a pageable
    // ggml_backend_tensor_set is a blocking copy, and it held the lock ~2 ms a span under the tier's PCIe traffic.
    // The span ends with a full synchronize, so the buffer is free again by the next span.
    const size_t need = (size_t) (im.SWA * n + 2 * n + im.RAW + 2) * 4 + 4 * 64;
    const bool staged = im.sub_mu && !im.cpu;
    if (staged && im.span_stage.size() < need) im.span_stage.resize(need, im.backend);
    size_t soff = 0;
    auto put = [&](ggml_tensor * t, const void * data, size_t off, size_t size) {
        if (!staged || im.span_stage.size() < soff + size) { ggml_backend_tensor_set(t, data, off, size); return; }
        std::memcpy(im.span_stage.data() + soff, data, size);
        ggml_backend_tensor_set_async(im.backend, t, im.span_stage.data() + soff, off, size);
        soff = (soff + size + 63) & ~(size_t) 63;
    };'''),
    ('''        ggml_backend_tensor_set(im.i_widx, widx.data(), (size_t) col0 * im.SWA * 4, widx.size() * 4);''',
     '''        put(im.i_widx, widx.data(), (size_t) col0 * im.SWA * 4, widx.size() * 4);'''),
    ('''        ggml_backend_tensor_set(im.i_sidx, si.data(), (size_t) col0 * 2 * 4, si.size() * 4);''',
     '''        put(im.i_sidx, si.data(), (size_t) col0 * 2 * 4, si.size() * 4);'''),
    ('''        ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, ws.size() * 4);''',
     '''        put(im.i_wslot, ws.data(), 0, ws.size() * 4);'''),
    ('''        ggml_backend_tensor_set(im.i_sslot, ss.data(), 0, ss.size() * 4);''',
     '''        put(im.i_sslot, ss.data(), 0, ss.size() * 4);'''),
    # the Engram host wait before the lock
    ('''bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    std::unique_lock<std::mutex> sl;''', '''bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    if (im.sub_mu && im.ly[(size_t) il].eslot >= 0 && !engram_wait_host(im)) return false;
    std::unique_lock<std::mutex> sl;'''),
    ('''bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;
    std::unique_lock<std::mutex> sl;''', '''bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;
    if (im.sub_mu && im.ly[(size_t) il].eslot >= 0 && !engram_wait_host(im)) return false;
    std::unique_lock<std::mutex> sl;'''),
])
# eng_waited must reset when a new batch starts reading
s = open(dc).read()
n0 = s.count('im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);')
s = s.replace('im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);', 'im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);\n            im.eng_waited = false;')
open(dc, 'w').write(s)
print('dense patched', n0)

cp = root + 'xe/ds41_xe_tier.cpp'
patch(cp, [
    ('''struct BigProf {
    double prep = 0, cards = 0, combine = 0, total = 0;''', '''struct BigProf {
    double prep = 0, cards = 0, combine = 0, total = 0;
    std::atomic<int64_t> lk_gemm_us{0}, lk_jobs_us{0}, lk_n{0};   ///< submit-lock hold time of the GEMM / job sites'''),
    ('''                for (int32_t e : ge) {
                    std::lock_guard<std::mutex> lk(g_submit);
                    const int64_t m = (int64_t) by_e[(size_t) e].size();''', '''                for (int32_t e : ge) {
                    std::lock_guard<std::mutex> lk(g_submit);
                    const double tlk = prof ? now_ms() : 0;
                    struct LkT { bool on; double t0; ~LkT() { if (on) { g_bp.lk_gemm_us += (int64_t) ((now_ms() - t0) * 1000.0); ++g_bp.lk_n; } } } lkt{prof, tlk};
                    const int64_t m = (int64_t) by_e[(size_t) e].size();'''),
    ('''                    std::lock_guard<std::mutex> lk(g_submit);
                    if (pl) xe::xe_experts_y_q8(c.q, c.cjobs, (int) nb, c.cxq, c.chq, c.cY, (int) H, (int) F, im.L, max_nt);''',
     '''                    std::lock_guard<std::mutex> lk(g_submit);
                    const double tlk = prof ? now_ms() : 0;
                    struct LkT { bool on; double t0; ~LkT() { if (on) g_bp.lk_jobs_us += (int64_t) ((now_ms() - t0) * 1000.0); } } lkt{prof, tlk};
                    if (pl) xe::xe_experts_y_q8(c.q, c.cjobs, (int) nb, c.cxq, c.chq, c.cY, (int) H, (int) F, im.L, max_nt);'''),
    ('''                         g_bp.small_us / 1000.0, (long long) g_bp.small_j.load());''',
     '''                         g_bp.small_us / 1000.0, (long long) g_bp.small_j.load());
        if (prof && g_bp.calls % 40 == 0)
            std::fprintf(stderr, "xe prof: submit-lock holds: GEMM experts %.0f ms (%lld), job batches %.0f ms\\n",
                         g_bp.lk_gemm_us / 1000.0, (long long) g_bp.lk_n.load(), g_bp.lk_jobs_us / 1000.0);'''),
])
print('tier patched')
