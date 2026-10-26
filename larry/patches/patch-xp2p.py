# patch-xp2p.py - pipelined batch (STRATA_XE_XP2P, default 1): the expert cards copy their X (the dense half's fn
# rows) card-to-card from card 0's o_fn instead of from the host mirror, and the dense half skips the fn readback -
# 3 x n x 20 KB a call went over the shared host uplink, ~21 GB in an 8.5K prompt, beside the prefetch's 61 GB
root = '/opt/ai/dsv41/xolotl/tools/v41/engine/'
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:90])
        s = s.replace(old, new)
    open(path, 'w').write(s)

patch(root + 'ds41_dense.hpp', [('''    const float * batch_fn() const;''', '''    const float * batch_fn() const;
    /// larry: the fn rows on the device (o_fn, [D, NC] floats on card 0; null on the CPU backend), and whether the
    /// batch spans read them back into batch_fn() at all (set_fn_host(false): a GPU tier copies them card-to-card)
    const float * batch_fn_dev() const;
    void set_fn_host(bool on);''')])
patch(root + 'ds41_dense.cpp', [
    ('''    std::mutex * sub_mu = nullptr;            // larry: set_submit_lock''', '''    std::mutex * sub_mu = nullptr;            // larry: set_submit_lock
    bool fn_host = true;                      // larry: set_fn_host'''),
    ('''    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data() + (size_t) col0 * im.D, (size_t) col0 * im.D * 4, (size_t) n * im.D * 4);''',
     '''    if (im.fn_host) ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data() + (size_t) col0 * im.D, (size_t) col0 * im.D * 4, (size_t) n * im.D * 4);'''),
    ('''const float * Ds41Dense::batch_fn() const { return p_->host_fn.data(); }''', '''const float * Ds41Dense::batch_fn() const { return p_->host_fn.data(); }
const float * Ds41Dense::batch_fn_dev() const { return p_->cpu || !p_->o_fn ? nullptr : (const float *) p_->o_fn->data; }
void Ds41Dense::set_fn_host(bool on) { p_->fn_host = on; }'''),
])
patch(root + 'xe/ds41_xe_tier.hpp', [('''    bool run_chunk_dev_async(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, void* out_dev,
                             bool last_of_layer);''', '''    bool run_chunk_dev_async(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, void* out_dev,
                             bool last_of_layer, const float* x_dev = nullptr);   ///< x_dev: X on the dense card (copied card-to-card)''')])
patch(root + 'xe/ds41_xe_tier.cpp', [
    ('''             void* out_dev, bool last_call = true) {''', '''             void* out_dev, bool last_call = true, const float* x_dev = nullptr) {'''),
    ('''        c.q.memcpy(c.cX, x, (size_t) n * H * 4);
        c.q.memcpy(c.cW, w, (size_t) n * K * 4);''', '''        c.q.memcpy(c.cX, x_dev ? x_dev : x, (size_t) n * H * 4);   // x_dev: card-to-card from the dense card
        c.q.memcpy(c.cW, w, (size_t) n * K * 4);'''),
    ('''bool Ds41XeTier::run_chunk_dev_async(int64_t l, int n, const int32_t* ids, const float* w, const float* x, void* out_dev,
                                     bool last_of_layer) {''', '''bool Ds41XeTier::run_chunk_dev_async(int64_t l, int n, const int32_t* ids, const float* w, const float* x, void* out_dev,
                                     bool last_of_layer, const float* x_dev) {'''),
    ('''    im.run_thr = std::thread([&im, l, n, ids, w, x, out_dev, last_of_layer] {''', '''    im.run_thr = std::thread([&im, l, n, ids, w, x, out_dev, last_of_layer, x_dev] {'''),
    ('''                                        : run_big(im, l, n, ids, w, x, nullptr, out_dev, last_of_layer);''',
     '''                                        : run_big(im, l, n, ids, w, x, nullptr, out_dev, last_of_layer, x_dev);'''),
])
patch(root + 'ds41_generate.cpp', [
    ('''                    dense.set_submit_lock(&Ds41XeTier::submit_mutex(), &Ds41XeTier::submit_priority());''',
     '''                    dense.set_submit_lock(&Ds41XeTier::submit_mutex(), &Ds41XeTier::submit_priority());
                    static const bool xp2p = !(std::getenv("STRATA_XE_XP2P") && std::getenv("STRATA_XE_XP2P")[0] == '0');
                    const float * fn_dev = xp2p ? dense.batch_fn_dev() : nullptr;
                    if (fn_dev) dense.set_fn_host(false);'''),
    ('''                                                            dense.batch_fn() + (size_t) a0 * n_embd, prdev + (size_t) a0 * n_embd, last);''',
     '''                                                            dense.batch_fn() + (size_t) a0 * n_embd, prdev + (size_t) a0 * n_embd, last,
                                                            fn_dev ? fn_dev + (size_t) a0 * n_embd : nullptr);'''),
    ('''                    if (!okp) tier.run_chunk_wait();   // never leave the tier thread running
                    dense.set_submit_lock(nullptr);''', '''                    if (!okp) tier.run_chunk_wait();   // never leave the tier thread running
                    dense.set_submit_lock(nullptr);
                    dense.set_fn_host(true);'''),
])
print('ok')
