# patch-pipe4.py - the pipelined batch with the serve DSpark seed capture (bcap_n > 0: the service runs DSpark, so its
# batches carried bcap and never pipelined): the rows are read after the finish of a dm layer's last part
p = '/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp'
s = open(p).read()
pairs = [
    ('''                bool pipe = pipe_env && prdev && !mc.cpu_only && bcap_n == 0, any2 = false;''',
     '''                bool pipe = pipe_env && prdev && !mc.cpu_only, any2 = false;'''),
    ('''                    auto PJ = [&]() -> bool {   // join the tier call in flight, then finish its part
                        if (!inflight) return true;
                        inflight = false;
                        const double t = now_ms();
                        const bool ok = tier.run_chunk_wait();
                        fd_tier += now_ms() - t;
                        if (!ok) { ferr = "tier run refused"; return false; }
                        return PF(fl, fa, fb);
                    };''', '''                    // serve DSpark seed (bcap_n > 0): the batch's last bcap_n rows of the state leaving dm_layers[j] (as
                    // the serial loop's capture), read once that layer's last part is finished
                    auto PC = [&](int l) -> bool {
                        if (!(bcap_n > 0 && c0 + B == n)) return true;
                        for (int j = 0; j < 3; ++j) {
                            if (l != dm_layers[j]) continue;
                            const int cn = std::min(bcap_n, B);
                            bcap_got = cn;
                            bool okr;
                            {
                                Ds41XeTier::submit_priority().fetch_add(1);
                                std::lock_guard<std::mutex> lk(Ds41XeTier::submit_mutex());
                                Ds41XeTier::submit_priority().fetch_sub(1);
                                okr = dense.read_state_cols(B - cn, cn, vx);
                            }
                            if (!okr) { ferr = dense.last_error(); return false; }
                            const size_t HC = vx.size() / ((size_t) cn * n_embd);
                            for (int t = 0; t < cn; ++t) {
                                float * dst = bcap_vh.data() + (size_t) (bcap_n - cn + t) * 3 * n_embd + (size_t) j * n_embd;
                                for (size_t h = 0; h < HC; ++h) {
                                    const float * xs = vx.data() + ((size_t) t * HC + h) * n_embd;
                                    for (int64_t d = 0; d < n_embd; ++d) dst[d] += xs[d] / (float) HC;
                                }
                            }
                        }
                        return true;
                    };
                    auto PJ = [&]() -> bool {   // join the tier call in flight, then finish its part
                        if (!inflight) return true;
                        inflight = false;
                        const double t = now_ms();
                        const bool ok = tier.run_chunk_wait();
                        fd_tier += now_ms() - t;
                        if (!ok) { ferr = "tier run refused"; return false; }
                        return PF(fl, fa, fb) && (fb < B || PC(fl));
                    };'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(p, 'w').write(s)
print('ok')
