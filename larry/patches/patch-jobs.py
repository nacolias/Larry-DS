# patch-jobs.py - the prompt job table in device memory, filled from a pinned host copy by a queued memcpy: the host
# writes into malloc_shared memory went through the driver's page-fault (signal) handler, which deadlocked against a
# concurrent enqueue on another thread (pipelined batch, 2026-10-20 16:2x: a card thread in the handler, the main
# thread in urEnqueueUSMMemcpy, both on driver mutexes)
p = '/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp'
s = open(p).read()
pairs = [
    ('''    XeJob* cjobs = nullptr; float* cH = nullptr; float* cY = nullptr; int cjobs_cap = 0;''',
     '''    XeJob* cjobs = nullptr; float* cH = nullptr; float* cY = nullptr; int cjobs_cap = 0;
    XeJob* cjobs_host = nullptr;    ///< pinned: cjobs' host copy (no shared USM: its page-fault handler deadlocked)'''),
    ('''            c.cjobs = sycl::malloc_shared<XeJob>((size_t) BJ, c.q);''',
     '''            if (c.cjobs_host) sycl::free(c.cjobs_host, c.q);
            c.cjobs = sycl::malloc_device<XeJob>((size_t) BJ, c.q);
            c.cjobs_host = sycl::malloc_host<XeJob>((size_t) BJ, c.q);'''),
    ('''            if (!c.cjobs || !c.cH || !c.cY) {
                std::fprintf(stderr, "xe tier: card %d: chunk job buffers allocation failed''',
     '''            if (!c.cjobs || !c.cjobs_host || !c.cH || !c.cY) {
                std::fprintf(stderr, "xe tier: card %d: chunk job buffers allocation failed'''),
    ('''                    c.q.wait();   // cjobs (shared memory) and cS are read by the previous batch / the GEMM gather''',
     '''                    c.q.wait();   // cjobs / cjobs_host and cS are read by the previous batch / the GEMM gather'''),
    ('''                        c.cjobs[q] = js[b0 + q];''', '''                        c.cjobs_host[q] = js[b0 + q];'''),
    ('''                    {
                        std::lock_guard<std::mutex> lk(g_submit);
                        c.q.memcpy(c.cS, S.data(), S.size() * 4);
                    }
                    c.q.wait();''', '''                    {
                        std::lock_guard<std::mutex> lk(g_submit);
                        c.q.memcpy(c.cjobs, c.cjobs_host, nb * sizeof(XeJob));
                        c.q.memcpy(c.cS, S.data(), S.size() * 4);
                    }
                    c.q.wait();'''),
    ('''fr(c.cjobs); fr(c.cH);''', '''fr(c.cjobs); fr(c.cjobs_host); fr(c.cH);'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(p, 'w').write(s)
print('ok')
