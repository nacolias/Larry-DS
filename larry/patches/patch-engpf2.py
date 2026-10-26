#!/usr/bin/env python3
# patch-engpf2.py - larry: the Engram next-batch prefetch (patch-engpf.py) deferred until the current batch's own
# reads are done, and planned off the dense thread: started with them, its lookups (under the cache lock) and reads
# slowed them (a 16K batch's wait went 0 -> ~0.4 s in 2 of 3 runs).  Idempotent.
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
rep(H, """    void pump();
    void task_run(const std::shared_ptr<Task> & t);""", """    void pump();
    void prefetch_run(Batch * b, int64_t pos0, int n);   // larry: prefetch()'s planning + reads, on its own thread
    void task_run(const std::shared_ptr<Task> & t);""")

C = EN + "ds41_engram.cpp"
rep(C, """    struct CachedRow { uint32_t e, k, idx; };
    std::vector<CachedRow> cached;
};
""", """    struct CachedRow { uint32_t e, k, idx; };
    std::vector<CachedRow> cached;
    std::thread th;   // larry: a prefetch's planner (joined by wait())
    ~Batch() { if (th.joinable()) th.join(); }
};
""")
rep(C, """bool Ds41Engram::wait(const std::shared_ptr<Batch> & b, std::string & err) {
    if (!b) { err = "engram: null batch"; return false; }
    if (b->read_task) b->read_task->wait();""", """bool Ds41Engram::wait(const std::shared_ptr<Batch> & b, std::string & err) {
    if (!b) { err = "engram: null batch"; return false; }
    if (b->th.joinable()) b->th.join();   // a prefetch: its planner fired and waited its reads
    if (b->read_task) b->read_task->wait();""")
# prefetch(): the body moves to prefetch_run (a thread); prefetch() only launches it
s = open(C).read()
a = s.find("std::shared_ptr<Ds41Engram::Batch> Ds41Engram::prefetch(int64_t pos0, int n, std::string & err) {")
z = s.find("bool Ds41Engram::rows(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err) {")
if a < 0 or z < 0 or z < a: sys.exit("prefetch body anchors")
body = s[a:z]
if "prefetch_run" in body:
    print(C + ": prefetch already threaded")
else:
    old_head = """std::shared_ptr<Ds41Engram::Batch> Ds41Engram::prefetch(int64_t pos0, int n, std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    // (a full row cache still gains: the reads warm the block device's page cache for the batch's own preads)
    if (nel == 0 || n <= 0 || nthreads_ <= 0) return nullptr;
    auto b = std::make_shared<Batch>();
    b->n = n;
    const int64_t cols = cols_, dim = dim_, srsz = dim / 32, m = n * cols;"""
    if body.count(old_head) != 1: sys.exit("prefetch head")
    new_head = """std::shared_ptr<Ds41Engram::Batch> Ds41Engram::prefetch(int64_t pos0, int n, std::string & err) {
    // (a full row cache still gains: the reads warm the block device's page cache for the batch's own preads)
    if (tables_.empty() || n <= 0 || nthreads_ <= 0) return nullptr;
    auto b = std::make_shared<Batch>();
    b->n = n;
    // hashing / sorting a 16K batch is ~0.1 s: its own thread plans it and waits its reads (hist_ is not written until
    // the caller's wait(): the dense half waits for it before its next set_token)
    Batch * bp = b.get();
    b->th = std::thread([this, bp, pos0, n] { prefetch_run(bp, pos0, n); });
    (void) err;
    return b;
}

void Ds41Engram::prefetch_run(Batch * b, int64_t pos0, int n) {
    const int64_t nel = (int64_t) tables_.size();
    const int64_t cols = cols_, dim = dim_, srsz = dim / 32, m = n * cols;"""
    body2 = body.replace(old_head, new_head)
    old_tail = """    if (b->groups.empty()) return nullptr;
    auto t = std::make_shared<Task>();
    t->n = (int64_t) b->groups.size();
    Batch * bp = b.get();"""
    if body2.count(old_tail) != 1: sys.exit("prefetch tail 1")
    body2 = body2.replace(old_tail, """    if (b->groups.empty()) return;
    auto t = std::make_shared<Task>();
    t->n = (int64_t) b->groups.size();
    Batch * bp = b;""")
    old_end = """    b->read_task = t;
    fire(t, std::min<int64_t>(t->n, 8 * nthreads_));
    (void) err;
    return b;
}
"""
    if body2.count(old_end) != 1: sys.exit("prefetch tail 2")
    body2 = body2.replace(old_end, """    run_sync(t, std::min<int64_t>(t->n, 8 * nthreads_));
}
""")
    # the planner's cache lookups: chunked under the lock (the dense half's next start() is the only other reader then)
    s = s[:a] + body2 + s[z:]
    open(C, "w").write(s)
    print(C + ": prefetch threaded")

D = EN + "ds41_dense.cpp"
rep(D, """    std::shared_ptr<Ds41Engram::Batch> eng_pf;   // larry: the next batch's rows being read into the row cache
""", """    std::shared_ptr<Ds41Engram::Batch> eng_pf;   // larry: the next batch's rows being read into the row cache
    std::vector<int> eng_pf_tids;                // larry: a prefetch request waiting for this batch's own reads
    int eng_pf_pos0 = -1;
""")
rep(D, """static bool engram_wait_host(Ds41Dense::Impl & im) {
    if (!im.eng_batch || im.eng_waited) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    if (g_eng_prof) eng_t_wait += eng_now() - t0;
    im.eng_waited = true;
    return true;
}
""", """// larry: the deferred next-batch prefetch (Ds41Dense::engram_prefetch), launched once this batch's own reads are done
static void eng_pf_launch(Ds41Dense::Impl & im) {
    if (im.eng_pf_pos0 < 0 || im.eng_pf) return;
    const int n = (int) im.eng_pf_tids.size();
    for (int t = 0; t < n; ++t) im.engram.set_token(im.eng_pf_pos0 + t, im.eng_pf_tids[(size_t) t]);
    std::string e2;
    im.eng_pf = im.engram.prefetch(im.eng_pf_pos0, n, e2);
    im.eng_pf_pos0 = -1;
    im.eng_pf_tids.clear();
}

static bool engram_wait_host(Ds41Dense::Impl & im) {
    if (!im.eng_batch || im.eng_waited) return true;
    const double t0 = g_eng_prof ? eng_now() : 0;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    if (g_eng_prof) eng_t_wait += eng_now() - t0;
    im.eng_waited = true;
    eng_pf_launch(im);   // outside the submit lock
    return true;
}
""")
rep(D, """    if (!im.eng_waited && !im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    const double t1 = g_eng_prof ? eng_now() : 0;
    im.eng_waited = false;
    im.eng_batch.reset();""", """    if (!im.eng_waited && !im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    const double t1 = g_eng_prof ? eng_now() : 0;
    im.eng_waited = false;
    im.eng_batch.reset();
    eng_pf_launch(im);""")
rep(D, """        if (im.eng_pf) {   // larry: the prefetched rows are in the cache once it is done (start() reads the cache)""",
    """        im.eng_pf_pos0 = -1;   // larry: a prefetch request never launched is dropped (this pass reads its own rows)
        im.eng_pf_tids.clear();
        if (im.eng_pf) {   // larry: the prefetched rows are in the cache once it is done (start() reads the cache)""")
rep(D, """    if (!on || !im.engram.enabled() || n <= 64 || im.eng_pf || pos0 < 0 || pos0 + n > im.cfg.ctx_max) return;
    for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);   // rewritten by that batch's own inputs
    std::string e2;
    im.eng_pf = im.engram.prefetch(pos0, n, e2);
}""", """    if (!on || !im.engram.enabled() || n <= 64 || im.eng_pf || pos0 < 0 || pos0 + n > im.cfg.ctx_max) return;
    // launched once this batch's own reads are done (eng_pf_launch); the tokens are set then (that batch's own
    // inputs rewrite them)
    im.eng_pf_tids.assign(tids, tids + n);
    im.eng_pf_pos0 = pos0;
    if (!im.eng_batch) eng_pf_launch(im);
}""")
