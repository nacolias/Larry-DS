#!/usr/bin/env python3
# patch-stallwatch.py [DENSE_CPP] - larry (diagnostic, after patch-logitsdiag.py): a batch head whose QUEUEING took
# 1.27-1.5 s (abwT65: ggml-sycl's submission of a one-row head graph, the GPU then 12-25 ms) - where does the host thread
# block?  While Ds41Dense::logits runs past 300 ms a helper thread samples the calling thread's kernel stack
# (/proc/self/task/TID/stack, root) and wchan every 100 ms; at the end the distinct stacks are printed with their counts
# (a driver call - VMM mapping, residency, TTM eviction - shows its xe / ttm frames; user-space waits show futex).
# STRATA_STALL_WATCH=0 turns it off.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_STALL_WATCH" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


for inc in ("#include <condition_variable>\n", "#include <map>\n", "#include <sys/syscall.h>\n", "#include <unistd.h>\n", "#include <fstream>\n", "#include <sstream>\n"):
    if inc not in s:
        rep("#include <thread>\n", "#include <thread>\n" + inc)
rep("""bool Ds41Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
""", """// larry, STRATA_STALL_WATCH (patch-stallwatch.py, diagnostic, default on): while a watched call runs past 300 ms, the
// calling thread's kernel stack is sampled every 100 ms (/proc/self/task/TID/stack: root only) and printed at the end
struct StallWatch {
    const char * what;
    pid_t tid = 0;
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    std::map<std::string, int> stacks;
    explicit StallWatch(const char * w) : what(w) {
        static const bool on = !(std::getenv("STRATA_STALL_WATCH") && std::getenv("STRATA_STALL_WATCH")[0] == '0');
        if (!on) return;
        tid = (pid_t) syscall(SYS_gettid);
        th = std::thread([this] {
            std::unique_lock<std::mutex> lk(mu);
            if (cv.wait_for(lk, std::chrono::milliseconds(300), [this] { return done; })) return;
            while (!done) {
                char path[64];
                std::snprintf(path, sizeof path, "/proc/self/task/%d/stack", (int) tid);
                std::ifstream f(path);
                std::stringstream ss;
                ss << f.rdbuf();
                std::snprintf(path, sizeof path, "/proc/self/task/%d/wchan", (int) tid);
                std::ifstream g(path);
                std::string wc;
                std::getline(g, wc);
                ++stacks["wchan " + wc + "\\n" + ss.str()];
                cv.wait_for(lk, std::chrono::milliseconds(100), [this] { return done; });
            }
        });
    }
    ~StallWatch() {
        if (!th.joinable()) return;
        { std::lock_guard<std::mutex> lk(mu); done = true; }
        cv.notify_all();
        th.join();
        for (const auto & kv : stacks)
            std::fprintf(stderr, "stall watch (%s): %d samples of 100 ms:\\n%s", what, kv.second, kv.first.c_str());
    }
};

bool Ds41Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    StallWatch sw("batch head");   // larry, STRATA_STALL_WATCH
""")
open(P, "w").write(s)
print("patched", P)
