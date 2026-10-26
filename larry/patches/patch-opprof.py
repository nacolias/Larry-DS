#!/usr/bin/env python3
# patch-opprof.py [GGML_SYCL_CPP] - larry, GGML_SYCL_OP_PROF=1 (diagnostic, default off): card time op by op.  After
# every computed node of an eager graph (cgraph->uid == 0: the prompt path's spans; recorded decode graphs are left
# alone) a one-thread kernel writes a sequence number into pinned host memory; a host thread timestamps each arrival,
# so a node's time is the gap since the mark before it (its kernels plus the launch gaps).  Summed by label (the op,
# a unary / glu op's own name, the node name without digits), the top GGML_SYCL_OP_PROF_TOP (40) printed every
# GGML_SYCL_OP_PROF_EVERY marks (200000) and at exit.  One device (the first that marks).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(P).read()
if "GGML_SYCL_OP_PROF" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""#include "ggml-backend-impl.h"
""", """#include "ggml-backend-impl.h"

#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>

// larry, GGML_SYCL_OP_PROF=1 (patch-opprof.py): see larry_opprof::mark
namespace larry_opprof {
static bool on() { static const bool v = std::getenv("GGML_SYCL_OP_PROF") != nullptr; return v; }
constexpr uint32_t kCap = 1u << 16;
struct State {
    volatile uint32_t * flags = nullptr;            // pinned host memory, a slot a mark
    std::atomic<const char *> labels[kCap];         // the interval each mark ends
    std::atomic<uint32_t> issued{0};
    std::atomic<uint32_t> seen{0};                  // the poller's position (marks consumed)
    std::atomic<bool> stop{false};
    std::thread poller;
    int device = -1;
    int64_t dropped = 0;                            // marks skipped with the ring full
    std::mutex mu;                                  // acc (the poller adds, the dump reads)
    std::map<std::string, std::pair<double, int64_t>> acc;
    std::set<std::string> names;                    // interned labels (the submitting thread only)
};
static State & st() { static State s; return s; }
static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void dump(const char * why) {
    State & S = st();
    static const int top = std::getenv("GGML_SYCL_OP_PROF_TOP") ? std::atoi(std::getenv("GGML_SYCL_OP_PROF_TOP")) : 40;
    std::vector<std::pair<std::string, std::pair<double, int64_t>>> v;
    {
        std::lock_guard<std::mutex> lk(S.mu);
        v.assign(S.acc.begin(), S.acc.end());
    }
    std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.first > b.second.first; });
    double tot = 0;
    for (const auto & e : v) tot += e.second.first;
    fprintf(stderr, "larry op prof (%s): %.1f ms over %zu labels, %lld marks dropped\\n", why, tot, v.size(), (long long) S.dropped);
    for (size_t i = 0; i < v.size() && (int) i < top; ++i)
        fprintf(stderr, "  %-44.44s %10.1f ms %5.1f %%  %8lld x  %8.1f us\\n", v[i].first.c_str(), v[i].second.first,
                100.0 * v[i].second.first / std::max(tot, 1e-9), (long long) v[i].second.second,
                1e3 * v[i].second.first / std::max<int64_t>(v[i].second.second, 1));
}
static void start(sycl::queue & q, int device) {
    State & S = st();
    S.device = device;
    S.flags = sycl::malloc_host<uint32_t>(kCap, q.get_context());
    for (uint32_t i = 0; i < kCap; ++i) { S.flags[i] = 0; S.labels[i].store(nullptr); }
    S.poller = std::thread([] {
        State & S = st();
        static const int64_t every = std::getenv("GGML_SYCL_OP_PROF_EVERY") ? std::atoll(std::getenv("GGML_SYCL_OP_PROF_EVERY")) : 200000;
        uint32_t seen = 0;
        double last = -1;
        while (!S.stop.load(std::memory_order_relaxed)) {
            if (seen == S.issued.load(std::memory_order_acquire)) { std::this_thread::yield(); continue; }
            const uint32_t k = seen % kCap;
            if (S.flags[k] != seen + 1) { __builtin_ia32_pause(); continue; }   // its kernel has not run yet
            const double t = now_ms();
            const char * lab = S.labels[k].load(std::memory_order_acquire);
            if (last >= 0 && lab) {
                std::lock_guard<std::mutex> lk(S.mu);
                auto & e = S.acc[lab];
                e.first += t - last;
                ++e.second;
            }
            last = t;
            ++seen;
            S.seen.store(seen, std::memory_order_release);
            if (every > 0 && seen % (uint32_t) every == 0) dump("periodic");
        }
    });
    std::atexit([] {
        State & S = st();
        S.stop.store(true);
        if (S.poller.joinable()) S.poller.join();
        dump("exit");
    });
}
/// the interval `label` ends here: a mark behind everything queued on `q` so far
static void mark(sycl::queue & q, int device, const char * label) {
    State & S = st();
    if (S.device < 0) start(q, device);
    if (device != S.device || !S.flags) return;
    const uint32_t i = S.issued.load(std::memory_order_relaxed);
    const uint32_t k = i % kCap;
    if (i - S.seen.load(std::memory_order_acquire) >= kCap) { ++S.dropped; return; }   // the ring is full
    S.labels[k].store(label, std::memory_order_release);
    volatile uint32_t * f = S.flags + k;
    const uint32_t seq = i + 1;
    q.single_task([=] { *f = seq; });
    S.issued.store(i + 1, std::memory_order_release);
}
/// a node's label: the op (a unary / glu op by its own name) + the node's name without digits
static const char * label(const ggml_tensor * t) {
    std::string l = ggml_op_name(t->op);
    if (t->op == GGML_OP_UNARY) l = ggml_unary_op_name(ggml_get_unary_op(t));
    if (t->op == GGML_OP_GLU) l = ggml_glu_op_name(ggml_get_glu_op(t));
    std::string n;
    for (const char * p = t->name; *p && n.size() < 24; ++p) if (!(*p >= '0' && *p <= '9')) n += *p;
    if (!n.empty()) l += " " + n;
    auto it = st().names.insert(l).first;
    return it->c_str();
}
}  // namespace larry_opprof
""")
rep("""static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_sycl_is_view_or_noop(node)) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
""", """static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);
    // larry, GGML_SYCL_OP_PROF (patch-opprof.py): eager graphs only (a recorded graph would replay fixed marks)
    const bool lop = larry_opprof::on() && cgraph->uid == 0;
    const char * lop_prev = "(idle before a graph)";
    if (lop) larry_opprof::mark(*sycl_ctx->stream(), sycl_ctx->device, lop_prev);
    lop_prev = nullptr;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_sycl_is_view_or_noop(node)) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (lop) {   // the previous node's interval ends; this node's begins
            if (lop_prev) larry_opprof::mark(*sycl_ctx->stream(), sycl_ctx->device, lop_prev);
            lop_prev = larry_opprof::label(node);
        }
""")
rep("""        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }
}
""", """        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }
    if (lop && lop_prev) larry_opprof::mark(*sycl_ctx->stream(), sycl_ctx->device, lop_prev);   // larry: the last node's end
}
""")
open(P, "w").write(s)
print("patched", P)
