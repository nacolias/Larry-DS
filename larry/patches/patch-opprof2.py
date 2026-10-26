#!/usr/bin/env python3
# patch-opprof2.py [GGML_SYCL_CPP] - larry, GGML_SYCL_OP_PROF (after patch-opprof.py): the host's side too - per label
# the host time from its mark to the next one (its dispatch: building and submitting the node's kernels), beside the
# card's interval.  A label whose host time is close to its card time ran host-bound (the card waited for the host).
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(P).read()
if "patch-opprof2" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    std::map<std::string, std::pair<double, int64_t>> acc;
    std::set<std::string> names;                    // interned labels (the submitting thread only)""",
    """    std::map<std::string, std::pair<double, int64_t>> acc;
    std::map<std::string, double> host;             // larry, patch-opprof2: host ms a label (its dispatch)
    double last_host = -1;                          // the submitting thread's time after its last mark
    std::set<std::string> names;                    // interned labels (the submitting thread only)""")
rep("""    std::vector<std::pair<std::string, std::pair<double, int64_t>>> v;
    {
        std::lock_guard<std::mutex> lk(S.mu);
        v.assign(S.acc.begin(), S.acc.end());
    }""", """    std::vector<std::pair<std::string, std::pair<double, int64_t>>> v;
    std::map<std::string, double> hv;
    {
        std::lock_guard<std::mutex> lk(S.mu);
        v.assign(S.acc.begin(), S.acc.end());
        hv = S.host;
    }""")
rep("""        fprintf(stderr, "  %-44.44s %10.1f ms %5.1f %%  %8lld x  %8.1f us\\n", v[i].first.c_str(), v[i].second.first,
                100.0 * v[i].second.first / std::max(tot, 1e-9), (long long) v[i].second.second,
                1e3 * v[i].second.first / std::max<int64_t>(v[i].second.second, 1));""",
    """        fprintf(stderr, "  %-44.44s %10.1f ms %5.1f %%  %8lld x  %8.1f us  host %8.1f us\\n", v[i].first.c_str(), v[i].second.first,
                100.0 * v[i].second.first / std::max(tot, 1e-9), (long long) v[i].second.second,
                1e3 * v[i].second.first / std::max<int64_t>(v[i].second.second, 1),
                1e3 * hv[v[i].first] / std::max<int64_t>(v[i].second.second, 1));""")
rep("""    S.labels[k].store(label, std::memory_order_release);
    volatile uint32_t * f = S.flags + k;
    const uint32_t seq = i + 1;
    q.single_task([=] { *f = seq; });
    S.issued.store(i + 1, std::memory_order_release);
}""", """    const double th = now_ms();   // larry, patch-opprof2: the interval's host side (since the last mark)
    if (S.last_host >= 0 && label) {
        std::lock_guard<std::mutex> lk(S.mu);
        S.host[label] += th - S.last_host;
    }
    S.labels[k].store(label, std::memory_order_release);
    volatile uint32_t * f = S.flags + k;
    const uint32_t seq = i + 1;
    q.single_task([=] { *f = seq; });
    S.issued.store(i + 1, std::memory_order_release);
    S.last_host = now_ms();
}""")
open(P, "w").write(s)
print("patched", P)
