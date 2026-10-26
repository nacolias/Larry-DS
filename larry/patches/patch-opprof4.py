#!/usr/bin/env python3
# patch-opprof4.py [GGML_SYCL_CPP] - larry, GGML_SYCL_OP_PROF_SKIP=N (after patch-opprof.py..3): the first N marks are
# not counted (a one-shot run's first batch compiles oneDNN's partitions and pays first-use allocations - the warm
# picture is the second batch's).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(P).read()
if "GGML_SYCL_OP_PROF_SKIP" in s:
    print("already patched"); sys.exit(0)
old = """            const double t = now_ms();
            const char * lab = S.labels[k].load(std::memory_order_acquire);
            if (last >= 0 && lab) {"""
new = """            const double t = now_ms();
            const char * lab = S.labels[k].load(std::memory_order_acquire);
            static const int64_t skip = std::getenv("GGML_SYCL_OP_PROF_SKIP") ? std::atoll(std::getenv("GGML_SYCL_OP_PROF_SKIP")) : 0;
            if (last >= 0 && lab && (int64_t) seen >= skip) {   // larry, patch-opprof4: the first `skip` marks uncounted"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
old2 = """        std::lock_guard<std::mutex> lk(S.mu);
        S.host[label] += th - S.last_host;"""
if s.count(old2) != 1:
    sys.exit("anchor2 x%d" % s.count(old2))
s = s.replace(old2, """        static const int64_t skip_h = std::getenv("GGML_SYCL_OP_PROF_SKIP") ? std::atoll(std::getenv("GGML_SYCL_OP_PROF_SKIP")) : 0;
        std::lock_guard<std::mutex> lk(S.mu);
        if ((int64_t) i >= skip_h) S.host[label] += th - S.last_host;""")
open(P, "w").write(s)
print("patched", P)
