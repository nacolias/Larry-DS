#!/usr/bin/env python3
# patch-stallwatch3.py [DENSE_CPP] - larry (diagnostic, after patch-stallwatch2.py): the second kind of slow batch head
# (abwT70: 6.04 s of "queue", the thread RUNNING in user space the whole time - wchan 0, no kernel stack, no other
# thread inside the driver) needs a user-space stack.  STRATA_STALL_PERF=1 (default 0): once a watched call passes
# 300 ms, `perf record --call-graph dwarf` samples that thread for 0.8 s into logs/stall.PID.N.perf (perf report /
# perf script it afterwards).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_STALL_PERF" in s:
    print("already patched"); sys.exit(0)
old = """            std::unique_lock<std::mutex> lk(mu);
            if (cv.wait_for(lk, std::chrono::milliseconds(300), [this] { return done; })) return;
            while (!done) {
"""
new = """            std::unique_lock<std::mutex> lk(mu);
            if (cv.wait_for(lk, std::chrono::milliseconds(300), [this] { return done; })) return;
            // larry, STRATA_STALL_PERF (patch-stallwatch3): a perf sample of this thread's user + kernel stacks
            static const bool perf_on = std::getenv("STRATA_STALL_PERF") && std::getenv("STRATA_STALL_PERF")[0] == '1';
            static int perf_n = 0;
            if (perf_on && perf_n < 4) {   // posix_spawn (vfork-like): no fork() of a process holding GiBs of DMA-pinned memory
                char ts[16], out[160];
                std::snprintf(ts, sizeof ts, "%d", (int) tid);
                std::snprintf(out, sizeof out, "/opt/ai/dsv41/logs/stall.%d.%d.perf", (int) getpid(), perf_n++);
                const char * av[] = { "perf", "record", "-q", "-F", "499", "--call-graph", "dwarf,32768", "-t", ts, "-o", out, "--", "sleep", "0.8", nullptr };
                pid_t pp = 0;
                if (posix_spawnp(&pp, "perf", nullptr, nullptr, const_cast<char * const *>(av), ::environ) == 0)
                    std::fprintf(stderr, "stall watch (%s): perf sampling thread %d -> %s\\n", what, (int) tid, out);
            }
            while (!done) {
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
if "#include <spawn.h>\n" not in s:
    s = s.replace("#include <thread>\n", "#include <thread>\n#include <spawn.h>\n", 1)
open(P, "w").write(s)
print("patched", P)
