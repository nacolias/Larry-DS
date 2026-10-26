#!/usr/bin/env python3
# patch-stallwatch4.py [DENSE_CPP] - larry (diagnostic, after patch-stallwatch3.py): perf's DWARF unwinding stops at
# urEventWait (abwT71: the stalled head's thread spinning in urEventWait -> __sched_yield, a host wait on an event inside
# the head graph's submission; the caller unknown).  STRATA_STALL_GDB=1 (default 0): once a watched call passes 400 ms,
# `gdb -p PID -batch -ex "thread apply all bt 40"` (posix_spawn) writes every thread's backtrace to
# logs/stall.PID.N.gdb - the waiting frame's caller, and whatever the other threads are doing.  Root only (yama).
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_STALL_GDB" in s:
    print("already patched"); sys.exit(0)
old = """            while (!done) {
                char path[64];
"""
new = """            // larry, STRATA_STALL_GDB (patch-stallwatch4): every thread's backtrace, once the stall is 400 ms old
            static const bool gdb_on = std::getenv("STRATA_STALL_GDB") && std::getenv("STRATA_STALL_GDB")[0] == '1';
            static int gdb_n = 0;
            if (gdb_on && gdb_n < 3 && !cv.wait_for(lk, std::chrono::milliseconds(100), [this] { return done; })) {
                char pidb[16], out[160];
                std::snprintf(pidb, sizeof pidb, "%d", (int) getpid());
                std::snprintf(out, sizeof out, "/opt/ai/dsv41/logs/stall.%d.%d.gdb", (int) getpid(), gdb_n++);
                posix_spawn_file_actions_t fa;
                posix_spawn_file_actions_init(&fa);
                posix_spawn_file_actions_addopen(&fa, 1, out, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                posix_spawn_file_actions_adddup2(&fa, 1, 2);
                const char * av[] = { "gdb", "-p", pidb, "-batch", "-nx", "-ex", "set pagination off", "-ex", "thread apply all bt 40", nullptr };
                pid_t pp = 0;
                if (posix_spawnp(&pp, "gdb", &fa, nullptr, const_cast<char * const *>(av), ::environ) == 0)
                    std::fprintf(stderr, "stall watch (%s): gdb backtraces -> %s\\n", what, out);
                posix_spawn_file_actions_destroy(&fa);
            }
            while (!done) {
                char path[64];
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
if "#include <fcntl.h>\n" not in s:
    s = s.replace("#include <thread>\n", "#include <thread>\n#include <fcntl.h>\n", 1)
open(P, "w").write(s)
print("patched", P)
