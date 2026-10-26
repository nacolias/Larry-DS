#!/usr/bin/env python3
# patch-stallwatch2.py [DENSE_CPP] - larry (diagnostic, after patch-stallwatch.py): the slow batch heads block in
# xe_exec_ioctl's down_read_interruptible(&vm->lock) (abwT69: 8 samples, xe_exec_ioctl+0x547 = the call right after
# xe_vm_userptr_check_repin) - another thread holds an xe VM's lock for write (a VM_BIND map / unmap, a userptr
# repin).  Now each sample also reads every thread of the process (/proc/self/task/*: comm, wchan, kernel stack) and
# keeps those inside the GPU driver (xe / drm / ttm / dma_fence / a VM lock) - the holder shows up beside the waiter.
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "patch-stallwatch2" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


if "#include <dirent.h>\n" not in s:
    rep("#include <thread>\n", "#include <thread>\n#include <dirent.h>\n")
rep("""                ++stacks["wchan " + wc + "\\n" + ss.str()];
                cv.wait_for(lk, std::chrono::milliseconds(100), [this] { return done; });
""", """                ++stacks["wchan " + wc + "\\n" + ss.str()];
                // larry, patch-stallwatch2: every other thread inside the GPU driver (the lock holder)
                if (DIR * d = opendir("/proc/self/task")) {
                    while (dirent * de = readdir(d)) {
                        if (de->d_name[0] == '.' || std::atoi(de->d_name) == (int) tid) continue;
                        const std::string base = std::string("/proc/self/task/") + de->d_name;
                        std::ifstream gw(base + "/wchan");
                        std::string w2;
                        std::getline(gw, w2);
                        std::ifstream gs(base + "/stack");
                        std::stringstream s2;
                        s2 << gs.rdbuf();
                        const std::string st = s2.str();
                        auto has = [&](const char * k) { return st.find(k) != std::string::npos || w2.find(k) != std::string::npos; };
                        if (!(has("xe_") || has("drm_") || has("ttm_") || has("dma_fence") || has("rwsem") || has("i915"))) continue;
                        std::ifstream gc(base + "/comm");
                        std::string cm;
                        std::getline(gc, cm);
                        ++stacks["  other thread " + std::string(de->d_name) + " (" + cm + ") wchan " + w2 + "\\n" + st];
                    }
                    closedir(d);
                }
                cv.wait_for(lk, std::chrono::milliseconds(100), [this] { return done; });
""")
open(P, "w").write(s)
print("patched", P)
