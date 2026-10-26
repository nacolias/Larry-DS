# patch-prio.py - the submit lock with a priority waiter: while the pipelined batch's dense half waits for it, the
# tier's threads hold off their next acquisition (std::mutex is not fair: three card threads re-acquiring it starved
# the dense half's span submissions, ~2 s of lock waits in an 8.5K prompt)
import re
root = '/opt/ai/dsv41/xolotl/tools/v41/engine/'
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:90])
        s = s.replace(old, new)
    open(path, 'w').write(s)

cp = root + 'xe/ds41_xe_tier.cpp'
s = open(cp).read()
assert s.count('std::mutex g_submit;') == 1
s = s.replace('std::mutex g_submit;', '''/// larry: a priority waiter (the pipelined batch's dense half: Ds41XeTier::submit_priority) makes the tier's threads
/// hold off their next acquisition while it waits.
struct PrioMutex {
    std::mutex m;
    std::atomic<int> hi{0};
    void lock() {
        while (hi.load(std::memory_order_acquire) > 0) std::this_thread::yield();
        m.lock();
    }
    void unlock() { m.unlock(); }
};
PrioMutex g_submit;''')
s = s.replace('std::lock_guard<std::mutex> lk(g_submit)', 'std::lock_guard<PrioMutex> lk(g_submit)')
s = s.replace('std::unique_lock<std::mutex> lk(g_submit, std::defer_lock)', 'std::unique_lock<PrioMutex> lk(g_submit, std::defer_lock)')
s = s.replace('std::unique_lock<std::mutex> prep_lk(g_submit)', 'std::unique_lock<PrioMutex> prep_lk(g_submit)')
s = s.replace('std::mutex& Ds41XeTier::submit_mutex() { return g_submit; }',
              'std::mutex& Ds41XeTier::submit_mutex() { return g_submit.m; }\nstd::atomic<int>& Ds41XeTier::submit_priority() { return g_submit.hi; }')
assert 'std::mutex> lk(g_submit' not in s and 'std::mutex> prep_lk' not in s
open(cp, 'w').write(s)

hp = root + 'xe/ds41_xe_tier.hpp'
patch(hp, [('''    static std::mutex& submit_mutex();''', '''    static std::mutex& submit_mutex();
    /// While > 0 the tier's threads hold off taking the submit lock (the dense half increments it while it waits).
    static std::atomic<int>& submit_priority();''')])
s = open(hp).read()
if '#include <atomic>' not in s:
    s = s.replace('#include <mutex>\n', '#include <mutex>\n#include <atomic>\n', 1)
    open(hp, 'w').write(s)

dh = root + 'ds41_dense.hpp'
patch(dh, [('''    void set_submit_lock(std::mutex * m);''', '''    void set_submit_lock(std::mutex * m, std::atomic<int> * prio = nullptr);''')])
s = open(dh).read()
if '#include <atomic>' not in s:
    s = s.replace('#include <mutex>\n', '#include <mutex>\n#include <atomic>\n', 1)
    open(dh, 'w').write(s)

dc = root + 'ds41_dense.cpp'
s = open(dc).read()
s = s.replace('''    std::mutex * sub_mu = nullptr;            // larry: set_submit_lock''',
              '''    std::mutex * sub_mu = nullptr;            // larry: set_submit_lock
    std::atomic<int> * sub_prio = nullptr;    // larry: raised while waiting for sub_mu (the tier's threads hold off)''')
s = s.replace('void Ds41Dense::set_submit_lock(std::mutex * m) { p_->sub_mu = m; }',
              'void Ds41Dense::set_submit_lock(std::mutex * m, std::atomic<int> * prio) { p_->sub_mu = m; p_->sub_prio = m ? prio : nullptr; }')
# every acquisition of sub_mu raises the priority flag while it waits
n0 = s.count('sl = std::unique_lock<std::mutex>(*im.sub_mu);')
s = s.replace('sl = std::unique_lock<std::mutex>(*im.sub_mu);',
              '{ if (im.sub_prio) im.sub_prio->fetch_add(1); sl = std::unique_lock<std::mutex>(*im.sub_mu); if (im.sub_prio) im.sub_prio->fetch_sub(1); }')
assert n0 == 3, n0
open(dc, 'w').write(s)

gp = root + 'ds41_generate.cpp'
patch(gp, [('''                    dense.set_submit_lock(&Ds41XeTier::submit_mutex());''',
            '''                    dense.set_submit_lock(&Ds41XeTier::submit_mutex(), &Ds41XeTier::submit_priority());''')])
print('ok', n0)
