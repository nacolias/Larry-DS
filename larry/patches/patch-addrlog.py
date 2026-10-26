#!/usr/bin/env python3
# patch-addrlog.py [ENGINE_DIR] - larry, STRATA_ADDR_LOG=1 (diagnostic, off by default): every device allocation the
# xe tier makes or frees (local_alloc / local_free, grow_dev / grow_host) and every virtual-memory reserve / map /
# unmap of the TP core's KV rows and the pager, with its PCI device, address and size - so a GPU fault's address can
# be matched to its buffer (a card-0 CAT error at 01:24 on 2026-10-25 came right after a keep-cache shrink; with
# GGML_SYCL_MEMTRACE=2 ggml-sycl logs its own allocations the same way).  Idempotent.
import re
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine"
T, PG, CO = D + "/xe/ds41_xe_tier.cpp", D + "/tp/tp_pager.cpp", D + "/tp/tp_core.cpp"


def rep(s, old, new, n=1, path=""):
    if s.count(old) != n:
        sys.exit("%s: anchor x%d (want %d): %s" % (path, s.count(old), n, old[:90]))
    return s.replace(old, new)


# ---------------------------------------------------------------- tier
t = open(T).read()
if "STRATA_ADDR_LOG" not in t:
    t = rep(t, """inline bool local_alloc_on() {""", """/// larry, STRATA_ADDR_LOG=1: a line per device allocation / free (diagnostic)
inline bool addr_log_on() { static const bool on = std::getenv("STRATA_ADDR_LOG") != nullptr; return on; }
inline void addr_log(const char* what, sycl::queue& q, const void* p, size_t bytes) {
    if (!addr_log_on()) return;
    std::string pci = "?";
    try { pci = q.get_device().get_info<sycl::ext::intel::info::device::pci_address>(); } catch (...) {}
    std::fprintf(stderr, "addr: %s %s %p +0x%zx\\n", what, pci.c_str(), p, bytes);
}
inline bool local_alloc_on() {""", path=T)
    t = rep(t, """    if (zeContextMakeMemoryResident(zc, zd, p, bytes) != ZE_RESULT_SUCCESS) { zeMemFree(zc, p); return nullptr; }
    return (uint8_t*) p;
}
inline void local_free(sycl::queue& q, void* p) {
    if (!p) return;
""", """    if (zeContextMakeMemoryResident(zc, zd, p, bytes) != ZE_RESULT_SUCCESS) { zeMemFree(zc, p); return nullptr; }
    addr_log("tier local alloc", q, p, bytes);
    return (uint8_t*) p;
}
inline void local_free(sycl::queue& q, void* p) {
    if (!p) return;
    addr_log("tier local free", q, p, 0);
""", path=T)
    t = rep(t, """template <class T>
bool grow_dev(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); sycl::free(p, q); }
    p = sycl::malloc_device<T>(need, q);
    cap = p ? need : 0;
    return p != nullptr;
}
""", """template <class T>
bool grow_dev(T*& p, size_t& cap, size_t need, sycl::queue& q) {
    if (cap >= need) return true;
    if (p) { q.wait(); addr_log("tier usm free", q, p, 0); sycl::free(p, q); }
    p = sycl::malloc_device<T>(need, q);
    cap = p ? need : 0;
    if (p) addr_log("tier usm alloc", q, p, need * sizeof(T));
    return p != nullptr;
}
""", path=T)
    open(T, "w").write(t)
    print("patched", T)
else:
    print("already patched", T)

# ---------------------------------------------------------------- tp pager / core: virtual memory
WRAP = """
// larry, STRATA_ADDR_LOG=1: the Level Zero virtual-memory calls of this file, logged (diagnostic)
static inline bool vm_log_on() { static const bool on = std::getenv("STRATA_ADDR_LOG") != nullptr; return on; }
static inline ze_result_t larry_vm_map(ze_context_handle_t c, const void * p, size_t n, ze_physical_mem_handle_t h, size_t o,
                                       ze_memory_access_attribute_t a) {
    if (vm_log_on()) std::fprintf(stderr, "addr: vmm map %p +0x%zx\\n", p, n);
    return (zeVirtualMemMap)(c, p, n, h, o, a);
}
static inline ze_result_t larry_vm_unmap(ze_context_handle_t c, const void * p, size_t n) {
    if (vm_log_on()) std::fprintf(stderr, "addr: vmm unmap %p +0x%zx\\n", p, n);
    return (zeVirtualMemUnmap)(c, p, n);
}
static inline ze_result_t larry_vm_reserve(ze_context_handle_t c, const void * s, size_t n, void ** pp) {
    const ze_result_t r = (zeVirtualMemReserve)(c, s, n, pp);
    if (vm_log_on()) std::fprintf(stderr, "addr: vmm reserve %p +0x%zx\\n", pp ? *pp : nullptr, n);
    return r;
}
#define zeVirtualMemMap larry_vm_map
#define zeVirtualMemUnmap larry_vm_unmap
#define zeVirtualMemReserve larry_vm_reserve
"""
for P in (PG, CO):
    s = open(P).read()
    if "STRATA_ADDR_LOG" in s:
        print("already patched", P)
        continue
    incs = [m.end() for m in re.finditer(r"^#include [^\n]*\n", s, re.M)]
    if not incs:
        sys.exit("%s: no includes" % P)
    at = incs[-1]
    s = s[:at] + WRAP + s[at:]
    open(P, "w").write(s)
    print("patched", P)
