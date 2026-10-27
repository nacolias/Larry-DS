#!/usr/bin/env python3
# patch-kvresident.py [XOLOTL_DIR] - larry, 2026-10-27: card 0's lazily mapped KV state (Ds41KvMap, STRATA_DENSE_LAZY_KV)
# made resident on the other cards as it is mapped.  The DEVICE_LOST of 2026-10-24 and again today (card 3's copy engine
# faulted reading card 0's KV in the TP core's replica growth after a 131K prompt) is ONE peer copy spanning two physical
# allocations of the reservation: bench/conc/vmm_span_probe.cpp - F (split at the allocation boundary) exact, G (one
# copy across, the new allocation zeContextMakeMemoryResident on the peer) exact, E (one copy across, the new
# allocation only mapped) faulted card 1's bcs (Fault response: Unsuccessful -ENOENT).  The driver makes a peer copy's
# FIRST allocation resident on the peer by itself (vmm_peer_read B: a copy inside one later allocation was exact), not
# the ones after it.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/tp/tp_pager.cpp")
s = open(P).read()
if "patch-kvresident.py" in s:
    print("already patched", P)
    sys.exit(0)
pairs = [
("""    struct Unit { size_t off = 0, bytes = 0, tail = 0, done = 0; };   // tail: the spare row's page (mapped to the end)
    std::vector<Unit> units;
""",
"""    struct Unit { size_t off = 0, bytes = 0, tail = 0, done = 0; };   // tail: the spare row's page (mapped to the end)
    std::vector<Unit> units;
    std::vector<ze_device_handle_t> peers;   // larry (patch-kvresident.py): the context's other cards
"""),
("""        phys.push_back(h);
        maps.emplace_back(va + off, bytes);
        q0->memset(va + off, 0, bytes).wait();   // fresh pages: the indexer scores every row up to its cap before masking
""",
"""        phys.push_back(h);
        maps.emplace_back(va + off, bytes);
        // larry (patch-kvresident.py): resident on the other cards too - their copy engines pull these rows (the TP
        // core's replicas); a peer copy that spans into an allocation not resident there faults (vmm_span_probe E)
        for (ze_device_handle_t pd : peers)
            if (zeContextMakeMemoryResident(zc, pd, va + off, bytes) != ZE_RESULT_SUCCESS) {
                err = "kv map: zeContextMakeMemoryResident on a peer failed";
                return false;
            }
        q0->memset(va + off, 0, bytes).wait();   // fresh pages: the indexer scores every row up to its cap before masking
"""),
("""    im.zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(im.q0->get_device());
    size_t pg = 0;
    if (zeVirtualMemQueryPageSize(im.zc, im.zd, (size_t) 1 << 30, &pg) != ZE_RESULT_SUCCESS || !pg) { err = "kv map: page size query failed"; return false; }
""",
"""    im.zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(im.q0->get_device());
    for (const auto & d : im.q0->get_context().get_devices()) {   // (larry, patch-kvresident.py)
        ze_device_handle_t h = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);
        if (h != im.zd) im.peers.push_back(h);
    }
    size_t pg = 0;
    if (zeVirtualMemQueryPageSize(im.zc, im.zd, (size_t) 1 << 30, &pg) != ZE_RESULT_SUCCESS || !pg) { err = "kv map: page size query failed"; return false; }
"""),
]
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:80]))
    s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
