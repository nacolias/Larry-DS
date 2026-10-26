#!/usr/bin/env python3
# patch-xmxpar.py - test_ds41_xe parity: each checked expert also through the PLANAR paths at the verify form (NT
# tokens, token t -> output row t, weight 1) - the dp4a kernels (xe_experts_q8) and the XMX kernels (xe_experts_x) -
# against the CPU reference.  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/test_ds41_xe.cpp"
s = open(P).read()
if "planar verify" in s:
    print("already"); sys.exit(0)
def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"anchor x{s.count(old)}: {old[:80]!r}")
    s = s.replace(old, new)
rep("""            const double rc = rel(cpuo.data(), ref.data(), E);""", """            const double rc = rel(cpuo.data(), ref.data(), E);
            double rq = -1, rx = -1;
            {   // planar verify form: the dp4a kernels and the XMX kernels
                static xe::XeQ8 pXq{}, pHq{};
                static sycl::half *pxt = nullptr, *pht = nullptr;
                if (!pxt) {
                    pXq = {sycl::malloc_device<int8_t>((size_t) 8 * E, q), sycl::malloc_device<float>((size_t) 8 * E / 32, q),
                           sycl::malloc_device<int32_t>((size_t) 8 * E / 16, q)};
                    pHq = {sycl::malloc_device<int8_t>((size_t) 8 * F, q), sycl::malloc_device<float>((size_t) 8 * F / 32, q),
                           sycl::malloc_device<int32_t>((size_t) 8 * F / 16, q)};
                    pxt = sycl::malloc_device<sycl::half>((size_t) 8 * E, q);
                    pht = sycl::malloc_device<sycl::half>((size_t) F * 8, q);
                }
                uint8_t* pb = sycl::malloc_device<uint8_t>(f.bytes, q);
                xe::xe_repack_blob(q, db, pb, f.gu_type, f.d_type, f.up_off, f.down_off, (int) E, (int) F).wait();
                xe::XeJob pj = j;
                pj.blob = pb;
                q.memcpy(dj, &pj, sizeof pj).wait();
                xe::xe_quant_rows(q, dX, NT, (int) E, pXq);
                xe::xe_experts_q8(q, dj, 1, NT, pXq, pHq, dO, (int) E, (int) F, L, false, NT).wait();
                q.memcpy(gpu.data(), dO, (size_t) NT * E * 4).wait();
                rq = rel(gpu.data(), ref.data(), (int64_t) NT * E);
                xe::xe_xt(q, dX, NT, (int) E, pxt);
                xe::xe_experts_x(q, dj, 1, NT, pxt, pht, dO, (int) E, (int) F, L, false).wait();
                q.memcpy(gpu.data(), dO, (size_t) NT * E * 4).wait();
                rx = rel(gpu.data(), ref.data(), (int64_t) NT * E);
                if (rx > 0.05) {   // which tokens are off
                    std::printf("    xmx per token:");
                    for (int t = 0; t < NT; ++t) std::printf(" %.2e", rel(gpu.data() + (size_t) t * E, ref.data() + (size_t) t * E, E));
                    std::printf("\\n");
                }
                sycl::free(pb, q);
            }""")
rep("""            std::printf("L%-2d e%-3d class %d (gu %3d / d %3d): xe decode %.2e verify %.2e prompt %.2e | cpu native %.2e  %s\\n", l, e, c,
                        f.gu_type, f.d_type, r1, r3, ry, rc, ok ? "ok" : "FAIL");""",
    """            std::printf("L%-2d e%-3d class %d (gu %3d / d %3d): xe decode %.2e verify %.2e prompt %.2e | planar verify dp4a %.2e xmx %.2e | cpu native %.2e  %s\\n",
                        l, e, c, f.gu_type, f.d_type, r1, r3, ry, rq, rx, rc, ok ? "ok" : "FAIL");""")
open(P, "w").write(s)
print("patched")
