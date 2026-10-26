#!/usr/bin/env python3
# patch-glue3.py [GGML_SYCL_DIR] - larry, GGML_SYCL_LARRY_CPYKERNEL=1 (default 0 until measured): a single plain copy
# in a graph (CPY / DUP / CONT between contiguous tensors of one type) as a kernel too, like the CPYBATCH runs.  A queue
# memcpy goes to the card's one user copy engine (Level Zero copy offload on the in-order queue): FIFO behind the
# tier's card-to-card pulls and the uploads on card 0 (its copy engine 90-100 % busy in a prompt, bench/speed/engmon.sh)
# and D2D runs ~92 GB/s there (unitrace) - while the next kernel of the dense span waits for it.  Idempotent.
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl"
P = D + "/ggml-sycl.cpp"
s = open(P).read()
if "GGML_SYCL_LARRY_CPYKERNEL" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""            if (ns >= 2) {
                larry_cpy_batch(*sycl_ctx->stream(), seg, ns);
                i = last;
                continue;
            }
        }""", """            if (ns >= 2) {
                larry_cpy_batch(*sycl_ctx->stream(), seg, ns);
                i = last;
                continue;
            }
            // larry, GGML_SYCL_LARRY_CPYKERNEL (patch-glue3.py): the single copy on the compute engine, 16 bytes a work-item
            static const int cpy_kernel = ggml_sycl_get_env("GGML_SYCL_LARRY_CPYKERNEL", 0);
            if (cpy_kernel) {
                const size_t nb = (size_t) seg[0].n * 4;
                if (nb % 16 == 0 && (uintptr_t) seg[0].s % 16 == 0 && (uintptr_t) seg[0].d % 16 == 0) {
                    using v4 = sycl::vec<uint32_t, 4>;
                    const v4 * vs = (const v4 *) seg[0].s;
                    v4 * vd = (v4 *) seg[0].d;
                    sycl_ctx->stream()->parallel_for(sycl::range<1>(nb / 16), [=](sycl::id<1> k) { vd[k] = vs[k]; });
                } else {
                    larry_cpy_batch(*sycl_ctx->stream(), seg, 1);
                }
                continue;
            }
        }""")
open(P, "w").write(s)
print("patched", P)
