#!/usr/bin/env python3
# patch-argmax2.py - GGML_SYCL_LARRY_ARGMAX_CHECK=1 (diagnostic): the stock kernel into a scratch buffer, larry_argmax
# into dst, indices compared on the host (first 40 calls printed when they differ, a count always).  Idempotent.
import sys
S = "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(S).read()
if "GGML_SYCL_LARRY_ARGMAX_CHECK" in s: print("already"); sys.exit(0)
old = """    if (larry_argmax(ctx, src0_dd, dst_dd, ncols, nrows, main_stream)) return;
    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);"""
new = """    static const int chk = ggml_sycl_get_env("GGML_SYCL_LARRY_ARGMAX_CHECK", 0);
    if (chk) {
        ggml_sycl_pool_alloc<int32_t> ref(ctx.pool(), (size_t) nrows);
        argmax_f32_i32_sycl(src0_dd, ref.get(), ncols, nrows, main_stream);
        if (larry_argmax(ctx, src0_dd, dst_dd, ncols, nrows, main_stream)) {
            std::vector<int32_t> a((size_t) nrows), b((size_t) nrows);
            main_stream->memcpy(a.data(), ref.get(), nrows * 4).wait();
            main_stream->memcpy(b.data(), dst_dd, nrows * 4).wait();
            static long n = 0, bad = 0;
            ++n;
            for (int64_t r = 0; r < nrows; ++r)
                if (a[(size_t) r] != b[(size_t) r] && ++bad <= 40) GGML_LOG_INFO("larry argmax check: row %lld stock %d larry %d\\n", (long long) r, a[(size_t) r], b[(size_t) r]);
            if (n % 500 == 0) GGML_LOG_INFO("larry argmax check: %ld calls, %ld mismatches\\n", n, bad);
            return;
        }
        main_stream->memcpy(dst_dd, ref.get(), nrows * 4);
        return;
    }
    if (larry_argmax(ctx, src0_dd, dst_dd, ncols, nrows, main_stream)) return;
    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);"""
if s.count(old) != 1: sys.exit("anchor")
s = s.replace(old, new)
open(S, "w").write(s)
print("patched")
