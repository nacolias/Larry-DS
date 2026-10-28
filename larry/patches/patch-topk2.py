#!/usr/bin/env python3
# patch-topk2.py [XOLOTL_DIR] - larry, 2026-10-28 (optimization #5, long contexts): ggml-sycl's top-k of long rows in two
# stages (GGML_SYCL_TOPK2=1, default 0).  The prompt indexer ranks every visible compressed row of a query and keeps its
# 512 best: at a 967K prompt's depth one call is 256 queries x 512K rows, ggml-sycl's radix select reads each row up to
# five times (4.5 ms a call at the card's read bandwidth; TOP_K ~32 s of the prompt's ~347 s of card time).  With it
# (patches/topk2_kernel.inc, the kernel and its proof) a group still owns a row, but reads it once for its blocks' largest
# keys, finds the k-th largest of those, and runs the radix select and the emit over the blocks that reach it only - the
# same set.  bench/li/topk2_test.cpp checks it against the one-group kernel and times both.  Only the one-group path
# (rows >= 32K columns with >= 4k blocks; a few rows still split across groups as before).  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "third_party/llama.cpp/ggml/src/ggml-sycl/topk-radix.cpp")
INC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topk2_kernel.inc")
s = open(P).read()
if "top_k_radix_select2_f32" in s:
    print("already patched", P)
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:100]))
    return s.replace(old, new)


s = rep(s, "#include <algorithm>\n", "#include <algorithm>\n#include <cstdlib>\n", "include")
s = rep(s, "void ggml_sycl_top_k_radix(\n", open(INC).read() + "\nvoid ggml_sycl_top_k_radix(\n", "kernel")
s = rep(s, """    const int nparts = top_k_radix_split_groups(ctx.device, ncols, nrows);
    if (nparts > 1) {
        top_k_radix_split_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, nparts, main_stream);
    } else {
        top_k_radix_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, main_stream);
    }
""", """    const int nparts = top_k_radix_split_groups(ctx.device, ncols, nrows);
    // larry (patch-topk2.py): a long row's group in two stages (its blocks' maxima first) - GGML_SYCL_TOPK2=1
    static const bool two = std::getenv("GGML_SYCL_TOPK2") && std::atoi(std::getenv("GGML_SYCL_TOPK2")) != 0;
    if (nparts > 1) {
        top_k_radix_split_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, nparts, main_stream);
    } else if (two && top_k2_shape_ok(ncols, k)) {
        top_k_radix2_f32_sycl(*main_stream, ggml_sycl_info().max_work_group_sizes[ctx.device], src, dst_indices, ncols, nrows, k);
    } else {
        top_k_radix_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, main_stream);
    }
""", "dispatch")
open(P, "w").write(s)
print("patched", P)
