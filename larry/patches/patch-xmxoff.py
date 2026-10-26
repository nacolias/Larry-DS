#!/usr/bin/env python3
# patch-xmxoff.py - GGML_SYCL_LARRY_XMXMV default 0 (measured no end-to-end gain).  Idempotent.
import sys
p = "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/fastmv.cpp"
s = open(p).read()
old = """    static const int on = [] { const char * e = std::getenv("GGML_SYCL_LARRY_XMXMV"); return e ? std::atoi(e) : 1; }();"""
new = """    // default OFF: matches dp4a to 1e-4 and is 8-17 us faster on shexp down / q_a in isolation, but end to end
    // (DSpark verify, 2- and 5.5-token passes) no gain - the extra transposed quantisation eats the q_a / q_b gain and
    // the shared expert runs beside the routed experts (2026-10-21)
    static const int on = [] { const char * e = std::getenv("GGML_SYCL_LARRY_XMXMV"); return e ? std::atoi(e) : 0; }();"""
if new in s: print("already"); sys.exit(0)
if s.count(old) != 1: sys.exit("anchor")
open(p, "w").write(s.replace(old, new))
print("patched")
