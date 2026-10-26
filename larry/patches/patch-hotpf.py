#!/usr/bin/env python3
# patch-hotpf.py [TIER_CPP] - larry, STRATA_XE_HOTCOPY_PF=D (default 0 = off): hot_copy (the hot rings' CPU copy of
# pageable arena experts) prefetches D bytes ahead (one line a 128-byte step) - on the arena's 4 KiB pages (THP could
# not be had: memory too fragmented, ~1 of 55 GB) the hardware prefetchers stop at every page.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_HOTCOPY_PF" in s:
    print("already patched"); sys.exit(0)
old = """void hot_copy(uint8_t* dst, const uint8_t* src, size_t n) {
    size_t i = 0;
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
    if (((uintptr_t) dst & 31) == 0)
        for (; i + 128 <= n; i += 128) {
            const __m256i a"""
new = """void hot_copy(uint8_t* dst, const uint8_t* src, size_t n) {
    size_t i = 0;
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
    // larry, STRATA_XE_HOTCOPY_PF: software prefetch D bytes ahead (across the 4 KiB pages)
    static const size_t pf_d = (size_t) std::max(0, env_i("STRATA_XE_HOTCOPY_PF", 0));
    if (pf_d && ((uintptr_t) dst & 31) == 0) {
        for (; i + 128 <= n; i += 128) {
            if (i + pf_d < n) {
                _mm_prefetch((const char*) (src + i + pf_d), _MM_HINT_T0);
                _mm_prefetch((const char*) (src + i + pf_d + 64), _MM_HINT_T0);
            }
            const __m256i a = _mm256_loadu_si256((const __m256i*) (src + i));
            const __m256i b = _mm256_loadu_si256((const __m256i*) (src + i + 32));
            const __m256i c = _mm256_loadu_si256((const __m256i*) (src + i + 64));
            const __m256i d = _mm256_loadu_si256((const __m256i*) (src + i + 96));
            _mm256_store_si256((__m256i*) (dst + i), a);
            _mm256_store_si256((__m256i*) (dst + i + 32), b);
            _mm256_store_si256((__m256i*) (dst + i + 64), c);
            _mm256_store_si256((__m256i*) (dst + i + 96), d);
        }
    }
    if (((uintptr_t) dst & 31) == 0)
        for (; i + 128 <= n; i += 128) {
            const __m256i a"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
