// topk2_test.cpp - larry, 2026-10-28: patch-topk2.py's two-stage top-k (patches/topk2_kernel.inc) against ggml-sycl's
// one-group radix select (top_k_radix_select_f32, copied from topk-radix.cpp): the same set on every row - the values
// above the k-th largest exactly, as many equal to it - for rows like the prompt indexer's (relu-weighted sums: many
// zeros; -inf past each query's visible rows, fewer visible than k too), plain normals and few-level ties; then the time
// of both at 256 rows x 64K..1M columns, k 512.  Build: icpx -fsycl -O2 topk2_test.cpp -I../../patches.  Card: argv[1].
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static constexpr int SYCL_TOP_K_RADIX_BITS        = 8;
static constexpr int SYCL_TOP_K_RADIX_BUCKETS     = 1 << SYCL_TOP_K_RADIX_BITS;
static constexpr int SYCL_TOP_K_RADIX_HIST_COPIES = 8;
static constexpr int SYCL_TOP_K_RADIX_HIST_SIZE   = SYCL_TOP_K_RADIX_BUCKETS * SYCL_TOP_K_RADIX_HIST_COPIES;
static constexpr int SYCL_TOP_K_RADIX_SLM_WORDS   = SYCL_TOP_K_RADIX_HIST_SIZE + 5;

static inline uint32_t top_k_radix_key(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if (u == 0x80000000u) {
        u = 0u;
    }
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ggml-sycl's (topk-radix.cpp, 2026-10-28)
static void top_k_radix_select_f32(const float * src, int32_t * dst_idx, const int ncols, const int k, uint32_t * slm,
                                   const sycl::nd_item<1> & item_ct1) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;
    const int tid = item_ct1.get_local_id(0);
    const int block_size = item_ct1.get_local_range(0);
    uint32_t * hist = slm;
    uint32_t * s_digit = slm + SYCL_TOP_K_RADIX_HIST_SIZE;
    uint32_t * s_bucket = slm + SYCL_TOP_K_RADIX_HIST_SIZE + 1;
    uint32_t * s_need = slm + SYCL_TOP_K_RADIX_HIST_SIZE + 2;
    uint32_t * s_cnt_gt = slm + SYCL_TOP_K_RADIX_HIST_SIZE + 3;
    uint32_t * s_cnt_eq = slm + SYCL_TOP_K_RADIX_HIST_SIZE + 4;
    if (tid == 0) { *s_cnt_gt = 0; *s_cnt_eq = 0; }
    const int copy = tid & (SYCL_TOP_K_RADIX_HIST_COPIES - 1);
    uint32_t prefix = 0, mask = 0, need = (uint32_t) k;
    for (int shift = 32 - SYCL_TOP_K_RADIX_BITS; shift >= 0; shift -= SYCL_TOP_K_RADIX_BITS) {
        for (int i = tid; i < SYCL_TOP_K_RADIX_HIST_SIZE; i += block_size) hist[i] = 0;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        for (int col = tid; col < ncols; col += block_size) {
            const uint32_t key = top_k_radix_key(src[col]);
            if ((key & mask) == prefix) {
                const uint32_t bucket = (key >> shift) & (SYCL_TOP_K_RADIX_BUCKETS - 1);
                local_atomic(hist[bucket * SYCL_TOP_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        uint32_t cnt = 0;
        if (tid < SYCL_TOP_K_RADIX_BUCKETS) {
            const uint32_t * h = hist + (SYCL_TOP_K_RADIX_BUCKETS - 1 - tid) * SYCL_TOP_K_RADIX_HIST_COPIES;
            for (int c = 0; c < SYCL_TOP_K_RADIX_HIST_COPIES; c++) cnt += h[c];
        }
        const uint32_t incl = sycl::inclusive_scan_over_group(item_ct1.get_group(), cnt, sycl::plus<uint32_t>());
        if (tid < SYCL_TOP_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
            *s_digit = (uint32_t) (SYCL_TOP_K_RADIX_BUCKETS - 1 - tid);
            *s_bucket = cnt;
            *s_need = need - (incl - cnt);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        const uint32_t digit = *s_digit, bucket_cnt = *s_bucket;
        need = *s_need;
        prefix |= digit << shift;
        mask |= (uint32_t) (SYCL_TOP_K_RADIX_BUCKETS - 1) << shift;
        if (bucket_cnt == need) break;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const uint32_t base_eq = (uint32_t) k - need;
    for (int col = tid; col < ncols; col += block_size) {
        const uint32_t kp = top_k_radix_key(src[col]) & mask;
        if (kp > prefix) {
            const uint32_t pos = local_atomic(*s_cnt_gt).fetch_add(1u);
            dst_idx[pos] = col;
        } else if (kp == prefix) {
            const uint32_t pos = local_atomic(*s_cnt_eq).fetch_add(1u);
            if (pos < need) dst_idx[base_eq + pos] = col;
        }
    }
}

#include "topk2_kernel.inc"

static void run_old(sycl::queue & q, int bsz, const float * src, int32_t * dst, int64_t ncols, int64_t nrows, int k) {
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_TOP_K_RADIX_SLM_WORDS), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nrows * bsz), sycl::range<1>(bsz)), [=](sycl::nd_item<1> it) {
            const int64_t row = it.get_group(0);
            top_k_radix_select_f32(src + row * ncols, dst + row * k, (int) ncols, k,
                                   slm.get_multi_ptr<sycl::access::decorated::no>().get(), it);
        });
    });
}

// one row's check: the selections' values, sorted, equal (the set above the pivot exactly; as many pivot-equal)
static bool same(const float * row, const int32_t * a, const int32_t * b, int k, int64_t ncols, std::vector<float> & va,
                 std::vector<float> & vb) {
    va.resize(k); vb.resize(k);
    for (int i = 0; i < k; ++i) {
        if (a[i] < 0 || a[i] >= ncols || b[i] < 0 || b[i] >= ncols) return false;
        va[i] = row[a[i]]; vb[i] = row[b[i]];
    }
    std::vector<int32_t> sa(a, a + k), sb(b, b + k);
    std::sort(sa.begin(), sa.end()); std::sort(sb.begin(), sb.end());
    if (std::adjacent_find(sa.begin(), sa.end()) != sa.end() || std::adjacent_find(sb.begin(), sb.end()) != sb.end()) return false;
    auto key = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); if (u == 0x80000000u) u = 0; return (u & 0x80000000u) ? ~u : (u | 0x80000000u); };
    std::vector<uint32_t> ka(k), kb(k);
    for (int i = 0; i < k; ++i) { ka[i] = key(va[i]); kb[i] = key(vb[i]); }
    std::sort(ka.begin(), ka.end()); std::sort(kb.begin(), kb.end());
    if (ka != kb) return false;
    const uint32_t piv = ka[0];   // above the pivot the indices themselves agree
    std::vector<int32_t> ga, gb;
    for (int i = 0; i < k; ++i) { if (key(row[a[i]]) > piv) ga.push_back(a[i]); if (key(row[b[i]]) > piv) gb.push_back(b[i]); }
    std::sort(ga.begin(), ga.end()); std::sort(gb.begin(), gb.end());
    return ga == gb;
}

int main(int argc, char ** argv) {
    const int card = argc > 1 ? std::atoi(argv[1]) : 0;
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (card >= (int) gpus.size()) { std::printf("no card %d\n", card); return 1; }
    sycl::queue q(gpus[(size_t) card], sycl::property::queue::in_order());
    const int bsz = (int) q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int k = 512;
    std::mt19937_64 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0, checked = 0;
    // ---- correctness: kinds x widths, 64 rows each
    const int64_t widths[] = { 32768, 65536, 131072, 300000, 524288, 1048576 };
    for (int kind = 0; kind < 4; ++kind)
        for (int64_t ncols : widths) {
            if (!top_k2_shape_ok(ncols, k)) { std::printf("kind %d ncols %lld: two stages not taken\n", kind, (long long) ncols); continue; }
            const int64_t nrows = 64;
            std::vector<float> h((size_t) (nrows * ncols));
            for (int64_t r = 0; r < nrows; ++r) {
                float * row = h.data() + r * ncols;
                // visible rows: the whole row, or a prefix (some shorter than k: -inf ties fill the top-k)
                int64_t nvis = ncols;
                if (kind == 2) nvis = 1 + (int64_t) (rng() % (uint64_t) (r % 3 == 0 ? 2 * k : ncols));
                for (int64_t c = 0; c < ncols; ++c) {
                    float v;
                    if (kind == 0) v = nd(rng);
                    else if (kind == 1 || kind == 2) {   // relu-weighted head sum: ~half the rows exactly 0
                        float s = 0;
                        for (int hh = 0; hh < 4; ++hh) s += std::max(0.0f, nd(rng) - 0.7f) * (0.5f + 0.1f * hh);
                        v = s;
                    } else v = (float) (rng() % 7) * 0.25f;   // few levels: ties everywhere
                    row[c] = c < nvis ? v : -INFINITY;
                }
            }
            float * d = sycl::malloc_device<float>(h.size(), q);
            int32_t * oa = sycl::malloc_device<int32_t>((size_t) nrows * k, q);
            int32_t * ob = sycl::malloc_device<int32_t>((size_t) nrows * k, q);
            q.memcpy(d, h.data(), h.size() * 4).wait();
            run_old(q, bsz, d, oa, ncols, nrows, k);
            top_k_radix2_f32_sycl(q, bsz, d, ob, ncols, nrows, k);
            q.wait();
            std::vector<int32_t> ha((size_t) nrows * k), hb((size_t) nrows * k);
            q.memcpy(ha.data(), oa, ha.size() * 4).wait();
            q.memcpy(hb.data(), ob, hb.size() * 4).wait();
            int bad = 0;
            std::vector<float> va, vb;
            for (int64_t r = 0; r < nrows; ++r)
                if (!same(h.data() + r * ncols, ha.data() + r * k, hb.data() + r * k, k, ncols, va, vb)) ++bad;
            checked += (int) nrows;
            fails += bad;
            std::printf("kind %d ncols %7lld: %s (%d of %lld rows differ)\n", kind, (long long) ncols, bad ? "DIFFER" : "same", bad,
                        (long long) nrows);
            sycl::free(d, q); sycl::free(oa, q); sycl::free(ob, q);
        }
    // ---- time: 256 rows (a prompt indexer slice), relu-like rows
    for (int64_t ncols : { (int64_t) 65536, (int64_t) 131072, (int64_t) 262144, (int64_t) 524288, (int64_t) 1048576 }) {
        const int64_t nrows = 256;
        std::vector<float> h((size_t) (nrows * ncols));
        for (auto & v : h) { float s = 0; for (int hh = 0; hh < 4; ++hh) s += std::max(0.0f, nd(rng) - 0.7f); v = s; }
        float * d = sycl::malloc_device<float>(h.size(), q);
        int32_t * o = sycl::malloc_device<int32_t>((size_t) nrows * k, q);
        q.memcpy(d, h.data(), h.size() * 4).wait();
        auto tm = [&](bool two) {
            std::vector<double> t;
            for (int rep = 0; rep < 7; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                if (two) top_k_radix2_f32_sycl(q, bsz, d, o, ncols, nrows, k);
                else run_old(q, bsz, d, o, ncols, nrows, k);
                q.wait();
                t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(t.begin(), t.end());
            return t[t.size() / 2];
        };
        tm(false); tm(true);   // warm-up (JIT)
        const double a = tm(false), b = tm(true);
        std::printf("256 rows x %7lld cols, k %d: one stage %.3f ms, two stages %.3f ms (%.2fx)\n", (long long) ncols, k, a, b, a / b);
        sycl::free(d, q); sycl::free(o, q);
    }
    std::printf("%s (%d rows checked)\n", fails ? "FAILURES" : "all same", checked);
    return fails ? 2 : 0;
}
