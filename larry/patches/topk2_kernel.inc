// larry (patch-topk2.py, GGML_SYCL_TOPK2): one row's top-k in two stages for long rows.  Stage 0 takes every block's
// largest radix key (1 << bs_log2 columns a block, at most SYCL_TOP_K2_MAX_BLOCKS blocks, into SLM - the one full read of
// the row); stage 1 the k-th largest of those, T (all four digits: its exact key); stage 2 the radix select and its emit
// of top_k_radix_select_f32 over the columns of the blocks whose maximum reaches T only, in block order.  The same set:
// k blocks hold a column >= T each, so the row's k-th largest key E >= T, and every column >= E lies in a block whose
// maximum is >= T - the candidates hold every column at or above the pivot (ties too), and the selection over them is
// the selection over the row (pivot-equal columns are taken in no fixed order by both).  The passes read ~k blocks.
// Needs top_k_radix_key and the SYCL_TOP_K_RADIX_* constants above.
static constexpr int SYCL_TOP_K2_MAX_BLOCKS = 8192;

static inline int top_k2_bs_log2(const int64_t ncols) {
    int b = 5;   // >= 32 columns a block
    while (((ncols + ((int64_t) 1 << b) - 1) >> b) > SYCL_TOP_K2_MAX_BLOCKS) {
        ++b;
    }
    return b;
}

// rows long enough that their block maxima cut the passes, and k small beside the blocks
static inline bool top_k2_shape_ok(const int64_t ncols, const int k) {
    if (ncols < 32768 || ncols > ((int64_t) 1 << 30)) {
        return false;
    }
    const int b = top_k2_bs_log2(ncols);
    return ((ncols + ((int64_t) 1 << b) - 1) >> b) >= 4 * (int64_t) k;
}

static void top_k_radix_select2_f32(
    const float *   src,
    int32_t *       dst_idx,
    const int       ncols,
    const int       k,
    const int       bs_log2,
    uint32_t *      slm,
    const sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    const auto grp        = item_ct1.get_group();
    const auto sg         = item_ct1.get_sub_group();
    const int  tid        = item_ct1.get_local_id(0);
    const int  block_size = item_ct1.get_local_range(0);
    const int  bs         = 1 << bs_log2;
    const int  nblk       = (ncols + bs - 1) >> bs_log2;

    uint32_t * hist     = slm;
    uint32_t * s_digit  = slm + SYCL_TOP_K_RADIX_HIST_SIZE;
    uint32_t * s_bucket = s_digit + 1;
    uint32_t * s_need   = s_digit + 2;
    uint32_t * s_cnt_gt = s_digit + 3;
    uint32_t * s_cnt_eq = s_digit + 4;
    uint32_t * s_ncand  = s_digit + 5;
    uint32_t * bmax     = s_digit + 8;   // [nblk]
    uint32_t * cand     = bmax + nblk;   // [nblk]: the candidate blocks, in order

    const int copy = tid & (SYCL_TOP_K_RADIX_HIST_COPIES - 1);

    // stage 0: every block's largest key, a sub-group a block
    {
        const int lane = sg.get_local_id()[0];
        const int sgw  = sg.get_local_range()[0];
        const int sgi  = sg.get_group_id()[0];
        const int nsg  = sg.get_group_range()[0];
        for (int b = sgi; b < nblk; b += nsg) {
            const int c0 = b << bs_log2;
            const int c1 = sycl::min(ncols, c0 + bs);
            uint32_t  m  = 0;
            for (int c = c0 + lane; c < c1; c += sgw) {
                m = sycl::max(m, top_k_radix_key(src[c]));
            }
            m = sycl::reduce_over_group(sg, m, sycl::maximum<uint32_t>());
            if (lane == 0) {
                bmax[b] = m;
            }
        }
    }
    if (tid == 0) {
        *s_cnt_gt = 0;
        *s_cnt_eq = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // the radix select of top_k_radix_select_f32 over n keys (key_at(i, key) false: no key there); every digit when
    // `full` (then prefix is the pivot's whole key)
    uint32_t prefix = 0;
    uint32_t mask   = 0;
    uint32_t need   = (uint32_t) k;
    auto select = [&](const int n, auto key_at, const bool full) {
        prefix = 0;
        mask   = 0;
        need   = (uint32_t) k;
        for (int shift = 32 - SYCL_TOP_K_RADIX_BITS; shift >= 0; shift -= SYCL_TOP_K_RADIX_BITS) {
            for (int i = tid; i < SYCL_TOP_K_RADIX_HIST_SIZE; i += block_size) {
                hist[i] = 0;
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (int i = tid; i < n; i += block_size) {
                uint32_t key;
                if (key_at(i, key) && (key & mask) == prefix) {
                    const uint32_t bucket = (key >> shift) & (SYCL_TOP_K_RADIX_BUCKETS - 1);
                    local_atomic(hist[bucket * SYCL_TOP_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
                }
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);

            uint32_t cnt = 0;
            if (tid < SYCL_TOP_K_RADIX_BUCKETS) {
                const uint32_t * h = hist + (SYCL_TOP_K_RADIX_BUCKETS - 1 - tid) * SYCL_TOP_K_RADIX_HIST_COPIES;
                for (int c = 0; c < SYCL_TOP_K_RADIX_HIST_COPIES; c++) {
                    cnt += h[c];
                }
            }
            const uint32_t incl = sycl::inclusive_scan_over_group(grp, cnt, sycl::plus<uint32_t>());
            if (tid < SYCL_TOP_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
                *s_digit  = (uint32_t) (SYCL_TOP_K_RADIX_BUCKETS - 1 - tid);
                *s_bucket = cnt;
                *s_need   = need - (incl - cnt);
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);

            const uint32_t digit      = *s_digit;
            const uint32_t bucket_cnt = *s_bucket;
            need   = *s_need;
            prefix |= digit << shift;
            mask   |= (uint32_t) (SYCL_TOP_K_RADIX_BUCKETS - 1) << shift;
            const bool done = !full && bucket_cnt == need;
            // the next pass rewrites hist and s_*; the reads above must land first
            item_ct1.barrier(sycl::access::fence_space::local_space);
            if (done) {
                break;
            }
        }
    };

    // stage 1: T = the k-th largest block maximum (its exact key)
    select(nblk, [&](const int i, uint32_t & key) { key = bmax[i]; return true; }, true);
    const uint32_t T = prefix;

    // the candidate blocks (maximum >= T) in order: each lane a run of blocks, an exclusive scan for their places
    {
        const int per = (nblk + block_size - 1) / block_size;
        const int b0  = sycl::min(nblk, tid * per);
        const int b1  = sycl::min(nblk, b0 + per);
        uint32_t  cnt = 0;
        for (int b = b0; b < b1; ++b) {
            cnt += bmax[b] >= T ? 1u : 0u;
        }
        uint32_t off = sycl::exclusive_scan_over_group(grp, cnt, sycl::plus<uint32_t>());
        for (int b = b0; b < b1; ++b) {
            if (bmax[b] >= T) {
                cand[off++] = (uint32_t) b;
            }
        }
        const uint32_t tot = sycl::reduce_over_group(grp, cnt, sycl::plus<uint32_t>());
        if (tid == 0) {
            *s_ncand = tot;
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int nc = (int) (*s_ncand) << bs_log2;   // the candidates' columns (the last block may run past the row)

    // stage 2: the select over the candidates' columns
    select(nc, [&](const int i, uint32_t & key) {
        const int c = (int) (cand[i >> bs_log2] << bs_log2) + (i & (bs - 1));
        if (c >= ncols) {
            return false;
        }
        key = top_k_radix_key(src[c]);
        return true;
    }, false);

    // exactly k - need candidates have (key & mask) > prefix; the first need pivot-equal ones fill the tail
    const uint32_t base_eq = (uint32_t) k - need;
    for (int i = tid; i < nc; i += block_size) {
        const int c = (int) (cand[i >> bs_log2] << bs_log2) + (i & (bs - 1));
        if (c >= ncols) {
            continue;
        }
        const uint32_t kp = top_k_radix_key(src[c]) & mask;
        if (kp > prefix) {
            const uint32_t pos = local_atomic(*s_cnt_gt).fetch_add(1u);
            dst_idx[pos] = c;
        } else if (kp == prefix) {
            const uint32_t pos = local_atomic(*s_cnt_eq).fetch_add(1u);
            if (pos < need) {
                dst_idx[base_eq + pos] = c;
            }
        }
    }
}

// a group a row, as top_k_radix_f32_sycl; block_size >= SYCL_TOP_K_RADIX_BUCKETS
static void top_k_radix2_f32_sycl(
    sycl::queue &   q,
    const int       block_size,
    const float *   src,
    int32_t *       dst_indices,
    const int64_t   ncols,
    const int64_t   nrows,
    const int       k
) {
    const int    bs_log2   = top_k2_bs_log2(ncols);
    const int    nblk      = (int) ((ncols + ((int64_t) 1 << bs_log2) - 1) >> bs_log2);
    const size_t slm_words = (size_t) SYCL_TOP_K_RADIX_HIST_SIZE + 8 + 2 * (size_t) nblk;
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(slm_words), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nrows * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) {
                const int64_t row = item_ct1.get_group(0);
                top_k_radix_select2_f32(
                    src + row * ncols, dst_indices + row * k,
                    (int) ncols, k, bs_log2,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    item_ct1);
            });
    });
}
