#define LIF_HSUM(s, x0, x1, x2, x3)                                                                        \
    {                                                                                                     \
        auto v0 = sycl::ext::oneapi::detail::get_wi_data(sg, x0);                                         \
        auto v1 = sycl::ext::oneapi::detail::get_wi_data(sg, x1);                                         \
        auto v2 = sycl::ext::oneapi::detail::get_wi_data(sg, x2);                                         \
        auto v3 = sycl::ext::oneapi::detail::get_wi_data(sg, x3);                                         \
        _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                   \
            s += sycl::select_from_group(sg, wl0, i) * sycl::fmax((float) v0[i], 0.0f);                   \
            s += sycl::select_from_group(sg, wl0, 8 + i) * sycl::fmax((float) v1[i], 0.0f);               \
            s += sycl::select_from_group(sg, wl1, i) * sycl::fmax((float) v2[i], 0.0f);                   \
            s += sycl::select_from_group(sg, wl1, 8 + i) * sycl::fmax((float) v3[i], 0.0f);               \
        }                                                                                                 \
    }

// qh: fp16 [B][32][128]; kh: the keys in fp16 [NKVp][128] (zero rows past NKV, NKVp a multiple of 32)
// larry (patch-lifbf16.py): T = sycl::half (those copies, ld_k 128) or li_bf16 (GGML_SYCL_LI_FUSED=2: the queries in bf16,
// the BF16 keys read where they are, rows ld_k apart, NKV a multiple of 32)
template <typename T>
static void lif_run_t(const T * qh, const T * kh, int64_t ld_k, int64_t NKV, const float * w, int64_t ldw,
                      const sycl::half * m, int64_t ldm, const float * nvis, float * od, int64_t ldo, int64_t B, int QW, int64_t KR,
                      queue_ptr stream) {
    using lif_A = li_mx::joint_matrix<sycl::sub_group, T, li_mx::use::a, 8, 16, li_mx::layout::row_major>;
    using lif_B = li_mx::joint_matrix<sycl::sub_group, T, li_mx::use::b, 16, 16, li_mx::layout::col_major>;
    const int64_t nkr = (NKV + KR - 1) / KR, nqb = (B + QW - 1) / QW;
    stream->parallel_for(sycl::nd_range<1>((size_t) (nqb * nkr * QW * 16), (size_t) (QW * 16)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t g = (int64_t) it.get_group(0);
        const int64_t kr = g % nkr, b = (g / nkr) * QW + (int64_t) sg.get_group_linear_id();
        if (b >= B) return;
        const float wl0 = w[b * ldw + lane], wl1 = w[b * ldw + 16 + lane];
        const T * qr = qh + b * 32 * 128;
        const sycl::half * mr = m + b * ldm;
        const float nv = nvis ? nvis[b] : 0.0f;   // nvis: rows j < nv visible (patch-linvis.py)
        float * orow = od + b * ldo;
        const int64_t j1 = sycl::min(NKV, (kr + 1) * KR);
        for (int64_t j0 = kr * KR; j0 < j1; j0 += 32) {
            const int64_t ja = j0 + lane, jb = j0 + 16 + lane;
            const float ma = ja >= NKV ? -INFINITY : nvis ? ((float) ja < nv ? 0.0f : -INFINITY) : (float) mr[ja];
            const float mb = jb >= NKV ? -INFINITY : nvis ? ((float) jb < nv ? 0.0f : -INFINITY) : (float) mr[jb];
            if (sycl::all_of_group(sg, ma == -INFINITY && mb == -INFINITY)) {
                if (ja < NKV) orow[ja] = -INFINITY;
                if (jb < NKV) orow[jb] = -INFINITY;
                continue;
            }
            lif_C c00, c01, c10, c11, c20, c21, c30, c31;
            li_mx::joint_matrix_fill(sg, c00, 0.0f); li_mx::joint_matrix_fill(sg, c01, 0.0f);
            li_mx::joint_matrix_fill(sg, c10, 0.0f); li_mx::joint_matrix_fill(sg, c11, 0.0f);
            li_mx::joint_matrix_fill(sg, c20, 0.0f); li_mx::joint_matrix_fill(sg, c21, 0.0f);
            li_mx::joint_matrix_fill(sg, c30, 0.0f); li_mx::joint_matrix_fill(sg, c31, 0.0f);
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                lif_A a0, a1, a2, a3;
                lif_B b0, b1;
                li_mx::joint_matrix_load(sg, a0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qr + 0 * 1024 + ks * 16), 128);
                li_mx::joint_matrix_load(sg, a1, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qr + 1 * 1024 + ks * 16), 128);
                li_mx::joint_matrix_load(sg, a2, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qr + 2 * 1024 + ks * 16), 128);
                li_mx::joint_matrix_load(sg, a3, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qr + 3 * 1024 + ks * 16), 128);
                li_mx::joint_matrix_load(sg, b0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kh + j0 * ld_k + ks * 16), ld_k);
                li_mx::joint_matrix_load(sg, b1, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kh + (j0 + 16) * ld_k + ks * 16), ld_k);
                li_mx::joint_matrix_mad(sg, c00, a0, b0, c00); li_mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                li_mx::joint_matrix_mad(sg, c10, a1, b0, c10); li_mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                li_mx::joint_matrix_mad(sg, c20, a2, b0, c20); li_mx::joint_matrix_mad(sg, c21, a2, b1, c21);
                li_mx::joint_matrix_mad(sg, c30, a3, b0, c30); li_mx::joint_matrix_mad(sg, c31, a3, b1, c31);
            }
            float s0 = 0.0f, s1 = 0.0f;
            LIF_HSUM(s0, c00, c10, c20, c30)
            LIF_HSUM(s1, c01, c11, c21, c31)
            if (ja < NKV) orow[ja] = s0 + ma;
            if (jb < NKV) orow[jb] = s1 + mb;
        }
    });
}

