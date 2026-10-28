// dqv2_kernel.inc - larry (#3, 2026-10-28): xe_dqgemm v2's kernel (patch-dqv2.py inserts it into xe_dqgemm.cpp's
// anonymous namespace, after v1: it uses v1's Dec decoders and constants; bench/dqb/xe_dqgemm_v2.cpp has the same code)
using A2_t = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major>;
using B2_t = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::ext_intel_packed>;
using C2_t = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN>;
#define DQ2_16(X) X(0, 0) X(0, 1) X(0, 2) X(0, 3) X(1, 0) X(1, 1) X(1, 2) X(1, 3) X(2, 0) X(2, 1) X(2, 2) X(2, 3) X(3, 0) X(3, 1) X(3, 2) X(3, 3)

template <int TYPE, bool PLANAR>
sycl::event dqgemm2_t(sycl::queue& q, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> Bs(sycl::range<1>((size_t) KC * NW), h);   // [k pair][row][2]: the chunk's weights
        h.parallel_for(sycl::nd_range<1>((size_t) ntiles * NW, NW), syclex::properties{sycl::ext::intel::experimental::grf_size<256>},
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const DqTile tl = tiles[it.get_group(0)];
            const DqJob jb = jobs[tl.job];
            const int lid = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id();
            const int tg = s >> 1, rh = s & 1;   // tokens 32 tg .. +31 (M-tiles 4 tg ..), output rows 64 rh .. +63
            const int live = sycl::min(MT, (jb.m - tl.m0 + TM - 1) / TM);
            const int nl = sycl::clamp(live - 4 * tg, 0, 4);   // this sub-group's live M-tiles
#define DQ2_DECL(i, j) C2_t c##i##j; mx::joint_matrix_fill(sg, c##i##j, 0.0f);
            DQ2_16(DQ2_DECL)
#undef DQ2_DECL
            const int wrow = tl.n0 + lid;   // the weight row this lane decodes
            Dec<TYPE, PLANAR> dec;
            auto bp = Bs.template get_multi_ptr<sycl::access::decorated::no>();
            uint32_t* bw = reinterpret_cast<uint32_t*>(&Bs[0]);
            auto ga = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                jb.A + (size_t) (tl.m0 + 32 * tg) * jb.lda);
            for (int k0 = 0; k0 < K; k0 += KC) {
                dec.load(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0);
                auto put = [&](auto kkc) {   // k-step KK's 16 values of this lane's row -> 8 dwords, k pairs 8 KK ..
                    constexpr int KK = decltype(kkc)::value;
                    sycl::half o[16];
                    dec.template get<KK>(o);
#pragma unroll
                    for (int p = 0; p < 8; ++p)
                        bw[(size_t) (8 * KK + p) * NW + lid] = sycl::bit_cast<uint32_t>(sycl::vec<sycl::half, 2>(o[2 * p], o[2 * p + 1]));
                };
                put(std::integral_constant<int, 0>{});
                put(std::integral_constant<int, 1>{});
                put(std::integral_constant<int, 2>{});
                put(std::integral_constant<int, 3>{});
                it.barrier(sycl::access::fence_space::local_space);
                if (nl > 0) {
#pragma unroll
                    for (int ks = 0; ks < KC / TK; ++ks) {
                        A2_t a0, a1, a2, a3;
                        B2_t b0, b1, b2, b3;
                        mx::joint_matrix_load(sg, b0, bp + (size_t) (8 * ks) * (2 * NW) + 2 * (64 * rh + 0), 2 * NW);
                        mx::joint_matrix_load(sg, b1, bp + (size_t) (8 * ks) * (2 * NW) + 2 * (64 * rh + 16), 2 * NW);
                        mx::joint_matrix_load(sg, b2, bp + (size_t) (8 * ks) * (2 * NW) + 2 * (64 * rh + 32), 2 * NW);
                        mx::joint_matrix_load(sg, b3, bp + (size_t) (8 * ks) * (2 * NW) + 2 * (64 * rh + 48), 2 * NW);
                        mx::joint_matrix_load(sg, a0, ga + (size_t) 0 * TM * jb.lda + k0 + TK * ks, jb.lda);
                        if (nl > 1) mx::joint_matrix_load(sg, a1, ga + (size_t) 1 * TM * jb.lda + k0 + TK * ks, jb.lda);
                        if (nl > 2) mx::joint_matrix_load(sg, a2, ga + (size_t) 2 * TM * jb.lda + k0 + TK * ks, jb.lda);
                        if (nl > 3) mx::joint_matrix_load(sg, a3, ga + (size_t) 3 * TM * jb.lda + k0 + TK * ks, jb.lda);
#define DQ2_MAD(i, j) if (i < nl) mx::joint_matrix_mad(sg, c##i##j, a##i, b##j, c##i##j);
                        DQ2_16(DQ2_MAD)
#undef DQ2_MAD
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
#define DQ2_ST(i, j)                                                                                                       \
            if (i < nl)                                                                                                    \
                mx::joint_matrix_store(sg, c##i##j,                                                                        \
                    sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(      \
                        jb.C + (size_t) (tl.m0 + 32 * tg + i * TM) * jb.ldc + tl.n0 + 64 * rh + 16 * j), jb.ldc, mx::layout::row_major);
            DQ2_16(DQ2_ST)
#undef DQ2_ST
        });
    });
}
#undef DQ2_16
