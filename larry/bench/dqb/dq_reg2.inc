// v4: B from registers (each lane its weight row) + A staged TRANSPOSED in local memory (As[k][m], row stride MPS
// halfs, padded against bank conflicts) so each lane loads its A fragment (8 rows at k = lane) as one 16-byte read.
constexpr int MPS = MT * TM + 8;
template <int TYPE, bool PLANAR, int MODE, int KC = 64>
sycl::event dqgemm_r2(sycl::queue& q, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> As(sycl::range<1>((size_t) KC * MPS), h);
        h.parallel_for(sycl::nd_range<1>((size_t) ntiles * NW, NW), syclex::properties{sycl::ext::intel::experimental::grf_size<256>},
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const DqTile tl = tiles[it.get_group(0)];
            const DqJob jb = jobs[tl.job];
            const int lid = (int) it.get_local_id(0);
            const int ln = lid & 15;
            const sycl::sub_group sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id();
            const int live = sycl::min(MT, (jb.m - tl.m0 + TM - 1) / TM);
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) mx::joint_matrix_fill(sg, acc[i], 0.0f);
            const int wrow = tl.n0 + lid;
            const sycl::half* arow = jb.A + (size_t) (tl.m0 + lid) * jb.lda;
            const bool astage = lid < live * TM;
            for (int k0 = 0; k0 < K; k0 += KC) {
                if (MODE != 3 && MODE != 4 && astage) {
#pragma unroll
                    for (int v = 0; v < KC; v += 8) {
                        const sycl::vec<sycl::half, 8> x = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(arow + k0 + v);
#pragma unroll
                        for (int j = 0; j < 8; ++j) As[(size_t) (v + j) * MPS + lid] = x[j];
                    }
                }
                if constexpr (MODE != 4) it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int kk = 0; kk < KC; kk += TK) {
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> mb;
                    float o[16];
                    if constexpr (MODE == 5) {
                        mx::joint_matrix_fill(sg, mb, (sycl::half) 0.5f);
                    } else if constexpr (MODE == 1 || MODE == 4 || MODE == 6) {
#pragma unroll
                        for (int j = 0; j < 16; ++j) o[j] = (float) (lid + j + k0);
                    } else {
                        dq8<TYPE, PLANAR>(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0 + kk, o);
                        dq8<TYPE, PLANAR>(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0 + kk + 8, o + 8);
                    }
                    if constexpr (MODE != 5) {
                        auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
#pragma unroll
                        for (int j = 0; j < 16; ++j) wb[j] = (sycl::half) o[j];
                    }
                    const sycl::half* ap = &As[(size_t) (kk + ln) * MPS];
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (MODE != 2 && (MODE == 6 || i < live)) {
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ma;
                            const sycl::vec<sycl::half, 8> a8 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(ap + i * TM);
                            auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
#pragma unroll
                            for (int r = 0; r < TM; ++r) wa[r] = a8[r];
                            mx::joint_matrix_mad(sg, acc[i], ma, mb, acc[i]);
                        }
                    }
                }
                if constexpr (MODE != 4) it.barrier(sycl::access::fence_space::local_space);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
                if (i < live)
                    mx::joint_matrix_store(sg, acc[i],
                        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                            jb.C + (size_t) (tl.m0 + i * TM) * jb.ldc + tl.n0 + s * TN), jb.ldc, mx::layout::row_major);
        });
    });
}
