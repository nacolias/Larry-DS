// v3: each lane decodes its own weight row straight into the B fragment (B[k][n]: lane n holds column n, element i =
// row i); local memory only for the activations (shared by the 8 sub-groups).
template <int TYPE, bool PLANAR, int MODE, int KC = 64>
sycl::event dqgemm_r(sycl::queue& q, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> As(sycl::range<1>((size_t) MT * TM * KC), h);
        h.parallel_for(sycl::nd_range<1>((size_t) ntiles * NW, NW), syclex::properties{sycl::ext::intel::experimental::grf_size<256>},
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const DqTile tl = tiles[it.get_group(0)];
            const DqJob jb = jobs[tl.job];
            const int lid = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id();
            const int live = sycl::min(MT, (jb.m - tl.m0 + TM - 1) / TM);
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) mx::joint_matrix_fill(sg, acc[i], 0.0f);
            const int wrow = tl.n0 + lid;
            auto asp = As.template get_multi_ptr<sycl::access::decorated::no>();
            const sycl::half* arow = jb.A + (size_t) (tl.m0 + lid) * jb.lda;
            const bool astage = lid < live * TM;
            for (int k0 = 0; k0 < K; k0 += KC) {
                if (MODE != 3 && astage) {
#pragma unroll
                    for (int v = 0; v < KC; v += 8)
                        *reinterpret_cast<sycl::vec<sycl::half, 8>*>(&As[(size_t) lid * KC + v]) =
                            *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(arow + k0 + v);
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int kk = 0; kk < KC; kk += TK) {
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> mb;
                    float o[16];
                    if constexpr (MODE == 1) {
#pragma unroll
                        for (int j = 0; j < 16; ++j) o[j] = (float) (lid + j + k0);
                    } else {
                        dq8<TYPE, PLANAR>(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0 + kk, o);
                        dq8<TYPE, PLANAR>(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0 + kk + 8, o + 8);
                    }
                    auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
#pragma unroll
                    for (int j = 0; j < 16; ++j) wb[j] = (sycl::half) o[j];
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (MODE != 2 && i < live) {
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ma;
                            mx::joint_matrix_load(sg, ma, asp + (size_t) (i * TM) * KC + kk, KC);
                            mx::joint_matrix_mad(sg, acc[i], ma, mb, acc[i]);
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            (void) s;
#pragma unroll
            for (int i = 0; i < MT; ++i)
                if (i < live)
                    mx::joint_matrix_store(sg, acc[i],
                        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                            jb.C + (size_t) (tl.m0 + i * TM) * jb.ldc + tl.n0 + s * TN), jb.ldc, mx::layout::row_major);
        });
    });
}
