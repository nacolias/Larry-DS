// Chunk decoders: load(r, k0) fetches row r's raw bytes for the 64 values k0 .. k0 + 63 (k0 % 64 == 0) with wide
// loads, get<KK>(o) decodes k-step KK's 16 values (k0 + 16 KK ..) from registers.  Same arithmetic as dq8.
template <int TYPE, bool PLANAR> struct Dec;

template <> struct Dec<kQ20, true> {   // planes: qs 16 B a block, then d
    sycl::vec<uint32_t, 4> q; float d;
    inline void load(const uint8_t* mat, size_t NB, int n, int, int r, int k0) {
        const size_t blk = (size_t) r * (n / 64) + k0 / 64;
        q = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(mat + blk * 16);
        d = hf(*reinterpret_cast<const uint16_t*>(mat + 16 * NB + blk * 2));
    }
    template <int KK> inline void get(float* o) const {
        const uint32_t w = q[KK];
#pragma unroll
        for (int t = 0; t < 16; ++t) o[t] = (float) ((int) ((w >> (8 * (t % 4) + 2 * (t / 4))) & 3u) - 1) * d;
    }
};
template <> struct Dec<kQ20, false> {   // 18-byte blocks: d, qs[16] (value j: byte j / 4, bits 2 (j % 4))
    uint16_t u[8]; float d;
    inline void load(const uint8_t* mat, size_t, int, int row_bytes, int r, int k0) {
        const uint16_t* b = reinterpret_cast<const uint16_t*>(mat + (size_t) r * row_bytes + (size_t) (k0 / 64) * 18);
        d = hf(b[0]);
#pragma unroll
        for (int i = 0; i < 8; ++i) u[i] = b[1 + i];
    }
    template <int KK> inline void get(float* o) const {
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int j = KK * 16 + t;
            o[t] = (float) ((int) ((u[j / 8] >> (2 * (j % 8))) & 3u) - 1) * d;
        }
    }
};
// PTQ1_0: 128 values a block: 16 + 8 + 2 trit bytes (5, 5 and 4 trits a byte), d; planar = the same fields as planes
struct DecPtq1 {
    sycl::vec<uint32_t, 4> a; sycl::vec<uint32_t, 2> b; uint32_t p; float d; int c;
    static inline int trit(uint32_t bv, int dig) {
        const uint32_t pw = dig == 0 ? 1u : dig == 1 ? 3u : dig == 2 ? 9u : dig == 3 ? 27u : 81u;
        return (int) ((((bv * pw) & 0xFFu) * 3u) >> 8) - 1;
    }
    inline uint32_t ab(int byte) const { return (a[byte / 4] >> (8 * (byte % 4))) & 0xFFu; }
    inline uint32_t bb(int byte) const { return (b[byte / 4] >> (8 * (byte % 4))) & 0xFFu; }
    template <int KK> inline void get(float* o) const {
        if (c == 0) {   // values 16 KK .. : bytes 0..15 of a, digit KK
#pragma unroll
            for (int t = 0; t < 16; ++t) o[t] = (float) trit(ab(t), KK) * d;
        } else if constexpr (KK == 0) {   // 64..79: a, digit 4
#pragma unroll
            for (int t = 0; t < 16; ++t) o[t] = (float) trit(ab(t), 4) * d;
        } else if constexpr (KK == 1 || KK == 2) {   // 80..95 / 96..111: b, digits 0,1 / 2,3
#pragma unroll
            for (int t = 0; t < 16; ++t) o[t] = (float) trit(bb(t % 8), 2 * (KK - 1) + t / 8) * d;
        } else {   // 112..119: b digit 4; 120..127: p bytes 0, 1 (value 120 + i: byte i % 2, digit i / 2)
#pragma unroll
            for (int t = 0; t < 8; ++t) o[t] = (float) trit(bb(t), 4) * d;
#pragma unroll
            for (int t = 0; t < 8; ++t) o[8 + t] = (float) trit((p >> (8 * (t % 2))) & 0xFFu, t / 2) * d;
        }
    }
};
template <> struct Dec<kPtq1, true> : DecPtq1 {
    inline void load(const uint8_t* mat, size_t NB, int n, int, int r, int k0) {
        const size_t blk = (size_t) r * (n / 128) + k0 / 128;
        a = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(mat + blk * 16);
        b = *reinterpret_cast<const sycl::vec<uint32_t, 2>*>(mat + 16 * NB + blk * 8);
        p = *reinterpret_cast<const uint32_t*>(mat + 24 * NB + blk * 4);
        d = hf((uint16_t) (p >> 16));
        c = (k0 / 64) % 2;
    }
};
template <> struct Dec<kPtq1, false> : DecPtq1 {   // 28-byte blocks (4-byte aligned)
    inline void load(const uint8_t* mat, size_t, int, int row_bytes, int r, int k0) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(mat + (size_t) r * row_bytes + (size_t) (k0 / 128) * 28);
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = w[i];
        b[0] = w[4]; b[1] = w[5]; p = w[6];
        d = hf((uint16_t) (p >> 16));
        c = (k0 / 64) % 2;
    }
};
// Q3_K: 256 values a block; a 64-value chunk c is half h = c / 2, 32-groups jj = 2 (c % 2) + {0, 1}
struct DecQ3K {
    sycl::vec<uint32_t, 8> qs, hm; uint32_t sc[3]; float d; int c;
    inline uint32_t scb(int k) const { return (sc[k / 4] >> (8 * (k % 4))) & 0xFFu; }
    template <int KK> inline void get(float* o) const {
        const int is = c * 4 + KK, h = c / 2, jj = 2 * (c % 2) + KK / 2;
        const uint32_t lo4 = (scb(is % 8) >> (4 * (is / 8))) & 0x0fu;
        const uint32_t hi2 = (scb(8 + is % 4) >> (2 * (is / 4))) & 0x03u;
        const float dl = d * (float) ((int) (lo4 | (hi2 << 4)) - 32);
        const int mbit = h * 4 + jj;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int l = (KK % 2) * 16 + t;
            const uint32_t qb = (qs[l / 4] >> (8 * (l % 4))) & 0xFFu, hb = (hm[l / 4] >> (8 * (l % 4))) & 0xFFu;
            const int qv = (int) ((qb >> (2 * jj)) & 3u) - (((hb >> mbit) & 1u) ? 0 : 4);
            o[t] = dl * (float) qv;
        }
    }
};
template <> struct Dec<kQ3K, true> : DecQ3K {   // planes: qs 64, hmask 32, scales 12, d 2 bytes a block
    inline void load(const uint8_t* mat, size_t NB, int n, int, int r, int k0) {
        const size_t blk = (size_t) r * (n / 256) + k0 / 256;
        c = (k0 / 64) % 4;
        const sycl::vec<uint32_t, 4>* q4 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(mat + blk * 64 + (c / 2) * 32);
        const sycl::vec<uint32_t, 4>* h4 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(mat + 64 * NB + blk * 32);
        const sycl::vec<uint32_t, 4> q0 = q4[0], q1 = q4[1], h0 = h4[0], h1 = h4[1];
#pragma unroll
        for (int i = 0; i < 4; ++i) { qs[i] = q0[i]; qs[4 + i] = q1[i]; hm[i] = h0[i]; hm[4 + i] = h1[i]; }
        const uint32_t* s4 = reinterpret_cast<const uint32_t*>(mat + 96 * NB + blk * 12);
        sc[0] = s4[0]; sc[1] = s4[1]; sc[2] = s4[2];
        d = hf(*reinterpret_cast<const uint16_t*>(mat + 108 * NB + blk * 2));
    }
};
template <> struct Dec<kQ3K, false> : DecQ3K {   // 110-byte blocks (2-byte aligned): hmask[32] qs[64] scales[12] d
    inline void load(const uint8_t* mat, size_t, int, int row_bytes, int r, int k0) {
        const uint16_t* b = reinterpret_cast<const uint16_t*>(mat + (size_t) r * row_bytes + (size_t) (k0 / 256) * 110);
        c = (k0 / 64) % 4;
        const uint16_t* qp = b + 16 + (c / 2) * 16;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            hm[i] = (uint32_t) b[2 * i] | ((uint32_t) b[2 * i + 1] << 16);
            qs[i] = (uint32_t) qp[2 * i] | ((uint32_t) qp[2 * i + 1] << 16);
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) sc[i] = (uint32_t) b[48 + 2 * i] | ((uint32_t) b[49 + 2 * i] << 16);
        d = hf(b[54]);
    }
};

// v5: v4 (B in registers, A transposed in local memory) with the chunk decoders
template <int TYPE, bool PLANAR, int MODE>
sycl::event dqgemm_f(sycl::queue& q, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    constexpr int KC = 64;
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
            Dec<TYPE, PLANAR> dec;
            for (int k0 = 0; k0 < K; k0 += KC) {
                dec.load(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0);
                if (astage) {
#pragma unroll
                    for (int v = 0; v < KC; v += 8) {
                        const sycl::vec<sycl::half, 8> x = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(arow + k0 + v);
#pragma unroll
                        for (int j = 0; j < 8; ++j) As[(size_t) (v + j) * MPS + lid] = x[j];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                auto step = [&](auto kkc) {
                    constexpr int KK = decltype(kkc)::value;
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> mb;
                    float o[16];
                    dec.template get<KK>(o);
                    auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
#pragma unroll
                    for (int j = 0; j < 16; ++j) wb[j] = (sycl::half) o[j];
                    const sycl::half* ap = &As[(size_t) (KK * TK + ln) * MPS];
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (MODE != 2 && i < live) {
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ma;
                            const sycl::vec<sycl::half, 8> a8 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(ap + i * TM);
                            auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
#pragma unroll
                            for (int r = 0; r < TM; ++r) wa[r] = a8[r];
                            mx::joint_matrix_mad(sg, acc[i], ma, mb, acc[i]);
                        }
                    }
                };
                step(std::integral_constant<int, 0>{});
                step(std::integral_constant<int, 1>{});
                step(std::integral_constant<int, 2>{});
                step(std::integral_constant<int, 3>{});
                it.barrier(sycl::access::fence_space::local_space);
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

// v6: v5 with the next chunk's weight bytes (PF & 1) and activations (PF & 2) fetched into registers before this chunk's products
template <int TYPE, bool PLANAR, int MODE, int PF>
sycl::event dqgemm_p(sycl::queue& q, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    constexpr int KC = 64;
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
            Dec<TYPE, PLANAR> dec, nxt;
            dec.load(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, 0);
            sycl::vec<sycl::half, 8> xa[KC / 8];
            if (astage) {
#pragma unroll
                for (int v = 0; v < KC / 8; ++v) xa[v] = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(arow + 8 * v);
            }
            for (int k0 = 0; k0 < K; k0 += KC) {
                const bool more = k0 + KC < K;
                if constexpr (PF & 1) { if (more) nxt.load(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0 + KC); }
                else if (k0 > 0) dec.load(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0);
                if (astage) {
                    if constexpr (!(PF & 2)) {
                        if (k0 > 0) {
#pragma unroll
                            for (int v = 0; v < KC / 8; ++v) xa[v] = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(arow + k0 + 8 * v);
                        }
                    }
#pragma unroll
                    for (int v = 0; v < KC / 8; ++v)
#pragma unroll
                        for (int j = 0; j < 8; ++j) As[(size_t) (8 * v + j) * MPS + lid] = xa[v][j];
                    if constexpr ((PF & 2) != 0) {
                        if (more) {
#pragma unroll
                            for (int v = 0; v < KC / 8; ++v) xa[v] = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(arow + k0 + KC + 8 * v);
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                auto step = [&](auto kkc) {
                    constexpr int KK = decltype(kkc)::value;
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> mb;
                    float o[16];
                    dec.template get<KK>(o);
                    auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
#pragma unroll
                    for (int j = 0; j < 16; ++j) wb[j] = (sycl::half) o[j];
                    const sycl::half* ap = &As[(size_t) (KK * TK + ln) * MPS];
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (MODE != 2 && i < live) {
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ma;
                            const sycl::vec<sycl::half, 8> a8 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(ap + i * TM);
                            auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
#pragma unroll
                            for (int r = 0; r < TM; ++r) wa[r] = a8[r];
                            mx::joint_matrix_mad(sg, acc[i], ma, mb, acc[i]);
                        }
                    }
                };
                step(std::integral_constant<int, 0>{});
                step(std::integral_constant<int, 1>{});
                step(std::integral_constant<int, 2>{});
                step(std::integral_constant<int, 3>{});
                it.barrier(sycl::access::fence_space::local_space);
                if constexpr (PF & 1) dec = nxt;
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

// packed-half decoders (two values per 32-bit op): h(1024 + q) = 0x6400 | q
inline sycl::vec<sycl::half, 2> h2(uint32_t bits) { return sycl::bit_cast<sycl::vec<sycl::half, 2>>(bits); }
template <int TYPE, bool PLANAR> struct DecH;
template <> struct DecH<kQ20, true> : Dec<kQ20, true> {   // exact: (q - 1) * d with q - 1 in {-1, 0, 1, 2}
    template <int KK> inline void geth(sycl::half* o) const {
        const uint32_t w = q[KK];
        const sycl::vec<sycl::half, 2> dd((sycl::half) d, (sycl::half) d), off((sycl::half) 1025.0f, (sycl::half) 1025.0f);
#pragma unroll
        for (int g = 0; g < 4; ++g) {   // values 4g .. 4g + 3: bits 2g + 8i
            const uint32_t x = w >> (2 * g);
            const sycl::vec<sycl::half, 2> lo = (h2((x & 0x00030003u) | 0x64006400u) - off) * dd;          // 4g, 4g + 2
            const sycl::vec<sycl::half, 2> hi = (h2(((x >> 8) & 0x00030003u) | 0x64006400u) - off) * dd;   // 4g + 1, 4g + 3
            o[4 * g] = lo[0]; o[4 * g + 2] = lo[1]; o[4 * g + 1] = hi[0]; o[4 * g + 3] = hi[1];
        }
    }
};
template <> struct DecH<kQ3K, true> : Dec<kQ3K, true> {   // dl rounded to half, then (v - 4) * dl in half
    template <int KK> inline void geth(sycl::half* o) const {
        const int is = c * 4 + KK, h = c / 2, jj = 2 * (c % 2) + KK / 2;
        const uint32_t lo4 = (scb(is % 8) >> (4 * (is / 8))) & 0x0fu;
        const uint32_t hi2 = (scb(8 + is % 4) >> (2 * (is / 4))) & 0x03u;
        const sycl::half dlh = (sycl::half) (d * (float) ((int) (lo4 | (hi2 << 4)) - 32));
        const sycl::vec<sycl::half, 2> dl(dlh, dlh), off((sycl::half) 1028.0f, (sycl::half) 1028.0f);
        const int mbit = h * 4 + jj;
#pragma unroll
        for (int g = 0; g < 4; ++g) {   // values l = 16 (KK % 2) + 4g .. + 3: dword l / 4 of qs and hmask
            const int wd = (KK % 2) * 4 + g;
            const uint32_t v = ((qs[wd] >> (2 * jj)) & 0x03030303u) | (((hm[wd] >> mbit) & 0x01010101u) << 2);   // q + 4 hbit
            const sycl::vec<sycl::half, 2> lo = (h2((v & 0x00FF00FFu) | 0x64006400u) - off) * dl;
            const sycl::vec<sycl::half, 2> hi = (h2(((v >> 8) & 0x00FF00FFu) | 0x64006400u) - off) * dl;
            o[4 * g] = lo[0]; o[4 * g + 2] = lo[1]; o[4 * g + 1] = hi[0]; o[4 * g + 3] = hi[1];
        }
    }
};
template <int TYPE, bool PLANAR>
sycl::event dqgemm_h(sycl::queue& q, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    constexpr int KC = 64;
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
            DecH<TYPE, PLANAR> dec;
            for (int k0 = 0; k0 < K; k0 += KC) {
                dec.load(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0);
                if (astage) {
#pragma unroll
                    for (int v = 0; v < KC; v += 8) {
                        const sycl::vec<sycl::half, 8> x = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(arow + k0 + v);
#pragma unroll
                        for (int j = 0; j < 8; ++j) As[(size_t) (v + j) * MPS + lid] = x[j];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                auto step = [&](auto kkc) {
                    constexpr int KK = decltype(kkc)::value;
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> mb;
                    sycl::half o[16];
                    dec.template geth<KK>(o);
                    auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
#pragma unroll
                    for (int j = 0; j < 16; ++j) wb[j] = o[j];
                    const sycl::half* ap = &As[(size_t) (KK * TK + ln) * MPS];
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (i < live) {
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ma;
                            const sycl::vec<sycl::half, 8> a8 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(ap + i * TM);
                            auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
#pragma unroll
                            for (int r = 0; r < TM; ++r) wa[r] = a8[r];
                            mx::joint_matrix_mad(sg, acc[i], ma, mb, acc[i]);
                        }
                    }
                };
                step(std::integral_constant<int, 0>{});
                step(std::integral_constant<int, 1>{});
                step(std::integral_constant<int, 2>{});
                step(std::integral_constant<int, 3>{});
                it.barrier(sycl::access::fence_space::local_space);
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
