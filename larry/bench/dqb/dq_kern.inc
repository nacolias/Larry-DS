namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

inline float hf(uint16_t u) { return (float) sycl::bit_cast<sycl::half>(u); }
inline uint16_t ld16(const uint8_t* p) { return (uint16_t) ((uint32_t) p[0] | ((uint32_t) p[1] << 8)); }

template <int TYPE>
constexpr int qkc() { return TYPE == kQ3K ? 256 : TYPE == kQ20 ? 64 : 128; }

/// 8 consecutive values v0 .. v0 + 7 (v0 % 8 == 0) of row r of an [nrows x n] matrix: PLANAR (NB = nrows * n / QK
/// blocks, xe_planar.cpp's layout; the decode of xe_dequant_planar_f16) or the file layout (row_bytes a row; the
/// decode of xe_dequant_rows_f16)
template <int TYPE, bool PLANAR>
inline void dq8(const uint8_t* mat, size_t NB, int n, int row_bytes, int r, int v0, float* o) {
    if constexpr (PLANAR) {
        if constexpr (TYPE == kQ20) {
            const size_t blk = (size_t) r * (n / 64) + v0 / 64;
            const float d = hf(*reinterpret_cast<const uint16_t*>(mat + 16 * NB + blk * 2));
            const int u = (v0 % 64) / 16, half = (v0 % 16) / 8;
            const uint32_t w = *reinterpret_cast<const uint32_t*>(mat + blk * 16 + 4 * u);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k = (8 * half + j) / 4, i = j % 4;
                o[j] = (float) ((int) ((w >> (8 * i + 2 * k)) & 3u) - 1) * d;
            }
        } else if constexpr (TYPE == kPtq1) {
            const size_t blk = (size_t) r * (n / 128) + v0 / 128;
            const uint8_t* p16 = mat + blk * 16;
            const uint8_t* p8 = mat + 16 * NB + blk * 8;
            const uint8_t* p4 = mat + 24 * NB + blk * 4;
            const float d = hf((uint16_t) (p4[2] | (p4[3] << 8)));
            const int j0 = v0 % 128;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = j0 + i;
                int byte, dig;
                if (j < 80) { byte = j % 16; dig = j / 16; }
                else if (j < 120) { byte = 16 + (j - 80) % 8; dig = (j - 80) / 8; }
                else { byte = 24 + (j - 120) % 2; dig = (j - 120) / 2; }
                const uint32_t bv = byte < 16 ? p16[byte] : byte < 24 ? p8[byte - 16] : p4[byte - 24];
                const uint32_t p = dig == 0 ? 1u : dig == 1 ? 3u : dig == 2 ? 9u : dig == 3 ? 27u : 81u;
                o[i] = (float) ((int) ((((bv * p) & 0xFFu) * 3u) >> 8) - 1) * d;
            }
        } else {   // Q3_K: 8 values of one 32-group (h, jj), one half (one scale)
            const size_t blk = (size_t) r * (n / 256) + v0 / 256;
            const int j256 = v0 % 256;
            const int h = j256 / 128, jj = (j256 % 128) / 32, l0 = j256 % 32;
            const uint8_t* qs = mat + blk * 64;
            const uint8_t* hm = mat + 64 * NB + blk * 32;
            const uint8_t* sc = mat + 96 * NB + blk * 12;
            const int is = h * 8 + 2 * jj + (l0 >= 16 ? 1 : 0);
            const uint32_t lo4 = (sc[is % 8] >> (4 * (is / 8))) & 0x0fu;
            const uint32_t hi2 = (sc[8 + is % 4] >> (2 * (is / 4))) & 0x03u;
            const float dl = hf(*reinterpret_cast<const uint16_t*>(mat + 108 * NB + blk * 2)) * (float) ((int) (lo4 | (hi2 << 4)) - 32);
            const int mbit = h * 4 + jj;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int l = l0 + i;
                const int qv = (int) ((qs[h * 32 + l] >> (2 * jj)) & 3u) - (((hm[l] >> mbit) & 1u) ? 0 : 4);
                o[i] = dl * (float) qv;
            }
        }
    } else {
        const uint8_t* row = mat + (size_t) r * row_bytes;
        if constexpr (TYPE == kQ20) {
            const uint8_t* blk = row + (size_t) (v0 / 64) * 18;
            const float d = hf(ld16(blk));
            const int j = v0 % 64;
            const uint32_t w = (uint32_t) blk[2 + j / 4] | ((uint32_t) blk[3 + j / 4] << 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = (float) ((int) ((w >> (2 * i)) & 3u) - 1) * d;
        } else if constexpr (TYPE == kPtq1) {
            const uint8_t* blk = row + (size_t) (v0 / 128) * 28;
            const float d = hf(ld16(blk + 26));
            const int j = v0 % 128;
            if (j < 80) {
                const int m0 = j % 16, dg = j / 16;
                const uint32_t p = dg == 0 ? 1u : dg == 1 ? 3u : dg == 2 ? 9u : dg == 3 ? 27u : 81u;
#pragma unroll
                for (int i = 0; i < 8; ++i) o[i] = (float) ((int) ((((blk[m0 + i] * p) & 0xFFu) * 3u) >> 8) - 1) * d;
            } else if (j < 120) {
                const int dg = (j - 80) / 8;
                const uint32_t p = dg == 0 ? 1u : dg == 1 ? 3u : dg == 2 ? 9u : dg == 3 ? 27u : 81u;
#pragma unroll
                for (int i = 0; i < 8; ++i) o[i] = (float) ((int) ((((blk[16 + i] * p) & 0xFFu) * 3u) >> 8) - 1) * d;
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int h = i % 2, dg = i / 2;
                    const uint32_t p = dg == 0 ? 1u : dg == 1 ? 3u : dg == 2 ? 9u : 27u;
                    o[i] = (float) ((int) ((((blk[24 + h] * p) & 0xFFu) * 3u) >> 8) - 1) * d;
                }
            }
        } else {
            const uint8_t* blk = row + (size_t) (v0 / 256) * 110;
            const int j256 = v0 % 256;
            const int h = j256 / 128, jj = (j256 % 128) / 32, l0 = j256 % 32;
            const uint8_t* sc = blk + 96;
            const int is = h * 8 + 2 * jj + (l0 >= 16 ? 1 : 0);
            const uint32_t lo4 = (sc[is % 8] >> (4 * (is / 8))) & 0x0fu;
            const uint32_t hi2 = (sc[8 + is % 4] >> (2 * (is / 4))) & 0x03u;
            const float dl = hf(ld16(blk + 108)) * (float) ((int) (lo4 | (hi2 << 4)) - 32);
            const int mbit = h * 4 + jj;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int l = l0 + i;
                const int qv = (int) ((blk[32 + h * 32 + l] >> (2 * jj)) & 3u) - (((blk[l] >> mbit) & 1u) ? 0 : 4);
                o[i] = dl * (float) qv;
            }
        }
    }
}

constexpr int TM = 8, TN = 16, TK = 16, NW = 128, MT = 16;   // tile: 128 output rows x 128 token rows

template <int TYPE, bool PLANAR, int MODE, int KC = 64, int GRF = 256>
sycl::event dqgemm_t(sycl::queue& q, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    // local memory: B (the decoded weights, [KC][NW]) and A (the tile's token rows for the chunk, [MT * TM][KC]) - the
    // 8 sub-groups read their activation tiles from here, not 8 times over from global memory
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> Bs(sycl::range<1>((size_t) KC * NW), h);
        sycl::local_accessor<sycl::half, 1> As(sycl::range<1>((size_t) MT * TM * KC), h);
        h.parallel_for(sycl::nd_range<1>((size_t) ntiles * NW, NW), syclex::properties{sycl::ext::intel::experimental::grf_size<GRF>},
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const DqTile tl = tiles[it.get_group(0)];
            const DqJob jb = jobs[tl.job];
            const int lid = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id();
            const int live = sycl::min(MT, (jb.m - tl.m0 + TM - 1) / TM);   // the tile's 8-row blocks with tokens
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) mx::joint_matrix_fill(sg, acc[i], 0.0f);
            const int wrow = tl.n0 + lid;   // the weight row this lane decodes
            auto bsp = Bs.template get_multi_ptr<sycl::access::decorated::no>();
            auto asp = As.template get_multi_ptr<sycl::access::decorated::no>();
            const sycl::half* arow = jb.A + (size_t) (tl.m0 + lid) * jb.lda;   // the token row this lane stages
            const bool astage = lid < live * TM;
            for (int k0 = 0; k0 < K; k0 += KC) {
                if (MODE != 3 && astage) {   // one token row's 64 values: 8 x 16-byte loads
#pragma unroll
                    for (int v = 0; v < KC; v += 8)
                        *reinterpret_cast<sycl::vec<sycl::half, 8>*>(&As[(size_t) lid * KC + v]) =
                            *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(arow + k0 + v);
                }
#pragma unroll
                for (int v = 0; v < KC; v += 8) {
                    float o[8];
                    if constexpr (MODE == 1) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) o[j] = (float) (lid + j + k0);
                    } else dq8<TYPE, PLANAR>(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0 + v, o);
#pragma unroll
                    for (int j = 0; j < 8; ++j) Bs[(size_t) (v + j) * NW + lid] = (sycl::half) o[j];
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int kk = 0; kk < KC; kk += TK) {
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> mb;
                    mx::joint_matrix_load(sg, mb, bsp + kk * NW + s * TN, NW);
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
#pragma unroll
            for (int i = 0; i < MT; ++i)
                if (i < live)
                    mx::joint_matrix_store(sg, acc[i],
                        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                            jb.C + (size_t) (tl.m0 + i * TM) * jb.ldc + tl.n0 + s * TN), jb.ldc, mx::layout::row_major);
        });
    });
}

