// tools/v41/engine/xe/xe_dqgemm.cpp - larry: the prompt path's expert GEMMs with the weights dequantised ON THE FLY
// (XMX joint_matrix, fp16 x fp16 -> f32).  One work-group owns one tile: one expert matrix's 128 output rows (columns
// of C) x up to 128 token rows.  Each lane decodes its own weight row straight into the B fragment of its sub-group
// (a B fragment's lane n holds column n, element i = row i: no local-memory round trip for the weights): per 64-value
// chunk one set of wide loads into registers (planar: 16-byte vectors; the file layout's 18- / 110-byte blocks: 2-byte
// words; 28-byte blocks: 4-byte words), then four 16-value k-steps unpacked two values per 32-bit op as fp16
// (h(0x6400 | q) = 1024 + q).  The tile's activations are staged in local memory TRANSPOSED (As[k][m]) so that a lane
// reads its A fragment (8 rows at k = lane) in one 16-byte load.  A launch covers every tile of a group of experts (one
// quant type, one layout).  The form it replaces wrote each expert's three matrices out as fp16 (6.7x their packed
// bytes) for oneMKL's GEMMs to read back.
// Rounding: Q2_0 / PTQ1_0 values (q - 1) * d are exact in fp16 (as before); Q3_K rounds d * (scale - 32) to fp16
// before the product (at most one more fp16 rounding than the f32 product).
#include "xe_kernels.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstdlib>
#include <type_traits>

namespace strata::v41::xe {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

using h2 = sycl::vec<sycl::half, 2>;
using u4 = sycl::vec<uint32_t, 4>;
inline float hf(uint16_t u) { return (float) sycl::bit_cast<sycl::half>(u); }
inline h2 h2b(uint32_t bits) { return sycl::bit_cast<h2>(bits); }
inline h2 splat(sycl::half x) { return h2(x, x); }

/// four values from two packed pairs: lo = (v0, v2), hi = (v1, v3)
inline void put4(sycl::half* o, h2 lo, h2 hi) { o[0] = lo[0]; o[1] = hi[0]; o[2] = lo[1]; o[3] = hi[1]; }

// ---- chunk decoders: load(r, k0) fetches row r's bytes for values k0 .. k0 + 63 (k0 % 64 == 0) into registers
// (the chunks come in order: a block's shared fields are loaded at its first chunk and kept);
// get<KK>(o) unpacks k-step KK's 16 values (k0 + 16 KK ..).  NB: an [nrows x n] matrix's block count (planar);
// row_bytes: a row's bytes (file layout).
template <int TYPE, bool PLANAR> struct Dec;

/// Q2_0 (64 values a block, value = (q - 1) d) planar: the qs plane (16 bytes a block; value t of 16-group u: bits
/// 8 (t % 4) + 2 (t / 4) of 32-bit word u), then the d plane
template <> struct Dec<kQ20, true> {
    u4 q; sycl::half d;
    void load(const uint8_t* mat, size_t NB, int n, int, int r, int k0) {
        const size_t blk = (size_t) r * (n / 64) + k0 / 64;
        q = *reinterpret_cast<const u4*>(mat + blk * 16);
        d = sycl::bit_cast<sycl::half>(*reinterpret_cast<const uint16_t*>(mat + 16 * NB + blk * 2));
    }
    template <int KK> void get(sycl::half* o) const {
        const uint32_t w = q[KK];
        const h2 dd = splat(d), off = splat((sycl::half) 1025.0f);
#pragma unroll
        for (int g = 0; g < 4; ++g) {   // values 4g .. 4g + 3: bits 2g + 8i
            const uint32_t x = w >> (2 * g);
            put4(o + 4 * g, (h2b((x & 0x00030003u) | 0x64006400u) - off) * dd, (h2b(((x >> 8) & 0x00030003u) | 0x64006400u) - off) * dd);
        }
    }
};
/// Q2_0 file layout: 18-byte blocks (d, qs[16]; value j: bits 2 (j % 8) of 16-bit word j / 8)
template <> struct Dec<kQ20, false> {
    uint32_t w[4]; sycl::half d;
    void load(const uint8_t* mat, size_t, int, int row_bytes, int r, int k0) {
        const uint16_t* b = reinterpret_cast<const uint16_t*>(mat + (size_t) r * row_bytes + (size_t) (k0 / 64) * 18);
        d = sycl::bit_cast<sycl::half>(b[0]);
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = (uint32_t) b[1 + 2 * i] | ((uint32_t) b[2 + 2 * i] << 16);
    }
    template <int KK> void get(sycl::half* o) const {
        const h2 dd = splat(d), off = splat((sycl::half) 1025.0f);
#pragma unroll
        for (int g = 0; g < 8; ++g) {   // values g (bits 2g) and g + 8 (bits 16 + 2g)
            const h2 v = (h2b(((w[KK] >> (2 * g)) & 0x00030003u) | 0x64006400u) - off) * dd;
            o[g] = v[0];
            o[g + 8] = v[1];
        }
    }
};
/// PTQ1_0: 128 values a block: 16 + 8 + 2 trit bytes (5, 5 and 4 trits a byte; trit = ((b 3^dig) & 255) 3 >> 8),
/// then d; planar = the same three fields as planes.  Two bytes at a time in 16-bit lanes.
struct DecPtq1 {
    u4 a; uint32_t b[2]; uint32_t p; sycl::half d; int c;
    template <int DIG> static h2 trits(uint32_t x16, h2 dd) {   // x16: two bytes in 16-bit lanes
        constexpr uint32_t pw = DIG == 0 ? 1u : DIG == 1 ? 3u : DIG == 2 ? 9u : DIG == 3 ? 27u : 81u;
        const uint32_t t = ((((x16 * pw) & 0x00FF00FFu) * 3u) >> 8) & 0x00030003u;
        return (h2b(t | 0x64006400u) - splat((sycl::half) 1025.0f)) * dd;
    }
    template <int DIG> static void bytes16(sycl::half* o, const uint32_t* wv, h2 dd) {   // 16 values, one digit, 4 words
#pragma unroll
        for (int g = 0; g < 4; ++g) put4(o + 4 * g, trits<DIG>(wv[g] & 0x00FF00FFu, dd), trits<DIG>((wv[g] >> 8) & 0x00FF00FFu, dd));
    }
    template <int DIG> static void bytes8(sycl::half* o, const uint32_t* wv, h2 dd) {   // 8 values, one digit, 2 words
#pragma unroll
        for (int g = 0; g < 2; ++g) put4(o + 4 * g, trits<DIG>(wv[g] & 0x00FF00FFu, dd), trits<DIG>((wv[g] >> 8) & 0x00FF00FFu, dd));
    }
    template <int KK> void get(sycl::half* o) const {
        const h2 dd = splat(d);
        const uint32_t av[4] = {a[0], a[1], a[2], a[3]};
        if (c == 0) {   // values 16 KK .. : bytes 0..15, digit KK
            bytes16<KK>(o, av, dd);
        } else if constexpr (KK == 0) {   // 64..79: bytes 0..15, digit 4
            bytes16<4>(o, av, dd);
        } else if constexpr (KK == 1 || KK == 2) {   // 80..95 / 96..111: bytes 16..23, digits 0, 1 / 2, 3
            bytes8<2 * (KK - 1)>(o, b, dd);
            bytes8<2 * (KK - 1) + 1>(o + 8, b, dd);
        } else {   // 112..119: bytes 16..23 digit 4; 120..127: bytes 24, 25 (value 120 + i: byte i % 2, digit i / 2)
            bytes8<4>(o, b, dd);
            const uint32_t x = (p & 0xFFu) | ((p & 0xFF00u) << 8);
            const h2 t0 = trits<0>(x, dd), t1 = trits<1>(x, dd), t2 = trits<2>(x, dd), t3 = trits<3>(x, dd);
            o[8] = t0[0]; o[9] = t0[1]; o[10] = t1[0]; o[11] = t1[1]; o[12] = t2[0]; o[13] = t2[1]; o[14] = t3[0]; o[15] = t3[1];
        }
    }
};
template <> struct Dec<kPtq1, true> : DecPtq1 {   // a block's bytes once, at its first chunk
    void load(const uint8_t* mat, size_t NB, int n, int, int r, int k0) {
        c = (k0 / 64) % 2;
        if (c != 0) return;
        const size_t blk = (size_t) r * (n / 128) + k0 / 128;
        a = *reinterpret_cast<const u4*>(mat + blk * 16);
        const sycl::vec<uint32_t, 2> bv = *reinterpret_cast<const sycl::vec<uint32_t, 2>*>(mat + 16 * NB + blk * 8);
        b[0] = bv[0]; b[1] = bv[1];
        p = *reinterpret_cast<const uint32_t*>(mat + 24 * NB + blk * 4);
        d = sycl::bit_cast<sycl::half>((uint16_t) (p >> 16));
    }
};
template <> struct Dec<kPtq1, false> : DecPtq1 {   // 28-byte blocks (4-byte aligned rows and blocks)
    void load(const uint8_t* mat, size_t, int, int row_bytes, int r, int k0) {
        c = (k0 / 64) % 2;
        if (c != 0) return;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(mat + (size_t) r * row_bytes + (size_t) (k0 / 128) * 28);
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = w[i];
        b[0] = w[4]; b[1] = w[5]; p = w[6];
        d = sycl::bit_cast<sycl::half>((uint16_t) (p >> 16));
    }
};
/// Q3_K: 256 values a block (hmask[32] qs[64] scales[12] d); a 64-value chunk c is half h = c / 2, 32-groups
/// jj = 2 (c % 2) + {0, 1}; value (h, jj, l): ((qs[32 h + l] >> 2 jj) & 3) - (hmask[l] bit 4 h + jj ? 0 : 4), times
/// d (scale_is - 32), is = 8 h + 2 jj + l / 16
struct DecQ3K {
    uint32_t qs[8], hm[8], sc[3]; float d; int c;
    uint32_t scb(int k) const { return (sc[k / 4] >> (8 * (k % 4))) & 0xFFu; }
    template <int KK> void get(sycl::half* o) const {
        const int is = c * 4 + KK, h = c / 2, jj = 2 * (c % 2) + KK / 2;
        const uint32_t lo4 = (scb(is % 8) >> (4 * (is / 8))) & 0x0fu;
        const uint32_t hi2 = (scb(8 + is % 4) >> (2 * (is / 4))) & 0x03u;
        const h2 dl = splat((sycl::half) (d * (float) ((int) (lo4 | (hi2 << 4)) - 32))), off = splat((sycl::half) 1028.0f);
        const int mbit = h * 4 + jj;
#pragma unroll
        for (int g = 0; g < 4; ++g) {   // values l = 16 (KK % 2) + 4g .. + 3: 32-bit word l / 4 of qs and hmask
            const int wd = (KK % 2) * 4 + g;
            const uint32_t v = ((qs[wd] >> (2 * jj)) & 0x03030303u) | (((hm[wd] >> mbit) & 0x01010101u) << 2);   // q + 4 hbit
            put4(o + 4 * g, (h2b((v & 0x00FF00FFu) | 0x64006400u) - off) * dl, (h2b(((v >> 8) & 0x00FF00FFu) | 0x64006400u) - off) * dl);
        }
    }
};
template <> struct Dec<kQ3K, true> : DecQ3K {   // planes: qs 64, hmask 32, scales 12, d 2 bytes a block
    void load(const uint8_t* mat, size_t NB, int n, int, int r, int k0) {   // hmask / scales / d a block, qs a half
        const size_t blk = (size_t) r * (n / 256) + k0 / 256;
        c = (k0 / 64) % 4;
        if (c == 0) {
            const u4* h4 = reinterpret_cast<const u4*>(mat + 64 * NB + blk * 32);
            const u4 h0 = h4[0], h1 = h4[1];
#pragma unroll
            for (int i = 0; i < 4; ++i) { hm[i] = h0[i]; hm[4 + i] = h1[i]; }
            const uint32_t* s4 = reinterpret_cast<const uint32_t*>(mat + 96 * NB + blk * 12);
            sc[0] = s4[0]; sc[1] = s4[1]; sc[2] = s4[2];
            d = hf(*reinterpret_cast<const uint16_t*>(mat + 108 * NB + blk * 2));
        }
        if (c % 2 == 0) {
            const u4* q4 = reinterpret_cast<const u4*>(mat + blk * 64 + (c / 2) * 32);
            const u4 q0 = q4[0], q1 = q4[1];
#pragma unroll
            for (int i = 0; i < 4; ++i) { qs[i] = q0[i]; qs[4 + i] = q1[i]; }
        }
    }
};
template <> struct Dec<kQ3K, false> : DecQ3K {   // 110-byte blocks (2-byte aligned)
    void load(const uint8_t* mat, size_t, int, int row_bytes, int r, int k0) {   // hmask / scales / d a block, qs a half
        const uint16_t* b = reinterpret_cast<const uint16_t*>(mat + (size_t) r * row_bytes + (size_t) (k0 / 256) * 110);
        c = (k0 / 64) % 4;
        if (c == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) hm[i] = (uint32_t) b[2 * i] | ((uint32_t) b[2 * i + 1] << 16);
#pragma unroll
            for (int i = 0; i < 3; ++i) sc[i] = (uint32_t) b[48 + 2 * i] | ((uint32_t) b[49 + 2 * i] << 16);
            d = hf(b[54]);
        }
        if (c % 2 == 0) {
            const uint16_t* qp = b + 16 + (c / 2) * 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) qs[i] = (uint32_t) qp[2 * i] | ((uint32_t) qp[2 * i + 1] << 16);
        }
    }
};

constexpr int TM = 8, TN = 16, TK = 16, NW = 128, KC = 64, MT = 16;   // tile: 128 output rows x 128 token rows
constexpr int MPS = MT * TM + 8;   // As row stride (halves): 272 bytes, so a fragment read spreads over the banks

template <int TYPE, bool PLANAR>
sycl::event dqgemm_t(sycl::queue& q, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
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
            const int live = sycl::min(MT, (jb.m - tl.m0 + TM - 1) / TM);   // the tile's 8-row blocks with tokens
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) mx::joint_matrix_fill(sg, acc[i], 0.0f);
            const int wrow = tl.n0 + lid;   // the weight row this lane decodes
            const sycl::half* arow = jb.A + (size_t) (tl.m0 + lid) * jb.lda;   // the token row this lane stages
            const bool astage = lid < live * TM;
            Dec<TYPE, PLANAR> dec;
            for (int k0 = 0; k0 < K; k0 += KC) {
                dec.load(jb.W, (size_t) jb.NB, K, jb.row_bytes, wrow, k0);
                if (astage) {   // one token row's 64 values: 8 x 16-byte loads, stored transposed
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
                    dec.template get<KK>(o);
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

// ---- larry, STRATA_XE_DQ2 (patch-dqv2.py): v2 - the chunk's weights decoded once into local memory VNNI-packed, a
// sub-group 32 tokens x 64 output rows with its token rows by 2D block loads (half v1's local-memory bytes a multiply;
// the same decode and k order: the same bits)
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

}  // namespace

sycl::event xe_dqgemm(sycl::queue& q, int type, bool planar, const DqJob* jobs, const DqTile* tiles, int ntiles, int K) {
    if (ntiles <= 0) return {};
    static const bool v2 = [] { const char* e = std::getenv("STRATA_XE_DQ2"); return e && *e && *e != '0'; }();   // (patch-dqv2.py)
    if (v2) {
        if (type == kQ20) return planar ? dqgemm2_t<kQ20, true>(q, jobs, tiles, ntiles, K) : dqgemm2_t<kQ20, false>(q, jobs, tiles, ntiles, K);
        if (type == kPtq1) return planar ? dqgemm2_t<kPtq1, true>(q, jobs, tiles, ntiles, K) : dqgemm2_t<kPtq1, false>(q, jobs, tiles, ntiles, K);
        return planar ? dqgemm2_t<kQ3K, true>(q, jobs, tiles, ntiles, K) : dqgemm2_t<kQ3K, false>(q, jobs, tiles, ntiles, K);
    }
    if (type == kQ20) return planar ? dqgemm_t<kQ20, true>(q, jobs, tiles, ntiles, K) : dqgemm_t<kQ20, false>(q, jobs, tiles, ntiles, K);
    if (type == kPtq1) return planar ? dqgemm_t<kPtq1, true>(q, jobs, tiles, ntiles, K) : dqgemm_t<kPtq1, false>(q, jobs, tiles, ntiles, K);
    return planar ? dqgemm_t<kQ3K, true>(q, jobs, tiles, ntiles, K) : dqgemm_t<kQ3K, false>(q, jobs, tiles, ntiles, K);
}

}  // namespace strata::v41::xe
