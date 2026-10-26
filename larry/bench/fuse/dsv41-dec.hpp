// ggml-sycl/dsv41-dec.hpp - larry (DeepSeek-V4.1 on Intel Arc B70): the decode / verify-width (< 8 queries) attention
// core as three kernels, pointer-based (dsv41-fused.cpp wraps them as ggml ops; an engine without ggml can call them).
//
//   kv_fin    the window row: rms_norm(kv) * w, RoPE, FP8 fake-quant -> this pass's rows and straight into the window
//             ring (the slots it writes hold positions >= RAW - SWA = 16 older than any query's window: nothing this
//             pass reads); and the pass's RoPE table (cos, sin per pair and query: the layer's q, o and indexer q use
//             the same angles - the transcendentals once, not per work-group: computed per work-group they cost the
//             attention ~30 us a layer)
//   attn_dec  per query its SWA window rows (ring or this pass's rows) and its top-K compressed rows, gathered straight
//             into local memory (f16); q RoPE'd while loaded; flash-decoding over 64-key chunks with XMX (as
//             fattn-dsv4-dec.hpp's launch_xmx); sinks; the output de-RoPE'd in the combine
//
// They replace, per decode layer: rms_norm * w, rope, fake_quant, set_rows (the window row); rope (q), concat (the
// window source), kgather2, concat + cast (the mask), rope back (o) - and keep the attention's two kernels.
// Tensor parallel: attn_dec reads q / sinks and writes o through pointers + strides over NH heads, so a shard of the
// heads passes its slice (NH % 32 == 0).
#pragma once

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace dsv41dec {

// ---- RoPE as ggml-sycl's rope_norm (NORMAL mode: adjacent pairs) on the LAST n_dims values of each row; YaRN when
// ext_factor != 0.  theta_scale = powf(freq_base, -2 / n_dims) and the corr dims (ggml_rope_yarn_corr_dims) come from
// the host, as ggml-sycl computes them.
struct Rope {
    int n_dims = 0;
    float theta_scale = 1.0f, freq_scale = 1.0f, ext_factor = 0.0f, attn_factor = 1.0f;
    float corr0 = 0.0f, corr1 = 0.0f;
};

// cos / sin of the pair starting at iw (even, relative to the rotated range) at position p: rope_yarn's arithmetic
inline void rope_cs(const Rope & r, int p, int iw, bool forward, float & c, float & s) {
    const float theta_extrap = (float) p * sycl::pow(r.theta_scale, iw / 2.0f);
    const float theta_interp = r.freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = r.attn_factor;
    if (r.ext_factor != 0.0f) {
        const float y = (iw / 2 - r.corr0) / sycl::max(0.001f, r.corr1 - r.corr0);
        const float ramp_mix = (1.0f - sycl::min(1.0f, sycl::max(0.0f, y))) * r.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * sycl::log(1.0f / r.freq_scale);
    }
    c = sycl::cos(theta) * mscale;
    s = sycl::sin(theta) * mscale;
    if (!forward) s *= -1.0f;
}

inline void rope_pair(const Rope & r, int p, int iw, bool forward, float & x0, float & x1) {
    float c, s;
    rope_cs(r, p, iw, forward, c, s);
    const float a = x0, b = x1;
    x0 = a * c - b * s;
    x1 = a * s + b * c;
}

// the same rotation from a table entry (tab[iw] = cos, tab[iw + 1] = sin of the forward rotation; back: sin negated,
// as rope_cs does)
inline void rope_tab(const float * tab, int iw, bool forward, float & x0, float & x1) {
    const float c = tab[iw];
    float s = tab[iw + 1];
    if (!forward) s *= -1.0f;
    const float a = x0, b = x1;
    x0 = a * c - b * s;
    x1 = a * s + b * c;
}

// ---- the model's FP8 activation rounding (ds41_ops.hpp; dsv41-fused.cpp's fake_quant_blocks mode 0) with exact IEEE
// arithmetic (icpx defaults to fast-math; the pragma is only allowed at file scope)
}  // namespace dsv41dec
#pragma float_control(precise, on, push)
namespace dsv41dec {
inline float fq8_pow2_ceil(float a) {
    const uint32_t u = sycl::bit_cast<uint32_t>(a);
    const uint32_t e = (u >> 23) & 0xffu;
    if (e == 0) return 0.0f;
    if (e == 0xffu) return a;
    return (u & 0x7fffffu) == 0 ? sycl::bit_cast<float>(e << 23) : sycl::bit_cast<float>((e + 1) << 23);
}
inline float fq8_pow2_floor(float a) {
    const uint32_t u = sycl::bit_cast<uint32_t>(a);
    const uint32_t e = (u >> 23) & 0xffu;
    if (e == 0) return 0.0f;
    if (e == 0xffu) return a;
    return sycl::bit_cast<float>(e << 23);
}
// v rounded to E4M3 (3 mantissa bits, smallest normal 2^-6), round half to even
inline float fq8_round_e4m3(float v) {
    const float p = sycl::fmax(fq8_pow2_floor(sycl::fabs(v)), 0x1p-6f);
    const float r = sycl::rint(v / p * 8.0f) + 0.0f;
    return r / 8.0f * p;
}
// the block scale from the block's max |x| (FP8_POW2_32: >= 1e-4, / 448, power-of-two ceiling)
inline float fq8_scale(float amax) { return fq8_pow2_ceil(sycl::fmax(amax, 1e-4f) / 448.0f); }
inline float fq8_apply(float x, float s) { return fq8_round_e4m3(sycl::fmin(sycl::fmax(x / s, -448.0f), 448.0f)) * s; }
// FP4_POW2_32 (the indexer's q and k): E2M1 (1 mantissa bit, smallest normal 1), scale 2^ceil(log2(max(amax, 6 2^-126) / 6))
inline float fq4_round_e2m1(float v) {
    const float p = sycl::fmax(fq8_pow2_floor(sycl::fabs(v)), 1.0f);
    const float r = sycl::rint(v / p * 2.0f) + 0.0f;
    return r / 2.0f * p;
}
inline float fq4_scale(float amax) { return fq8_pow2_ceil(sycl::fmax(amax, 6.0f * 0x1p-126f) / 6.0f); }
inline float fq4_apply(float x, float s) { return fq4_round_e2m1(sycl::fmin(sycl::fmax(x / s, -6.0f), 6.0f)) * s; }
}  // namespace dsv41dec
#pragma float_control(pop)
namespace dsv41dec {

// ---- int8 activations for fastmv's Q6K_R8 mat-vecs ("q8", quant_x's layout and arithmetic): per 32-value block
// q = round(x * 127 / amax), scale amax / 127 (amax 0: all 0).  A producer that holds a block across lanes takes the
// block's amax with sub-group shuffles (max is exact in any order) and writes the same bytes quant_x would.
inline float q8_inv(float amax) { return amax > 0.f ? 127.0f / amax : 0.f; }
inline uint32_t q8_pack4(float a, float b, float c, float d, float id) {
    return ((uint32_t) ((int) sycl::rint(a * id) & 0xFF)) | ((uint32_t) ((int) sycl::rint(b * id) & 0xFF) << 8) |
           ((uint32_t) ((int) sycl::rint(c * id) & 0xFF) << 16) | ((uint32_t) ((int) sycl::rint(d * id) & 0xFF) << 24);
}

// ================================================================== rms_q8
// y = rms_norm(x) * w (ggml-sycl's rms_norm_f32 at decode shapes: a 256-lane work-group below 1024 values, 1024 lanes
// from there - same sums in the same order) and y's q8: y goes through local memory, then a sub-group a 32-value block
struct RmsQ8Args {
    const float * x = nullptr; int64_t x_cs = 0;
    const float * w = nullptr;
    float eps = 0.0f;
    float * y = nullptr;        // [n][K]
    int8_t * q = nullptr;       // [n][K]
    float * d = nullptr;        // [n][K / 32]
    int K = 0, n = 0;
};

template <int WG>
inline sycl::event rms_q8_t(sycl::queue & q, const RmsQ8Args & a) {
    const RmsQ8Args A = a;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> s_sum(sycl::range<1>(WG / 16), h);
        sycl::local_accessor<float, 1> ys(sycl::range<1>((size_t) A.K), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) A.n * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int t = (int) it.get_group(0), l = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const float * x = A.x + (size_t) t * A.x_cs;
            float tmp = 0.0f;
            for (int col = l; col < A.K; col += WG) {
                const float xi = x[col];
                tmp += xi * xi;
            }
#pragma unroll
            for (int m = 8; m > 0; m >>= 1) tmp += sycl::permute_group_by_xor(sg, tmp, m);
            if (lane == 0) s_sum[sgi] = tmp;
            sycl::group_barrier(it.get_group());
            tmp = 0.0f;
            for (int i = 0; i < (WG / 16 + 15) / 16; ++i) tmp += s_sum[lane + i * 16];
#pragma unroll
            for (int m = 8; m > 0; m >>= 1) tmp += sycl::permute_group_by_xor(sg, tmp, m);
            const float mean = tmp / A.K;
            const float scale = sycl::rsqrt(mean + A.eps);
            float * y = A.y + (size_t) t * A.K;
            for (int col = l; col < A.K; col += WG) {
                const float v = scale * x[col] * A.w[col];
                y[col] = v;
                ys[col] = v;
            }
            sycl::group_barrier(it.get_group());
            const int nblk = A.K / 32;
            for (int bk = sgi; bk < nblk; bk += WG / 16) {
                const float v0 = ys[bk * 32 + 2 * lane], v1 = ys[bk * 32 + 2 * lane + 1];
                const float am = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
                const float id = q8_inv(am);
                const uint16_t pk = (uint16_t) (((int) sycl::rint(v0 * id) & 0xFF) | (((int) sycl::rint(v1 * id) & 0xFF) << 8));
                *(uint16_t *) (A.q + (size_t) t * A.K + bk * 32 + 2 * lane) = pk;
                if (lane == 0) A.d[(size_t) t * nblk + bk] = am / 127.0f;
            }
        });
    });
}

inline sycl::event rms_q8(sycl::queue & q, const RmsQ8Args & a) {
    return a.K < 1024 ? rms_q8_t<256>(q, a) : rms_q8_t<1024>(q, a);
}

// ================================================================== kv_fin
struct KvFinArgs {
    const float * x = nullptr;  int64_t x_cs = 0;      // n columns of D values (column stride in floats)
    const float * w = nullptr;                         // [D] the norm weight
    const int32_t * pos = nullptr;                     // [n]
    float eps = 0.0f;
    Rope rope;
    bool fq = true;                                    // the FP8 rounding (act_quant)
    float * y = nullptr;        int64_t y_cs = 0;      // n columns out
    float * ring = nullptr;     int64_t ring_cs = 0;   // column t >= slot_from also -> ring row slot[t - slot_from]
    const int32_t * slot = nullptr;
    int slot_from = 0;
    float * tab = nullptr;                             // the RoPE table: tab[t * n_dims + iw] = cos, [.. + 1] = sin
    int D = 0, n = 0;
};

// one 256-lane work-group a column, D == 512: the sum of squares as ggml-sycl's rms_norm_f32 at this shape (2 values a
// lane, sub-group xor butterflies, per-sub-group partials through local memory), then pairs (2 l, 2 l + 1) per lane -
// a 16-lane sub-group holds exactly one 32-value FP8 block
inline sycl::event kv_fin(sycl::queue & q, const KvFinArgs & a) {
    constexpr int WG = 256, D = 2 * WG;
    const KvFinArgs A = a;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> s_sum(sycl::range<1>(WG / 16), h);
        sycl::local_accessor<float, 1> ys(sycl::range<1>(D), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) A.n * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int t = (int) it.get_group(0), l = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const float * x = A.x + (size_t) t * A.x_cs;
            const float x0 = x[l], x1 = x[l + WG];
            float tmp = 0.0f;
            tmp += x0 * x0;
            tmp += x1 * x1;
#pragma unroll
            for (int m = 8; m > 0; m >>= 1) tmp += sycl::permute_group_by_xor(sg, tmp, m);
            if (lane == 0) s_sum[sgi] = tmp;
            sycl::group_barrier(it.get_group());
            tmp = 0.0f;
            tmp += s_sum[lane];
#pragma unroll
            for (int m = 8; m > 0; m >>= 1) tmp += sycl::permute_group_by_xor(sg, tmp, m);
            const float mean = tmp / D;
            const float scale = sycl::rsqrt(mean + A.eps);
            ys[l] = scale * x0 * A.w[l];
            ys[l + WG] = scale * x1 * A.w[l + WG];
            sycl::group_barrier(it.get_group());
            const int i0 = 2 * l;
            float v0 = ys[i0], v1 = ys[i0 + 1];
            const int off = D - A.rope.n_dims;
            if (i0 >= off) {
                float c, sn;
                rope_cs(A.rope, A.pos[t], i0 - off, true, c, sn);
                const float a0 = v0, a1 = v1;
                v0 = a0 * c - a1 * sn;
                v1 = a0 * sn + a1 * c;
                if (A.tab) {
                    A.tab[(size_t) t * A.rope.n_dims + (i0 - off)] = c;
                    A.tab[(size_t) t * A.rope.n_dims + (i0 - off) + 1] = sn;
                }
            }
            if (A.fq) {
                const float am = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
                const float s = fq8_scale(am);
                v0 = fq8_apply(v0, s);
                v1 = fq8_apply(v1, s);
            }
            *(sycl::float2 *) (A.y + (size_t) t * A.y_cs + i0) = sycl::float2(v0, v1);
            if (A.ring && t >= A.slot_from)
                *(sycl::float2 *) (A.ring + (size_t) A.slot[t - A.slot_from] * A.ring_cs + i0) = sycl::float2(v0, v1);
        });
    });
}

// ================================================================== attn_dec
constexpr int DD = 512;            // head dim
constexpr int XKC = 64, XHG = 32;  // keys per chunk, heads per work-group (fattn-dsv4-dec.hpp's XMX form)

struct AttnArgs {
    const float * q = nullptr;  size_t q_h = 0, q_b = 0;   // q[b * q_b + h * q_h + d], before RoPE
    const float * tab = nullptr;                           // kv_fin's RoPE table [n][n_dims]: q forward, o back,
    int n_dims = 0;                                        // the last n_dims of each head
    // the window: key j < swa of query b is row widx[b * wi_b + j] of [ring (RAW rows) | rows (this pass)]
    const float * ring = nullptr; int64_t ring_cs = 0; int RAW = 0;
    const float * rows = nullptr; int64_t rows_cs = 0;
    const int32_t * widx = nullptr; int64_t wi_b = 0;
    const float * wmask = nullptr;  int64_t wm_b = 0;      // 0 / -inf
    // the compressed keys: key swa + i of query b is row sel[b * sel_b + i] of comp (0 f32, 1 bf16, 2 f16)
    const void * comp = nullptr; int comp_t = 0; size_t comp_rb = 0;
    const int32_t * sel = nullptr; int64_t sel_b = 0;
    const float * selm = nullptr;  int64_t sm_b = 0;
    int swa = 0, K = 0;                                    // M = swa + K keys a query
    const float * sinks = nullptr;                         // [NH] or null
    float * o = nullptr; size_t o_h = 0, o_b = 0;          // o[b * o_b + h * o_h + d], de-RoPE'd
    int NH = 0, n = 0;
    float scale = 1.0f;
    float * part = nullptr;                                // scratch_floats()
    // optional: o's q8 for the grouped output projection - row (h / hpg) * n + b of K8 = hpg * 512 values holds query
    // b's heads of group h / hpg (head h at (h % hpg) * 512)
    int hpg = 0;
    int8_t * oq = nullptr;
    float * od = nullptr;
};

inline int n_chunks(int M) { return (M + XKC - 1) / XKC; }
inline size_t scratch_floats(int NH, int M, int n) { return (size_t) n * n_chunks(M) * NH * (DD + 2); }
inline bool covered(int NH, int M) { return NH % XHG == 0 && M >= 1 && n_chunks(M) <= 32; }

namespace jm = sycl::ext::oneapi::experimental::matrix;

inline sycl::event attn_dec(sycl::queue & q, const AttnArgs & a, sycl::event * first = nullptr) {
    const int M = a.swa + a.K;
    const int nch = n_chunks(M);
    const int NH = a.NH, n = a.n;
    constexpr int D = DD;
    constexpr int KS = D + 8;     // ks row stride (halves): ks[key][d]
    constexpr int QS = XHG + 8;   // qt row stride (halves): qt[d][head]
    constexpr int SS = XHG + 4;   // ss row stride (floats): ss[key][head]
    constexpr int PS = XKC + 16;  // ps row stride (halves): ps[head][key]
    const AttnArgs A = a;
    float * pO = a.part;
    float * pML = a.part + (size_t) n * nch * NH * D;
    sycl::event e1 = q.submit([&](sycl::handler & h) {
        sycl::local_accessor<sycl::half, 1> ks(sycl::range<1>((size_t) XKC * KS), h);
        sycl::local_accessor<sycl::half, 1> qt(sycl::range<1>((size_t) D * QS), h);
        sycl::local_accessor<float, 1> ss(sycl::range<1>((size_t) XKC * SS), h);
        sycl::local_accessor<sycl::half, 1> ps(sycl::range<1>((size_t) XHG * PS), h);
        sycl::local_accessor<float, 1> mk(sycl::range<1>((size_t) XKC), h);
        sycl::local_accessor<uint64_t, 1> rp(sycl::range<1>((size_t) XKC), h);   // the chunk's key rows: address | type
        sycl::local_accessor<float, 1> tl(sycl::range<1>((size_t) DD), h);       // the query's RoPE table
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>((size_t) n, (size_t) nch, (size_t) (NH / XHG) * 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const int b = (int) it.get_group(0), c = (int) it.get_group(1), hg = (int) it.get_group(2);
            const int lid = (int) it.get_local_id(2);
            const sycl::sub_group sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int j0 = c * XKC;
            const int nk = sycl::min(XKC, M - j0);
            // 0. per key of the chunk its source row (an address; bit 0: a compressed row) and its mask (-1e30 for keys
            // past the count); the query's RoPE table -> local memory (the gather below then has no dependent loads)
            if (lid < XKC) {
                uint64_t ra = 0;
                float mv = -1e30f;
                if (lid < nk) {
                    const int m = j0 + lid;
                    if (m < A.swa) {
                        const int r = A.widx[(size_t) b * A.wi_b + m];
                        ra = (uint64_t) (r < A.RAW ? A.ring + (size_t) r * A.ring_cs : A.rows + (size_t) (r - A.RAW) * A.rows_cs);
                        mv = A.wmask[(size_t) b * A.wm_b + m];
                    } else {
                        ra = (uint64_t) ((const char *) A.comp + (size_t) A.sel[(size_t) b * A.sel_b + (m - A.swa)] * A.comp_rb) | 1u;
                        mv = A.selm[(size_t) b * A.sm_b + (m - A.swa)];
                    }
                    mv = sycl::fmax(mv, -1e30f);
                }
                rp[lid] = ra;
                mk[lid] = mv;
            } else if (lid - XKC < A.n_dims) {
                tl[lid - XKC] = A.tab[(size_t) b * A.n_dims + (lid - XKC)];
            }
            sycl::group_barrier(it.get_group());
            // 1. the chunk's keys -> ks (f16; rows past the key count are zero), each straight from its source row
#pragma unroll 8
            for (int idx = lid * 4; idx < XKC * D; idx += 256 * 4) {
                const int row = idx / D, d = idx % D;
                sycl::half4 hv(0.0f, 0.0f, 0.0f, 0.0f);
                if (row < nk) {
                    const uint64_t ra = rp[row];
                    if (!(ra & 1u)) {
                        const sycl::float4 fv = *reinterpret_cast<const sycl::float4 *>((const float *) ra + d);
                        hv = sycl::half4((sycl::half) fv.x(), (sycl::half) fv.y(), (sycl::half) fv.z(), (sycl::half) fv.w());
                    } else {
                        const char * src = (const char *) (ra & ~(uint64_t) 1);
                        if (A.comp_t == 1) {
                            const sycl::uint2 u = *reinterpret_cast<const sycl::uint2 *>(src + (size_t) d * 2);
                            hv = sycl::half4((sycl::half) sycl::bit_cast<float>(u.x() << 16), (sycl::half) sycl::bit_cast<float>(u.x() & 0xffff0000u),
                                             (sycl::half) sycl::bit_cast<float>(u.y() << 16), (sycl::half) sycl::bit_cast<float>(u.y() & 0xffff0000u));
                        } else if (A.comp_t == 2) {
                            hv = *reinterpret_cast<const sycl::half4 *>(src + (size_t) d * 2);
                        } else {
                            const sycl::float4 fv = *reinterpret_cast<const sycl::float4 *>(src + (size_t) d * 4);
                            hv = sycl::half4((sycl::half) fv.x(), (sycl::half) fv.y(), (sycl::half) fv.z(), (sycl::half) fv.w());
                        }
                    }
                }
                *reinterpret_cast<sycl::half4 *>(&ks[(size_t) row * KS + d]) = hv;
            }
            // 2. Q^T of this work-group's 32 heads -> qt (RoPE'd from the local table, scaled, f16)
            const int roff = D - A.n_dims;
            const float * tb = tl.template get_multi_ptr<sycl::access::decorated::no>().get();
#pragma unroll 8
            for (int idx = lid * 4; idx < XHG * D; idx += 256 * 4) {
                const int hl = idx / D, d = idx % D;
                sycl::float4 v = *reinterpret_cast<const sycl::float4 *>(A.q + (size_t) b * A.q_b + (size_t) (hg * XHG + hl) * A.q_h + d);
                if (d >= roff) {
                    float v0 = v.x(), v1 = v.y(), v2 = v.z(), v3 = v.w();
                    rope_tab(tb, d - roff, true, v0, v1);
                    rope_tab(tb, d + 2 - roff, true, v2, v3);
                    v = sycl::float4(v0, v1, v2, v3);
                }
                qt[(size_t) d * QS + hl] = (sycl::half) (v.x() * A.scale);
                qt[(size_t) (d + 1) * QS + hl] = (sycl::half) (v.y() * A.scale);
                qt[(size_t) (d + 2) * QS + hl] = (sycl::half) (v.z() * A.scale);
                qt[(size_t) (d + 3) * QS + hl] = (sycl::half) (v.w() * A.scale);
            }
            sycl::group_barrier(it.get_group());
            auto kp = ks.template get_multi_ptr<sycl::access::decorated::no>();
            // 3. S^T: one 8-key x 16-head tile per sub-group (keys 8 mt.., heads 16 nt..), 32 steps over D
            {
                const int mt = s >> 1, nt = s & 1;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> ma;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> mb;
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> mc;
                jm::joint_matrix_fill(sg, mc, 0.0f);
                auto qp = qt.template get_multi_ptr<sycl::access::decorated::no>();
                for (int k = 0; k < D / 16; ++k) {
                    jm::joint_matrix_load(sg, ma, kp + (size_t) (mt * 8) * KS + k * 16, KS);
                    jm::joint_matrix_load(sg, mb, qp + (size_t) (k * 16) * QS + nt * 16, QS);
                    jm::joint_matrix_mad(sg, mc, ma, mb, mc);
                }
                auto sp = ss.template get_multi_ptr<sycl::access::decorated::no>();
                jm::joint_matrix_store(sg, mc, sp + (size_t) (mt * 8) * SS + nt * 16, SS, jm::layout::row_major);
            }
            sycl::group_barrier(it.get_group());
            // 4. the chunk's softmax per head: sub-groups 0, 1, lane = head; P (f16) -> ps[head][key]
            if (s < 2) {
                const int hl = s * 16 + lane, hh = hg * XHG + hl;
                float xv[XKC];
                float mx = -1e30f;
#pragma unroll
                for (int j = 0; j < XKC; ++j) {
                    xv[j] = sycl::fmax(ss[(size_t) j * SS + hl] + mk[j], -1e30f);
                    mx = sycl::fmax(mx, xv[j]);
                }
                float l = 0.0f;
#pragma unroll
                for (int j = 0; j < XKC; ++j) {
                    const float pj = xv[j] > -1e29f ? sycl::exp(xv[j] - mx) : 0.0f;
                    l += pj;
                    ps[(size_t) hl * PS + j] = (sycl::half) pj;
                }
                float * ml = pML + (((size_t) b * nch + c) * NH + hh) * 2;
                ml[0] = mx;
                ml[1] = l;
            }
            sycl::group_barrier(it.get_group());
            // 5. this chunk's O = P . K: 4 head tiles x 32 dim tiles, 8 per sub-group, 4 steps over the keys
            {
                auto pp = ps.template get_multi_ptr<sycl::access::decorated::no>();
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> ma;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> mb;
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> mc;
                for (int ti = 0; ti < 8; ++ti) {
                    const int tile = s * 8 + ti, ht = tile / 32, dt = tile % 32;
                    jm::joint_matrix_fill(sg, mc, 0.0f);
#pragma unroll
                    for (int k = 0; k < XKC / 16; ++k) {
                        jm::joint_matrix_load(sg, ma, pp + (size_t) (ht * 8) * PS + k * 16, PS);
                        jm::joint_matrix_load(sg, mb, kp + (size_t) (k * 16) * KS + dt * 16, KS);
                        jm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    float * dst = pO + (((size_t) b * nch + c) * NH + hg * XHG + ht * 8) * D + dt * 16;
                    jm::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                                             sycl::access::decorated::no>(dst), D, jm::layout::row_major);
                }
            }
        });
    });
    if (first) *first = e1;
    // 6. combine: one sub-group per (query, head) - the chunks' parts and the sink into O, then the de-RoPE of the
    // rotated dims (lane L holds dims 128 g + 8 L + e: the last 64 are g = 3, L >= 8, adjacent pairs in-register)
    constexpr int SGW = 4;
    const size_t nrows = (size_t) n * NH;
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((nrows + SGW - 1) / SGW * SGW * 16), sycl::range<1>(SGW * 16)),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t row = (size_t) it.get_group(0) * SGW + sg.get_group_linear_id();   // b * NH + h
        if (row >= nrows) return;
        const int lane = (int) sg.get_local_linear_id();
        const int b = (int) (row / NH), hh = (int) (row % NH);
        const float * ob = pO + ((size_t) b * nch * NH + hh) * D;
        const float * mb = pML + ((size_t) b * nch * NH + hh) * 2;
        const size_t cs = (size_t) NH * D, ms = (size_t) NH * 2;   // chunk strides
        constexpr int MAXC = 32;                                  // chunks: <= 2 per lane (M <= 2048)
        float mc[MAXC / 16], lc[MAXC / 16];
        float gm = A.sinks ? A.sinks[hh] : -1e30f;
#pragma unroll
        for (int r = 0; r < MAXC / 16; ++r) {
            const int cc = lane + 16 * r;
            mc[r] = cc < nch ? mb[cc * ms] : -1e30f;
            lc[r] = cc < nch ? mb[cc * ms + 1] : 0.0f;
            gm = sycl::fmax(gm, mc[r]);
        }
        gm = sycl::reduce_over_group(sg, gm, sycl::maximum<float>());
        float w[MAXC / 16], ls = 0.0f;
#pragma unroll
        for (int r = 0; r < MAXC / 16; ++r) { w[r] = mc[r] > -1e29f ? sycl::exp(mc[r] - gm) : 0.0f; ls += lc[r] * w[r]; }
        float L = sycl::reduce_over_group(sg, ls, sycl::plus<float>());
        if (A.sinks) L += sycl::exp(A.sinks[hh] - gm);
        const float inv = L > 0.0f ? 1.0f / L : 0.0f;
        // the lane's RoPE table entries (dims 384 + 8 lane .. + 7 are rotated when >= D - n_dims), loaded ahead
        const int roff = D - A.n_dims;
        float tv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int iw = 384 + 8 * lane + e - roff;
            tv[e] = iw >= 0 ? A.tab[(size_t) b * A.n_dims + iw] : 0.0f;
        }
        float acc[D / 16];
#pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[i] = 0.0f;
        for (int cc = 0; cc < nch; ++cc) {
            const float wc = sycl::select_from_group(sg, cc < 16 ? w[0] : w[1], cc % 16);
            const float * pc = ob + cc * cs;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const sycl::float4 x0 = *reinterpret_cast<const sycl::float4 *>(pc + 128 * g + 8 * lane);
                const sycl::float4 x1 = *reinterpret_cast<const sycl::float4 *>(pc + 128 * g + 8 * lane + 4);
                acc[8 * g] += x0.x() * wc; acc[8 * g + 1] += x0.y() * wc; acc[8 * g + 2] += x0.z() * wc; acc[8 * g + 3] += x0.w() * wc;
                acc[8 * g + 4] += x1.x() * wc; acc[8 * g + 5] += x1.y() * wc; acc[8 * g + 6] += x1.z() * wc; acc[8 * g + 7] += x1.w() * wc;
            }
        }
#pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[i] *= inv;
#pragma unroll
        for (int g = 0; g < 4; ++g)
#pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const int d = 128 * g + 8 * lane + e;
                if (d >= roff) {   // (g = 3 only when roff >= 384: tv holds those dims' entries)
                    const float cs = tv[e], sn = -tv[e + 1];   // the back rotation: rope_tab(.., false)
                    const float a0 = acc[8 * g + e], a1 = acc[8 * g + e + 1];
                    acc[8 * g + e] = a0 * cs - a1 * sn;
                    acc[8 * g + e + 1] = a0 * sn + a1 * cs;
                }
            }
        float * op = A.o + (size_t) b * A.o_b + (size_t) hh * A.o_h;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            *reinterpret_cast<sycl::float4 *>(op + 128 * g + 8 * lane) = sycl::float4(acc[8 * g], acc[8 * g + 1], acc[8 * g + 2], acc[8 * g + 3]);
            *reinterpret_cast<sycl::float4 *>(op + 128 * g + 8 * lane + 4) = sycl::float4(acc[8 * g + 4], acc[8 * g + 5], acc[8 * g + 6], acc[8 * g + 7]);
        }
        if (A.hpg) {   // q8: a 32-value block is 4 lanes x 8 dims (lanes 4 j .. 4 j + 3 of one g)
            const int K8 = A.hpg * D;
            const size_t r8 = (size_t) (hh / A.hpg) * n + b;
            const int koff = (hh % A.hpg) * D;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                float am = 0.0f;
#pragma unroll
                for (int e = 0; e < 8; ++e) am = sycl::fmax(am, sycl::fabs(acc[8 * g + e]));
                am = sycl::fmax(am, sycl::permute_group_by_xor(sg, am, 1));
                am = sycl::fmax(am, sycl::permute_group_by_xor(sg, am, 2));
                const float id = q8_inv(am);
                const int k = koff + 128 * g + 8 * lane;
                *reinterpret_cast<sycl::uint2 *>(A.oq + r8 * K8 + k) =
                    sycl::uint2(q8_pack4(acc[8 * g], acc[8 * g + 1], acc[8 * g + 2], acc[8 * g + 3], id),
                                q8_pack4(acc[8 * g + 4], acc[8 * g + 5], acc[8 * g + 6], acc[8 * g + 7], id));
                if ((lane & 3) == 0) A.od[r8 * (K8 / 32) + k / 32] = am / 127.0f;
            }
        }
    });
}

// ================================================================== the lightning indexer at decode widths
// (lightning-indexer.cpp's GGML_SYCL_LI_XMX kernel with its inputs prepared in-kernel): per query b and visible key row
// i < nvis[b]:  score = sum_h relu(q_h . k_i) * w_h * wscale  (+ -inf outside the candidate blocks when cand is set),
// -inf for i >= nvis.  q: 32 heads x 128 dims, RoPE'd (pos) on the last n_dims and FP4-rounded (FP4_POW2_32) here.
constexpr int LIH = 32, LID = 128;
using li_bf16 = sycl::ext::oneapi::bfloat16;

struct LiArgs {
    const float * q = nullptr; size_t q_h = 0, q_b = 0;   // q[b * q_b + h * q_h + d] (the indexer's q_b product, raw)
    const float * tab = nullptr;                           // kv_fin's RoPE table [n][n_dims] (same positions, same angles)
    int n_dims = 0;
    bool fq = true;
    const float * w = nullptr; size_t w_b = 0;            // head weights w[b * w_b + h] (the proj product, raw)
    float wscale = 1.0f;
    const li_bf16 * k = nullptr; int64_t ld_k = 0;         // key rows (bf16, ld_k elements apart), NKV of them
    int64_t NKV = 0;
    const float * nvis = nullptr;                          // [n] visible rows per query
    const int32_t * cand = nullptr; size_t cand_b = 0;     // ncb candidate 8-row blocks per query (null: all blocks)
    int ncb = 0;
    const float * lastblk = nullptr;                       // block max out: + 1e30 at block lastblk[b] (the newest)
    float * o = nullptr; int64_t ld_o = 0;                 // scores [n][NKV]
    float * bmax = nullptr; int64_t ld_bm = 0;             // optional: per 8-row block the max score (+ the pin) [n][NKV / 8]
    int n = 0;
    // scratch (li_scratch_bytes): packed q, scaled weights, candidate bitmaps
    li_bf16 * qp = nullptr;
    float * ws = nullptr;
    uint32_t * bits = nullptr;
};

inline int64_t li_words(int64_t NKV) { return (NKV / 8 + 31) / 32; }

// kernel 1: (query, head, 32-dim block) sub-groups -> q RoPE'd, rounded and packed for the DPAS B tiles (lightning-
// indexer.cpp's li_xmx_pack_q layout), the scaled head weights; then one work-group per query builds its candidate
// bitmap (bit = block id) when cand is set
inline sycl::event li_prep(sycl::queue & q, const LiArgs & a) {
    const LiArgs A = a;
    const int64_t nq_sg = (int64_t) A.n * LIH * (LID / 32);   // q sub-groups
    constexpr int WG = 256;
    const int64_t nq_wg = (nq_sg * 16 + WG - 1) / WG;
    const int64_t nwg = nq_wg + (A.cand ? A.n : 0);
    const int64_t nwords = li_words(A.NKV);
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<uint32_t, 1> lb(sycl::range<1>((size_t) (A.cand ? nwords : 1)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nwg * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t g = (int64_t) it.get_group(0);
            const int l = (int) it.get_local_id(0);
            if (g < nq_wg) {
                const sycl::sub_group sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id();
                const int64_t sgg = g * (WG / 16) + (int64_t) sg.get_group_linear_id();
                if (sgg >= nq_sg) return;   // whole sub-groups
                const int j = (int) (sgg % (LID / 32)), hh = (int) ((sgg / (LID / 32)) % LIH);
                const int64_t b = sgg / ((LID / 32) * LIH);
                const int d = 32 * j + 2 * lane;
                const float * qr = A.q + b * A.q_b + (size_t) hh * A.q_h;
                float x0 = qr[d], x1 = qr[d + 1];
                const int roff = LID - A.n_dims;
                if (d >= roff) rope_tab(A.tab + (size_t) b * A.n_dims, d - roff, true, x0, x1);
                if (A.fq) {
                    const float am = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(x0), sycl::fabs(x1)), sycl::maximum<float>());
                    const float sc = fq4_scale(am);
                    x0 = fq4_apply(x0, sc);
                    x1 = fq4_apply(x1, sc);
                }
                const int nt = hh / 16, nn = hh % 16, ks = d / 16, k2 = (d % 16) / 2;
                li_bf16 * t = A.qp + ((b * 2 + nt) * 8 + ks) * 256;
                t[(k2 * 16 + nn) * 2 + 0] = (li_bf16) x0;
                t[(k2 * 16 + nn) * 2 + 1] = (li_bf16) x1;
                if (j == 0 && lane == 0) A.ws[b * LIH + hh] = A.w[b * A.w_b + hh] * A.wscale;
                return;
            }
            // a query's candidate bitmap
            const int64_t b = g - nq_wg;
            for (int64_t i = l; i < nwords; i += WG) lb[i] = 0u;
            sycl::group_barrier(it.get_group());
            const int64_t nblk = A.NKV / 8;
            for (int i = l; i < A.ncb; i += WG) {
                const int32_t blk = A.cand[b * A.cand_b + i];
                if (blk >= 0 && blk < nblk) {
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space> r(lb[blk / 32]);
                    r.fetch_or(1u << (blk % 32));
                }
            }
            sycl::group_barrier(it.get_group());
            for (int64_t i = l; i < nwords; i += WG) A.bits[b * nwords + i] = lb[i];
        });
    });
}

// kernel 2: the XMX scores (li_xmx_run's tiles: 8 key rows x 32 heads per sub-group), then per row the head sum, the
// visibility (i < nvis: score + 0 - normalising -0 as the mask add did; else -inf) and the candidate mask; with bmax
// the 8-row block's max (+ 0 or + 1e30 for the newest block, as the pin add did)
inline sycl::event li_run(sycl::queue & q, const LiArgs & a) {
    namespace mx = sycl::ext::oneapi::experimental::matrix;
    constexpr int SG = 16, SGPWG = 8, TPS = 4;
    const LiArgs A = a;
    const int64_t ntiles = A.NKV / 8;
    const int64_t nsg = (ntiles + TPS - 1) / TPS;
    const int64_t nwg = (nsg + SGPWG - 1) / SGPWG;
    const int64_t nwords = li_words(A.NKV);
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> scr(sycl::range<1>((size_t) SGPWG * 256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nwg * SGPWG * SG), sycl::range<1>(SGPWG * SG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int sgi = (int) sg.get_group_linear_id();
            const int lane = (int) sg.get_local_linear_id();
            const int64_t gsg = (int64_t) it.get_group(0) * SGPWG + sgi;
            auto sp = scr.get_multi_ptr<sycl::access::decorated::no>() + (size_t) sgi * 256;
            for (int tt = 0; tt < TPS; ++tt) {
                const int64_t tile = gsg * TPS + tt;
                if (tile >= ntiles) break;   // (uniform across the sub-group)
                const int64_t row0 = tile * 8;
                for (int64_t b = 0; b < A.n; ++b) {
                    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> c0, c1;
                    mx::joint_matrix_fill(sg, c0, 0.0f);
                    mx::joint_matrix_fill(sg, c1, 0.0f);
                    for (int ks = 0; ks < 8; ++ks) {
                        mx::joint_matrix<sycl::sub_group, li_bf16, mx::use::a, 8, 16, mx::layout::row_major> ma;
                        mx::joint_matrix<sycl::sub_group, li_bf16, mx::use::b, 16, 16, mx::layout::ext_intel_packed> mb0, mb1;
                        mx::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space,
                                              sycl::access::decorated::no>(A.k + row0 * A.ld_k + ks * 16), A.ld_k);
                        mx::joint_matrix_load(sg, mb0, sycl::address_space_cast<sycl::access::address_space::global_space,
                                              sycl::access::decorated::no>(A.qp + ((b * 2 + 0) * 8 + ks) * 256), 32);
                        mx::joint_matrix_load(sg, mb1, sycl::address_space_cast<sycl::access::address_space::global_space,
                                              sycl::access::decorated::no>(A.qp + ((b * 2 + 1) * 8 + ks) * 256), 32);
                        mx::joint_matrix_mad(sg, c0, ma, mb0, c0);
                        mx::joint_matrix_mad(sg, c1, ma, mb1, c1);
                    }
                    mx::joint_matrix_store(sg, c0, sp, 16, mx::layout::row_major);         // [8 rows][16 heads]
                    mx::joint_matrix_store(sg, c1, sp + 128, 16, mx::layout::row_major);   // heads 16-31
                    sycl::group_barrier(sg);
                    float v = -INFINITY;
                    if (lane < 8) {
                        const float * wb = A.ws + b * LIH;
                        float acc = 0.0f;
                        for (int nn = 0; nn < 16; ++nn) acc += sycl::fmax(sp[lane * 16 + nn], 0.0f) * wb[nn];
                        for (int nn = 0; nn < 16; ++nn) acc += sycl::fmax(sp[128 + lane * 16 + nn], 0.0f) * wb[16 + nn];
                        const int64_t row = row0 + lane;
                        bool on = (float) row < A.nvis[b];   // the mask's 0 / -inf: visible iff nvis - i > 0
                        if (on && A.cand) on = (A.bits[b * nwords + tile / 32] >> (tile % 32)) & 1u;
                        // acc + 0: -0 becomes +0 as the mask add made it (bit test: fast-math may fold a float add of 0)
                        v = on ? sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(acc) == 0x80000000u ? 0u : sycl::bit_cast<uint32_t>(acc))
                               : -INFINITY;
                        A.o[b * A.ld_o + row] = v;
                    }
                    if (A.bmax) {   // pool max over the block's 8 rows (lanes 8-15 carry -inf), then the pin add
                        float m = sycl::reduce_over_group(sg, lane < 8 ? v : -INFINITY, sycl::maximum<float>());
                        if (m == -INFINITY) m = -3.402823466e+38f;   // pool_2d starts at -FLT_MAX
                        if (lane == 0) {
                            const float pin = (float) tile == A.lastblk[b] ? 1e30f : 0.0f;
                            const float r = m + pin;
                            A.bmax[b * A.ld_bm + tile] = sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(r) == 0x80000000u ? 0u : sycl::bit_cast<uint32_t>(r));
                        }
                    }
                    sycl::group_barrier(sg);
                }
            }
        });
    });
}

// ================================================================== the top-k ids, canonicalised
// per query one work-group: its K top-k ids (any order) sorted ascending (bitonic over 512, padded with INT_MAX) ->
// sel, and selm = 0 where the row is visible (id < nvis) else -inf
struct SelArgs {
    const int32_t * ids = nullptr; size_t ids_b = 0;
    const float * nvis = nullptr;
    int32_t * sel = nullptr; size_t sel_b = 0;
    float * selm = nullptr; size_t selm_b = 0;
    int K = 0, n = 0;
};

inline sycl::event sel_fin(sycl::queue & q, const SelArgs & a) {
    constexpr int NS = 512, WG = 256;
    const SelArgs A = a;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int32_t, 1> v(sycl::range<1>(NS), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) A.n * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t b = (int64_t) it.get_group(0);
            const int l = (int) it.get_local_id(0);
            for (int i = l; i < NS; i += WG) v[i] = i < A.K ? A.ids[b * A.ids_b + i] : 0x7fffffff;
            sycl::group_barrier(it.get_group());
            for (int kk = 2; kk <= NS; kk <<= 1) {
                for (int jj = kk >> 1; jj > 0; jj >>= 1) {
                    for (int i = l; i < NS; i += WG) {
                        const int ixj = i ^ jj;
                        if (ixj > i) {
                            const int32_t x = v[i], y = v[ixj];
                            const bool up = (i & kk) == 0;
                            if (up ? x > y : x < y) { v[i] = y; v[ixj] = x; }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }
            const float nv = A.nvis[b];
            for (int i = l; i < A.K; i += WG) {
                const int32_t s = v[i];
                A.sel[b * A.sel_b + i] = s;
                A.selm[b * A.selm_b + i] = (float) s < nv ? 0.0f : -INFINITY;
            }
        });
    });
}

}  // namespace dsv41dec
