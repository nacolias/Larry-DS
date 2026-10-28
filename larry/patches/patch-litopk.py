#!/usr/bin/env python3
# patch-litopk.py [XOLOTL_DIR] - larry, 2026-10-28 (optimization #5, long contexts): the prompt indexer's scores and their
# top-k as ONE op, its queries shared with the expert cards.  A 967K prompt spends ~86 of its ~347 s of card-0 time in the
# indexer of layers 2 / 8 / 14 (every token reads every earlier compressed row: O(n^2)) and its top-k, while cards 1-3 sit
# mostly idle at that depth.  Here:
#  - ggml_lightning_indexer_topk(q, k, w, nvis, top_k) (GGML_OP_LIGHTNING_INDEXER, op_params[0] = 2, [1] = top_k): I32
#    [top_k, n], each query's top_k best visible rows (no particular order, as ggml_top_k); ggml-sycl only, BF16 keys in
#    rows of 128.  It scores a slice of queries at a time into scratch (GGML_SYCL_LI_TOPK_MIB, 512) with mode 2's kernel
#    (patch-lifbf16.py) and takes each slice's top-k there (ggml_sycl_top_k_rows: the two stages of patch-topk2.py when
#    GGML_SYCL_TOPK2) - the [cap, n] score tensor never exists in the graph.
#  - ggml_backend_sycl_li_remote_prepare(nkv, layers, nq) (before a long prompt's first batch: the cards' buffers sized for
#    its end) / _remote(on) / _remote_release() (after it): a call of >= GGML_SYCL_LI_REMOTE_MIN_KEYS keys (65536) and >= 64
#    queries keeps a share of its queries on card 0 and hands the rest to the other cards of card 0's context.  Card 0
#    stages what they lack in a plain buffer of its own - the keys' rows they have not seen yet (each card keeps a copy of
#    every keys tensor of the prompt) and their queries / weights / visible rows; card 0's KV pages and ggml's pool are never
#    read by the peers - each card pulls its part, scores and ranks it on a queue of its own, and card 0 pulls the ids back
#    after its own share: in queue order, no host wait (a key catch-up past the stage goes first, chunk by chunk).  The
#    other cards keep their key copies in fp16 (converted as the rows arrive: mode 1's kernel, 6 % faster than bf16's at
#    512K keys); card 0 keeps GGML_SYCL_LI_REMOTE_SHARE % of a call's queries (25).
# GGML_SYCL_LI_TOPK_CHECK=N (debug): the first N shared calls checked against card 0's scores of every query.
# Needs patch-lifbf16.py and patch-topk2.py first.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
G = os.path.join(X, "third_party/llama.cpp/ggml")


def edit(path, pairs, marker):
    s = open(path).read()
    if marker in s:
        print("already patched", path)
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit("%s: anchor x%d: %s" % (path, s.count(old), old[:70]))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


# ---- ggml.h / ggml.c: the constructor
edit(os.path.join(G, "include/ggml.h"), [(
"""    GGML_API struct ggml_tensor * ggml_lightning_indexer_nvis(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * nvis);
""",
"""    GGML_API struct ggml_tensor * ggml_lightning_indexer_nvis(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * nvis);

    // larry (patch-litopk.py): ggml_lightning_indexer_nvis's scores straight into their top-k (never a tensor): I32
    // [top_k, n_batch], every query's top_k best visible rows in no particular order (as ggml_top_k).  rows_final: the
    // rows of k already written for good when it runs (>= every nvis; the other cards copy no further).  ggml-sycl only
    // (op_params[0] = 2, op_params[1] = top_k, op_params[2] = rows_final; k BF16 rows of 128).
    GGML_API struct ggml_tensor * ggml_lightning_indexer_topk(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * nvis,
        int                   top_k,
        int                   rows_final);
""")], "ggml_lightning_indexer_topk")

edit(os.path.join(G, "src/ggml.c"), [(
"""// larry: ggml_lightning_indexer with nvis F32 [n_batch] (rows i < nvis[t] visible to query t) for the mask
""",
"""// larry (patch-litopk.py): ggml_lightning_indexer_nvis's scores straight into their top-k: I32 [top_k, n_batch]
struct ggml_tensor * ggml_lightning_indexer_topk(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * nvis,
        int                   top_k,
        int                   rows_final) {

    GGML_ASSERT(       q->type == GGML_TYPE_F32);
    GGML_ASSERT( weights->type == GGML_TYPE_F32);
    GGML_ASSERT(    nvis->type == GGML_TYPE_F32);
    GGML_ASSERT(      q->ne[0] == k->ne[0]);
    GGML_ASSERT(      q->ne[1] == weights->ne[0]);
    GGML_ASSERT(      k->ne[1] == 1);
    GGML_ASSERT(      q->ne[2] == weights->ne[1]);
    GGML_ASSERT(ggml_nelements(nvis) == q->ne[2]);
    GGML_ASSERT(ggml_is_contiguous(nvis));
    GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && weights->ne[3] == 1);
    GGML_ASSERT(top_k > 0 && top_k <= k->ne[2]);

    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, q->ne[2]);

    int32_t params[3] = { 2, top_k, rows_final };
    ggml_set_op_params(result, params, sizeof(params));

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = nvis;

    return result;
}

// larry: ggml_lightning_indexer with nvis F32 [n_batch] (rows i < nvis[t] visible to query t) for the mask
""")], "ggml_lightning_indexer_topk")

# ---- ggml-sycl.h: the remote share's switches
edit(os.path.join(G, "include/ggml-sycl.h"), [(
"""GGML_BACKEND_API void ggml_backend_sycl_larry_f16cache_release(void);
""",
"""GGML_BACKEND_API void ggml_backend_sycl_larry_f16cache_release(void);
// larry (patch-litopk.py): the prompt indexer's top-k op (ggml_lightning_indexer_topk) shares its calls' queries with the
// other cards of card 0's context.  prepare: card 0's queue (the dense backend's: ggml_backend_sycl_get_queue), their buffers
// for calls of up to nkv keys, `layers` keys tensors and nq queries a card (before a long prompt's first batch; false: no
// room - nothing shared); on: share (1) or not (0); release: their buffers freed (after the prompt).
GGML_BACKEND_API bool ggml_backend_sycl_li_remote_prepare(void * queue0, int64_t nkv, int layers, int64_t nq);
GGML_BACKEND_API void ggml_backend_sycl_li_remote(int on);
GGML_BACKEND_API void ggml_backend_sycl_li_remote_release(void);
""")], "ggml_backend_sycl_li_remote")

# ---- ggml-sycl.cpp: supports_op
edit(os.path.join(G, "src/ggml-sycl/ggml-sycl.cpp"), [(
"""        case GGML_OP_LIGHTNING_INDEXER:
            return op->src[0]->type == GGML_TYPE_F32 &&""",
"""        case GGML_OP_LIGHTNING_INDEXER:
            if (op->op_params[0] == 2)   // larry (patch-litopk.py): the top-k mode
                return op->type == GGML_TYPE_I32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_BF16 &&
                       op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 128 &&
                       op->src[0]->ne[1] == 32;
            return op->src[0]->type == GGML_TYPE_F32 &&""")], "the top-k mode")

# ---- topk-radix: a queue-based entry for other files / other cards
edit(os.path.join(G, "src/ggml-sycl/topk-radix.hpp"), [(
"""void ggml_sycl_top_k_radix(
""",
"""// larry (patch-litopk.py): the same on any queue of the context (rows of ncols >= 1; the two stages of patch-topk2.py
// with GGML_SYCL_TOPK2 where they pay, else the one-group radix select); block_size = the device's largest work-group
void ggml_sycl_top_k_rows(sycl::queue & q, int block_size, const float * src, int32_t * dst, int64_t ncols, int64_t nrows, int k);

void ggml_sycl_top_k_radix(
""")], "ggml_sycl_top_k_rows")
edit(os.path.join(G, "src/ggml-sycl/topk-radix.cpp"), [(
"""void ggml_sycl_top_k_radix(
    ggml_backend_sycl_context & ctx,
""",
"""// larry (patch-litopk.py)
void ggml_sycl_top_k_rows(sycl::queue & q, int block_size, const float * src, int32_t * dst, int64_t ncols, int64_t nrows, int k) {
    static const bool two = std::getenv("GGML_SYCL_TOPK2") && std::atoi(std::getenv("GGML_SYCL_TOPK2")) != 0;
    if (two && top_k2_shape_ok(ncols, k)) {
        top_k_radix2_f32_sycl(q, block_size, src, dst, ncols, nrows, k);
        return;
    }
    GGML_ASSERT(ncols <= INT32_MAX && block_size >= SYCL_TOP_K_RADIX_BUCKETS);
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_TOP_K_RADIX_SLM_WORDS), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nrows * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) {
                const int64_t row = item_ct1.get_group(0);
                top_k_radix_select_f32(src + row * ncols, dst + row * k, (int) ncols, k,
                                       slm.get_multi_ptr<sycl::access::decorated::no>().get(), item_ct1);
            });
    });
}

void ggml_sycl_top_k_radix(
    ggml_backend_sycl_context & ctx,
""")], "ggml_sycl_top_k_rows")

# ---- lightning-indexer.cpp: the top-k mode and the remote share
LIT = r'''
// ---- larry (patch-litopk.py): the top-k mode - ggml_lightning_indexer_topk (op_params[0] = 2, [1] = k).  Mode 2's
// scores (the BF16 keys as stored) a slice of queries at a time into scratch (<= GGML_SYCL_LI_TOPK_MIB, 512), each slice's
// top-k taken there.  With ggml_backend_sycl_li_remote(1) a call of >= GGML_SYCL_LI_REMOTE_MIN_KEYS keys (65536) and >= 64
// queries keeps one share of its queries (1 / (1 + the other cards)) and hands the others to the other cards of card 0's
// context: card 0 stages the keys' rows they lack and their queries / weights / visible rows in a plain buffer of its own
// (card 0's KV pages and ggml's pool are never read by a peer), each card pulls its part, scores and ranks it on its own
// queue, and card 0 pulls their ids after its own share - all in queue order, no host wait.
#include "topk-radix.hpp"
#include "ggml-sycl.h"

namespace {
struct LirKeys {   // a card's copy of one keys tensor of the prompt: rows [0, have), in fp16 (the faster kernel at depth)
    const void * src = nullptr;
    sycl::half * dev = nullptr;
    int64_t have = 0;
};
struct LirCard {
    std::unique_ptr<sycl::queue> q;
    int bsz = 1024;
    std::vector<LirKeys> keys;
    sycl::half * kpool = nullptr;    // [layers][nkv][128] - the keys' copies, a layer each (fp16)
    li_bf16 * kin = nullptr;         // rows as they arrive (bf16, a stage-full), converted into kpool
    li_bf16 * qb = nullptr;          // its queries as they arrive, bf16 [nq][32][128]
    sycl::half * qh = nullptr;       // the same in fp16
    float * w = nullptr;             // their weights [nq][32]
    float * nv = nullptr;            // their visible rows [nq]
    float * sc = nullptr;            // their scores [nq][nkv]
    int32_t * sel = nullptr;         // their ids [nq][k <= 512]
    sycl::event done;
};
struct Lir {
    int on = 0;
    bool ready = false;
    int64_t nkv = 0, nq = 0;
    int layers = 0;
    sycl::context * ctx = nullptr;   // card 0's
    std::vector<LirCard> cards;
    uint8_t * stage = nullptr;       // card 0, plain device memory: what the others pull
    size_t stage_cap = 0;
};
Lir g_lir;
sycl::queue * g_lir_q0 = nullptr;   // card 0's queue (the dense backend's), set at the first top-k call

constexpr size_t kLirKeyChunk = (size_t) 32 << 20;   // a key catch-up a stage-full at a time
constexpr int kLirMaxK = 512;

void lir_free() {
    for (LirCard & c : g_lir.cards) {
        c.q->wait();
        for (void * p : { (void *) c.kpool, (void *) c.kin, (void *) c.qb, (void *) c.qh, (void *) c.w, (void *) c.nv, (void *) c.sc,
                          (void *) c.sel })
            if (p) sycl::free(p, *c.q);
        c.kpool = nullptr; c.kin = nullptr; c.qb = nullptr; c.qh = nullptr; c.w = nullptr; c.nv = nullptr; c.sc = nullptr; c.sel = nullptr;
        c.keys.clear();
    }
    if (g_lir.stage && g_lir_q0) { g_lir_q0->wait(); sycl::free(g_lir.stage, *g_lir_q0); }
    g_lir.stage = nullptr;
    g_lir.stage_cap = 0;
    g_lir.ready = false;
}

// the other cards of card 0's context, a queue each (once)
bool lir_cards(sycl::queue & q0) {
    if (!g_lir.cards.empty()) return true;
    const sycl::context ctx = q0.get_context();
    for (const sycl::device & d : ctx.get_devices()) {
        if (d == q0.get_device() || !d.is_gpu()) continue;
        LirCard c;
        // GGML_SYCL_LI_REMOTE_PRIO=1: its queue at low priority (the card's own work - the prompt's expert calls - first);
        // measured at 984K worse: card 0 then waited longer for the shares (322.3 s against 313.2 at the default 0)
        static const int prio = [] { const char * e = std::getenv("GGML_SYCL_LI_REMOTE_PRIO"); return e && *e ? std::atoi(e) : 0; }();
        if (prio) c.q = std::make_unique<sycl::queue>(ctx, d, sycl::property_list{ sycl::property::queue::in_order(),
                                                                                    sycl::ext::oneapi::property::queue::priority_low() });
        else c.q = std::make_unique<sycl::queue>(ctx, d, sycl::property::queue::in_order());
        c.bsz = (int) d.get_info<sycl::info::device::max_work_group_size>();
        g_lir.cards.push_back(std::move(c));
    }
    return !g_lir.cards.empty();
}

int lir_min_keys() { static const int v = [] { const char * e = std::getenv("GGML_SYCL_LI_REMOTE_MIN_KEYS"); return e && *e ? std::atoi(e) : 65536; }(); return v; }

// queries [n0, n) of the call to the other cards (false: not taken - no room, a shape it was not prepared for)
// (the keys' rows past `fin` are not written yet - the cards copy rows [have, fin) only; every query sees fewer)
bool lir_begin(sycl::queue & q0, const li_bf16 * kd, int64_t NKV, int64_t fin, const li_bf16 * qb, const float * w, const float * nv,
               int64_t B, int K, int qw, int64_t KR, int64_t & n0, std::vector<int64_t> & bnd) {
    const int nc = (int) g_lir.cards.size();
    if (!g_lir.ready || NKV > g_lir.nkv || K > kLirMaxK) return false;
    // card 0's share: GGML_SYCL_LI_REMOTE_SHARE % of the queries (default 25); the rest evenly over the others
    static const int pct = [] { const char * e = std::getenv("GGML_SYCL_LI_REMOTE_SHARE"); return e && *e ? std::atoi(e) : 25; }();
    const int64_t share = std::max<int64_t>(0, std::min<int64_t>(B - nc, (B * pct + 50) / 100));
    if (B - share > (int64_t) nc * g_lir.nq) return false;
    n0 = share;
    bnd.assign((size_t) nc + 1, 0);
    for (int c = 0; c <= nc; ++c) bnd[(size_t) c] = n0 + (B - n0) * c / nc;
    // the keys tensor's copy on every card (the same rows on each)
    int64_t have = fin;
    for (LirCard & c : g_lir.cards) {
        LirKeys * e = nullptr;
        for (LirKeys & k : c.keys) if (k.src == kd) e = &k;
        if (!e) {
            if ((int) c.keys.size() >= g_lir.layers) return false;
            LirKeys k;
            k.src = kd;
            k.dev = c.kpool + (size_t) c.keys.size() * (size_t) g_lir.nkv * 128;
            c.keys.push_back(k);
            e = &c.keys.back();
        }
        have = std::min(have, e->have);
    }
    // the stage: [the keys' new rows][queries n0..B][weights n0..B][visible rows n0..B] - rows that do not fit beside the
    // queries go first, a stage at a time (a prompt that starts deep: its first call per keys tensor), the host waiting
    const size_t qbytes = (size_t) (B - n0) * 8192, wbytes = (size_t) (B - n0) * 128, vbytes = (size_t) (B - n0) * 4;
    const size_t tail = ((qbytes + 255) & ~(size_t) 255) + ((wbytes + 255) & ~(size_t) 255) + vbytes + 256;
    if (tail + 256 > g_lir.stage_cap) return false;
    const int64_t fit = (int64_t) ((g_lir.stage_cap - tail) / 256), full = (int64_t) (g_lir.stage_cap / 256);
    while (fin - have > fit) {
        const int64_t nr = std::min<int64_t>(full, fin - have);
        q0.memcpy(g_lir.stage, kd + have * 128, (size_t) nr * 256);
        const sycl::event e0 = q0.ext_oneapi_submit_barrier();
        for (LirCard & c : g_lir.cards) {
            LirKeys * k = nullptr;
            for (LirKeys & x : c.keys) if (x.src == kd) k = &x;
            c.q->ext_oneapi_submit_barrier({ e0 });
            c.q->memcpy(c.kin, g_lir.stage, (size_t) nr * 256);
            sycl::half * kdst = k->dev + have * 128;
            const li_bf16 * kin = c.kin;
            c.q->parallel_for(sycl::range<1>((size_t) nr * 128), [=](sycl::id<1> i) { kdst[i[0]] = (sycl::half) (float) kin[i[0]]; });
        }
        for (LirCard & c : g_lir.cards) {
            c.q->wait();
            for (LirKeys & x : c.keys) if (x.src == kd) x.have = have + nr;
        }
        have += nr;
    }
    const size_t kb = (size_t) std::max<int64_t>(0, fin - have) * 256;
    const size_t qo = (kb + 255) & ~(size_t) 255, wo = qo + ((qbytes + 255) & ~(size_t) 255), vo = wo + ((wbytes + 255) & ~(size_t) 255);
    if (kb) q0.memcpy(g_lir.stage, kd + have * 128, kb);
    q0.memcpy(g_lir.stage + qo, qb + n0 * 4096, qbytes);
    q0.memcpy(g_lir.stage + wo, w + n0 * 32, wbytes);
    q0.memcpy(g_lir.stage + vo, nv + n0, vbytes);
    const sycl::event es = q0.ext_oneapi_submit_barrier();
    for (int ci = 0; ci < nc; ++ci) {
        LirCard & c = g_lir.cards[(size_t) ci];
        LirKeys * k = nullptr;
        for (LirKeys & x : c.keys) if (x.src == kd) k = &x;
        const int64_t b0 = bnd[(size_t) ci], nq = bnd[(size_t) ci + 1] - b0, off = b0 - n0;
        c.q->ext_oneapi_submit_barrier({ es });
        if (kb) {
            c.q->memcpy(c.kin, g_lir.stage, kb);
            sycl::half * kdst = k->dev + have * 128;
            const li_bf16 * kin = c.kin;
            c.q->parallel_for(sycl::range<1>(kb / 2), [=](sycl::id<1> i) { kdst[i[0]] = (sycl::half) (float) kin[i[0]]; });
        }
        c.q->memcpy(c.qb, g_lir.stage + qo + (size_t) off * 8192, (size_t) nq * 8192);
        c.q->memcpy(c.w, g_lir.stage + wo + (size_t) off * 128, (size_t) nq * 128);
        c.q->memcpy(c.nv, g_lir.stage + vo + (size_t) off * 4, (size_t) nq * 4);
        if (nq > 0) {
            const li_bf16 * qin = c.qb;
            sycl::half * qh = c.qh;
            c.q->parallel_for(sycl::range<1>((size_t) nq * 4096), [=](sycl::id<1> i) { qh[i[0]] = (sycl::half) (float) qin[i[0]]; });
            lif_run_t<sycl::half>(c.qh, k->dev, 128, NKV, c.w, 32, nullptr, 0, c.nv, c.sc, NKV, nq, qw, KR, c.q.get());
            ggml_sycl_top_k_rows(*c.q, c.bsz, c.sc, c.sel, NKV, nq, K);
        }
        c.done = c.q->ext_oneapi_submit_barrier();
        k->have = std::max(k->have, fin);
    }
    return true;
}

void lir_end(sycl::queue & q0, int32_t * out, int K, const std::vector<int64_t> & bnd) {
    std::vector<sycl::event> ev;
    for (LirCard & c : g_lir.cards) ev.push_back(c.done);
    q0.ext_oneapi_submit_barrier(ev);
    for (size_t ci = 0; ci < g_lir.cards.size(); ++ci) {
        const int64_t b0 = bnd[ci], nq = bnd[ci + 1] - b0;
        if (nq > 0) q0.memcpy(out + b0 * K, g_lir.cards[ci].sel, (size_t) nq * K * 4);
    }
}
}  // namespace

bool ggml_backend_sycl_li_remote_prepare(void * queue0, int64_t nkv, int layers, int64_t nq) {
    if (queue0) g_lir_q0 = (sycl::queue *) queue0;   // card 0's: the dense backend's
    if (!g_lir_q0) return false;
    sycl::queue & q0 = *g_lir_q0;
    if (!lir_cards(q0)) return false;
    nkv = (nkv + 31) / 32 * 32;
    if (g_lir.ready && nkv <= g_lir.nkv && layers <= g_lir.layers && nq <= g_lir.nq) return true;
    lir_free();
    g_lir.nkv = nkv; g_lir.layers = layers; g_lir.nq = nq;
    // the stage: a key chunk, or one call's new rows + the others' queries (nc * nq of them), whichever is larger
    const size_t call = (size_t) 2 * 1024 * 256 + (size_t) g_lir.cards.size() * (size_t) nq * (8192 + 128 + 4) + 4096;
    g_lir.stage_cap = std::max(kLirKeyChunk, call);
    bool ok = true;
    for (LirCard & c : g_lir.cards) {
        c.kpool = sycl::malloc_device<sycl::half>((size_t) layers * (size_t) nkv * 128, *c.q);
        c.kin = sycl::malloc_device<li_bf16>(g_lir.stage_cap / 2, *c.q);
        c.qb = sycl::malloc_device<li_bf16>((size_t) nq * 4096, *c.q);
        c.qh = sycl::malloc_device<sycl::half>((size_t) nq * 4096, *c.q);
        c.w = sycl::malloc_device<float>((size_t) nq * 32, *c.q);
        c.nv = sycl::malloc_device<float>((size_t) nq, *c.q);
        c.sc = sycl::malloc_device<float>((size_t) nq * (size_t) nkv, *c.q);
        c.sel = sycl::malloc_device<int32_t>((size_t) nq * kLirMaxK, *c.q);
        ok = ok && c.kpool && c.kin && c.qb && c.qh && c.w && c.nv && c.sc && c.sel;
    }
    g_lir.stage = sycl::malloc_device<uint8_t>(g_lir.stage_cap, q0);
    ok = ok && g_lir.stage;
    if (!ok) { lir_free(); GGML_LOG_INFO("larry li remote: no room for %lld keys x %d layers, %lld queries a card\n", (long long) nkv, layers, (long long) nq); return false; }
    g_lir.ready = true;
    GGML_LOG_INFO("larry li remote: %zu cards, %lld keys x %d layers, %lld queries a card (%.0f MiB a card)\n", g_lir.cards.size(),
                  (long long) nkv, layers, (long long) nq,
                  ((double) layers * nkv * 256 + (double) g_lir.stage_cap + (double) nq * (16384 + 132 + 4.0 * nkv + 4.0 * kLirMaxK)) / 1048576.0);
    return true;
}

void ggml_backend_sycl_li_remote(int on) { g_lir.on = on; }

void ggml_backend_sycl_li_remote_release(void) {
    g_lir.on = 0;
    if (!g_lir.cards.empty() || g_lir.stage) lir_free();
}

static void li_topk(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * nvt = dst->src[3];
    const int K = dst->op_params[1];
    const int64_t B = q->ne[2], NKV = k->ne[2];
    const size_t nbq1 = q->nb[1], nbq2 = q->nb[2];
    GGML_ASSERT(k->type == GGML_TYPE_BF16 && k->nb[2] == 256 && NKV % 32 == 0 && ((uintptr_t) k->data) % 64 == 0);
    GGML_ASSERT(w->nb[1] == 32 * sizeof(float) && dst->nb[1] == (size_t) K * sizeof(int32_t));
    queue_ptr stream = ctx.stream();
    ggml_sycl_pool & pool = ctx.pool();
    if (!g_lir_q0) g_lir_q0 = stream;
    ggml_sycl_pool_alloc<li_bf16> Qb(pool, (size_t) B * 4096);
    li_bf16 * qb = Qb.get();
    const char * qd = (const char *) q->data;
    stream->parallel_for(sycl::range<1>((size_t) B * 4096), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int64_t e = (int64_t) (i % 128), hh = (int64_t) ((i / 128) % 32), b = (int64_t) (i / 4096);
        qb[i] = (li_bf16) *(const float *) (qd + b * nbq2 + hh * nbq1 + e * sizeof(float));
    });
    static const int qw_env = [] { const char * v = std::getenv("GGML_SYCL_LI_FUSED_QW"); return v && *v ? std::atoi(v) : 0; }();
    static const int64_t kr_env = [] { const char * v = std::getenv("GGML_SYCL_LI_FUSED_KR"); return v && *v ? std::atoll(v) : 0; }();
    const int qw = qw_env > 0 ? qw_env : NKV > 131072 ? 16 : 8;
    const int64_t KR = kr_env > 0 ? (kr_env + 31) / 32 * 32 : NKV <= 16384 ? 128 : NKV <= 65536 ? 256 : 512;
    const li_bf16 * kd = (const li_bf16 *) k->data;
    const float * wd = (const float *) w->data;
    const float * nv = (const float *) nvt->data;
    int32_t * out = (int32_t *) dst->data;
    int64_t n0 = B;
    std::vector<int64_t> bnd;
    if (g_lir.on && g_lir.ready && B >= 64 && NKV >= lir_min_keys() && stream == g_lir_q0)
        if (!lir_begin(*stream, kd, NKV, std::min<int64_t>(NKV, dst->op_params[2] > 0 ? dst->op_params[2] : NKV), qb, wd, nv, B, K, qw,
                       KR, n0, bnd)) n0 = B;
    static const int64_t mib = [] { const char * v = std::getenv("GGML_SYCL_LI_TOPK_MIB"); return v && *v ? std::atoll(v) : 512; }();
    const int64_t slice = std::max<int64_t>(16, std::min<int64_t>(std::max<int64_t>(n0, 1), (mib << 20) / (NKV * 4)));
    ggml_sycl_pool_alloc<float> S(pool, (size_t) slice * (size_t) NKV);
    const int bsz0 = ggml_sycl_info().max_work_group_sizes[ctx.device];
    for (int64_t b0 = 0; b0 < n0; b0 += slice) {
        const int64_t nq = std::min(slice, n0 - b0);
        lif_run_t<li_bf16>(qb + b0 * 4096, kd, 128, NKV, wd + b0 * 32, 32, nullptr, 0, nv + b0, S.get(), NKV, nq, qw, KR, stream);
        ggml_sycl_top_k_rows(*stream, bsz0, S.get(), out + b0 * K, NKV, nq, K);
    }
    if (n0 < B) lir_end(*stream, out, K, bnd);
    // debug, GGML_SYCL_LI_TOPK_CHECK=N: the first N calls of <= GGML_SYCL_LI_TOPK_CHECK_KEYS keys (131072) that shared
    // their queries (or every call when sharing is off) against card 0's scores of every query: a query's ids must be
    // distinct, in range, and a top-k of its row - all at or above its k-th score, as many above it as the row has
    static const int chk = [] { const char * v = std::getenv("GGML_SYCL_LI_TOPK_CHECK"); return v && *v ? std::atoi(v) : 0; }();
    static const int64_t chk_keys = [] { const char * v = std::getenv("GGML_SYCL_LI_TOPK_CHECK_KEYS"); return v && *v ? std::atoll(v) : 131072; }();
    static int nchk = 0;
    if (chk && nchk < chk && NKV <= chk_keys && (n0 < B || !g_lir.on)) {
        ++nchk;
        ggml_sycl_pool_alloc<float> SC(pool, (size_t) B * (size_t) NKV);
        lif_run_t<li_bf16>(qb, kd, 128, NKV, wd, 32, nullptr, 0, nv, SC.get(), NKV, B, qw, KR, stream);
        std::vector<int32_t> ids((size_t) B * K);
        std::vector<float> sc((size_t) B * (size_t) NKV);
        stream->memcpy(ids.data(), out, ids.size() * 4).wait();
        stream->memcpy(sc.data(), SC.get(), sc.size() * 4).wait();
        int64_t bad = 0;
        std::vector<float> v;
        std::vector<int32_t> s;
        for (int64_t b = 0; b < B; ++b) {
            const float * row = sc.data() + b * NKV;
            const int32_t * id = ids.data() + b * K;
            v.assign(row, row + NKV);
            std::nth_element(v.begin(), v.begin() + (K - 1), v.end(), std::greater<float>());
            const float kth = v[(size_t) K - 1];
            s.assign(id, id + K);
            std::sort(s.begin(), s.end());
            bool ok = s.front() >= 0 && s.back() < NKV && std::adjacent_find(s.begin(), s.end()) == s.end();
            int64_t gt_row = 0, gt_ids = 0, ge_ids = 0;
            for (int64_t j = 0; j < NKV; ++j) gt_row += row[j] > kth;
            for (int i = 0; ok && i < K; ++i) { gt_ids += row[s[(size_t) i]] > kth; ge_ids += row[s[(size_t) i]] >= kth; }
            ok = ok && ge_ids == K && gt_ids == gt_row;
            bad += !ok;
        }
        GGML_LOG_INFO("larry li_topk check %d: %lld queries x %lld keys (%lld on card 0, the rest on %zu other cards): %lld queries' ids wrong\n",
                      nchk, (long long) B, (long long) NKV, (long long) n0, n0 < B ? g_lir.cards.size() : (size_t) 0, (long long) bad);
    }
}

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    if (dst->op_params[0] == 2) { li_topk(ctx, dst); return; }   // larry (patch-litopk.py): the top-k mode
'''
edit(os.path.join(G, "src/ggml-sycl/lightning-indexer.cpp"), [(
"""
void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
""", LIT)], "li_topk")
