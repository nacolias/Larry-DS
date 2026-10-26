#!/usr/bin/env python3
# patch-qkv.py - larry, STRATA_QKV_FUSE (default 0 for the A/B): each layer's attn_q_a and attn_kv (both read xn) as ONE
# [n_embd, 1280 + 512] weight - one mat-vec / GEMM instead of two (as the shared expert's gate+up).  unitrace: q_a
# 16.8 us (321 GB/s) + kv 7.5 us (287 GB/s) a decode layer, 19.4 + 10.7 us a 2-token verify layer - the pair streams
# 7.5 MB, ~16 us at the big mat-vecs' ~480 GB/s, and one launch + dependency gap less.  The loader creates attn_kv
# right after attn_q_a (the file has q_b and q_a_norm between them) so the allocator puts them back to back; the
# fused tensor is their bytes (no copy).  Once it is reordered (STRATA_PREREORDER) the halves' bytes are not usable,
# so every reader takes the fused weight: build_attn and the enc-skip KV graph (which then computes q_a's rows too:
# prompts only, a small GEMM).  Decode (1 column): the same kernel configuration for every row -> the same bits;
# verify widths: kv's rows move from the generic to the row-blocked kernel (another summation order).
# Usage: patch-qkv.py [ds41_dense.cpp]   Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_QKV_FUSE" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


# ---- the loader: attn_kv created right after attn_q_a
rep("""        size_t rq_n = 0, rq_from = 0, rq_to = 0, pq_n = 0;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                w.t[ti.name] = make(ti, mem_type(ti));
""", """        size_t rq_n = 0, rq_from = 0, rq_to = 0, pq_n = 0;
        // larry, STRATA_QKV_FUSE (default 0): each layer's attn_kv created right after its attn_q_a - the allocator then
        // puts them back to back (one [K, Nq + Nkv] weight over both, below)
        const bool qkv_fuse = im.sycl && env_on("STRATA_QKV_FUSE", false);
        std::map<std::string, const TensorInfo *> kv_info;
        if (qkv_fuse)
            for (size_t s = 0; s < model.size(); ++s)
                for (const auto & ti : model.shard(s).tensors())
                    if (ends_with(ti.name, ".attn_kv.weight")) kv_info[ti.name] = &ti;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                if (qkv_fuse && w.t.count(ti.name)) continue;   // an attn_kv created early
                w.t[ti.name] = make(ti, mem_type(ti));
                if (qkv_fuse && ends_with(ti.name, ".attn_q_a.weight")) {
                    const std::string kn = ti.name.substr(0, ti.name.size() - std::strlen("attn_q_a.weight")) + "attn_kv.weight";
                    auto ik = kv_info.find(kn);
                    if (ik != kv_info.end() && !skip(kn)) w.t[kn] = make(*ik->second, mem_type(*ik->second));
                }
""")
# ---- the fused tensors, after the shared expert's pairs
rep("""            for (auto & a : add) w.t[a.first] = a.second;
            std::fprintf(stderr, "shared expert: %zu gate+up pairs as one weight\\n", add.size());
        }
""", """            for (auto & a : add) w.t[a.first] = a.second;
            std::fprintf(stderr, "shared expert: %zu gate+up pairs as one weight\\n", add.size());
        }
        if (qkv_fuse) {   // STRATA_QKV_FUSE: q_a | kv as one weight over their bytes (same type, rows, buffer, adjacent)
            size_t nf = 0;
            for (int64_t il = 0;; ++il) {
                const std::string pre = "blk." + std::to_string(il) + ".";
                auto iq = w.t.find(pre + "attn_q_a.weight"), ik = w.t.find(pre + "attn_kv.weight");
                if (iq == w.t.end() || ik == w.t.end()) break;
                ggml_tensor * q = iq->second, * k = ik->second;
                if (q->type != k->type || q->ne[0] != k->ne[0] || q->ne[2] != 1 || k->ne[2] != 1 || q->buffer != k->buffer ||
                    (const char *) k->data != (const char *) q->data + ggml_nbytes(q)) continue;
                ggml_tensor * f = ggml_new_tensor_2d(w.ctx, q->type, q->ne[0], q->ne[1] + k->ne[1]);
                ggml_set_name(f, (pre + "attn_qkv_a.weight").c_str());
                if (ggml_backend_tensor_alloc(q->buffer, f, q->data) != GGML_STATUS_SUCCESS) continue;
                w.t[f->name] = f;
                ++nf;
            }
            std::fprintf(stderr, "attention: %zu q_a+kv pairs as one weight (STRATA_QKV_FUSE)\\n", nf);
        }
""")
# ---- the reorder list: not the halves of a fused pair
rep("""        if (ends(kv.first, "ffn_gate_shexp.weight") || ends(kv.first, "ffn_up_shexp.weight")) {
            const std::string pre = kv.first.substr(0, kv.first.size() - (ends(kv.first, "ffn_gate_shexp.weight") ? 21 : 19));
            if (p_->w.t.count(pre + "ffn_gate_up_shexp.weight")) continue;
        }
""", """        if (ends(kv.first, "ffn_gate_shexp.weight") || ends(kv.first, "ffn_up_shexp.weight")) {
            const std::string pre = kv.first.substr(0, kv.first.size() - (ends(kv.first, "ffn_gate_shexp.weight") ? 21 : 19));
            if (p_->w.t.count(pre + "ffn_gate_up_shexp.weight")) continue;
        }
        if (ends(kv.first, "attn_q_a.weight") || ends(kv.first, "attn_kv.weight")) {   // STRATA_QKV_FUSE: the pair as one
            const std::string pre = kv.first.substr(0, kv.first.size() - (ends(kv.first, "attn_q_a.weight") ? 15 : 14));
            if (p_->w.t.count(pre + "attn_qkv_a.weight")) continue;
        }
""")
# ---- build_attn
rep("""    // ---- q (no per-head norm in V4.1)
    ggml_tensor * qr = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_q_a.weight"), xn), im.BL(il, "attn_q_a_norm.weight"));
""", """    // ---- q (no per-head norm in V4.1).  larry, STRATA_QKV_FUSE: q_a and kv as one product (rows: q_a, then kv)
    ggml_tensor * qa_y = nullptr, * kv_y = nullptr;
    if (im.w.has("blk." + std::to_string(il) + ".attn_qkv_a.weight")) {
        ggml_tensor * wf = im.BL(il, "attn_qkv_a.weight");
        const int64_t nq = im.BL(il, "attn_q_a.weight")->ne[1], nk = wf->ne[1] - nq;
        ggml_tensor * y = ggml_mul_mat(gc, wf, xn);                                             // [nq + nk, n]
        qa_y = ggml_view_2d(gc, y, nq, n, y->nb[1], 0);
        kv_y = ggml_view_2d(gc, y, nk, n, y->nb[1], (size_t) nq * ggml_element_size(y));
    } else {
        qa_y = ggml_mul_mat(gc, im.BL(il, "attn_q_a.weight"), xn);
        kv_y = ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn);
    }
    ggml_tensor * qr = b.rms_w(qa_y, im.BL(il, "attn_q_a_norm.weight"));
""")
rep("""    // ---- window K (= V): [ring | this pass's rows] -> each query's SWA rows; the ring keeps the last RAW
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
""", """    // ---- window K (= V): [ring | this pass's rows] -> each query's SWA rows; the ring keeps the last RAW
    ggml_tensor * kv = b.rms_w(kv_y, im.BL(il, "attn_kv_a_norm.weight"));
""")
# ---- the enc-skip KV graph
rep("""    // ---- window K (= V): the same post-RoPE raw row build_attn writes into the ring
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
""", """    // ---- window K (= V): the same post-RoPE raw row build_attn writes into the ring.  STRATA_QKV_FUSE: through the fused
    // weight (attn_kv alone is not usable once the pair is reordered) - its q_a rows are computed and dropped
    ggml_tensor * kv_y = nullptr;
    if (im.w.has("blk." + std::to_string(il) + ".attn_qkv_a.weight")) {
        ggml_tensor * wf = im.BL(il, "attn_qkv_a.weight");
        const int64_t nq = im.BL(il, "attn_q_a.weight")->ne[1], nk = wf->ne[1] - nq;
        ggml_tensor * y = ggml_mul_mat(gc, wf, xn);
        kv_y = ggml_view_2d(gc, y, nk, n, y->nb[1], (size_t) nq * ggml_element_size(y));
    } else {
        kv_y = ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn);
    }
    ggml_tensor * kv = b.rms_w(kv_y, im.BL(il, "attn_kv_a_norm.weight"));
""")
open(P, "w").write(s)
print("patched")
