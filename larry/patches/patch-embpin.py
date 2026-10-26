#!/usr/bin/env python3
# patch-embpin.py [DENSE_CPP] - larry, STRATA_EMB_PIN=1 (default 0 until measured): a prompt batch's embeddings without
# the host conversion.  begin_batch spent 0.4-0.6 s converting a batch's rows f16 -> f32 on one thread (random rows of
# the 1.3 GB table on the mmap: page faults, the page cache under pressure) and 0.09-0.18 s more uploading the 168-335 MB
# result from pageable memory (STRATA_DENSE_PROF, 2026-10-25).  Now: the table copied once at init into pinned host
# memory, a batch's rows gathered as they are (f16, 8 threads for big batches) into a pinned stage and uploaded async,
# i_emb f16 (half its VRAM), the init graph casts it (f16 -> f32 is exact: the same values).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_EMB_PIN" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


if "#include <thread>" not in s:
    rep("#include <vector>\n", "#include <thread>\n#include <vector>\n")
rep("""    std::vector<float> host_logits, host_emb;
""", """    std::vector<float> host_logits, host_emb;
    // larry, STRATA_EMB_PIN: the token table pinned (f16 rows), a batch's rows staged pinned, i_emb f16
    bool emb_f16 = false;
    HostRows<uint16_t> emb_pin, emb_stage;
    ggml_backend_event_t emb_ev = nullptr;
""")
rep("""    im.i_emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
""", """    im.i_emb = ggml_new_tensor_2d(c, im.emb_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, D, NC);   // larry, STRATA_EMB_PIN: f16
""")
rep("""    if (!load_weights(im, err)) return false;
""", """    if (!load_weights(im, err)) return false;
    im.emb_f16 = im.sycl && env_on("STRATA_EMB_PIN", false);   // larry (before the columns are allocated)
""")
rep("""    im.host_ids.resize((size_t) (im.NU * NCM), pin_be);
    im.host_wts.resize((size_t) (im.NU * NCM), pin_be);
""", """    im.host_ids.resize((size_t) (im.NU * NCM), pin_be);
    im.host_wts.resize((size_t) (im.NU * NCM), pin_be);
    if (im.emb_f16) {   // larry, STRATA_EMB_PIN: the table into pinned memory once (8 threads), the batch stage
        const double te0 = dprof_now();
        const size_t nrow = (size_t) im.g.vocab, rowb = (size_t) D * 2;
        im.emb_pin.resize(nrow * (size_t) D, pin_be);
        im.emb_stage.resize((size_t) D * (size_t) NCM, pin_be);
        if (im.emb_pin.buf && im.emb_stage.buf) {
            std::vector<std::thread> th;
            for (int k = 0; k < 8; ++k)
                th.emplace_back([&, k] {
                    const size_t r0 = nrow * (size_t) k / 8, r1 = nrow * (size_t) (k + 1) / 8;
                    std::memcpy(im.emb_pin.data() + r0 * (size_t) D, im.embd + r0 * rowb, (r1 - r0) * rowb);
                });
            for (auto & t : th) t.join();
            std::fprintf(stderr, "ds41_dense: token table pinned (%.2f GB, %.0f ms), batch embeddings staged f16\\n",
                         nrow * rowb / 1e9, dprof_now() - te0);
        } else {
            std::fprintf(stderr, "ds41_dense: STRATA_EMB_PIN: pinned allocation failed - embeddings from the mmap (f16)\\n");
            im.emb_pin.resize(0);
        }
    }
""")
rep("""    im.host_emb.resize((size_t) (D * n));
    for (int t = 0; t < n; ++t) {
        if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
        const uint16_t * row = reinterpret_cast<const uint16_t *>(im.embd) + (size_t) tids[t] * (size_t) D;
        for (int64_t i = 0; i < D; ++i) im.host_emb[(size_t) (t * D + i)] = ggml_fp16_to_fp32(row[i]);
    }
""", """    for (int t = 0; t < n; ++t)
        if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
    double tu1 = 0;
    if (im.emb_f16) {   // larry, STRATA_EMB_PIN: the rows as they are (f16), from the pinned table
        const uint16_t * tab = !im.emb_pin.empty() ? im.emb_pin.data() : reinterpret_cast<const uint16_t *>(im.embd);
        const size_t rowb = (size_t) D * 2;
        if (n > 8 && (size_t) n * (size_t) D <= im.emb_stage.size()) {
            if (im.emb_ev) ggml_backend_event_synchronize(im.emb_ev);   // the previous batch's upload has read the stage
            else im.emb_ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
            uint16_t * st = im.emb_stage.data();
            const int nt = n >= 1024 ? 8 : 1;
            std::vector<std::thread> th;
            auto part = [&](int k) {
                for (int t = n * k / nt; t < n * (k + 1) / nt; ++t)
                    std::memcpy(st + (size_t) t * (size_t) D, tab + (size_t) tids[t] * (size_t) D, rowb);
            };
            for (int k = 1; k < nt; ++k) th.emplace_back(part, k);
            part(0);
            for (auto & t : th) t.join();
            tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
            ggml_backend_tensor_set_async(im.backend, im.i_emb, st, 0, (size_t) n * rowb);
            if (im.emb_ev) ggml_backend_event_record(im.emb_ev, im.backend);
            else ggml_backend_synchronize(im.backend);
        } else {
            std::vector<uint16_t> rows((size_t) n * (size_t) D);
            for (int t = 0; t < n; ++t) std::memcpy(rows.data() + (size_t) t * (size_t) D, tab + (size_t) tids[t] * (size_t) D, rowb);
            im.up_set(im.i_emb, rows.data(), rows.size() * 2);
        }
    } else {
    im.host_emb.resize((size_t) (D * n));
    for (int t = 0; t < n; ++t) {
        const uint16_t * row = reinterpret_cast<const uint16_t *>(im.embd) + (size_t) tids[t] * (size_t) D;
        for (int64_t i = 0; i < D; ++i) im.host_emb[(size_t) (t * D + i)] = ggml_fp16_to_fp32(row[i]);
    }
""")
rep("""    const double tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
    im.up_set(im.i_emb, im.host_emb.data(), im.host_emb.size() * 4);
    const double tu2 = g_dprof_on && n > 64 ? dprof_now() : 0;""", """    tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
    im.up_set(im.i_emb, im.host_emb.data(), im.host_emb.size() * 4);
    }
    const double tu2 = g_dprof_on && n > 64 ? dprof_now() : 0;""")
# the init graph: cast an f16 i_emb
rep("""    ggml_tensor * e = ggml_reshape_3d(gc, b.cols2(im.i_emb, D, n, col0), D, 1, n);
""", """    ggml_tensor * e = b.cols2(im.i_emb, D, n, col0);
    if (e->type != GGML_TYPE_F32) e = ggml_cast(gc, e, GGML_TYPE_F32);   // larry, STRATA_EMB_PIN: f16 rows (exact)
    e = ggml_reshape_3d(gc, e, D, 1, n);
""")
open(P, "w").write(s)
print("patched", P)
