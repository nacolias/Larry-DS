#!/usr/bin/env python3
# patch-vision-engine.py [XOLOTL_DIR] - larry, 2026-10-26: images for the ds41 engine (DeepSeek-V4.1's own VL path,
# model.py: merge_image_embeddings, Gate's bias_vl, the Engram's DEAD tokens).
#   protocol   GENI <max_new> [sampling keys] [img_keys=k,..] <embeddings file> <ids csv> (Strata's server, as for its
#              own vision engine): the file is the vision encoder's SVE1 records, one per picture in prompt order
#              (tools/v41/vision/ds41_vision.py: int32 {magic, n, nx, ny, n_embd} + float32[n][n_embd]); every position
#              of a picture's span carries the image token id (129264, <｜deepseek_image｜>; STRATA_IMAGE_TOKEN).
#              `--vision` is accepted (the server adds it when its config has a "vision" section).
#   embeddings the span's rows replace the token table's at those positions (upload_inputs; f16 as the table's).
#   routing    an image position selects its experts with exp_probs_b_vl instead of exp_probs_b (model.py Gate:
#              `bias_vl where image_mask`); a span with images takes the composite router (the fused one has one bias).
#   Engram     an image position is DEAD in the n-gram history (no n-gram crosses it) and adds nothing (gate 0).
#   reuse      every image position has the same token id, so the conversation cache compares the pictures' keys
#              (img_keys: the server's hash of each picture) as well: a prefix match ends at the first span that differs.
#   path       a request whose new tokens hold an image runs as a layer-major batch, its DSpark tail taken from it.
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
E = os.path.join(X, "tools/v41/engine")


def edit(path, pairs, marker):
    s = open(path).read()
    if marker in s:
        print("already patched", path)
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit("%s: anchor x%d: %s" % (path, s.count(old), old[:90]))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


# ---------------------------------------------------------------------------------------------- the Engram history
edit(os.path.join(E, "ds41_engram.cpp"), [
("""    hist_[(size_t) pos] = g_->engram_token_map[(size_t) tid];
}""",
"""    // larry (vision, patch-vision-engine.py): tid < 0 = an image position - model.py's DEAD: no n-gram crosses it
    hist_[(size_t) pos] = tid < 0 ? (int64_t) -1 : (int64_t) g_->engram_token_map[(size_t) tid];
}"""),
("""    for (int64_t k = 0; k < N; ++k) tok[k] = pos - k >= 0 ? hist_[(size_t) (pos - k)] : g_->engram_pad;
""",
"""    // the look-back stops at the sequence's start and (larry, vision) at an image position: pad from there on
    bool blocked = false;
    for (int64_t k = 0; k < N; ++k) {
        if (pos - k < 0 || hist_[(size_t) (pos - k)] < 0) blocked = true;
        tok[k] = blocked ? g_->engram_pad : hist_[(size_t) (pos - k)];
    }
"""),
], "patch-vision-engine.py")

# ------------------------------------------------------------------------------------------------- the dense half
edit(os.path.join(E, "ds41_dense.hpp"), [(
"""    bool begin_batch(const int * tids, int B, int pos0);
""",
"""    bool begin_batch(const int * tids, int B, int pos0);
    /// larry (vision, patch-vision-engine.py): the prompt's image spans - absolute positions [p0, p0 + n) take their
    /// input embeddings from `rows` (the vision encoder's F32 [n][n_embd], reading order; valid until clear_images),
    /// select their experts with the model's VL bias (exp_probs_b_vl) and get no Engram (model.py image spans).  Set
    /// around a prompt's feed; begin_batch / begin read them.
    struct ImageSpan { int64_t p0; int n; const float * rows; };
    void set_images(const std::vector<ImageSpan> & spans);
    void clear_images();
    /// larry (vision): the Engram's token history at positions [pos0, pos0 + n) again (an image position as DEAD)
    void engram_tokens(int64_t pos0, const int * tids, int n);
""")], "patch-vision-engine.py")

edit(os.path.join(E, "ds41_dense.cpp"), [
# Impl: the image mask input + the spans
("""    ggml_tensor * i_lastblk = nullptr;    // F32 [NC]   (nvis - 1) / block size: the pinned candidate block (ratio 1)
""",
"""    ggml_tensor * i_lastblk = nullptr;    // F32 [NC]   (nvis - 1) / block size: the pinned candidate block (ratio 1)
    ggml_tensor * i_img = nullptr;        // larry (vision): F32 [NC]  1 at an image position (VL routing bias, no Engram)
    std::vector<Ds41Dense::ImageSpan> img;   // larry (vision): the prompt's image spans (set_images)
    const float * img_row(int64_t p) const {
        for (const auto & s : img) if (p >= s.p0 && p < s.p0 + s.n) return s.rows + (size_t) (p - s.p0) * (size_t) D;
        return nullptr;
    }
    bool img_in(int64_t p0, int64_t n) const {
        for (const auto & s : img) if (s.p0 < p0 + n && s.p0 + s.n > p0) return true;
        return false;
    }
"""),
("""    im.i_lastblk = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
""",
"""    im.i_lastblk = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
    im.i_img = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);   // larry (vision)
"""),
# embeddings: the image rows over the token table's
("""            for (auto & t : th) t.join();
            tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
            ggml_backend_tensor_set_async(im.backend, im.i_emb, st, 0, (size_t) n * rowb);""",
"""            for (auto & t : th) t.join();
            if (!im.img.empty())   // larry (vision): an image position's row is the encoder's, not the token table's
                for (int t = 0; t < n; ++t)
                    if (const float * r = im.img_row((int64_t) pos0 + t))
                        for (int64_t i = 0; i < D; ++i) st[(size_t) t * (size_t) D + (size_t) i] = ggml_fp32_to_fp16(r[i]);
            tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
            ggml_backend_tensor_set_async(im.backend, im.i_emb, st, 0, (size_t) n * rowb);"""),
("""            for (int t = 0; t < n; ++t) std::memcpy(rows.data() + (size_t) t * (size_t) D, tab + (size_t) tids[t] * (size_t) D, rowb);
            im.up_set(im.i_emb, rows.data(), rows.size() * 2);""",
"""            for (int t = 0; t < n; ++t) std::memcpy(rows.data() + (size_t) t * (size_t) D, tab + (size_t) tids[t] * (size_t) D, rowb);
            for (int t = 0; t < n && !im.img.empty(); ++t)   // larry (vision)
                if (const float * r = im.img_row((int64_t) pos0 + t))
                    for (int64_t i = 0; i < D; ++i) rows[(size_t) t * (size_t) D + (size_t) i] = ggml_fp32_to_fp16(r[i]);
            im.up_set(im.i_emb, rows.data(), rows.size() * 2);"""),
("""    tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
    im.up_set(im.i_emb, im.host_emb.data(), im.host_emb.size() * 4);""",
"""    for (int t = 0; t < n && !im.img.empty(); ++t)   // larry (vision)
        if (const float * r = im.img_row((int64_t) pos0 + t)) std::memcpy(&im.host_emb[(size_t) (t * D)], r, (size_t) D * 4);
    tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
    im.up_set(im.i_emb, im.host_emb.data(), im.host_emb.size() * 4);"""),
("""    im.up_set(im.i_pos, pos.data(), (size_t) n * 4);
""",
"""    im.up_set(im.i_pos, pos.data(), (size_t) n * 4);
    if (im.img_in(pos0, n)) {   // larry (vision): these columns' image mask
        std::vector<float> m((size_t) n);
        for (int t = 0; t < n; ++t) m[(size_t) t] = im.img_row((int64_t) pos0 + t) ? 1.0f : 0.0f;
        im.up_set(im.i_img, m.data(), (size_t) n * 4);
    }
"""),
# the Engram history: an image position DEAD
("""    for (int t = 0; t < n; ++t) im.engram.set_token(im.eng_pf_pos0 + t, im.eng_pf_tids[(size_t) t]);
""",
"""    for (int t = 0; t < n; ++t)   // (larry, vision: an image position as DEAD)
        im.engram.set_token(im.eng_pf_pos0 + t, im.img_row((int64_t) im.eng_pf_pos0 + t) ? -1 : im.eng_pf_tids[(size_t) t]);
"""),
("""            im.eng_pf.reset();
        }
        for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);
        static const bool eng_async = env_on("STRATA_ENG_ASYNC", true);
""",
"""            im.eng_pf.reset();
        }
        for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, im.img_row((int64_t) pos0 + t) ? -1 : tids[t]);   // (larry, vision)
        static const bool eng_async = env_on("STRATA_ENG_ASYNC", true);
"""),
# build_attn: the image columns
("""    const bool q8f = hcf && q8_fused(im, il, n);
    GGML_ASSERT(part == 0 || (hcf && col0 == 0 && n <= im.o_att->ne[1]));
""",
"""    const bool q8f = hcf && q8_fused(im, il, n);
    GGML_ASSERT(part == 0 || (hcf && col0 == 0 && n <= im.o_att->ne[1]));
    // larry (vision): image positions among this graph's columns - their VL routing bias and no Engram for them
    ggml_tensor * mimg = !im.img.empty() && im.img_in((int64_t) im.pos0 + col0, n) ? b.v1(im.i_img, n, (size_t) col0 * 4) : nullptr;
"""),
("""        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        ggml_tensor * gv = ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate);""",
"""        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (mimg) gate = ggml_mul(gc, gate, ggml_reshape_3d(gc, ggml_scale_bias(gc, mimg, -1.0f, 1.0f), 1, 1, n));   // (larry, vision)
        ggml_tensor * gv = ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate);"""),
("""    ggml_tensor * s0 = ggml_add(gc, sc, im.BL(il, "exp_probs_b.bias"));                        // the model's own selection score
""",
"""    ggml_tensor * s0 = ggml_add(gc, sc, im.BL(il, "exp_probs_b.bias"));                        // the model's own selection score
    if (mimg) {   // larry (vision): an image position selects with the VL bias (model.py Gate: bias_vl where image_mask)
        ggml_tensor * dvl = ggml_sub(gc, im.BL(il, "exp_probs_b_vl.bias"), im.BL(il, "exp_probs_b.bias"));   // [NE]
        s0 = ggml_add(gc, s0, ggml_mul(gc, ggml_repeat(gc, ggml_reshape_2d(gc, dvl, im.NE, 1), s0),
                                         ggml_repeat(gc, ggml_reshape_2d(gc, mimg, 1, n), s0)));
    }
"""),
("""    if (im.sycl && fused_router && rb_protect <= 0 && rb_margin < 0.0f) {""",
"""    if (im.sycl && fused_router && rb_protect <= 0 && rb_margin < 0.0f && !mimg) {   // (larry, vision: one bias only)"""),
# build_attn_kv_only: the Engram mask
("""    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {   // the Engram add still feeds the K/V + compressor projections""",
"""    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);
    ggml_tensor * mimg = !im.img.empty() && im.img_in((int64_t) im.pos0 + col0, n) ? b.v1(im.i_img, n, (size_t) col0 * 4) : nullptr;   // (larry, vision)

    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {   // the Engram add still feeds the K/V + compressor projections"""),
("""        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        x = ggml_add(gc, x, ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate));""",
"""        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (mimg) gate = ggml_mul(gc, gate, ggml_reshape_3d(gc, ggml_scale_bias(gc, mimg, -1.0f, 1.0f), 1, 1, n));   // (larry, vision)
        x = ggml_add(gc, x, ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate));"""),
# the API
("""void Ds41Dense::engram_prefetch(const int * tids, int n, int pos0) {""",
"""// larry (vision, patch-vision-engine.py)
void Ds41Dense::set_images(const std::vector<ImageSpan> & spans) { p_->img = spans; }
void Ds41Dense::clear_images() { p_->img.clear(); }
void Ds41Dense::engram_tokens(int64_t pos0, const int * tids, int n) {
    Impl & im = *p_;
    if (!im.engram.enabled()) return;
    for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, im.img_row(pos0 + t) ? -1 : tids[t]);
}

void Ds41Dense::engram_prefetch(const int * tids, int n, int pos0) {"""),
], "patch-vision-engine.py")

# ------------------------------------------------------------------------------------------------- the serve loop
edit(os.path.join(E, "ds41_generate.cpp"), [
("""    double chunk_prefetch_gib = 0.0;
};""",
"""    double chunk_prefetch_gib = 0.0;
    bool vision = false;          // larry: --vision (the server adds it for a config with a "vision" section; GENI is always read)
};"""),
("""        else if (k == "--enc-replay") a.enc_replay = std::atoi(next().c_str());
""",
"""        else if (k == "--enc-replay") a.enc_replay = std::atoi(next().c_str());
        else if (k == "--vision") a.vision = true;
"""),
("""std::vector<int> parse_csv(const std::string & s) {""",
"""// larry (vision, patch-vision-engine.py): a GENI request's pictures - the vision encoder's SVE1 records (int32 {magic,
// n, nx, ny, n_embd} + float32[n][n_embd]), one per picture in prompt order - and the spans they fill
struct ImgRec { int n = 0; std::vector<float> rows; };
static bool load_sve1(const std::string & path, int64_t n_embd, std::vector<ImgRec> & out, std::string & err) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open the image embeddings " + path; return false; }
    for (;;) {
        int32_t h[5];
        const size_t r = std::fread(h, 4, 5, f);
        if (r == 0) break;
        if (r != 5 || h[0] != 0x31455653 || h[1] <= 0 || (int64_t) h[4] != n_embd) {
            err = "the image embeddings are not SVE1 records of this model's width";
            std::fclose(f);
            return false;
        }
        ImgRec rec;
        rec.n = h[1];
        rec.rows.resize((size_t) h[1] * (size_t) n_embd);
        if (std::fread(rec.rows.data(), 4, rec.rows.size(), f) != rec.rows.size()) { err = "short SVE1 record"; std::fclose(f); return false; }
        out.push_back(std::move(rec));
    }
    std::fclose(f);
    return true;
}
// an image span of a conversation: positions [p0, p0 + n) and the picture it holds
struct ImgId { int64_t p0; int n; uint64_t key; };
static uint64_t fnv64(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= ((const unsigned char *) p)[i]; h *= 1099511628211ull; }
    return h;
}
// the prefix two token streams share once their pictures are compared too (every image position carries the same
// token id, so another picture of the same size would pass for the same one): cut at the first span either has that
// the other does not have with the same extent and key
static size_t img_cap(size_t lcp, const std::vector<ImgId> & a, const std::vector<ImgId> & b) {
    auto cut = [&](const std::vector<ImgId> & x, const std::vector<ImgId> & y) {
        for (const ImgId & s : x) {
            if ((size_t) s.p0 >= lcp) break;
            bool same = false;
            for (const ImgId & t : y) if (t.p0 == s.p0) { same = t.n == s.n && t.key == s.key; break; }
            if (!same) { lcp = (size_t) s.p0; break; }
        }
    };
    cut(a, b);
    cut(b, a);
    return lcp;
}
static std::vector<ImgId> img_trim(const std::vector<ImgId> & v, int64_t len) {
    std::vector<ImgId> o;
    for (const ImgId & s : v) if (s.p0 + s.n <= len) o.push_back(s);
    return o;
}

std::vector<int> parse_csv(const std::string & s) {"""),
("""        std::vector<int> hist;
""",
"""        std::vector<int> hist;
        std::vector<ImgId> conv_img;   // larry (vision): the live conversation's image spans (hist's pictures)
"""),
("""            std::vector<float> logits;
            int64_t saved = 0;            // positions saved""",
"""            std::vector<float> logits;
            std::vector<ImgId> img;       // larry (vision): its image spans
            int64_t saved = 0;            // positions saved"""),
("""            if (line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\\n"); std::fflush(stdout); continue; }
            std::istringstream ss(line);
            std::string w, csv;""",
"""            // larry (vision): GENI <max_new> [keys] [img_keys=k,..] <embeddings file> <ids csv> - a request with pictures
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\\n"); std::fflush(stdout); continue; }
            std::istringstream ss(line);
            std::string w, csv, img_path;
            std::vector<std::string> img_keys;"""),
("""                if (eq == std::string::npos) { csv = w; continue; }""",
"""                if (eq == std::string::npos) { if (geni && img_path.empty()) img_path = w; else csv = w; continue; }"""),
("""                else if (k == "seed") rng.seed((uint64_t) std::atoll(v.c_str()));
            }""",
"""                else if (k == "seed") rng.seed((uint64_t) std::atoll(v.c_str()));
                else if (k == "img_keys") {   // larry (vision): one key a picture (the server's hash of it)
                    std::stringstream ks(v);
                    for (std::string x; std::getline(ks, x, ',');) img_keys.push_back(x);
                }
            }"""),
("""            size_t lcp = 0;
            while (lcp < hist.size() && lcp < ids.size() && hist[lcp] == ids[lcp]) ++lcp;""",
"""            // larry (vision): the request's pictures - each a run of image tokens, record by record
            std::vector<ImgRec> img_recs;
            std::vector<ImgId> req_img;
            std::vector<Ds41Dense::ImageSpan> img_spans;
            if (geni) {
                static const int img_tok = std::getenv("STRATA_IMAGE_TOKEN") ? std::atoi(std::getenv("STRATA_IMAGE_TOKEN")) : 129264;
                std::string ierr;
                if (load_sve1(img_path, n_embd, img_recs, ierr)) {
                    size_t k = 0;
                    int j = 0;
                    int64_t ps = -1;
                    for (size_t p = 0; p < ids.size() && ierr.empty(); ++p) {
                        if (ids[p] != img_tok) continue;
                        if (k >= img_recs.size()) { ierr = "more image tokens than image rows"; break; }
                        if (j == 0) ps = (int64_t) p;
                        if ((int64_t) p != ps + j) { ierr = "a picture's image tokens are not consecutive"; break; }
                        if (++j == img_recs[k].n) {
                            const uint64_t key = k < img_keys.size() ? fnv64(img_keys[k].data(), img_keys[k].size())
                                                                     : fnv64(img_recs[k].rows.data(), img_recs[k].rows.size() * 4);
                            req_img.push_back({ ps, img_recs[k].n, key });
                            img_spans.push_back({ ps, img_recs[k].n, img_recs[k].rows.data() });
                            ++k;
                            j = 0;
                        }
                    }
                    if (ierr.empty() && (k != img_recs.size() || j != 0)) ierr = "the prompt's image tokens and the image rows do not match";
                }
                if (!ierr.empty()) { std::printf("ERR %s\\n", ierr.c_str()); std::fflush(stdout); continue; }
            }
            size_t lcp = 0;
            while (lcp < hist.size() && lcp < ids.size() && hist[lcp] == ids[lcp]) ++lcp;
            lcp = img_cap(lcp, conv_img, req_img);   // (larry, vision)"""),
("""                while (l < pk[s].hist.size() && l < ids.size() && pk[s].hist[l] == ids[l]) ++l;""",
"""                while (l < pk[s].hist.size() && l < ids.size() && pk[s].hist[l] == ids[l]) ++l;
                l = img_cap(l, pk[s].img, req_img);   // (larry, vision)"""),
("""                    hist = park_hist;
                    snap_len = -1;        // the tail under the park's end is re-fed, so no snapshot survives this""",
"""                    hist = park_hist;
                    conv_img = pk[ub].img;   // (larry, vision)
                    snap_len = -1;        // the tail under the park's end is re-fed, so no snapshot survives this"""),
("""                    hist = park_hist;
                    snap_len = (int) park_hist.size();""",
"""                    hist = park_hist;
                    conv_img = pk[ub].img;   // (larry, vision)
                    snap_len = (int) park_hist.size();"""),
("""                hist.resize((size_t) snap_len);   // the same prompt again (regenerate): its end state + saved logits""",
"""                hist.resize((size_t) snap_len);   // the same prompt again (regenerate): its end state + saved logits
                conv_img = img_trim(conv_img, snap_len);   // (larry, vision)"""),
("""                hist.resize((size_t) snap_len);
                reused = snap_len;
            } else {""",
"""                hist.resize((size_t) snap_len);
                conv_img = img_trim(conv_img, snap_len);   // (larry, vision)
                reused = snap_len;
            } else {"""),
("""                        pk[ps].hist.assign(hist.begin(), hist.begin() + snap_len);""",
"""                        pk[ps].hist.assign(hist.begin(), hist.begin() + snap_len);
                        pk[ps].img = img_trim(conv_img, snap_len);   // (larry, vision)"""),
("""                hist.clear();
                snap_len = -1;
            }
            std::fprintf(stderr, "serve: reuse lcp=%zu park=%zu/%zu lp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d\\n", lcp,""",
"""                hist.clear();
                conv_img.clear();   // (larry, vision)
                snap_len = -1;
            }
            std::fprintf(stderr, "serve: reuse lcp=%zu park=%zu/%zu lp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d\\n", lcp,"""),
("""            bool by_step = defer_vram && dh >= 1 && dh <= step_max;""",
"""            bool by_step = defer_vram && dh >= 1 && dh <= step_max;
            // larry (vision): new tokens with a picture run as a batch (the passes have no image path) and take their
            // DSpark tail from it (a tail re-run as passes would re-feed the picture's positions as plain tokens)
            bool img_new = false;
            for (const ImgId & s : req_img) if (s.p0 + s.n > reused) img_new = true;
            if (img_new) by_step = false;"""),
("""            if (batch_tail && sp_tail > 0 && !by_step && dh >= 2 && PB > 1) { bcap = sp_tail; dh = delta; }""",
"""            if ((batch_tail || img_new) && sp_tail > 0 && !by_step && dh >= 2 && PB > 1) { bcap = sp_tail; dh = delta; }"""),
("""            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));""",
"""            if (!img_spans.empty()) {   // larry (vision): the pictures' rows for the feed; the Engram's look-back history again
                dense.set_images(img_spans);
                const int kb = std::min(reused, 8);
                if (kb > 0) dense.engram_tokens(reused - kb, ids.data() + reused - kb, kb);
            }
            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
            dense.clear_images();"""),
("""                std::printf("ERR prefill failed: %s\\n", ferr.c_str());
                std::fflush(stdout);
                hist.clear();""",
"""                std::printf("ERR prefill failed: %s\\n", ferr.c_str());
                std::fflush(stdout);
                hist.clear();
                conv_img.clear();   // (larry, vision)"""),
("""            hist = ids;
            if (total > reused && dense.snapshot(total)) {""",
"""            hist = ids;
            conv_img = req_img;   // (larry, vision)
            if (total > reused && dense.snapshot(total)) {"""),
("""                std::fprintf(stderr, "serve: decode failed: %s\\n", ferr.c_str());
                hist.clear();""",
"""                std::fprintf(stderr, "serve: decode failed: %s\\n", ferr.c_str());
                hist.clear();
                conv_img.clear();   // (larry, vision)"""),
], "patch-vision-engine.py")
