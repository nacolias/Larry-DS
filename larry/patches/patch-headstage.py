#!/usr/bin/env python3
# patch-headstage.py [TIER_CPP] - larry: the LM head split's (mv_hook) peer copies go through card-0 staging the tier
# owns, never through ggml's buffers.  The expert cards read the quantised input from / wrote their rows into the
# ggml tensors directly; a card-0 buffer an expert card touches the first time is imported there, and xe's dma-buf
# export (ttm_bo_setup_export) populates a system-memory copy of the WHOLE buffer first: xetrace (abwT77) 35-80 ms a
# prompt even warm, and the slow batch heads (abwT76: 2.2 s / 6.9 s inside DRM_IOCTL_PRIME_HANDLE_TO_FD, the head's
# mv_hook memcpy) when a graph's buffer was re-allocated (GiBs).  Now card 0 stages xq + xd (one small kernel), every
# card copies its input from that and its rows back into card-0 staging (one copy each way instead of 2 + ncols), and
# a scatter kernel on card 0 puts them into y.  Same products, byte copies: bit-identical logits.
# STRATA_XE_HEAD_STAGE=0: the direct copies as before.  STRATA_XE_HEAD_STAGE_CHECK=1: the first 4 staged calls of each width
# also run the direct way into a scratch, compared on the host ("head stage check").  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_HEAD_STAGE" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""        std::vector<int8_t*> xq;
        std::vector<float*> xd, y;
    };
    std::map<const void*, HeadSplit> heads;
""", """        std::vector<int8_t*> xq;
        std::vector<float*> xd, y;
        // larry, patch-headstage.py: card-0 staging, the only card-0 memory the other cards' copies touch (imported once)
        uint8_t* sx = nullptr;          ///< ncols * K int8 then ncols * K / 32 scales (as the cards' xq: xd follows it)
        float* sy = nullptr;            ///< part i >= 1: its rows x ncols at (r[i] - r[1]) * 8
        sycl::queue* sq = nullptr;      ///< the dense queue sx / sy were made for (another queue: the direct copies)
    };
    std::map<const void*, HeadSplit> heads;
""")
# the cards' xq and xd one buffer (the staged copy puts xd right after the ncols * K int8): one copy in
rep("""        int8_t* xq = sycl::malloc_device<int8_t>((size_t) 8 * K, c.q);
        float* xd = sycl::malloc_device<float>((size_t) 8 * K / 32, c.q);
        float* y = sycl::malloc_device<float>((size_t) 8 * rows, c.q);
        if (!wk || !xq || !xd || !y) { std::fprintf(stderr, "xe tier: head split allocation failed - not split\\n"); return Ds41XeImpl::HeadSplit(); }
""", """        // larry, patch-headstage: xq and xd one buffer (staged: xd right after the used int8; direct: at 8 * K)
        int8_t* xq = sycl::malloc_device<int8_t>((size_t) 8 * K + (size_t) 8 * K / 32 * sizeof(float), c.q);
        float* xd = xq ? reinterpret_cast<float*>(xq + (size_t) 8 * K) : nullptr;
        float* y = sycl::malloc_device<float>((size_t) 8 * rows, c.q);
        if (!wk || !xq || !y) { std::fprintf(stderr, "xe tier: head split allocation failed - not split\\n"); return Ds41XeImpl::HeadSplit(); }
""")
rep("""        sycl::queue& q0 = *static_cast<sycl::queue*>(queue);
        const int64_t bpr = K / 256;
        const sycl::event e0 = q0.ext_oneapi_submit_barrier();
        std::vector<sycl::event> ev;
        for (size_t i = 0; i < hs.c.size(); ++i) {
            XeCard& c = *hs.c[i];
            const int64_t r0 = hs.r[i + 1], rows = hs.r[i + 2] - r0;
            c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.memcpy(hs.xq[i], xq, (size_t) ncols * K); });
            c.q.memcpy(hs.xd[i], xd, (size_t) ncols * K / 32 * sizeof(float));
            fastmv::Args a;
            a.w = hs.w[i]; a.K = K; a.N = rows; a.G = 1; a.g_div = 1; a.n_blocks_total = rows * bpr;
            a.x_col_stride = K; a.x_batch_stride = K * ncols;
            a.y = hs.y[i]; a.y_col_stride = rows; a.y_batch_stride = rows * ncols; a.ncols = ncols;
            a.xq = hs.xq[i]; a.xd = hs.xd[i];
            fastmv::launch(c.q, fastmv::WType::Q6K_R8, a);
            for (int t = 0; t < ncols; ++t) c.q.memcpy(y + t * y_col_stride + r0, hs.y[i] + t * rows, (size_t) rows * 4);
            ev.push_back(c.q.ext_oneapi_submit_barrier());
        }
        fastmv::Args a0;
        a0.w = w; a0.K = K; a0.N = hs.r[1]; a0.G = 1; a0.g_div = 1; a0.n_blocks_total = N * bpr;
        a0.x_col_stride = K; a0.x_batch_stride = K * ncols;
        a0.y = y; a0.y_col_stride = y_col_stride; a0.y_batch_stride = y_col_stride * ncols; a0.ncols = ncols;
        a0.xq = xq; a0.xd = xd;
        fastmv::launch(q0, fastmv::WType::Q6K_R8, a0);
        q0.ext_oneapi_submit_barrier(ev);
        return true;
""", """        sycl::queue& q0 = *static_cast<sycl::queue*>(queue);
        const int64_t bpr = K / 256, R1 = hs.r[1];
        auto own = [&](float* yo, int64_t ys) {   // part 0, on the dense card
            fastmv::Args a0;
            a0.w = w; a0.K = K; a0.N = R1; a0.G = 1; a0.g_div = 1; a0.n_blocks_total = N * bpr;
            a0.x_col_stride = K; a0.x_batch_stride = K * ncols;
            a0.y = yo; a0.y_col_stride = ys; a0.y_batch_stride = ys * ncols; a0.ncols = ncols;
            a0.xq = xq; a0.xd = xd;
            fastmv::launch(q0, fastmv::WType::Q6K_R8, a0);
        };
        auto direct = [&](float* yo, int64_t ys) {   // the cards copy from xq / xd and into yo themselves
            const sycl::event e0 = q0.ext_oneapi_submit_barrier();
            std::vector<sycl::event> ev;
            for (size_t i = 0; i < hs.c.size(); ++i) {
                XeCard& c = *hs.c[i];
                const int64_t r0 = hs.r[i + 1], rows = hs.r[i + 2] - r0;
                c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.memcpy(hs.xq[i], xq, (size_t) ncols * K); });
                c.q.memcpy(hs.xd[i], xd, (size_t) ncols * K / 32 * sizeof(float));
                fastmv::Args a;
                a.w = hs.w[i]; a.K = K; a.N = rows; a.G = 1; a.g_div = 1; a.n_blocks_total = rows * bpr;
                a.x_col_stride = K; a.x_batch_stride = K * ncols;
                a.y = hs.y[i]; a.y_col_stride = rows; a.y_batch_stride = rows * ncols; a.ncols = ncols;
                a.xq = hs.xq[i]; a.xd = hs.xd[i];
                fastmv::launch(c.q, fastmv::WType::Q6K_R8, a);
                for (int t = 0; t < ncols; ++t) c.q.memcpy(yo + t * ys + r0, hs.y[i] + t * rows, (size_t) rows * 4);
                ev.push_back(c.q.ext_oneapi_submit_barrier());
            }
            own(yo, ys);
            q0.ext_oneapi_submit_barrier(ev);
        };
        // larry, STRATA_XE_HEAD_STAGE (patch-headstage.py, default 1): the peer copies through card-0 staging
        static const bool stage_on = env_i("STRATA_XE_HEAD_STAGE", 1) != 0;
        if (stage_on && !hs.sq) {
            hs.sq = &q0;
            hs.sx = sycl::malloc_device<uint8_t>((size_t) 8 * K + (size_t) 8 * K / 32 * sizeof(float), q0);
            hs.sy = sycl::malloc_device<float>((size_t) 8 * (N - R1), q0);
            if (!hs.sx || !hs.sy) {
                if (hs.sx) sycl::free(hs.sx, q0);
                if (hs.sy) sycl::free(hs.sy, q0);
                hs.sx = nullptr; hs.sy = nullptr;
                std::fprintf(stderr, "xe tier: head split staging allocation failed - direct copies\\n");
            }
        }
        if (!stage_on || !hs.sx || hs.sq != &q0) {
            direct(y, y_col_stride);
            return true;
        }
        const size_t nq = (size_t) ncols * K, nd = nq / 32;   // int8 bytes, scales
        uint8_t* sx = hs.sx;
        float* sy = hs.sy;
        const sycl::event e0 = q0.parallel_for(sycl::range<1>(nq / 4 + nd), [=](sycl::id<1> id) {
            const size_t j = id[0];
            if (j < nq / 4) reinterpret_cast<uint32_t*>(sx)[j] = reinterpret_cast<const uint32_t*>(xq)[j];
            else reinterpret_cast<float*>(sx + nq)[j - nq / 4] = xd[j - nq / 4];
        });
        std::vector<sycl::event> ev;
        for (size_t i = 0; i < hs.c.size(); ++i) {
            XeCard& c = *hs.c[i];
            const int64_t r0 = hs.r[i + 1], rows = hs.r[i + 2] - r0;
            int8_t* cx = hs.xq[i];
            c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.memcpy(cx, sx, nq + nd * sizeof(float)); });
            fastmv::Args a;
            a.w = hs.w[i]; a.K = K; a.N = rows; a.G = 1; a.g_div = 1; a.n_blocks_total = rows * bpr;
            a.x_col_stride = K; a.x_batch_stride = K * ncols;
            a.y = hs.y[i]; a.y_col_stride = rows; a.y_batch_stride = rows * ncols; a.ncols = ncols;
            a.xq = cx; a.xd = reinterpret_cast<const float*>(cx + nq);
            fastmv::launch(c.q, fastmv::WType::Q6K_R8, a);
            ev.push_back(c.q.memcpy(sy + (r0 - R1) * 8, hs.y[i], (size_t) rows * ncols * sizeof(float)));
        }
        own(y, y_col_stride);
        q0.ext_oneapi_submit_barrier(ev);
        int64_t rr[9];
        for (int i = 0; i < 9; ++i) rr[i] = hs.r[i];
        const int np = (int) hs.c.size() + 1;
        const int64_t NR = N - R1, ys = y_col_stride;
        q0.parallel_for(sycl::range<1>((size_t) (NR * ncols)), [=](sycl::id<1> id) {
            const int64_t t = (int64_t) id[0] / NR, row = R1 + (int64_t) id[0] % NR;
            int p = 1;
            while (p + 1 < np && row >= rr[p + 1]) ++p;   // part p: rows [rr[p], rr[p + 1])
            const int64_t p0 = rr[p], rows = rr[p + 1] - p0;
            y[t * ys + row] = sy[(p0 - R1) * 8 + t * rows + (row - p0)];
        });
        // STRATA_XE_HEAD_STAGE_CHECK=1: the first 4 staged calls of each width the direct way too, into a scratch, compared
        static const bool check = env_i("STRATA_XE_HEAD_STAGE_CHECK", 0) != 0;
        static int checked = 0, by_n[9] = {};
        if (check && by_n[ncols] < 4) {
            ++checked; ++by_n[ncols];
            float* yc = sycl::malloc_device<float>((size_t) N * ncols, q0);
            if (yc) {
                direct(yc, N);
                std::vector<float> ha((size_t) N * ncols), hb((size_t) N * ncols);
                for (int t = 0; t < ncols; ++t) q0.memcpy(ha.data() + (size_t) t * N, y + t * y_col_stride, (size_t) N * 4);
                q0.memcpy(hb.data(), yc, (size_t) N * ncols * 4);
                q0.wait();
                size_t bad = 0;
                for (size_t j = 0; j < ha.size(); ++j) bad += std::memcmp(&ha[j], &hb[j], 4) != 0;
                std::fprintf(stderr, "xe tier: head stage check %d (%d columns): %zu of %zu logits differ\\n", checked, ncols,
                             bad, ha.size());
                sycl::free(yc, q0);
            }
        }
        return true;
""")
rep("""            sycl::free(kv.second.xq[i], q); sycl::free(kv.second.xd[i], q); sycl::free(kv.second.y[i], q);
        }
    im.heads.clear();
""", """            sycl::free(kv.second.xq[i], q); sycl::free(kv.second.y[i], q);   // (xd lives in xq's buffer: patch-headstage)
        }
    for (auto& kv : im.heads) {   // larry, patch-headstage: the card-0 staging (by the context: ggml's queue may be gone)
        if (kv.second.sx) sycl::free(kv.second.sx, im.ctx);
        if (kv.second.sy) sycl::free(kv.second.sy, im.ctx);
    }
    im.heads.clear();
""")
open(P, "w").write(s)
print("patched", P)
