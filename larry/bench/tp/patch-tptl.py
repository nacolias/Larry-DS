#!/usr/bin/env python3
# patch-tptl.py - larry, STRATA_GPU_TL stamps around the attention output split (STRATA_TP_OPROJ): dense ring 5 = the
# dense card reached the projection, 6 = its own quarter done, 7 = the parts summed; expert card rings 20 / 21 around
# its part (20 runs when the card reaches it - before the dense card's quantised o may be ready).  tl_print adds per card
# "oproj part ready->done" and "dense start (5) -> part done (21)".  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "tl_op_n" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


rep("""    std::vector<double> tl_chunk, tl_chunk_max;          // per card: promotion chunks' device time (stamps 12 -> 13)""",
    """    std::vector<double> tl_chunk, tl_chunk_max;          // per card: promotion chunks' device time (stamps 12 -> 13)
    std::vector<double> tl_op_busy, tl_op_lag;           // per card: oproj part 20 -> 21, dense 5 -> 21
    std::vector<long> tl_op_n;""")
# stamps in the hook
rep("""        std::lock_guard<PrioMutex> lk(g_submit);
        const sycl::event e0 = fastmv::quant_x(q0, o, ocs, OGD, n, OG, OGD, im.os_xq, im.os_xd);   // all groups""",
    """        std::lock_guard<PrioMutex> lk(g_submit);
        const bool tl = Ds41XeTier::tl_on();
        if (tl) tl_stamp(im, 0, q0, 5);
        const sycl::event e0 = fastmv::quant_x(q0, o, ocs, OGD, n, OG, OGD, im.os_xq, im.os_xd);   // all groups""")
rep("""                sx::command_graph<sx::graph_state::executable>& gr = *eg;
                ev.push_back(c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.ext_oneapi_graph(gr); }));
            } else {
                ev.push_back(enqueue(&e0));
            }""", """                sx::command_graph<sx::graph_state::executable>& gr = *eg;
                if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 20);
                ev.push_back(c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.ext_oneapi_graph(gr); }));
                if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 21);
            } else {
                if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 20);
                ev.push_back(enqueue(&e0));
                if (tl) tl_stamp(im, 1 + (size_t) c.idx, c.q, 21);
            }""")
rep("""        os_mv(q0, s.wb[0], KB, D, 1, 0, n, im.os_aq, im.os_ad, dst, dcs, dcs * n);
        const float* sl = im.os_slots;""", """        os_mv(q0, s.wb[0], KB, D, 1, 0, n, im.os_aq, im.os_ad, dst, dcs, dcs * n);
        if (tl) tl_stamp(im, 0, q0, 6);
        const float* sl = im.os_slots;""")
rep("""                for (int k = 0; k < nparts; ++k) v += sl[(size_t) k * 8 * D + c * D + r];
                dst[c * dcs + r] = v;
            });
        });
        ++im.os_calls;""", """                for (int k = 0; k < nparts; ++k) v += sl[(size_t) k * 8 * D + c * D + r];
                dst[c * dcs + r] = v;
            });
        });
        if (tl) tl_stamp(im, 0, q0, 7);
        ++im.os_calls;""")
# tl_flush: the dense stamps 5, the card pairs (20, 21)
rep("""    std::vector<double> t1, t3;
    for (uint32_t k = d.done; k < dn; ++k) {
        const int b = d.id[k % C];
        if (b == 1 || b == 9) t1.push_back(d.t[k % C]);""", """    std::vector<double> t1, t3, t5;
    for (uint32_t k = d.done; k < dn; ++k) {
        const int b = d.id[k % C];
        if (b == 5) t5.push_back(d.t[k % C]);
        if (b == 1 || b == 9) t1.push_back(d.t[k % C]);""")
rep("""    im.tl_copy_lag.resize(NC, 0.0); im.tl_copy_sum.resize(NC, 0.0); im.tl_copy_n.resize(NC, 0);""",
    """    im.tl_copy_lag.resize(NC, 0.0); im.tl_copy_sum.resize(NC, 0.0); im.tl_copy_n.resize(NC, 0);
    im.tl_op_busy.resize(NC, 0.0); im.tl_op_lag.resize(NC, 0.0); im.tl_op_n.resize(NC, 0);""")
rep("""            if (r.id[k % C] == 11 && r.id[(k + 1) % C] == 14) {   // end -> copy done; copy done -> the next sum""",
    """            if (r.id[k % C] == 20 && r.id[(k + 1) % C] == 21) {   // an oproj part (STRATA_TP_OPROJ)
                const double ts = r.t[k % C], te = r.t[(k + 1) % C];
                auto lb = std::upper_bound(t5.begin(), t5.end(), te);   // the dense card's start of that projection
                if (lb != t5.begin()) { im.tl_op_busy[c] += te - ts; im.tl_op_lag[c] += te - *(lb - 1); ++im.tl_op_n[c]; }
                continue;
            }
            if (r.id[k % C] == 11 && r.id[(k + 1) % C] == 14) {   // end -> copy done; copy done -> the next sum""")
rep("""    for (size_t c = 0; c < NC; ++c)
        if (c < im.tl_chunk_n.size() && im.tl_chunk_n[c])""", """    for (size_t c = 0; c < NC; ++c)
        if (c < im.tl_op_n.size() && im.tl_op_n[c])
            std::fprintf(stderr, "; card %zu oproj part ready->done %.1f us, dense start->part done %.1f us (x%ld)", c,
                         im.tl_op_busy[c] / im.tl_op_n[c], im.tl_op_lag[c] / im.tl_op_n[c], im.tl_op_n[c]);
    for (size_t c = 0; c < NC; ++c)
        if (c < im.tl_chunk_n.size() && im.tl_chunk_n[c])""")
open(P, "w").write(s)
print("patched")
