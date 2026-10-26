#!/usr/bin/env python3
# patch-streamfit.py [TIER_CPP] - larry, STRATA_XE_STREAM_FIT (default 1): a prompt call's stream buffer on an expert
# card grows only into what the card has free beyond the call's other buffers (STRATA_XE_STREAM_FIT_MARGIN_GIB, 0.5,
# plus 100 KiB a token for X, the part and the GEMM rows); a smaller buffer takes more rounds.  2026-10-26 03:16 the
# engine aborted: after a 49K-token conversation the TP core's KV rows had taken ~0.55 GiB of each expert card, the
# next (4.7K-token, not prefetched) prompt regrew two stream buffers to 864 MiB on a card with ~0.9 GiB free, the
# driver evicted to host memory past TTM's limit ("VM worker error: -12" from 03:08) and a submission failed with
# UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY.  The cap never shrinks a buffer already there, and a card keeping experts in
# its buffer keeps it as before.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_STREAM_FIT" in s:
    print("already patched"); sys.exit(0)
old = """        uint8_t* const st0 = c.stream;   // a card keeping experts keeps its buffer (no regrow)
        const size_t sneed = std::min<size_t>(stream_cap, std::max<size_t>((size_t) sbytes[(size_t) c.idx], 1));
        // with the prefetch on, a regrowth goes straight to the cap (planned into the prefetch's reserve)
        const size_t sreq = !kept[(size_t) c.idx].empty() ? c.stream_bytes : im.pf_on && sneed > c.stream_bytes ? stream_cap : sneed;
        if (!grow_stream(c, sreq, stream_cap)) {"""
new = """        uint8_t* const st0 = c.stream;   // a card keeping experts keeps its buffer (no regrow)
        // larry, STRATA_XE_STREAM_FIT (patch-streamfit.py): the buffer grows only into what the card has free beyond the
        // call's other buffers - a long context's KV rows or the promotion cache must not leave it overcommitted
        size_t cap_c = stream_cap;
        static const int stream_fit = env_i("STRATA_XE_STREAM_FIT", 1);
        static const double fit_margin = env_d("STRATA_XE_STREAM_FIT_MARGIN_GIB", 0.5);
        if (stream_fit && !c.dense) {
            const int64_t fr = c.free_vram();
            if (fr >= 0) {
                const int64_t room = fr + (int64_t) c.stream_bytes - (int64_t) (fit_margin * kGiB) - (int64_t) n * 100 * 1024;
                const size_t fit = (size_t) std::max<int64_t>(room, (int64_t) 64 << 20);
                if (fit < cap_c) {
                    static int64_t nfit = 0;
                    if (nfit++ < 20 || nfit % 1000 == 0)
                        std::fprintf(stderr, "xe tier: card %d: stream buffer capped at %.0f MiB (%.2f GiB free, %d tokens) - more rounds\\n",
                                     c.idx, fit / 1048576.0, fr / kGiB, n);
                    cap_c = fit;
                }
            }
        }
        const size_t sneed = std::min<size_t>(cap_c, std::max<size_t>((size_t) sbytes[(size_t) c.idx], 1));
        // with the prefetch on, a regrowth goes straight to the cap (planned into the prefetch's reserve)
        const size_t sreq = !kept[(size_t) c.idx].empty() ? c.stream_bytes : im.pf_on && sneed > c.stream_bytes ? cap_c : sneed;
        if (!grow_stream(c, sreq, cap_c)) {"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
