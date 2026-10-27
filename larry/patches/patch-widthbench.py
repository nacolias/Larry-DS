#!/usr/bin/env python3
# patch-widthbench.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions, the stage 2 measurement; diagnostic, default
# off): STRATA_TP_WIDTH_BENCH=R - after each request (test servers only), R passes of every width 1..7 through pass_v (the
# decode biases, DSpark's taps: a verify pass's shape) at the conversation's end, rows of (a) its own last tokens - one
# conversation's consecutive rows, as DSpark verifies - (b) random tokens - each row routing apart - and (c) two
# conversations: its own last ceil(n/2) tokens + the previous request's last floor(n/2) (another conversation's text, as a
# pass of two conversations' verify rows would route); median ms a width to stderr ("width bench: ...").  The passes
# write past the conversation, so it is dropped after (the next request reads its prompt fresh).  Also a solo request's
# DSpark cycles timed ("DSpark cycle: ..."): rows verified, verify ms, DSpark step ms (the next drafts), the rest.  What
# batched decode (two conversations' rows in one pass) would cost against today's turns (one pass each).  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
s = open(P).read()
if "STRATA_TP_WIDTH_BENCH" in s:
    print("already patched", P)
    sys.exit(0)
pairs = [
(r'''        auto dspark_cycle = [&](DecState & d, const std::function<bool()> & stopped) -> bool {
''',
r'''        // larry (patch-widthbench.py, diagnostic, STRATA_TP_WIDTH_BENCH): a solo request's DSpark cycles timed - passes,
        // rows verified, verify ms, DSpark step ms
        static const bool wb_prof = std::getenv("STRATA_TP_WIDTH_BENCH") != nullptr;
        double wb_cyc[4] = { 0, 0, 0, 0 };
        auto dspark_cycle = [&](DecState & d, const std::function<bool()> & stopped) -> bool {
'''),
(r'''            if (!pass_v(vt.data(), (int) vt.size(), p0)) { d.ferr = ferr; d.finish = "error"; return false; }
''',
r'''            const double wb_v0 = wb_prof ? now_ms() : 0.0;
            if (!pass_v(vt.data(), (int) vt.size(), p0)) { d.ferr = ferr; d.finish = "error"; return false; }
            if (wb_prof) { wb_cyc[0] += 1; wb_cyc[1] += (double) vt.size(); wb_cyc[2] += now_ms() - wb_v0; }
'''),
(r'''            if (!dsp->step(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, tn, d.sp_ids, d.sp_conf, pv_tp ? r0 : -1)) {
                d.ferr = "DSpark step: " + dsp->last_error(); d.finish = "error"; return false;
            }
''',
r'''            const double wb_s0 = wb_prof ? now_ms() : 0.0;
            if (!dsp->step(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, tn, d.sp_ids, d.sp_conf, pv_tp ? r0 : -1)) {
                d.ferr = "DSpark step: " + dsp->last_error(); d.finish = "error"; return false;
            }
            if (wb_prof) wb_cyc[3] += now_ms() - wb_s0;
'''),
(r'''                while (dspark_cycle(dd, stop_pending)) {}
                if (dd.lk_pass > 0)
''',
r'''                const double wb_c0 = now_ms();
                for (double & v : wb_cyc) v = 0;
                while (dspark_cycle(dd, stop_pending)) {}
                if (wb_prof && wb_cyc[0] > 0)   // (patch-widthbench.py)
                    std::fprintf(stderr, "serve: DSpark cycle: %.0f passes, %.2f rows verified, verify %.2f ms, step %.2f ms, the rest %.2f ms (a pass)\n",
                                 wb_cyc[0], wb_cyc[1] / wb_cyc[0], wb_cyc[2] / wb_cyc[0], wb_cyc[3] / wb_cyc[0],
                                 (now_ms() - wb_c0 - wb_cyc[2] - wb_cyc[3]) / wb_cyc[0]);
                if (dd.lk_pass > 0)
'''),
(r'''            if (admit >= 0) {   // larry (patch-slots-serve.py): BADM, and the slot decodes on in the windows''',
r'''            // larry (patch-widthbench.py, diagnostic): pass costs by width - one conversation's rows, rows routing apart,
            // two conversations' rows (its own + the previous request's)
            if (static const int wb_reps = std::getenv("STRATA_TP_WIDTH_BENCH") ? std::atoi(std::getenv("STRATA_TP_WIDTH_BENCH")) : 0;
                wb_reps > 0 && admit < 0 && tp_on && hist.size() > 16 && (int64_t) hist.size() + 16 < ctxv) {
                static std::vector<int> wb_prev;   // the previous request's conversation: another conversation's text
                if (!bias_now) { bias_now = true; apply_bias(); }
                std::mt19937 wr(1234);
                std::uniform_int_distribution<int> wtok(1000, 120000);
                const int p0 = (int) hist.size();
                std::string out = "width bench at " + std::to_string(p0) + " (median ms of " + std::to_string(wb_reps) + "):";
                for (int n = 1; n <= 7; ++n) {
                    double med[3] = { 0, 0, 0 };
                    for (int kind = 0; kind < 3; ++kind) {
                        if (kind == 2 && (n < 2 || wb_prev.size() < 16)) continue;
                        const int own = kind == 2 ? (n + 1) / 2 : n;   // two conversations: the first ceil(n/2) rows its own
                        std::vector<double> t;
                        for (int r = 0; r < wb_reps; ++r) {
                            std::vector<int> tk(n);
                            for (int i = 0; i < n; ++i)
                                tk[(size_t) i] = kind == 1 ? wtok(wr)
                                               : i < own ? hist[hist.size() - (size_t) own + (size_t) i]
                                                         : wb_prev[wb_prev.size() - (size_t) (n - own) + (size_t) (i - own)];
                            const double t0 = now_ms();
                            if (!pass_v(tk.data(), n, p0)) { std::fprintf(stderr, "width bench: %s\n", ferr.c_str()); break; }
                            t.push_back(now_ms() - t0);
                        }
                        if (t.empty()) break;
                        std::sort(t.begin(), t.end());
                        med[kind] = t[t.size() / 2];
                    }
                    char b[80];
                    if (med[2] > 0) std::snprintf(b, sizeof b, " n%d %.1f / %.1f / %.1f", n, med[0], med[1], med[2]);
                    else std::snprintf(b, sizeof b, " n%d %.1f / %.1f / -", n, med[0], med[1]);
                    out += b;
                }
                std::fprintf(stderr, "%s (own rows / random rows / two conversations' rows)\n", out.c_str());
                wb_prev = hist;
                hist.clear();   // the passes wrote past it: read fresh next time
                conv_img.clear();
                snap_len = -1;
#if defined(DS41_HAVE_DSPARK)
                dsp_next = -1;
#endif
            }
            if (admit >= 0) {   // larry (patch-slots-serve.py): BADM, and the slot decodes on in the windows'''),
]
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:90]))
    s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
