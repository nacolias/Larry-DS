#!/usr/bin/env python3
# patch-parkcap.py [GENERATE_CPP] - larry: a parked conversation saves only its OWN rows.  The park saved every
# position the side request could write (its prompt + max tokens + 64): an agent's side call asks for max_tokens 32000,
# so a 12,588-token main conversation was saved as 47,012 positions (2026-10-24 14:13 session) - 3.7x the copies, out
# and back in, for rows the main conversation never reads (it re-writes everything past its snapshot when it goes on).
# Now min(that, snapshot + 1) positions; the park still expires when a later request writes past the side request's
# reach (park_rows unchanged).  A side request whose reach exceeds STRATA_SERVE_PARK_MAX no longer skips the park when
# the conversation itself fits.  The park and unpark lines print their positions and seconds.  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "the conversation's own rows" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:90]))
    s = s.replace(old, new)


rep("""                const int64_t rows = (int64_t) ids.size() + (max_new > 0 ? max_new : park_max) + 64;
                if (park_on && !dense.parked() && snap_len >= park_min && hist.size() >= (size_t) snap_len && rows <= park_max
                    && dense.park(rows)) {
                    park_hist.assign(hist.begin(), hist.begin() + snap_len);
                    park_logits = snap_logits;
                    park_rows = rows;
                    std::fprintf(stderr, "serve: parked the %d-token conversation (%lld positions saved)\\n", snap_len, (long long) rows);
                }
""", """                const int64_t rows = (int64_t) ids.size() + (max_new > 0 ? max_new : park_max) + 64;
                // larry: the conversation's own rows only - it re-writes everything past its snapshot when it goes on
                // (an agent's side call asks for 32000 tokens: a 12.6K conversation was saved as 47K positions)
                const int64_t save = std::min<int64_t>(rows, (int64_t) snap_len + 1);
                const double tpk = now_ms();
                if (park_on && !dense.parked() && snap_len >= park_min && hist.size() >= (size_t) snap_len && save <= park_max
                    && dense.park(save)) {
                    park_hist.assign(hist.begin(), hist.begin() + snap_len);
                    park_logits = snap_logits;
                    park_rows = rows;
                    std::fprintf(stderr, "serve: parked the %d-token conversation (%lld positions saved, valid to %lld; %.2f s)\\n",
                                 snap_len, (long long) save, (long long) rows, (now_ms() - tpk) / 1000.0);
                }
""")
rep("""                if (park_partial && lp < park_hist.size() && lp >= lp_min && lp > lcp && dense.unpark()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation at lcp %zu (partial: re-feeds from there)\\n",
                                 park_hist.size(), lp);
""", """                const double tup = now_ms();
                if (park_partial && lp < park_hist.size() && lp >= lp_min && lp > lcp && dense.unpark()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation at lcp %zu (partial: re-feeds from there; %.2f s)\\n",
                                 park_hist.size(), lp, (now_ms() - tup) / 1000.0);
""")
rep("""                } else if (lp == park_hist.size() && lp > lcp && dense.unpark()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation\\n", park_hist.size());
""", """                } else if (lp == park_hist.size() && lp > lcp && dense.unpark()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation (%.2f s)\\n", park_hist.size(), (now_ms() - tup) / 1000.0);
""")
open(P, "w").write(s)
print("patched", P)
