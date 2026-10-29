#!/usr/bin/env python3
# patch-kvahead.py [XOLOTL_DIR] - larry, 2026-10-29 (decode at depth): the expert cards' evictions for the TP core's KV rows
# read ahead STRATA_TP_DEC_AHEAD tokens (2048; 0 = the old behaviour) before a conversation needs them.  When its KV
# replicas outgrow their bucket (65,536, 131,072, ... rows) STRATA_XE_TP_EVICT gives a 1 GiB slab of cards 1 and 3 back to
# the arena and reads its experts' host copies from the model file - 0.6-3.2 s in the middle of a turn in the 2026-10-29
# replays.  patch-roomahead.py's read-ahead covered long prompts only: it was checked after a prompt batch with more to
# come (a follow-up's single batch never was), and every request dropped what an earlier one had read.  Now every point
# that asks - a request's start, a batch's start and end, a reply's passes - asks for the rows of its position plus
# STRATA_TP_DEC_AHEAD, so they agree while a conversation sits near a bucket boundary over several short turns: the read
# runs once, beside the prompt or the reply, and the crossing adopts it.  (A first version asked from the replies only and
# kept the start-of-request drop: the read-ahead was dropped and queued again every turn near 65,536 - 2 GiB re-read a
# turn - and the crossing waited 0.8 s for the last one.)  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
G = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
s = open(G).read()
if "patch-kvahead.py" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new, what):
    global s
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:90]))
    s = s.replace(old, new)


# the lead, one place (an `ahead` lambda beside feed: the tokens past a position a read-ahead covers)
rep("""    std::function<void()> feed_seam;
""", """    std::function<void()> feed_seam;
    // larry (patch-kvahead.py, STRATA_TP_DEC_AHEAD, 2048 tokens; 0 = off): how far past a position the expert cards'
    // evictions for the TP core's KV rows are read ahead - the same at every point that asks, so they keep one read
    const int64_t kv_ahead = [] {
        const char * e = std::getenv("STRATA_TP_DEC_AHEAD");
        return e ? std::max<int64_t>(0, std::atoll(e)) : (int64_t) 2048;
    }();
""", "kv_ahead")
# a batch's end (patch-roomahead.py's check) and the last batch's start
rep("""                    if (ahead_s > 0 && left_ms <= ahead_s * 1000.0) tp->room_ahead((int64_t) pos0 + n + 64);""",
    """                    if (ahead_s > 0 && left_ms <= ahead_s * 1000.0) tp->room_ahead((int64_t) pos0 + n + std::max<int64_t>(64, kv_ahead));""",
    "after batch")
rep("""                btr("batch start", B);
""", """                btr("batch start", B);
#if defined(DS41_TP)
                // larry (patch-kvahead.py): the prompt's last batch - a follow-up's only one - asks as it starts (the
                // check after a batch below never ran for it): the reads run beside the batch, the sync after it adopts them
                if (tp && kv_ahead > 0 && c0 + B >= n) tp->room_ahead((int64_t) pos0 + n + kv_ahead);
#endif
""", "last batch")
# a request's start: what this conversation will need soon kept, not dropped
rep("""            if (tp) tp->room_ahead(-1);   // larry (patch-roomahead.py): an earlier prompt's read-ahead left over goes
""", """            // larry (patch-roomahead.py): an earlier prompt's read-ahead left over goes - (patch-kvahead.py) unless this
            // conversation's rows within STRATA_TP_DEC_AHEAD of its prompt's end want the same
            if (tp) tp->room_ahead(kv_ahead > 0 ? (int64_t) total + kv_ahead : -1);
""", "request start")
# a reply's passes: when its position moves into the lead of a new bucket
rep("""            int lk_cool = 0, lk_back = 1;
""", """            int lk_cool = 0, lk_back = 1;
            int64_t ahead_b = 0;   // larry (patch-kvahead.py): the KV bucket this reply asked a read-ahead for
""", "DecState")
rep("""            const bool more = dspark_accept(d, v, vlog.data(), vh.data(), pv_tp ? 0 : -1);
""", """            const bool more = dspark_accept(d, v, vlog.data(), vh.data(), pv_tp ? 0 : -1);
#if defined(DS41_TP)
            // larry (patch-kvahead.py): the reply's position within STRATA_TP_DEC_AHEAD of a KV replica bucket (a power
            // of two of positions, from 1024) not asked for yet: ask (the same rows the next request's start will want)
            if (tp && more && kv_ahead > 0) {
                int64_t b = 1024;
                while (b <= (int64_t) d.P + 1) b <<= 1;
                if ((int64_t) d.P + kv_ahead >= b && b > d.ahead_b) {
                    d.ahead_b = b;
                    tp->room_ahead((int64_t) d.P + kv_ahead);
                }
            }
#endif
""", "dspark_cycle")
open(G, "w").write(s)
print("patched", G)
