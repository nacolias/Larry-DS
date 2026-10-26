#!/usr/bin/env python3
# patch-pull.py - larry, STRATA_XE_PULL (default 0 for the A/B): the dense card PULLS the expert cards' decode / verify
# parts (a copy on the dense queue reading the card's OUT, after that card's expert event) instead of each card
# pushing its part into the dense card's slot (a P2P write on the card's own queue).  The pushed parts became visible
# to the dense queue's sum ~25 us (decode, 20 KB) / ~120 us (verify, 112 KB) after the card's copy completed - the
# lag scales with the bytes, as if the peer writes drained at ~1 GB/s behind the completion.  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_PULL" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


rep("""    bool cpu_async = true;
""", """    bool cpu_async = true;
    bool pull = false;   ///< STRATA_XE_PULL: the dense card copies the expert cards' parts in (no peer writes)
""")
rep("""    im.cpu_async = env_i("STRATA_XE_CPU_ASYNC", 1) != 0;
""", """    im.cpu_async = env_i("STRATA_XE_CPU_ASYNC", 1) != 0;
    im.pull = env_i("STRATA_XE_PULL", 0) != 0;
""")
# arm(): the armed sequence ends with the experts (pull) or the push copy
rep("""        card_experts(im, c, ng, nt, mv, xoff, c.OUT, im.arm_x);
        if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 11);
        c.arm_ev = c.q.memcpy(slots + (size_t) c.idx * kMaxJobTok * H, c.OUT, (size_t) nt * H * 4);""",
    """        sycl::event xev = card_experts(im, c, ng, nt, mv, xoff, c.OUT, im.arm_x);
        if (Ds41XeTier::tl_on()) tl_stamp(im, 1 + (size_t) c.idx, c.q, 11);
        // STRATA_XE_PULL: the dense card reads c.OUT itself after this event (the sum's call queues that copy)
        c.arm_ev = im.pull ? xev : c.q.memcpy(slots + (size_t) c.idx * kMaxJobTok * H, c.OUT, (size_t) nt * H * 4);""")
# dispatch(): no push copy when pulling
rep("""                if (!c.dense && !im.host_slots) ev = c.q.memcpy(slots + (size_t) c.idx * kMaxJobTok * H, c.OUT, n_out * 4);""",
    """                if (!c.dense && !im.host_slots && !im.pull) ev = c.q.memcpy(slots + (size_t) c.idx * kMaxJobTok * H, c.OUT, n_out * 4);""")
# the armed cards' events: a used card's part is pulled
rep("""        std::vector<sycl::event> deps;
        std::vector<int> used;""", """        std::vector<sycl::event> deps;
        std::vector<int> used;
        std::vector<std::pair<int, sycl::event>> pulls;   // STRATA_XE_PULL: (card, its experts' event) to copy in""")
rep("""            arm_release(im, c);
            c.armed = false;
            c.stage_ev = c.arm_ev;
            deps.push_back(c.arm_ev);""", """            arm_release(im, c);
            c.armed = false;
            c.stage_ev = c.arm_ev;
            if (im.pull && nj[(size_t) c.idx] > 0) pulls.push_back({c.idx, c.arm_ev});
            else deps.push_back(c.arm_ev);""")
rep("""        for (size_t k = 0; k < todo.size(); ++k) { deps.push_back(evs[k]); used.push_back(todo[k]->idx); }""",
    """        for (size_t k = 0; k < todo.size(); ++k) {
            if (im.pull && !todo[k]->dense && !im.host_slots) pulls.push_back({todo[k]->idx, evs[k]});
            else deps.push_back(evs[k]);
            used.push_back(todo[k]->idx);
        }""")
# before the sum: the pulls on the dense queue
rep("""            if (Ds41XeTier::tl_on() && nt == 1 && im.dense_q) tl_stamp(im, 0, qd, 15);   // the dense queue is free""",
    """            for (const auto& pr : pulls) {   // STRATA_XE_PULL: each part read by the dense card's queue
                XeCard& pc = *im.cards[(size_t) pr.first];
                const sycl::event pdep = pr.second;
                deps.push_back(qd.submit([&](sycl::handler& h) {
                    h.depends_on(pdep);
                    h.memcpy(slots + (size_t) pc.idx * kMaxJobTok * H, pc.OUT, n_out * 4);
                }));
            }
            if (Ds41XeTier::tl_on() && nt == 1 && im.dense_q) tl_stamp(im, 0, qd, 15);   // the dense queue is free""")
open(P, "w").write(s)
print("patched")
