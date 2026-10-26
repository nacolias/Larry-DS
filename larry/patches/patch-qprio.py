#!/usr/bin/env python3
# patch-qprio.py [TIER_CPP] - larry, STRATA_XE_QPRIO (default 0 until measured): queue priorities on the EXPERT cards.
# A card has one copy engine open to user space (Level Zero ordinal 1; xe keeps bcs8) and it ran 90-100 % busy through
# a 16K prompt (bench/speed/engmon.sh, the xe PMU): the prefetch's staging chunks (pq) and the tier call's X read (q:
# card 0 -> this card, 84-168 MB a call) share it, so a call's X lands ~24 ms in (STRATA_XE_CALL_TL "A") for ~8 ms of
# copy.  1 = the prefetch queue at low priority (it only carries prompt staging there); 2 = also the call queue at
# high priority (X, the experts, the parts).  Card 0's queues stay normal.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_QPRIO" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""        c->q = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        c->pq = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        c->oq = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        c->name = c->dev.get_info<sycl::info::device::name>();""", """        // larry, STRATA_XE_QPRIO (patch-qprio.py): the expert cards' prefetch queue low (1), their call queue high (2)
        static const int qprio = env_i("STRATA_XE_QPRIO", 0);
        const bool expert_card = !(dense_dev && devs[(size_t) i] == ddev);
        namespace qp = sycl::ext::oneapi::property::queue;
        if (qprio >= 2 && expert_card)
            c->q = sycl::queue(im.ctx, c->dev, sycl::property_list{sycl::property::queue::in_order(), qp::priority_high()});
        else
            c->q = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        if (qprio >= 1 && expert_card)
            c->pq = sycl::queue(im.ctx, c->dev, sycl::property_list{sycl::property::queue::in_order(), qp::priority_low()});
        else
            c->pq = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        c->oq = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        c->name = c->dev.get_info<sycl::info::device::name>();""")
open(P, "w").write(s)
print("patched", P)
