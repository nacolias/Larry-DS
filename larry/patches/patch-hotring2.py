#!/usr/bin/env python3
# patch-hotring2.py [TIER_CPP] - larry, after patch-hotring.py: with STRATA_XE_HOTRING=1 the prefetch's two pinned
# layer stages (2 x up to 1.5 GiB, taken out of STRATA_XE_PINNED_GIB) hold nothing - the stagers' rings carry every
# copy.  STRATA_XE_HOTRING_NOSTAGE (default 1 with the ring): no stage at all, its pinned budget goes to the arena (the
# hottest pageable experts DMA straight), the rings allocated at load (all or none: none = no prefetch) and the
# stage's size kept only as the planner's cap on a slot's pageable bytes.  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_HOTRING_NOSTAGE" in s:
    print("already patched"); sys.exit(0)
if "STRATA_XE_HOTRING" not in s:
    sys.exit("patch-hotring.py first")


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:90]))
    s = s.replace(old, new)


# forward declarations for build_arena_ranked (the helpers live in a later unnamed namespace)
rep("""bool Ds41XeTier::build_arena_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked_in, std::string& err) {
""", """namespace {
bool hot_on();
size_t hot_chunk();
int hot_slots();
}  // namespace

bool Ds41XeTier::build_arena_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked_in, std::string& err) {
""")

rep("""        const size_t want = (size_t) std::min<int64_t>(mx, (int64_t) (env_d("STRATA_XE_PREFETCH_STAGE_GIB", 1.5) * kGiB));
        for (auto& sl : im.pf) {
            sl.stage = want ? sycl::malloc_host<uint8_t>(want, im.ctx) : nullptr;
            if (!sl.stage) { im.pf_stage_cap = 0; break; }
            im.pf_stage_cap = want;
        }
        if (!im.pf_stage_cap) for (auto& sl : im.pf) if (sl.stage) { sycl::free(sl.stage, im.ctx); sl.stage = nullptr; }
        pin_cap -= 2 * (int64_t) im.pf_stage_cap;
        std::fprintf(stderr, "xe tier: prefetch staging 2 x %.2f GiB pinned (largest layer's arena share %.2f GiB)\\n",
                     im.pf_stage_cap / kGiB, mx / kGiB);
""", """        const size_t want = (size_t) std::min<int64_t>(mx, (int64_t) (env_d("STRATA_XE_PREFETCH_STAGE_GIB", 1.5) * kGiB));
        // larry, STRATA_XE_HOTRING with STRATA_XE_HOTRING_NOSTAGE (default 1): the stagers' rings carry every copy, so no
        // stage - its pinned budget goes to the arena; `want` stays the planner's cap on a slot's pageable bytes.  The
        // rings now, all or none (none: no prefetch at all - nothing could stage it)
        if (hot_on() && env_i("STRATA_XE_HOTRING_NOSTAGE", 1) && want) {
            const int ns = std::max(1, env_i("STRATA_XE_STAGERS", 4));
            bool all = true;
            for (int i = 0; i < ns; ++i) {
                im.pf_rings.push_back(sycl::malloc_host<uint8_t>((size_t) hot_slots() * hot_chunk(), im.ctx));
                all = all && im.pf_rings.back();
            }
            if (!all) {
                for (uint8_t* p : im.pf_rings) if (p) sycl::free(p, im.ctx);
                im.pf_rings.clear();
            }
            im.pf_stage_cap = all ? want : 0;
            std::fprintf(stderr, "xe tier: prefetch through %d hot rings of %d x %zu KiB (no stage: %.2f GiB more pinned arena; "
                         "largest layer's arena share %.2f GiB)%s\\n", ns, hot_slots(), hot_chunk() >> 10, 2 * want / kGiB, mx / kGiB,
                         all ? "" : " - RING ALLOCATION FAILED: no prefetch");
        } else {
            for (auto& sl : im.pf) {
                sl.stage = want ? sycl::malloc_host<uint8_t>(want, im.ctx) : nullptr;
                if (!sl.stage) { im.pf_stage_cap = 0; break; }
                im.pf_stage_cap = want;
            }
            if (!im.pf_stage_cap) for (auto& sl : im.pf) if (sl.stage) { sycl::free(sl.stage, im.ctx); sl.stage = nullptr; }
            pin_cap -= 2 * (int64_t) im.pf_stage_cap;
            std::fprintf(stderr, "xe tier: prefetch staging 2 x %.2f GiB pinned (largest layer's arena share %.2f GiB)\\n",
                         im.pf_stage_cap / kGiB, mx / kGiB);
        }
""")

# prefetch_begin: the rings may exist already (allocated at load)
rep("""        if (hot_on()) {   // STRATA_XE_HOTRING: a ring per stager (all or none: pf_enqueue counts on the stagers' copies)
""", """        if (hot_on() && im.pf_rings.empty()) {   // STRATA_XE_HOTRING: a ring per stager (all or none: pf_enqueue counts on the stagers' copies)
""")
open(P, "w").write(s)
print("patched", P)
