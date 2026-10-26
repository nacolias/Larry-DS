#!/usr/bin/env python3
# patch-shexpgen.py [GENERATE_CPP] - larry, STRATA_SHEXP_TIER=1 (default 0; with patch-shexptier.py and
# patch-shexpdense.py): the engine hands the TP core's shared-expert shards to the expert tier at load, and in a
# pipelined prompt batch every part of at least STRATA_SHEXP_TIER_MIN columns (default 256) has its shared expert
# computed by the expert cards - the dense half's spans of that part skip it (set_shexp_skip), its tier call does it
# (set_shexp_next), always the same part both ways.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "STRATA_SHEXP_TIER" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""            for (int k = 1; k < tp->n_cards(); ++k) {
                int64_t r0 = 0, r1 = 0;
                int gpu = -1;
                if (const void * p = tp->shard(nm, k, r0, r1, gpu)) tier.tp_head_slice(t->data, gpu, r0, r1, p);
            }
        }
    }
#endif
""", """            for (int k = 1; k < tp->n_cards(); ++k) {
                int64_t r0 = 0, r1 = 0;
                int gpu = -1;
                if (const void * p = tp->shard(nm, k, r0, r1, gpu)) tier.tp_head_slice(t->data, gpu, r0, r1, p);
            }
        }
        // larry, STRATA_SHEXP_TIER (patch-shexpgen.py): the shared expert's rank shards for the expert tier (prompt calls)
        if (std::getenv("STRATA_SHEXP_TIER") && std::atoi(std::getenv("STRATA_SHEXP_TIER")) != 0) {
            // every rank's slice of every layer, and together the whole n_ff (ffn_down_shexp's K) - else none
            struct Sl { int l, gpu; const void * gu; const void * dn; int64_t nf; };
            std::vector<Sl> sls;
            bool whole = true;
            for (int l = 0; l < n_layer && whole; ++l) {
                const ggml_tensor * wd = dense.weight("blk." + std::to_string(l) + ".ffn_down_shexp.weight");
                int64_t sum = 0;
                for (int k = 0; k < tp->n_cards(); ++k) {
                    Ds41TpShard gu, dn;
                    const std::string pre = "blk." + std::to_string(l) + ".";
                    if (!tp->shard_desc(pre + "ffn_gate_up_shexp.weight", k, gu) || !tp->shard_desc(pre + "ffn_down_shexp.weight", k, dn)) continue;
                    const int64_t nf = gu.a1 - gu.a0;
                    if (gu.kind != 1 || dn.kind != 2 || (dn.a1 - dn.a0) * 256 != nf) continue;
                    sls.push_back({l, gu.gpu, gu.p, dn.p, nf});
                    sum += nf;
                }
                whole = wd && sum == wd->ne[0];
                if (!whole) std::fprintf(stderr, "shared expert on the expert tier: layer %d's slices cover %lld of %lld - off\\n", l,
                                         (long long) sum, (long long) (wd ? wd->ne[0] : -1));
            }
            if (whole) for (const Sl & x : sls) tier.tp_shexp_slice(x.l, x.gpu, x.gu, x.dn, x.nf);
            std::fprintf(stderr, "shared expert on the expert tier: %zu slices, %s\\n", whole ? sls.size() : (size_t) 0,
                         whole && tier.shexp_ready() ? "every layer" : "off");
        }
    }
#endif
""")
rep("""                    auto PD = [&](int l, int a0, int a1) -> bool {
                        const double t = now_ms();
""", """                    // larry, STRATA_SHEXP_TIER: a part of >= STRATA_SHEXP_TIER_MIN columns has its shared expert on the expert cards
#if defined(DS41_XE)
                    static const bool sh_env = std::getenv("STRATA_SHEXP_TIER") && std::atoi(std::getenv("STRATA_SHEXP_TIER")) != 0;
                    static const int sh_min = std::getenv("STRATA_SHEXP_TIER_MIN") ? std::atoi(std::getenv("STRATA_SHEXP_TIER_MIN")) : 256;
                    const bool sh_on = sh_env && tier.shexp_ready();
#else
                    const bool sh_on = false;
                    const int sh_min = 0;
#endif
                    auto sh_part = [&](int a0, int a1) { return sh_on && a1 - a0 >= sh_min; };
                    auto PD = [&](int l, int a0, int a1) -> bool {
                        const double t = now_ms();
                        dense.set_shexp_skip(sh_part(a0, a1));   // larry, STRATA_SHEXP_TIER
""")
rep("""                        inflight = tier.run_chunk_dev_async(l, a1 - a0, dense.batch_ids() + (size_t) a0 * top_k,""",
    """#if defined(DS41_XE)
                        tier.set_shexp_next(sh_part(a0, a1));   // larry, STRATA_SHEXP_TIER: the same parts as PD's
#endif
                        inflight = tier.run_chunk_dev_async(l, a1 - a0, dense.batch_ids() + (size_t) a0 * top_k,""")
rep("""                    dense.set_pass_ring(false);
                    dense.set_submit_lock(nullptr);
                    dense.set_fn_host(true);
""", """                    dense.set_pass_ring(false);
                    dense.set_submit_lock(nullptr);
                    dense.set_fn_host(true);
                    dense.set_shexp_skip(false);   // larry, STRATA_SHEXP_TIER
""")
open(P, "w").write(s)
print("patched", P)
