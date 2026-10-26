#!/usr/bin/env python3
# patch-ltquant.py [TIER_CPP] - larry, STRATA_XE_LT_QUANT_THREADS (default 0 = the old serial form; 4 measured no gain, abwT69): a prompt call's
# CPU-light experts need their tokens' activations quantised (int8 / the native formats) before the pool runs them;
# the tier thread did that one row after another inside the job setup - ~7 ms a 512-token call (2.9 s of the 9.7 s the
# light part took over 400 calls, abwT68), and in short prompts the light part ends after the cards (it bounds the
# call).  Now each job batch's (token, format) pairs are listed first and quantised on helper threads (the same
# functions into the same buffers), then the jobs only point at them.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_LT_QUANT_THREADS" in s:
    print("already patched"); sys.exit(0)
old = """            while (light_ok && done < light.size()) {
                const size_t nb = std::min<size_t>(light.size() - done, (size_t) cpu::ExpertPool::kMaxSplitMulti);
                for (size_t i = 0; i < nb; ++i) {
                    const int32_t e = light[done + i];
"""
new = """            while (light_ok && done < light.size()) {
                const size_t nb = std::min<size_t>(light.size() - done, (size_t) cpu::ExpertPool::kMaxSplitMulti);
                // larry, STRATA_XE_LT_QUANT_THREADS (patch-ltquant.py): this batch's activations quantised first, on
                // helper threads (each (token, format) once; the job setup below then finds them done)
                static const int lq_thr = std::max(0, std::min(16, env_i("STRATA_XE_LT_QUANT_THREADS", 0)));
                if (lq_thr > 0) {
                    struct LQ { int t; int fs; const cpu::NativeFmt* f; };
                    std::vector<LQ> lq;
                    for (size_t i = 0; i < nb; ++i) {
                        const int32_t e = light[done + i];
                        const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
                        int fs = -1;
                        if (!cpu::q2_native_kernels(f.gu_type))
                            for (size_t b = 0; b < nf; ++b) if (im.nact_fmt[b] == f.gu_act) fs = (int) b;
                        for (int32_t sk : by_e[(size_t) e]) {
                            const int t = sk / K;
                            uint8_t& hv = im.lt_have[(size_t) t * (1 + nf) + (size_t) (fs + 1)];
                            if (!hv) { hv = 1; lq.push_back({t, fs, &f}); }
                        }
                    }
                    auto work = [&](size_t k0, size_t step) {
                        for (size_t k = k0; k < lq.size(); k += step) {
                            const LQ& q = lq[k];
                            if (q.fs < 0) cpu::act_quant_any(xrow(q.t), (int) H, im.lt_actq[(size_t) q.t]);
                            else cpu::native_quant_act(*q.f, xrow(q.t), im.lt_nact.data() + ((size_t) q.t * nf + (size_t) q.fs) * cpu::kNativeActBytes);
                        }
                    };
                    const size_t nh = std::min<size_t>((size_t) lq_thr, lq.size() / 16);
                    std::vector<std::thread> th;
                    for (size_t k = 1; k <= nh; ++k) th.emplace_back(work, k, nh + 1);
                    work(0, nh + 1);
                    for (auto& t : th) t.join();
                }
                for (size_t i = 0; i < nb; ++i) {
                    const int32_t e = light[done + i];
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
