#!/usr/bin/env python3
# patch-opprof5.py [GGML_SYCL_CPP] - larry, GGML_SYCL_OP_PROF_SHAPES=1 (with GGML_SYCL_OP_PROF=1): each label also
# carries the node's [ne0 x ne1] and its first source's name (digits dropped: the layers fold together) and type -
# which matrix a MUL_MAT is, which row size an RMS_NORM has.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(P).read()
if "GGML_SYCL_OP_PROF_SHAPES" in s:
    print("already patched"); sys.exit(0)
old = """    if (!n.empty()) l += " " + n;
    auto it = st().names.insert(l).first;
    return it->c_str();
}
}  // namespace larry_opprof"""
new = """    if (!n.empty()) l += " " + n;
    static const bool shapes = std::getenv("GGML_SYCL_OP_PROF_SHAPES") != nullptr;   // larry, patch-opprof5.py
    if (shapes) {
        char b[48];
        snprintf(b, sizeof b, " [%lld x %lld]", (long long) t->ne[0], (long long) t->ne[1]);
        l += b;
        if (const ggml_tensor * s0 = t->src[0]) {
            std::string w;
            for (const char * p = s0->name; *p && w.size() < 32; ++p) if (!(*p >= '0' && *p <= '9')) w += *p;
            l += " " + w + " " + ggml_type_name(s0->type);
        }
    }
    auto it = st().names.insert(l).first;
    return it->c_str();
}
}  // namespace larry_opprof"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
