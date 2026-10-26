import sys
p = sys.argv[1]
s = open(p).read()
if 'sycl_mqa' in s:
    print('already'); sys.exit(0)
old = "    bool own_backend = false, cpu = true;"
new = "    bool own_backend = false, cpu = true;\n    bool sycl = false;   // larry: the backend is ggml-sycl (its DSV4 prompt flash attention needs no 256-padding)"
assert s.count(old) == 1; s = s.replace(old, new)
old2 = "    im.cpu = ggml_backend_is_cpu(im.backend);"
new2 = "    im.cpu = ggml_backend_is_cpu(im.backend);\n    im.sycl = !im.cpu && std::strstr(ggml_backend_name(im.backend), \"SYCL\") != nullptr;"
assert s.count(old2) == 1; s = s.replace(old2, new2)
old3 = "    if (!im.cpu && M % 256) {   // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256"
new3 = """    // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256; ggml-sycl's DSV4 prompt path (>= 8
    // queries, fattn-dsv4.cpp in larry's ggml-sycl patch) takes any multiple of 16 - padding there only moves bytes
    const char * mqa_env = std::getenv("GGML_SYCL_DSV4_MQA");
    const bool sycl_mqa = im.sycl && n >= 8 && M % 16 == 0 && !(mqa_env && mqa_env[0] == '0');
    if (!im.cpu && !sycl_mqa && M % 256) {"""
assert s.count(old3) == 1; s = s.replace(old3, new3)
if '#include <cstring>' not in s:
    s = s.replace('#include <cmath>', '#include <cmath>\n#include <cstring>', 1)
open(p, 'w').write(s)
print('ok')
