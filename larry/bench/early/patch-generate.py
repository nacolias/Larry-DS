import re, sys
p = sys.argv[1]
s = open(p).read()
if 'DS41_XE' in s:
    print('already patched'); sys.exit(0)
s = s.replace('#include "ds41_moe.hpp"\n', '#include "ds41_moe.hpp"\n#if defined(DS41_XE)\n#include "ds41_xe_tier.hpp"   // larry: the multi-card Intel Arc (SYCL) expert tier\n#endif\n', 1)
s = s.replace('using namespace strata::v41;\n', 'using namespace strata::v41;\n#if defined(DS41_XE)\nusing Ds41Tier = Ds41XeTier;\n#else\nusing Ds41Tier = Ds41MoeTier;\n#endif\n', 1)
s = s.replace('    Ds41MoeTier tier;\n', '    Ds41Tier tier;\n', 1)
s = s.replace('Ds41MoeTier::', 'Ds41Tier::')
old = '    if (!tier.init(TG, &src, mc, err)) {'
new = ('#if defined(DS41_XE)\n'
       '    if (!tier.init(TG, &src, mc, mc.cpu_only ? nullptr : dense.routed_dev(), err)) {'
       ' std::fprintf(stderr, "ds41_generate: tier init: %s\\n", err.c_str()); return 1; }\n'
       '#else\n' + old)
i = s.index(old)
j = s.index('\n', i)
s = s[:i] + new + s[i:j] + '\n#endif' + s[j:]
open(p, 'w').write(s)
print('patched')
