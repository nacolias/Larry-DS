#!/bin/bash
# abwT93: the AVX2 Q2_0 dot (patch-q20avx2.py, bit-exact, 2.2x a core) in the CPU pool - g0 the scalar generic
# (GGML_Q20_GENERIC=1) against g1; short fresh prompts and the decode test
cd /opt/ai/dsv41
AB_PROMPTS=512,700,1000,2000,3000 setsid nohup bash bench/speed/abw.sh \
  g0:prompts,dec:GGML_Q20_GENERIC=1 g1:prompts,dec > logs/abwT93.log 2>&1 < /dev/null &
