#!/bin/bash
# abwT99: the cards' engine utilisation (xe PMU) through fresh 16K prompts - 512-column spans (k5) against 1024 (k10),
# where the expert cards' calls bound layers 0-19: are they GPU-bound or waiting on their hosts?
cd /opt/ai/dsv41
setsid nohup bash bench/speed/engwin.sh logs/abwT99.log k5 k10 > /dev/null 2>&1 < /dev/null &
T=STRATA_PIPE_TRACE=1
AB_PROMPTS=16000 setsid nohup bash bench/speed/abw.sh \
  k5:prompts:$T k10:prompts:GGML_SYCL_DSV4_SDPA_N=1024,AB_ARGS=--prefill-chunk:1024,$T > logs/abwT99.log 2>&1 < /dev/null &
