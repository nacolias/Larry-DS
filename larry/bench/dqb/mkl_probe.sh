#!/bin/bash
# mkl_probe.sh - mkl_gemm_probe on card 1 (an abw.sh probe: the service must be stopped)
set --
source /opt/intel/oneapi/2026.1/oneapi-vars.sh > /dev/null 2>&1
ONEAPI_DEVICE_SELECTOR=level_zero:1 /opt/ai/dsv41/bench/dqb/mkl_gemm_probe
