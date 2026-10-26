# xe-env.sh - the environment of every DeepSeek-V4.1 Xe engine run (sourced; clears "$@" - capture arguments first)
#   DS41_HOME (this file's directory)  DS41_SRV (/srv/dsv41: the second NVMe drive)  DS41_MODEL_DIR (DS41_HOME/models/gsq1017)
#   ONEAPI_VARS (/opt/intel/oneapi/2026.1/oneapi-vars.sh)  XE_BIN (DS41_HOME/xolotl/build-xe)
D=${DS41_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SRV=${DS41_SRV:-/srv/dsv41}
set -- ; source "${ONEAPI_VARS:-/opt/intel/oneapi/2026.1/oneapi-vars.sh}" >/dev/null
export ZES_ENABLE_SYSMAN=1                     # free-VRAM queries (sycl ext_intel_free_memory)
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu   # the four B70s, Level Zero only
export NEOReadDebugKeys=1 EnableRecoverablePageFaults=0   # larry's B70 stability keys (upstream Strata #2157)
export SYCL_CACHE_PERSISTENT=1
export GGML_NO_BACKTRACE=1
M=${DS41_MODEL_DIR:-$D/models/gsq1017}
GGUF=$M/V41-GSQ-1017.gguf
ENGRAM=$M/engram-layers.1:$SRV/gsq1017/engram-layers.14
RANK=$M/rank.bin
SIDECAR=$M/V41-MTP-Vision-1017.Q2_K.gguf
XE_BIN=${XE_BIN:-$D/xolotl/build-xe}
# larry: the NEO driver copies small H2D / D2H transfers on the CPU after waiting for the queue to go idle - with the
# pipelined prompt's span ring (two spans queued) every small index upload waited ~5 ms holding the submit lock (31.6K
# prompt: 7.4 s); decode measured the same either way
export NEOReadDebugKeys=1 ExperimentalH2DCpuCopyThreshold=0 ExperimentalD2HCpuCopyThreshold=0
