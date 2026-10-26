#!/bin/bash
# ds41-serve.sh - DeepSeek-V4.1-Flash-1017 (GSQ, 163 GiB) behind Xolotl's OpenAI / Anthropic server
# (tools/v41/serve/serve_ds41.py: DeepSeek's chat encoder, DSML tool calls, reasoning effort).  Run by ds41.service
# (host/), or by hand - one engine per box.  The engine (ds41_generate_xe: the dense half on ggml-sycl + the Intel
# multi-card expert tier) runs through ds41-xe-engine.sh; its flags and the model paths are in larry-ds41.json.
#   DS41_CONFIG (DS41_HOME/larry-ds41.json)  DS41_HOST (127.0.0.1; larry: its docker bridge, 172.23.0.1)  DS41_PORT (18080)
#   STRATA_API_KEY, or DS41_KEY_FILE: a file with a VLLM_API_KEY=... line (larry's is shared with its other servers);
#     neither = no key (anyone who reaches DS41_HOST:DS41_PORT can use the server)
#   DS41_UPLINK_PCI: a PCIe port whose link is checked at start (larry: 0000:00:03.1, the cards' switch uplink, which
#     should read 16.0 GT/s x16)
#   DS41_SERVER_EXTRA_ARGS: more serve_ds41.py flags
set -euo pipefail
D=${DS41_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
CFG=${DS41_CONFIG:-$D/larry-ds41.json}
HOST=${DS41_HOST:-127.0.0.1}
PORT=${DS41_PORT:-18080}
[ -f "$CFG" ] || { echo "ds41-serve: config missing: $CFG" >&2; exit 1; }
[ -x "$D/venv/bin/python" ] || { echo "ds41-serve: $D/venv missing" >&2; exit 1; }

# the server reads STRATA_API_KEY (an empty one is refused)
if [ -z "${STRATA_API_KEY:-}" ] && [ -n "${DS41_KEY_FILE:-}" ]; then
  umask 077
  key=$(grep -E '^VLLM_API_KEY=' "$DS41_KEY_FILE" | cut -d= -f2-)
  umask 022
  [ -n "$key" ] || { echo "ds41-serve: no VLLM_API_KEY in $DS41_KEY_FILE" >&2; exit 1; }
  export STRATA_API_KEY=$key
fi
if [ -z "${STRATA_API_KEY:-}" ]; then
  unset STRATA_API_KEY
  echo "ds41-serve: no API key - anyone who reaches $HOST:$PORT can use the server" >&2
fi

if [ -n "${DS41_UPLINK_PCI:-}" ]; then
  P=/sys/bus/pci/devices/$DS41_UPLINK_PCI
  UPLINK="$(cat $P/current_link_speed 2>/dev/null) x$(cat $P/current_link_width 2>/dev/null)"
  if [ "$UPLINK" = "16.0 GT/s PCIe x16" ]; then echo "ds41-serve: PCIe uplink $UPLINK"
  else echo "ds41-serve: WARNING: PCIe uplink is $UPLINK, not 16.0 GT/s PCIe x16" >&2; fi
fi

read -r -a EXTRA <<<"${DS41_SERVER_EXTRA_ARGS:-}"
cd "$D/xolotl"
exec "$D/venv/bin/python" tools/v41/serve/serve_ds41.py --engine strata --config "$CFG" --host "$HOST" --port "$PORT" "${EXTRA[@]}"
