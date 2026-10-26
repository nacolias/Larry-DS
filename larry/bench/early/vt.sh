#!/bin/bash
# vt.sh NAME [engine args] - the engine under VTune gpu-offload; result in /opt/ai/dsv41/vtune/NAME
N=$1; shift
ARGS=("$@")
cd /opt/ai/dsv41 && source xe-env.sh
source /opt/intel/oneapi/vtune/latest/env/vars.sh >/dev/null 2>&1
mkdir -p vtune; rm -rf vtune/$N
XE_BIN=${XE_BIN} vtune -collect xpu-offload -r vtune/$N -- $XE_BIN/ds41_generate_xe -m $GGUF --backend gpu --experts gpu --engram $ENGRAM --rank $RANK "${ARGS[@]}" > logs/$N.out 2> logs/$N.err < /dev/null
echo "rc=$?" >> logs/$N.err
vtune -report hotspots -r vtune/$N -group-by computing-task -format csv -csv-delimiter tab 2>/dev/null > vtune/$N.tasks.tsv
