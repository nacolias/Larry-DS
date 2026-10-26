#!/bin/bash
# smoke102.sh - abwT102's first probe: the new build (keephist + regrowasync, their flags off) one-shot on p2k's prompt
# path with the service's environment; its PPL against the old builds' 2.0035-2.0114.  On a failure the old binaries
# (logs/tier-rebuild/bin-20261025-2346) go back into build-xe and the window stops (AB_FLAG), so the service comes
# back on them.
cd /opt/ai/dsv41
source xe-env.sh
eval "$(grep '^export ' ds41-xe-engine.sh)"
unset STRATA_SERVE_DSPARK
RUN_TIMEOUT=900 ./run.sh smk102 --ids-file p2k.i32 -n 1 --ppl --prompt-batch 2048
P=$(grep -h PPL logs/smk102.out | awk '{print $2}')
rc=$(grep -h '^rc=' logs/smk102.err | tail -1)
ok=0
python3 -c "import sys; p = float(sys.argv[1] or 0); sys.exit(0 if 1.99 < p < 2.03 else 1)" "$P" && ok=1
echo "smoke102: PPL ${P:-none} $rc ok=$ok $(grep -h '^prompt:' logs/smk102.err | cut -c1-90)"
if [ "$ok" != 1 ] || [ "$rc" != "rc=0" ]; then
  B=logs/tier-rebuild/bin-20261025-2346
  cp $B/ds41_generate_xe xolotl/build-xe/ds41_generate_xe && cp $B/libds41_xe.so xolotl/build-xe/libds41_xe.so
  echo "smoke102: FAILED - the 23:46 binaries are back in build-xe; stopping the window"
  echo smoke102 > "${AB_FLAG:-/run/ds41-ab.yield}"
fi
