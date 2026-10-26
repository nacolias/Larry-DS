#!/bin/bash
# gpu-clock.sh pin|restore - pin or restore the compute-engine (GT0) minimum clock of every card bound to xe (larry's
# larry-gpu-clock; ds41.service runs it as root, ExecStartPre=+ / ExecStopPost=+).
#   pin      remember each card's min_freq, then min_freq = rp0 (the hardware max, 2800 MHz on a B70) while the server
#            runs: a layer-split server leaves each card idle part of the time and the cards downclock between bursts
#            (pinning measured +15-23 % decode on larry, 2026-09-24, with no visible change in idle power).
#   restore  only on a card that is still pinned (min_freq = rp0): min_freq = the value pin remembered (rpn, the
#            400 MHz hardware minimum, if none was saved), then the saved value is deleted.  ExecStopPost runs even
#            when ExecStartPre failed before pin ran; a card that is not pinned is left as it is, so an old saved
#            value is never written back.
# The cards are every PCI device bound to xe (a renumbered bus cannot skip one).  Never fails the caller: a card that
# cannot be set is reported and skipped.
set -u
mode=${1:-}
SAVE=/var/lib/ds41-gpu-clock
case "$mode" in pin|restore) ;; *) echo "usage: gpu-clock.sh pin|restore" >&2; exit 0 ;; esac
mkdir -p "$SAVE" 2>/dev/null
n=0
for dev in /sys/bus/pci/drivers/xe/0000:*; do
  [ -e "$dev" ] || continue
  bdf=${dev##*/}; f=$dev/tile0/gt0/freq0; n=$((n + 1))
  [ -d "$f" ] || { echo "gpu-clock: $bdf has no $f, skipped" >&2; continue; }
  if [ "$mode" = pin ]; then
    cur=$(cat "$f/min_freq"); v=$(cat "$f/rp0_freq")
    [ "$cur" = "$v" ] || echo "$cur" >"$SAVE/$bdf.min_freq"   # a second pin must not save the pinned value
  else
    [ "$(cat "$f/min_freq")" = "$(cat "$f/rp0_freq")" ] || continue   # not pinned: leave it
    v=$(cat "$SAVE/$bdf.min_freq" 2>/dev/null)
    case "$v" in ''|*[!0-9]*) v=$(cat "$f/rpn_freq") ;; esac
  fi
  if echo "$v" >"$f/min_freq" 2>/dev/null; then
    [ "$mode" = restore ] && rm -f "$SAVE/$bdf.min_freq"
  else
    echo "gpu-clock: could not set $bdf min_freq=$v" >&2
  fi
done
[ "$n" -gt 0 ] || echo "gpu-clock: no xe devices found" >&2
exit 0
