#!/bin/bash
# swapctl.sh off|on|show|guard PID - larry's swap for the first-prompt stall A/B (2026-10-26, the owner's go-ahead):
# with no swap TTM has no backup space (ttm_backup_bytes_avail = free swap), so the xe shrinker cannot write the
# engine's buffers out.  guard PID: wait for PID (an abw.sh window) to end, then make sure swap is on again.
SW=/swapfile
on() { swapon --show=NAME --noheadings | grep -qx "$SW" || swapon "$SW"; }
case ${1:-show} in
  off) swapoff "$SW" ;;
  on) on ;;
  show) ;;
  guard) while kill -0 "$2" 2>/dev/null; do sleep 10; done; on ;;
  *) echo "usage: swapctl.sh off|on|show|guard PID"; exit 2 ;;
esac
echo "swap ${1:-show}: $(swapon --show=NAME,SIZE,USED --noheadings | tr -s ' ' | tr '\n' ' ')free $(free -g | awk '/Mem:/ {print $4}')G $(date +%T)"
