#!/bin/bash
# memset.sh default|tuned|show [WSF] [BOOST] - larry's kernel memory settings for the first-prompt stall A/B (2026-10-26;
# the owner approved trying them).  tuned: vm.watermark_scale_factor WSF (default 200: kswapd wakes ~2.4 GiB above the
# min watermark instead of ~0.12 GiB, so allocation bursts find free pages instead of reclaiming in the engine's
# threads), vm.compaction_proactiveness 0 (no background compaction: its page migrations invalidate the cards' userptr
# mappings), THP defrag "defer" (no direct compaction on a THP fault); BOOST, when given, sets vm.watermark_boost_factor.
# default: the boot values (10, 20, madvise, boost 15000).  Runtime only - nothing here survives a reboot.
set -u
M=${1:-show}
case $M in
  default)
    sysctl -q -w vm.watermark_scale_factor=10 vm.compaction_proactiveness=20 vm.watermark_boost_factor=15000
    echo madvise > /sys/kernel/mm/transparent_hugepage/defrag ;;
  tuned)
    sysctl -q -w vm.watermark_scale_factor=${2:-200} vm.compaction_proactiveness=0
    [ -n "${3:-}" ] && sysctl -q -w vm.watermark_boost_factor=$3
    echo defer > /sys/kernel/mm/transparent_hugepage/defrag ;;
  show) ;;
  *) echo "usage: memset.sh default|tuned|show [WSF] [BOOST]"; exit 2 ;;
esac
echo "memset $M: watermark_scale_factor=$(sysctl -n vm.watermark_scale_factor)" \
  "compaction_proactiveness=$(sysctl -n vm.compaction_proactiveness)" \
  "watermark_boost_factor=$(sysctl -n vm.watermark_boost_factor)" \
  "thp_defrag=$(sed 's/.*\[\(.*\)\].*/\1/' /sys/kernel/mm/transparent_hugepage/defrag)" \
  "free=$(awk '/MemFree/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)G $(date +%T)"
