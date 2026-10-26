#!/bin/bash
# vmwatch.sh OUT SECONDS - /proc/vmstat deltas every 2 s (page migration, compaction, THP collapse, reclaim) + the
# engine's existence, for SECONDS: what the kernel does to host memory while a fresh engine runs its first prompts
OUT=$1; N=$(( ${2:-900} / 2 ))
K="pgmigrate_success compact_stall compact_migrate_scanned compact_isolated thp_collapse_alloc thp_fault_alloc pgscan_kswapd pgsteal_kswapd pgscan_direct numa_pages_migrated"
prev=""
for i in $(seq 1 $N); do
  cur=$(awk -v k="$K" 'BEGIN{n=split(k,a," "); for(i=1;i<=n;i++) w[a[i]]=1} ($1 in w){printf "%s=%s ", $1, $2}' /proc/vmstat)
  if [ -n "$prev" ]; then
    line="$(date +%T) eng=$(pgrep -x ds41_generate_x | head -1)"
    for kv in $cur; do k=${kv%%=*}; v=${kv#*=}; p=$(echo "$prev" | tr ' ' '\n' | grep "^$k=" | cut -d= -f2); d=$((v - p)); [ $d -ne 0 ] && line="$line $k+$d"; done
    echo "$line"
  fi
  prev=$cur
  sleep 2
done > $OUT
