#!/bin/bash
# xetrace.sh [ABW_LOG] - the slow batch heads, kernel side.  gdb (abwT76) caught the head's thread inside the LM-head
# split's enqueues (mv_hook: a card's memcpy / kernel launch) in DRM_IOCTL_PRIME_HANDLE_TO_FD (running, 6.9 s) and in
# DRM_IOCTL_XE_EXEC waiting for the VM lock: a dma-buf peer import of card 0's memory?  For the A/B window (two fresh
# baseline engines, 8K..31K prompts) this records (1) function_graph durations over 2 ms of the xe exec / bind / create
# ioctls, the PRIME ioctls and the dma-buf / TTM / rebind paths under them, (2) in the instance "xecalls" every call
# of the import / allocation ioctls, (3) the xe rebind / evict / userptr events; each line of the A/B log goes in as a
# marker.  Out: logs/xetrace.HHMM.graph / .calls (trace clock: mono).
cd /opt/ai/dsv41
T=/sys/kernel/tracing
I=$T/instances/xecalls
A=${1:-logs/abwT77.log}
OUT=logs/xetrace.$(date +%H%M)
EV="xe_vm_rebind_worker_enter xe_vm_rebind_worker_exit xe_vm_rebind_worker_retry xe_vma_evict xe_vma_userptr_invalidate xe_vm_restart xe_vm_kill"
trace_off() {
  echo 0 > $T/tracing_on
  [ -d $I ] && echo 0 > $I/tracing_on
  cat $T/trace > $OUT.graph 2>/dev/null
  [ -d $I ] && cat $I/trace > $OUT.calls 2>/dev/null
  echo nop > $T/current_tracer
  echo 0 > $T/tracing_thresh
  echo > $T/set_ftrace_filter
  for e in $EV; do echo 0 > $T/events/xe/$e/enable; done
  if [ -d $I ]; then echo nop > $I/current_tracer; echo > $I/set_ftrace_filter; rmdir $I; fi
  echo 1408 > $T/buffer_size_kb
  echo local > $T/trace_clock
  echo > $T/trace
  echo "$(date +%T) tracing off: $OUT.graph ($(wc -l < $OUT.graph) lines), $OUT.calls ($(wc -l < $OUT.calls 2>/dev/null) lines)"
}
trap trace_off EXIT
echo 0 > $T/tracing_on
echo nop > $T/current_tracer
echo > $T/trace
echo mono > $T/trace_clock
echo 8192 > $T/buffer_size_kb
echo "drm_prime_handle_to_fd_ioctl drm_prime_fd_to_handle_ioctl drm_gem_prime_handle_to_dmabuf drm_gem_prime_fd_to_handle xe_gem_prime_export xe_gem_prime_import ttm_bo_setup_export xe_dma_buf_attach xe_dma_buf_map dma_buf_dynamic_attach dma_buf_map_attachment xe_ttm_vram_mgr_alloc_sgt xe_dma_buf_move_notify xe_exec_ioctl xe_vm_bind_ioctl xe_gem_create_ioctl xe_bo_move xe_bo_evict xe_bo_validate xe_bo_migrate ttm_tt_populate xe_vm_userptr_pin xe_vm_rebind preempt_rebind_work_func" > $T/set_ftrace_filter
echo 2000 > $T/tracing_thresh
echo function_graph > $T/current_tracer
echo 1 > $T/options/funcgraph-abstime
echo 1 > $T/options/funcgraph-proc
for e in $EV; do echo 1 > $T/events/xe/$e/enable; done
mkdir -p $I
echo 0 > $I/tracing_on
echo mono > $I/trace_clock
echo 16384 > $I/buffer_size_kb
echo "drm_prime_handle_to_fd_ioctl drm_prime_fd_to_handle_ioctl xe_gem_create_ioctl xe_vm_bind_ioctl xe_dma_buf_map xe_ttm_vram_mgr_alloc_sgt" > $I/set_ftrace_filter
echo function > $I/current_tracer
echo 1 > $I/tracing_on
echo 1 > $T/tracing_on
echo "$(date +%T) tracing on ($(cat $T/current_tracer), thresh $(cat $T/tracing_thresh) us; instance: $(cat $I/current_tracer))"
n=0
while :; do
  if [ -f $A ]; then
    m=$(wc -l < $A)
    if [ "$m" -gt "$n" ]; then
      sed -n "$((n + 1)),${m}p" $A | while IFS= read -r line; do echo "abw: ${line:0:200}" > $T/trace_marker; echo "abw: ${line:0:200}" > $I/trace_marker; done
      n=$m
    fi
    grep -q "^service health\|nothing run\|aborting" $A && break
  fi
  sleep 1
done
echo "$(date +%T) window over"
