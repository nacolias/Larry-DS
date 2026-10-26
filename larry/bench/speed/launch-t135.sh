#!/bin/bash
# launch-t135.sh - abwT135: two fresh servers on the boot memory settings with memtrace2.bt (memtrace.bt + the xe
# shrinker's BO backups to shmem / swap, their restores, the preempt-rebind worker): are the first prompts' stalls
# restores of BOs the shrinker wrote out while the engine loaded?
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh "@BT=memtrace2.bt bash bench/speed/memtrace-start.sh T135 30" \
  "@bash bench/speed/memset.sh default" e1:prompts e2:prompts "@bash bench/speed/memtrace-stop.sh" > logs/abwT135.log 2>&1 < /dev/null &
echo "abwT135 launched: logs/abwT135.log"
