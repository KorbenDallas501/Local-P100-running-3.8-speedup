#!/usr/bin/env bash
# Thermal guard that aborts the REQUEST, not the server.
#
# STAYS ARMED after a trip. The previous version `break`ed out of its loop on the first
# trip, which left every later request in a multi-request sweep completely unguarded --
# cycle 2 ran two requests at 80 C with the clock collapsed to 405 MHz and accumulated
# 190 s of HW thermal slowdown. It also writes a TRIP FLAG so the caller can stop the
# sweep instead of continuing into a throttled card.
LIMIT=${LIMIT:-80}
LOG=${LOG:-$HOME/rc_guard.log}
FLAG=${FLAG:-$HOME/rc_tripped}
: > "$LOG"; rm -f "$FLAG"
while true; do
  R=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used \
      --format=csv,noheader,nounits 2>/dev/null)
  T=$(echo "$R" | cut -d, -f1 | tr -dc '0-9')
  if [ -z "$T" ]; then
    echo "$(date +%T) NO GPU READING - aborting request AND server" | tee -a "$LOG"
    date +%T > "$FLAG"
    pkill -f "[c]url.*809"; pkill -f "[l]lama-server"; break
  fi
  echo "$(date +%T), $R" >> "$LOG"
  if [ "$T" -ge "$LIMIT" ]; then
    echo "$(date +%T) !!! ${T}C >= ${LIMIT}C - ABORTING REQUEST" | tee -a "$LOG"
    date +%T > "$FLAG"
    pkill -f "[c]url.*809"
    # stay armed: a sweep has more requests queued and they must not run unguarded
  fi
  sleep 0.5
done
