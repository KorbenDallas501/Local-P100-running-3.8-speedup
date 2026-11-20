#!/usr/bin/env bash
# $1 = n-max   $2 = p_min   $3 = arm index
set -u
NM=$1; P=$2; IDX=$3
for x in $(pgrep -f "[l]lama-server"); do kill -9 $x 2>/dev/null; done; sleep 3
T0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -dc '0-9')
if [ "${T0:-99}" -ge 74 ]; then echo "arm$IDX nmax=$NM p=$P SKIPPED ${T0}C"; exit 0; fi
NMAX=$NM PMIN=$P nohup bash $HOME/run_ridge_mtp.sh > $HOME/b_$IDX.log 2>&1 &
ok=0
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:8081/health)" = 200 ] && { ok=1; break; }
  grep -qiE "CUDA error|out of memory|unknown argument" $HOME/b_$IDX.log && break
  sleep 3
done
[ $ok = 1 ] || { echo "arm$IDX nmax=$NM p=$P SERVER FAILED"; grep -iE "error|out of mem" $HOME/b_$IDX.log | tail -2; exit 1; }
ACT=$(grep -oE 'n_max=[0-9]+, n_min=[0-9]+, p_min=[0-9.]+' $HOME/b_$IDX.log | head -1)
LIMIT=78 LOG=$HOME/guard_b$IDX.log nohup bash $HOME/guard_req.sh >/dev/null 2>&1 &
curl -s -m 600 http://127.0.0.1:8081/completion -H 'Content-Type: application/json' \
  -d @$HOME/prose_p1.json -o /tmp/b_$IDX.json
for g in $(pgrep -f "[g]uard_req"); do kill $g 2>/dev/null; done
T1=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -dc '0-9')
# tau and mean width from the server's own draft accounting when present
R=$(python3 -c "
import json
try:
    d=json.load(open('/tmp/b_$IDX.json')); t=d.get('timings',{})
    print('%.2f %.1f'%(t.get('predicted_per_second',0), t.get('predicted_ms',0)))
except Exception: print('NA NA')
")
DR=$(grep -oE 'n_drafted = [0-9]+|n_accept = [0-9]+' $HOME/b_$IDX.log | tr '\n' ' ')
echo "arm$IDX nmax=$NM p=$P | log: $ACT | ${T0}->${T1}C | t/s+ms: $R | $DR"
for x in $(pgrep -f "[l]lama-server"); do kill -9 $x 2>/dev/null; done
