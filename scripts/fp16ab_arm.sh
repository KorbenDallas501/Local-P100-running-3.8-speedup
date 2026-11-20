#!/usr/bin/env bash
# CLEAN sm_60 FP16 CARVE-OUT A/B. $1 = bindir (fp16fix=flag ON | fp16ctl=flag OFF)  $2 = idx
#                                 $3 = "q" to also run the broader quality gate
#
# WHAT F145 GOT WRONG: it compared $HOME/fp16fix (its own single lib dir, built from tree HEAD)
# against $HOME/sub4_2:$HOME/lucefix (a two-dir split, built a day earlier). Two binaries, two
# lib stacks, two tree states -- and the arm NEVER set the knob, because the knob is COMPILE-time
# (common.cuh: #if LUCE_SM60_NO_FAST_FP16 -> excludes __CUDA_ARCH__ 600 from FAST_FP16_AVAILABLE).
# Here both arms are single-dir, same tree, same commit, same dirty state, same compiler
# invocation, differing ONLY in -DLUCE_SM60_NO_FAST_FP16. Provenance md5-verified end to end.
set -u
BD=$1; IDX=$2; QG=${3:-}
for x in $(pgrep -f "[l]lama-server"); do kill -9 $x 2>/dev/null; done; sleep 3
T0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -dc '0-9')
[ "${T0:-99}" -ge 74 ] && { echo "arm$IDX $BD SKIPPED ${T0}C"; exit 0; }
M=${MODEL:-$HOME/models/qwen38-27b-ridge/Ridge-q41nonffn.gguf}
LOG=$HOME/fpab_$IDX.log
GGML_CUDA_NO_PEER_COPY=1 LD_LIBRARY_PATH=$HOME/$BD nohup $HOME/$BD/llama-server --model "$M" \
    --cache-type-k turbo4 --cache-type-v turbo4 \
    --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.6 \
    --reasoning off --ctx-checkpoints 2 -ngl 99 -c 8192 --parallel 1 -b 512 -ub 512 \
    --host 127.0.0.1 --port 8092 > "$LOG" 2>&1 &
ok=0
for i in $(seq 1 150); do
  [ "$(curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:8092/health)" = 200 ] && { ok=1; break; }
  grep -qiE "CUDA error|out of memory|unknown argument" "$LOG" && break
  sleep 3
done
[ $ok = 1 ] || { echo "arm$IDX $BD SERVER FAILED"; grep -iE "error|out of mem" "$LOG"|tail -2; exit 1; }
LIMIT=78 LOG=$HOME/guard_fpab$IDX.log nohup bash $HOME/guard_req.sh >/dev/null 2>&1 &
curl -s -m 600 http://127.0.0.1:8092/completion -H 'Content-Type: application/json' \
  -d @$HOME/prose_p1.json -o /tmp/fpab_$IDX.json
T1=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -dc '0-9')
TS=$(python3 -c "
import json
try: print('%.2f'%json.load(open('/tmp/fpab_$IDX.json')).get('timings',{}).get('predicted_per_second',0))
except Exception: print('NA')")
echo "arm$IDX $BD ${T0}->${T1}C t/s=$TS"
[ "$QG" = "q" ] && bash $HOME/quality_gate.sh 8092 "$BD"
for g in $(pgrep -f "[g]uard_req"); do kill $g 2>/dev/null; done
for x in $(pgrep -f "[l]lama-server"); do kill -9 $x 2>/dev/null; done
