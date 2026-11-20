#!/usr/bin/env bash
# Reproduce the exact conditions that produced Ridge's 11.79 t/s, on IQ4_XS.
set -u
M=${MODEL:-$HOME/models/iq4xs/Qwen3.8-27B-UD-IQ4_XS.gguf}
CTX=${CTX:-32768}
export LD_LIBRARY_PATH=$HOME/lucebench:$HOME/lucebin

T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -dc '0-9')
echo "idle temp: ${T}C"
[ "${T:-99}" -gt 75 ] && { echo "ABORT: too hot"; exit 1; }

LIMIT=80 LOG=$HOME/p100_temps_mtp.log bash $HOME/p100_guard.sh &
GUARD=$!

# ngram draft length is size_m (default 48!), NOT --spec-draft-n-max, which the
# ngram path ignores entirely. On a compute-bound card a deep draft is ruinous.
NGRAM_ARGS=""
[ -n "${NGRAM_M:-}" ] && NGRAM_ARGS="$NGRAM_ARGS --spec-ngram-map-k-size-m $NGRAM_M --spec-ngram-map-k4v-size-m $NGRAM_M"
[ -n "${NGRAM_N:-}" ] && NGRAM_ARGS="$NGRAM_ARGS --spec-ngram-map-k-size-n $NGRAM_N --spec-ngram-map-k4v-size-n $NGRAM_N"
[ -n "${NGRAM_HITS:-}" ] && NGRAM_ARGS="$NGRAM_ARGS --spec-ngram-map-k-min-hits $NGRAM_HITS"
[ -n "$NGRAM_ARGS" ] && echo "ngram args:$NGRAM_ARGS"

$HOME/lucebench/llama-bench --help >/dev/null 2>&1
$HOME/lucebin/llama-server --model "$M" \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --spec-type "${SPEC:-draft-mtp}" --spec-draft-n-max "${DRAFTN:-3}" \
  --reasoning off \
  --ctx-checkpoints "${CTXCP:-2}" $NGRAM_ARGS \
  -ngl 99 -c "$CTX" --parallel 1 \
  -b "${BATCH:-2048}" -ub "${UBATCH:-2048}" \
  --host 127.0.0.1 --port 8081 > $HOME/mtp_server.log 2>&1 &
SRV=$!

for i in $(seq 1 150); do
  R=$(curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null)
  case "$R" in *'"status":"ok"'*) echo "model ready after $((i*6))s"; break;; esac
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED:"; tail -20 $HOME/mtp_server.log; kill $GUARD 2>/dev/null; exit 1; fi
  sleep 6
done
R=$(curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null)
case "$R" in *'"status":"ok"'*) : ;; *) echo "NOT READY: $R"; tail -20 $HOME/mtp_server.log; kill $SRV $GUARD 2>/dev/null; exit 1;; esac
echo "server up; VRAM:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

rm -f $HOME/mtp_resp.json          # never let a stale response masquerade as a fresh result
curl -s -m 300 http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Explain in about 150 words why passive-cooled datacenter GPUs need forced airflow."}],"max_tokens":160,"temperature":0}' \
  -o $HOME/mtp_resp.json
if [ ! -s $HOME/mtp_resp.json ]; then
  echo "!!! REQUEST FAILED - no response body. Server log tail:"
  tail -25 $HOME/mtp_server.log
  kill $SRV $GUARD 2>/dev/null
  exit 1
fi
python3 - <<'PY'
import json,os
d=json.load(open(os.path.expanduser('~/mtp_resp.json')))
t=d.get('timings',{})
print("predicted_n :", t.get('predicted_n'))
print("predicted_ms:", round(t.get('predicted_ms',0),1))
print("DECODE t/s  :", round(t.get('predicted_per_second',0),2))
print("prompt t/s  :", round(t.get('prompt_per_second',0),2))
txt=d.get('choices',[{}])[0].get('message',{}).get('content','')
print("content len :", len(txt))
print("first 120   :", repr(txt[:120]))

fail=[]
if len(txt) < 200:
    fail.append("output too short (%d chars)" % len(txt))
w=txt.split()
if w:
    top=max(set(w), key=w.count)
    frac=w.count(top)/len(w)
    if frac > 0.25:
        fail.append("degenerate repetition: %r is %.0f%% of tokens" % (top, 100*frac))
ok_chars=sum((c.isalnum() or c.isspace() or c in ".,;:'-()%/") for c in txt)
if txt and ok_chars < 0.9*len(txt):
    fail.append("high non-language character ratio - possible kernel corruption")
if fail:
    print("!!! QUALITY CHECK FAILED - THE t/s ABOVE IS NOT A VALID RESULT")
    for f in fail: print("   -", f)
    raise SystemExit(2)
print("quality     : OK")
PY

kill $SRV $GUARD 2>/dev/null; sleep 2
grep -i "draft acceptance\|n_drafted\|n_accept" $HOME/mtp_server.log | tail -3
echo "--- peak temp ---"
awk '{print $2}' $HOME/p100_temps_mtp.log | tr -dc '0-9\n' | sort -n | tail -1
