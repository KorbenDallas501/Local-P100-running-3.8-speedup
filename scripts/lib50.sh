# Paths: set these for your machine (originals used the author's layout).
: "${PROJECT_DIR:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${LLAMA_DIR:=$HOME/llama.cpp}"; : "${LUCE_DIR:=$HOME/luce}"; : "${DATA:=$HOME}"
# shared helpers for the 50 t/s experiment chain. PERSISTENT: $DATA, not /tmp.
# (The scratchpad under /tmp was wiped mid-chain on 2026-09-03 and took every batch script with
#  it. Nothing that must survive a session lives in /tmp any more.)
H=${HOST:?set HOST=user@<p100-host>}
MAC=${P100_MAC:?set P100_MAC for wake-on-LAN}
SSH="ssh -o BatchMode=yes"
D50=$PROJECT_DIR
LC=$LLAMA_DIR

cool_cycle(){
  # kill stuck GPU processes FIRST: a llama-server in Dl (uninterruptible) state holds the card
  # hot and blocks a clean shutdown. Found one at 63 C on 2026-09-04 pretending to be a cold box.
  $SSH $H 'for p in $(pgrep -f "[l]lama-server|[t]est-backend-ops"); do kill -9 $p 2>/dev/null; done
           for p in $(ps -eo pid,args | awk "/tb1[0-9]_remot[e]|batch[0-9]*_drive[r]/ {print \$1}"); do kill -9 $p 2>/dev/null; done
           sleep 3; sudo poweroff' 2>/dev/null || true
  # VERIFY it actually went down. This box auto-restarts after poweroff (BIOS restore-on-AC),
  # so "issued poweroff" is not evidence of anything. Confirm by PING, not by ssh.
  local down=0
  for i in $(seq 1 30); do ping -c1 -W2 ${HOST_IP:?set HOST_IP} >/dev/null 2>&1 || { down=1; break; }; sleep 5; done
  [ "$down" = 1 ] || { echo "ABORT: $1 -- host never went down; refusing to time a fake cold cycle"; return 1; }
  echo "[$(date +%T)] $1: confirmed DOWN; cooling 8 min"; sleep 480
  # and confirm it STAYED down -- an auto-restart mid-sleep silently reheats the card
  if ping -c1 -W2 ${HOST_IP:?set HOST_IP} >/dev/null 2>&1; then
    echo "[$(date +%T)] WARNING: $1 -- host came back up during the cooldown; card was NOT cooling"
    echo "  the start temperature below is the ground truth; treat a hot start as contaminated"
  fi
  python3 - "$MAC" <<'PY'
import socket,sys
mac=sys.argv[1].replace(':','')
pkt=b'\xff'*6+bytes.fromhex(mac)*16
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)
for t in ('255.255.255.255','255.255.255.255'):
    for p in (9,7): s.sendto(pkt,(t,p))
PY
  for i in $(seq 1 60); do $SSH -o ConnectTimeout=5 $H 'true' 2>/dev/null && break; sleep 10; done
  $SSH -o ConnectTimeout=10 $H 'true' 2>/dev/null || { echo "ABORT: Z640 did not return"; return 1; }
  echo "[$(date +%T)] $1: cold at $($SSH $H 'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader')"
}

# runcycle MARKER OUTFILE HARNESS 'run line' 'run line' ...
# Driver is built LOCALLY and shipped as bytes -- never a remote heredoc (F105).
runcycle(){ local M=$1 O=$2 HB=$3; shift 3
  local body=$(mktemp)
  { echo '#!/usr/bin/env bash'; echo 'set -u'
    echo "C=Ridge-q41nonffn.gguf"; echo "S=Ridge-splice.gguf"; echo "OUT=\$HOME/$O"; echo ': > "$OUT"'
    echo "echo \"=== $M \$(date +%F_%T) idle=\$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) ===\" >> \"\$OUT\""
    echo "run(){ local label=\$1; shift"
    echo '  { echo; echo "########## $label ##########"; } >> "$OUT"'
    echo "  bash \$HOME/${HB}_remote.sh \"\$@\" >> \"\$OUT\" 2>&1"
    echo '}'
    for l in "$@"; do echo "$l"; done
    echo "echo \"${M}_COMPLETE \$(date +%F_%T)\" >> \"\$OUT\""
  } > "$body"
  # sanity: every run line the caller passed must have landed, and no local-only variable may
  # have survived unexpanded into the driver text (F105). "$@" inside run() is intentional.
  local want=$#; local got=$(grep -c '^run "' "$body")
  [ "$want" = "$got" ] || { echo "ABORT: driver has $got run lines, expected $want"; rm -f "$body"; return 1; }
  # $C and $S are defined inside the driver itself, so exempt them; anything else with a $ in a
  # run line is a local-only variable that leaked through unexpanded.
  if grep -E '^run "' "$body" | sed -e 's/\$C//g' -e 's/\$S//g' | grep -qE '\$[A-Za-z_][A-Za-z0-9_]*'; then
      echo "ABORT: an unexpanded variable reached a run line:"
      grep -E '^run "' "$body" | sed -e 's/\$C//g' -e 's/\$S//g' | grep -E '\$[A-Za-z_]'
      rm -f "$body"; return 1
  fi
  cool_cycle "$M" || { rm -f "$body"; return 1; }
  $SSH $H "cat > \$HOME/${M}_driver.sh" < "$body"
  local nl=$(wc -l < "$body"); local rl=$($SSH $H "wc -l < \$HOME/${M}_driver.sh")
  rm -f "$body"
  [ "$nl" = "$rl" ] || { echo "ABORT: driver truncated in transfer ($nl vs $rl)"; return 1; }
  $SSH $H "chmod +x \$HOME/${M}_driver.sh; setsid nohup bash \$HOME/${M}_driver.sh >/dev/null 2>&1 & sleep 2; echo ${M}-running"
  for i in $(seq 1 160); do $SSH -o ConnectTimeout=8 $H "grep -q ${M}_COMPLETE \$HOME/$O" 2>/dev/null && break; sleep 30; done
  echo "[$(date +%T)] $M done"
  $SSH $H "grep -E '^##########|^==|eval time =|acceptance rate|ready VRAM|STARTUP ERROR|out of memory|SERVER FAILED' \$HOME/$O" | tee -a "$D50/results_$M.txt"
}
