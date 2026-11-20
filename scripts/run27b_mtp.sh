#!/usr/bin/env bash
# CHAMPION serve config.
# 2026-09-01: --spec-draft-n-max 3 -> 1. Interleaved A/B (3,1,3,1), PROF=0, 64-tok pulses:
#   n-max 3 = 12.438 / 12.365 t/s (acceptance 0.415)
#   n-max 1 = 15.690 / 15.625 t/s (acceptance 0.703)   = +26.3%
# n-max 3 makes the target forward run at 4 columns; the mmvq marginal-column cost is convex
# while the acceptance series is concave, and they cross between the 2nd and 3rd column.
# Speculative decoding is exact, so this changes no output. Override with NMAX=3 to compare.
# Prior figure, different thermal conditions: 14.23 t/s decode, measured 2026-08-28.
#   UD-IQ4_XS + draft-mtp + turbo4 K+V, -c 8192 -b 512 -ub 512
# -b/-ub 512 are NOT optional: the default 2048 sizes the compute buffer into OOM
# (VRAM sits at 15723/16384 MiB as it is). Raising -c beyond 8192 will not fit.
export LD_LIBRARY_PATH=$HOME/lucebin
export GGML_CUDA_NO_PEER_COPY=1
M=${MODEL:-$HOME/models/iq4xs/Qwen3.8-27B-UD-IQ4_XS.gguf}

# Thermal guard: the card has no blower yet. Kills the server at 80C, below the
# 82C throttle, so throughput is never quietly degraded.
LIMIT=${LIMIT:-80} LOG=$HOME/p100_temps_serve.log bash $HOME/p100_guard.sh &

exec $HOME/lucebin/llama-server --model "$M" \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --spec-type draft-mtp --spec-draft-n-max ${NMAX:-1} \
  --reasoning off \
  --ctx-checkpoints "${CTXCP:-2}" \
  -ngl 99 -c "${CTX:-8192}" --parallel 1 \
  -b "${BATCH:-512}" -ub "${UBATCH:-512}" \
  --host 0.0.0.0 --port "${PORT:-8081}"
