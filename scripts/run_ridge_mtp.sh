#!/usr/bin/env bash
# Champion launcher -- Ridge-q41nonffn + native MTP + p_min.
#
# 2026-09-04: n-max default 3 -> 2 (B=4 -> B=3). MEASURED, interleaved 2:3:2:3 from cold 51C:
#   B=3  19.41 / 19.40 @47C  (replication spread 0.02%)   mean 19.40
#   B=4  18.63 / 18.41 @47C  (spread 1.15%)               mean 18.52   => B=3 is +4.8%
# WHY: p_min=0.6 cuts MEAN draft width to 1.67, so verifying at B=4 pays for ~2.3 columns the
# drafter rarely proposes. B=3 matches the width actually produced. B=2 over-narrows (16.43).
#
# Config is F104 arm A3 verbatim, the MEASURED p_min point:
#   A1 p_min=0   B=4 control  43 C  17.51 t/s
#   A3 p_min=0.6 B=4          59 C  18.38 raw -> 18.77 @43C  = +7.2%   <- this config
# prose_p1, 192 tokens, cold, bracketed. Shipped 2026-09-04.
#
# n-max 1: DO NOT SHIP. Now MEASURED (F148, full 3x2 factorial, one boot, temp-corrected):
#   n-max 1 loses ~8% at BOTH p_min values -- (1, 0.60)=18.08, (1, 0.45)=18.19 -- vs 19.63 here.
# The reason this file used to give -- "both are width-reducing" so they compound -- was an
# INFERENCE and is WRONG: p_min barely matters at n-max 1 (0.6% apart). They do not compound;
# n-max 1 is simply a worse cell. The old +26.3% n-max-1 figure (run27b_mtp.sh, IQ4_XS model)
# does not transfer to Ridge-q41nonffn and is contradicted here under clean same-boot conditions.
# n-max 2 / p_min 0.60 is the MEASURED joint optimum of the full grid, not a sequential guess.
#
# Binary layout: lucefix supplies llama-server + cudart; sub4_2 supplies the FORK's ggml-cuda.
# This split is load-bearing -- lucefix alone gives the master kernels, not the fork's.
set -u
export GGML_CUDA_NO_PEER_COPY=1
export LD_LIBRARY_PATH=$HOME/sub4_2:$HOME/lucefix
M=${MODEL:-$HOME/models/qwen38-27b-ridge/Ridge-q41nonffn.gguf}
[ -f "$M" ] || { echo "ABORT: champion model missing at $M"; exit 1; }
exec $HOME/lucefix/llama-server --model "$M" \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --spec-type draft-mtp --spec-draft-n-max "${NMAX:-2}" \
  --spec-draft-p-min "${PMIN:-0.6}" \
  --reasoning off --ctx-checkpoints 2 \
  -ngl 99 -c "${CTX:-8192}" --parallel 1 -b 512 -ub 512 \
  --host 0.0.0.0 --port "${PORT:-8081}"
