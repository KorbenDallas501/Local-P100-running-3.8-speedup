#!/usr/bin/env bash
# Bonsai 2 27B PQ2_0 on Tesla P100-PCIE-16GB (sm_60).
#
# PQ2_0 storage default: 17.565 t/s versus PTQ1_0 8.506 (+106.5%),
# interleaved 64-token greedy controls, q8_0 KV, 32K, 2026-09-21.
# Outputs matched on those requests; PQ2 uses ~1144 MiB more GPU memory.
# PTQ1_0 and Ridge were removed after verified off-box copies. Neither is local.
#
# Four-row kernel: 24.40/24.29 and 24.52/24.47 t/s, versus stock 17.56/17.49
# on 192-token novel prose, no speculation, original content SHA every time.
# Width1 r4 promoted: code17.70->24.34, multilingual17.65->24.34,
# 3348-token context16.52->23.13; exact stock output SHAs on all three.
# All8 PQ2 shapes passed bitwise kernel checks; row-alignment guard retained.
# PQ2 trace: MMVQ 80.5%, FWHT 1.7%. The earlier 89.8% MMVQ trace was PTQ.
#
# THERMAL: blower NOT operational, NOT rewired, NOT spinning (user confirmed).
# September 11 flat 59-60 C was IDLE-ONLY, never evidence of load cooling.
# Under load +~0.15 C/s, no plateau; 68 C guard after ~4 minutes.
# Until blower works, use guarded bursts: 2-3 arms, full cooldown, start <=50 C,
# kill >=68 C, poweroff >=78 C. Prior Xid79/card loss near 84 C.
# The review burst_arm.py/control.py tools enforce the experiment limits.
#
# Speculation must stay at verification width <=8; wider batches use cuBLAS
# weight expansion and lose the ternary MMVQ advantage. Draft length <=7.
# ngram-simple n3/m5: 31.18 vs17.54 t/s on copy-heavy text (+77.8%), ~0 on prose
# with the STOCK kernel; do not assume this ratio composes with four rows.
# Explicit cross-column unpack sharing in the existing runtime is PTQ-only.
# Qwen3.5-0.8B is tokenizer-compatible, but draft-simple n2 lost on prose (14.1).
# Base PQ2 GGUF has no MTP head. Verified Bonsai-trained head model is at
# ~/models/bonsai2_mtp/bonsai2-mtp-q8.gguf; requires the patched MTP runtime.
# Selective verification r4: large aligned PQ2 projections at widths2,3,5..8.
# Width4 and small projections retain stock geometry after shape benchmarks.
# MTP n2 improved26.84/26.34 ->29.08/28.62 with identical acceptance and SHAs.
#
# Mixed-use default: strict ngram n4/m5 then trained MTP n1/p0.
# Repeats: prose28.083/28.261, copy36.955/37.271 t/s, original output SHAs.
# MTP alone: prose28.879/28.962, copy31.291/31.387. Chain wins total decode
# time above~16% copy-like generated tokens in this two-workload comparison.
# Optimizes a mixed workload; use SPEC=draft-mtp for mostly novel prose.
# Code/multilingual/context output checks pass with exact stock SHAs.
# Low-acceptance populated-context case: chain20.91, MTP21.55, no-spec23.13t/s.
# Use SPEC=none for this workload; speculation is not a universal speedup.
# SPEC=none (or SPEC="") disables speculation. CTX defaults to32768.
# Larger contexts need measurement; draft length must remain <=7.
set -u
: "${SPEC=ngram-simple,draft-mtp}"; : "${DRAFT_MODEL:=}"; : "${NMAX:=}"; : "${PMIN:=}"; : "${NGLD:=}"
: "${CTKD:=}"; : "${CTVD:=}"; : "${NGRAM_M:=5}"; : "${NGRAM_N:=4}"
DEFAULT_MODEL=$HOME/models/bonsai2_pq2/Ternary-Bonsai-2-27B-PQ2_0.gguf
case ",${SPEC}," in
    *,draft-mtp,*)
        DEFAULT_MODEL=$HOME/models/bonsai2_mtp/bonsai2-mtp-q8.gguf
        : "${NMAX:=1}"; : "${PMIN:=0}" ;;
esac
# Keep speculative target batches inside the P100 MMVQ width limit.
case ",${SPEC}," in
    *,draft-mtp,*|*,draft-simple,*|*,draft-dflash,*)
        : "${NMAX:=3}"
        if [[ ! "$NMAX" =~ ^[0-7]$ ]]; then
            echo "ABORT: NMAX must be 0..7 (verification width <=8)" >&2; exit 1
        fi ;;
esac
case ",${SPEC}," in
    *,ngram-simple,*)
        if [[ ! "$NGRAM_M" =~ ^[0-7]$ ]]; then
            echo "ABORT: NGRAM_M must be 0..7 (verification width <=8)" >&2; exit 1
        fi ;;
esac
export GGML_CUDA_NO_PEER_COPY=1
# Selective verification geometry passed shape, output, and timing checks.
export GGML_CUDA_PQ2_VERIFY_R4=${GGML_CUDA_PQ2_VERIFY_R4:-1}
B=${PRISM_BIN:-$HOME/prismbin_verify_r4}
M=${BONSAI_MODEL:-$DEFAULT_MODEL}
[ -f "$M" ] || { echo "ABORT: bonsai model missing at $M"; exit 1; }
[ -x "$B/llama-server" ] || { echo "ABORT: prism llama-server missing at $B"; exit 1; }
export LD_LIBRARY_PATH="$B${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ARGS=( -m "$M" --port "${PORT:-8099}" --host "${HOST:-0.0.0.0}"
       --ctx-size "${CTX:-32768}" --parallel 1 -ngl 99 -fit off
       -b "${B_LOG:-512}" -ub "${UB:-512}"
       --reasoning "${REASONING:-off}" --jinja
       --cache-type-k "${CTK:-q8_0}" --cache-type-v "${CTV:-q8_0}"
       --metrics )
ARGS+=( --spec-type "${SPEC:-none}" )
[ -n "$DRAFT_MODEL" ] && ARGS+=( --model-draft "$DRAFT_MODEL" )
[ -n "$NMAX" ]        && ARGS+=( --spec-draft-n-max "$NMAX" )
[ -n "$PMIN" ]        && ARGS+=( --spec-draft-p-min "$PMIN" )
[ -n "$NGLD" ]        && ARGS+=( --spec-draft-ngl "$NGLD" )
[ -n "$CTKD" ]        && ARGS+=( -ctkd "$CTKD" )
[ -n "$CTVD" ]        && ARGS+=( -ctvd "$CTVD" )
[ -n "$NGRAM_M" ] && [ "$NGRAM_M" != 0 ] && ARGS+=( --spec-ngram-simple-size-m "$NGRAM_M" )
[ -n "$NGRAM_N" ] && [ "$NGRAM_N" != 0 ] && ARGS+=( --spec-ngram-simple-size-n "$NGRAM_N" )
echo "LAUNCH: llama-server ${ARGS[*]}" >&2
exec "$B/llama-server" "${ARGS[@]}"
