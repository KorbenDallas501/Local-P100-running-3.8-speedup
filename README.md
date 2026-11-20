# Local Bonsai-2 27B on a single P100: 8.5 → 24.5 t/s (29 t/s on short prompts)

Bonsai-2 27B (PrismML's ternary model built on Qwen3.8-27B) with llama.cpp on one Tesla P100 16 GB
(Pascal, sm_60, no dp4a): 29.1 t/s decode with short prompts, about 23 t/s at a typical session context.

This project — code, patches, measurements and this text — was written by AI coding agents (Claude Code and Codex) under human supervision.

## Hardware

- HP Z640 workstation, one Intel Xeon E5-2630 v3 (2014; Haswell, 8 cores / 16 threads), 24 GB RAM
- NVIDIA Tesla P100-PCIE-16GB (Pascal, sm_60; 16,269 MiB usable), Ubuntu 24.04, NVIDIA driver 580.173.02
- Built with CUDA toolkit nvcc 12.0 for sm_60
- Fan required: the P100 is passively cooled and needs a fan or forced airflow.

Everything runs on the P100 (`-ngl 99`, no CPU offload). That is what makes an old workstation viable:
the 2014 Xeon only feeds the card, and offloading any layers to it would make generation far too slow.

| Model file | Size | Context used |
|---|---|---|
| Bonsai-2 27B PQ2_0 + MTP head (`Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`) | 7.66 GB | 32K (q8_0 KV) for the benchmarks; 262,144 allocated (q4_0 KV) in daily use |

## 1. Results

Decode speed, Bonsai-2 27B, one P100, single user, greedy:

| Step | t/s | Measured with | Output check |
|---|---|---|---|
| PTQ1_0 weights (stock build) | 8.5 | 64-token prompts, 32K context | reference |
| PQ2_0 weights (PrismML's packing, same model) | 17.6 | 64-token prompts, 32K context | outputs matched PTQ1_0 on the test requests |
| r4 four-row kernel (this repo) | 24.5 | 18-token prompt, 192 tokens generated, no speculation | byte-identical to the unpatched build on PQ2_0 |
| + MTP-1 speculative decoding | 29.1 | 18-63-token prompts | byte-identical (same output hashes) |
| after a ~3,300-token conversation | 22.8 with MTP / 23.3 without speculation | 3,348-token prompt, same file and session (F175) | |

The PTQ1_0 → PQ2_0 step is PrismML's format; this repo's contribution is the kernel and MTP steps
(17.6 → 29.1 t/s, +65%). The kernel and MTP changes give byte-identical output to the unpatched build
on the same weights. At long context MTP stops paying on this card: with a 3,348-token prompt,
speculation off (23.3 t/s) is faster than MTP (22.8 t/s, about 2% slower), so turn speculation off for
long-context work.

## 2. Why the P100 wants short drafts

- **No dp4a.** On sm_60 the quantised mat-vec kernel compiles to 16-bit XMAD:
  81,883 XMAD and 0 IDP4A, about **2.1x the integer instructions** of the same file on sm_86
  for the same bytes moved. The card is **ALU-bound, not bandwidth-bound** (ledger F1, F2, F154).
- **Verification is not free.** Each extra column in the verify batch costs real arithmetic:
  on the r4 build T1 = 39.6 ms, T2 = 54.5 ms, about 14.2 ms (36% of a step) per extra column (F174).
- **So:** n-max 1 is best for Bonsai; n-max 2 was tested and rejected (F175c).

## 3. How to reproduce

```bash
git clone https://github.com/PrismML-Eng/llama.cpp prism && cd prism
git checkout 1a07bfa5f4144274c8f1c9963821dd9d9a51854b
git apply ../patches/prism_r4_kernel.patch
cmake -B build-sm60 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
cmake --build build-sm60 -j
# built with nvcc 12.0; run on NVIDIA driver 580.173.02
PRISM_BIN=$PWD/build-sm60/bin BONSAI_MODEL=/path/to/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf \
  SPEC=draft-mtp ../scripts/run_bonsai_p100.sh
```

Base commit and build details: `patches/PRISM_BASE.txt`.

**Model files.** PrismML's PQ2_0 release (`Ternary-Bonsai-2-27B-PQ2_0.gguf`) has no MTP head; use it
with `SPEC=none` for the 24.5 t/s line. The 29.1 t/s line uses the community GGUF with a trained MTP
head, `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` from the Hugging Face repo
`ProCreations/Ternary-Bonsai-2-27B-MTP` (revision `efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51`,
7,657,489,728 bytes, sha256 `3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4`):
the unchanged PQ2_0 base tensors plus a 451 MB Q8_0 MTP head. Download it whole; check the hash.

Switches added by the patch. **The defaults reproduce the measured setup.**

| Variable | Default | Meaning |
|---|---|---|
| `GGML_CUDA_PQ2_VERIFY_R4` | 1 | four-row PQ2_0 path for decode and for MTP verify widths 2, 3, 5-8 |
| `GGML_CUDA_PQ2_ROWS` | 4 | `8` / `selective8` = experiment |
| `GGML_CUDA_PQ2_WIDTH_GEOMETRY` | 0 | experiment (F174, rejected) |
| `GGML_CUDA_PASCAL_GRAPHS` | 0 | experiment (CUDA graphs on Pascal, +0.2%, closed) |

The `qwen35.cpp` part of the patch applies the Hadamard inverse rotation to the MTP head's token
embedding when that weight is stored rotated, and does nothing otherwise.

Full measurement record: `docs/MEASUREMENT_LEDGER.md` (F-numbers), `docs/HISTORY_PAGE.md`,
`docs/EXPERIMENT_LOG.csv`, plus the planning notes `docs/PATH_TO_50.md` and `docs/DFLASH2_RESEARCH.md`.

## 4. What did not work

- f16 K cache: 0% (F173) (Qwen3.8).
- Width-geometry kernel for verify widths 2-8: real only at width 4, which no shipping profile uses (F174).
- MTP n-max 2 for Bonsai (F175c); the n-gram + MTP chain only won on a 63-token prompt (F175).
- sm_60 FP16 fast-path carve-out: slower and wrong answers (F145) (Qwen3.8).
- Naive half2 mat-vec instead of int8 on GP100: loses (fp16 probe, in `luce_mtp.patch`) (Qwen3.8).
- CUDA graphs on Pascal: -1.1 to -1.8% stock, +0.2% when forced (F19); closed (Qwen3.8).
- Eagle-3 and DFlash draft models: closed (see the "settled" list in the ledger) (Qwen3.8).
- GSQ-RCO IQ3_S quant: loses 20% (F151). Tensor splice: loses 5.4% (F136) (Qwen3.8).
- rows-per-block 4 at verify widths 3-8 for every type: iq2_s regresses 27% at n=5 (F73); only selected types kept (Qwen3.8).
- Thread count: default 8 already optimal (F144) (Qwen3.8).

## 5. Also tested: Qwen3.8 27B, 10.6 → 19.44 t/s

Same card, 3.7 bpw "Ridge" quant (12.5 GB): 10.6 t/s without speculation → **19.44 t/s** with a
q4_1-non-FFN variant (3.66 bpw), MTP n-max 2, p_min 0.6 and Pascal MMVQ tuning (F111, F129, F148);
15-token prompt (`prose_p1.json`), 192 tokens generated, 8K context. Here n-max 2 beats n-max 1 (19.63 vs 18.1); verify cost
`T(B) = 81.0 + 35.7(B-1)` ms. q4_1 is the fastest format on this card (2.27x faster per weight than q6_K).
TurboQuant supplies the KV cache: V as turbo4 is 6.1% faster than q8_0 (F139), and K is auto-upgraded
to q8_0 at this model's 6:1 GQA ratio (F43, F45). 96K of context fits with decode flat (-0.24%, F171,
F172) and 128K allocates at 18.55 t/s (F167); these are allocated, not filled, contexts.
Patch: `patches/luce_mtp.patch` on the base in `patches/LUCE_BASE.txt` (Luce dflash + TheTom
turboquant). Launchers: `scripts/run_ridge_mtp.sh` (the 19.44 config); `scripts/p100_mtp.sh` and
`scripts/run27b_mtp.sh` are the earlier IQ4_XS configurations.

## 6. Scripts

- `run_bonsai_p100.sh`: Bonsai launcher (above).
- `quality_gate.sh`: broader output-quality check; `guard_req.sh`: safety guard that aborts a request without stopping the server;
  `prose_p1.json`: the Qwen3.8 benchmark request (15-token prompt, 192 tokens generated, greedy).
- `lib50.sh`, `kernel_ab.sh`, `b_sweep.sh`, `rpb_sweep.sh`: A/B benchmark drivers that run the P100 box
  over ssh (set `HOST`, `HOST_IP`, `MODEL`, `LLAMA_DIR` first); `b_arm.sh`, `fp16ab_arm.sh`: single arms
  they run on the box.
- `gguf_splice.py`, `dump_types.py`: GGUF tensor-type tools (need llama.cpp's `gguf-py`).

## 7. How this was made

All of it — code, patches, measurements and this text — was written by AI coding agents (Claude Code and
OpenAI Codex) under human supervision; every result was measured and checked before it was kept.

## 8. Credits and links

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT)
- [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (MIT); Bonsai-2 weights by PrismML (Apache 2.0, not included)
- [ProCreations/Ternary-Bonsai-2-27B-MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP): the trained MTP head used for the 29.1 t/s line (not included; licence: Apache 2.0)
- [Luce-Org llama.cpp-dflash-ggml](https://github.com/Luce-Org/llama.cpp-dflash-ggml) (MIT)
- [TheTom llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (MIT)
- [qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) (MIT): patches for running llama.cpp fast on Tesla P100

No model weights or binaries are included.

## 9. Contributions

Review, corrections and contributions are welcome, especially on time to first token (TTFT / prefill).
Current starting point for Bonsai-2 27B on the P100: a cold prefill of a 6,000-token prompt takes
about 49 s (~120 tokens/s, F176). With prompt-cache reuse, follow-up turns re-read only a few hundred
tokens (F176).

## License

MIT, see `LICENSE`.
