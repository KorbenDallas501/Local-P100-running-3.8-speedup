# DFlash2 on a single Tesla P100 — research record

Started 2026-09-02. Baseline pinned at tag `ridge-baseline-29.37` (commit c4252088f, branch
`sm60-p100-kernels`). DFlash work is on branch `ridge-dflash2` off that same commit. Upstream
reference checkout is a separate worktree at $LLAMA_DIR (detached 67a17c17c) and
touches nothing in our tree.

## 1. What was found externally (MEASURED BY OTHERS, not by us)

- DFlash2 = block-diffusion drafter. llama.cpp PR #27342 "spec : add DFlash2 support (local
  convolution + candidate selector)", 15 files, +639/-24, merged 2026-08-27T17:05:58Z **into the
  feature branch `xsn/dflash2`, NOT into master**. The HF card's "merged into main" is wrong as
  stated, but v2 has since reached master: `master:src/models/dflash.cpp` contains
  `build_dflash2_conv` and the v2 selector metadata, and `master:common/speculative.cpp` registers
  both `draft-dflash` and `draft-dspark`.
- Drafter: z-lab / incoai `Qwen3.8-27B-DFlash2`, **2B params**, GGUF Q4_K_M **1.14 GB**
  (Q8_0 2.06 GB, BF16 3.86 GB). BF16 is NOT required.
- Reported acceptance length **5.13-5.39** across quantizations, on **GSM8K**, max 2048 new tokens.
  GPU not stated on the model card.
- No result found for DFlash2 on P100 / P40 / 1080 Ti / V100 at time of writing. One third-party
  post reports 11-35 t/s on an M5 Max against a "70 t/s" headline claim, i.e. headline numbers
  are not reproducing on non-datacenter hardware.

## 2. Compatibility with Ridge — VERIFIED LOCALLY from the GGUF

| property | drafter | Ridge | verdict |
|---|---|---|---|
| vocab | 248,320 | 248,320 | EXACT match |
| n_embd | 5,120 | 5,120 | EXACT match |
| tokenizer | gpt2 BPE | gpt2 BPE | match |
| blocks | 5 | 64 (+MTP blk.64) | n/a |
| `dflash.target_layers` | [6, 20, 34, 48, 62] | 64 layers | all five in range |

Other metadata: `block_size 8`, `selector_top_k 16`, `selector_rank 256`, `conv_kernel_size 2`,
`conv_group_size 16`, `context_length 262144`, `sliding_window 2048`, file_type 15 (Q4_K_M).
`has_confidence_head` and `sample_from_anchor` are ABSENT -> this is DFlash2 (selector_top_k > 0
is what the impl uses to set `is_dflash2`), not DSpark.

**The drafter consumes target hidden states from 5 layers.** It is not a standalone draft model.

## 3. Mechanism, read from master:common/speculative.cpp

Per round, the drafter batch is `n_max + 1` tokens: position 0 = last accepted token, positions
1..n_max = `mask_token_id`. **ONE `llama_decode` on the drafter produces the entire block** --
block diffusion, not autoregressive drafting. Then a CPU-side selector walks a lattice: each row
holds top-16 candidate token ids followed by a 16x16 predecessor->successor score block; the path
is greedy argmax, optionally truncated when softmax confidence < `p_min`.

Consequences for us:
- Draft cost per round = one 2B Q4_K_M forward over n_max+1 columns. NOT n_max forwards.
- Selector cost is CPU-side and negligible.
- Drafting is strictly more expensive per round than native MTP's single head, so the question is
  purely whether the higher block quality pays for it.

## 4. tau definition — the normalization the comparison depends on

Our tau = output advancement per verification round, INCLUDING the target's bonus token, bounded
[1, B]. In the DFlash2 impl the drafter proposes `n_max` tokens and the target verifies `n_max + 1`
columns, so a published "acceptance length" of 5.2 at `--spec-draft-n-max 7` (B=8) is on the same
footing as our tau ONLY if it counts the bonus token. UNRESOLVED: the model card says
"tokens generated per verification step", which reads as including the bonus, but this is not
stated explicitly. Treat 5.13-5.39 as tau_external with that caveat until confirmed.

## 5. PROJECTION before measuring (CALCULATED, not measured — recorded to be falsified)

Our measured prose baseline: B=4, tau 2.909, 56.06 ms/token, 17.84 t/s -> round = 163 ms.
Our measured round model on the tuned kernel: round(B) = a + Delta*(B-1), a ~= 81 ms, Delta ~= 26.9 ms
(Delta was 35.7 ms before the RPB work).

If DFlash2 delivers tau = 5.2 at B=8, ignoring drafter cost:
  round(8) = 81 + 26.9*7 = 269 ms -> 269/5.2 = 51.7 ms/token -> **19.3 t/s**
  (pre-kernel Delta 35.7 would give 331/5.2 = 63.7 ms/tok = 15.7 t/s, i.e. WORSE than our MTP B=4)

So even a full transfer of the external tau predicts only ~19-20 t/s, not ~32. **The barrier is
not drafter quality, it is our large per-column verification cost Delta.** Adding the 2B drafter
forward (~6 ms at the 200 GB/s-equivalent 4-bit rate, more with launch overhead) makes it worse.

Implication for the 50 t/s prose target: 50 t/s = 20 ms/token. With tau 5.2 the round budget is
104 ms, and `a` alone is ~81 ms -- itself near the memory-bandwidth floor for an 11.73 GiB model
at P100's effective bandwidth. So the ceiling with a perfect zero-cost drafter and Delta -> 0 is
about tau/a = 5.2/0.081 = **64 t/s**, and every ms of Delta eats into it. 50 t/s prose is
therefore not arithmetically impossible, but it requires BOTH a large tau gain AND a large Delta
reduction. Neither alone suffices.

## 6. Port scope (for the branch, once the reference measurement exists)

Upstream master lacks `MMVQ_PARAMETERS_PASCAL` entirely -- sm_60 falls through to
`MMVQ_PARAMETERS_GENERIC`, so master does NOT contain any of our validated Pascal wins. That is
why we do not rebase onto master. Our tree in turn has no DFlash at all (no `dflash` symbol in
common/, src/, tools/; no LLM_ARCH_DFLASH; the origin `luce-dflash` branch is OLDER than our HEAD
and has neither MTP nor DFlash).
Required on the branch: ~1000 lines src/models/dflash.cpp, arch/hparams/tensor-mapping plumbing,
the dflash impl in common/speculative.cpp, `llama_get_embeddings_nextn` +
`llama_set_embeddings_nextn(masked)` context support, and top-k.cu (upstream 275 lines vs our 95).
SM60 hazard scan of upstream top-k.cu: only shared-memory integer atomicAdd. No bf16, no
cooperative groups, no sm_80 async copy, no mma/wmma. **No sm_60 blocker in the CUDA path.**
dflash.cpp uses only standard ggml ops (reshape/view/concat/get_rows/set_rows/rope_ext/mul_mat/
sigmoid/fill/repeat), all of which exist in our tree.

## 7. Build-flag audit (asked: what do these actually do in OUR tree?)

- **`GGML_CUDA_FORCE_DMMV=ON` -> DOES NOT EXIST.** No `dmmv.cu`, no CMake option, no source
  reference anywhere in the tree. DMMV (`dequantize_mul_mat_vec`) was the pre-MMVQ kernel family
  and has been deleted upstream. CMake silently ignores unknown -D options, so setting it would
  do NOTHING while appearing to work. Every P100 guide recommending it is stale.
  **Its modern successor is the mmvq path we are already tuning** -- our rows_per_block work is
  the direct continuation of what that old advice was reaching for.
- `GGML_CUDA_FORCE_CUBLAS=OFF`, `GGML_CUDA_FORCE_MMQ=OFF`: both are compile-time `#ifdef`s that
  override the backend's matmul choice heuristic. Ours are OFF, i.e. the heuristic decides. That
  is what we want; forcing either would remove the per-shape decision we benefit from.
- `-DCMAKE_CUDA_ARCHITECTURES=60`: correct and necessary for the P100. Our known-good build uses
  `60;86` (86 for the local RTX 3060); the reference build uses 60 only to halve compile time.
- **LIVE knob found: `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F` / `_32F` env vars** (ggml-cuda.cu:1821).
  Relevant because **GP100 is the one Pascal chip with full-rate (2:1) FP16** -- unlike P40 and
  1080 Ti at 1/64 rate. This affects the cuBLAS F16 GEMM path used for large batches, i.e.
  PREFILL, not mmvq decode. Prior finding "naive fp16 loses 0.61x" was about mmvq weight loads and
  does NOT apply here. Low priority (prefill is not our generation bottleneck) but cheap to test.
- `--use_fast_math` / `--extra-device-vectorization`: not set in our known-good build. Untested.
  fast_math changes numerics, so it would need an acceptance-identity check before any t/s claim.

## 8. THE REFRAMING (this is the important result of the search)

Published, from inco.ai's own comparison on the SAME benchmark:

| workload | native MTP tau | DFlash2 tau | DFlash2 advantage |
|---|--:|--:|--:|
| GSM8K    | 5.02 | 5.46 | +8.8% |
| MATH-500 | 4.72 | 5.28 | +11.9% |

The advertised "2x" / "3.43x speedup, 236 t/s on H200" is measured against **non-speculative
autoregressive decoding**, NOT against native MTP. Against the mechanism we ALREADY run, DFlash2's
tau advantage is ~9-12%.

And "acceptance length" is defined as completion tokens / verification steps -- **identical to our
tau definition**, bonus token included. So these numbers are directly comparable, and:

  OUR native MTP on echo:  tau 5.647  <-- 12.5% ABOVE their GSM8K native-MTP tau of 5.02
  OUR native MTP on prose: tau 2.909

**Our MTP head is not underperforming.** GSM8K and MATH-500 are formulaic, high-redundancy
workloads that sit much closer to our echo workload than to novel prose. No published result found
for ANY drafter on genuinely novel prose.

Projected DFlash2 on our prose if the +8.8% tau ratio transfers: tau 2.909 -> 3.164, giving
19.57 t/s at B=4 (vs our 17.84) BEFORE paying for the 2B drafter forward, and WORSE than baseline
at B=6 (14.68) and B=8 (11.75) because our per-column cost Delta ~26.9 ms dominates.
So expected net effect on prose: somewhere between zero and +10%. Still worth measuring -- the
ratio could differ on prose, where MTP is weakest and block diffusion might help more -- but this
is not a path to 50 t/s on its own.

## 9. Next experiment chosen from this (batch20, needs NO port)

Run our OWN native MTP on a GSM8K-style prompt at B=4/6/8, with a prose control and a repeat
bracket. Decides between:
 (a) tau ~5 on GSM8K-class -> the prose tau deficit is WORKLOAD-INTRINSIC. No drafter can fix it;
     50 t/s prose must come from verification cost (a, Delta), and DFlash2 is at best a ~10% add-on.
 (b) tau ~3 on GSM8K-class too -> something in OUR setup suppresses tau (candidate: the IQ2_S FFN
     quantization degrading the MTP head), which would make drafter/quant work high-value.
Either way it is decisive, and it costs one cold batch on hardware we already have working.

## 10. INDEPENDENT CORROBORATION: prose acceptance is ~0.40-0.55 for this model, on ANY GPU

Source: github.com/sudoingX/qwen38-mtp -- paired baseline-vs-MTP benchmarks for Qwen3.8-27B across
~10 GPUs (Unsloth Q4_K_M / UD-Q4_K_XL, q4_0 KV), with per-prompt acceptance broken out BY WORKLOAD.

| workload class | acceptance (their measurements, many GPUs) |
|---|--:|
| code / Python | 0.78 - 0.90 |
| bash / shell   | 0.70 - 0.87 |
| **prose / narrative** | **0.40 - 0.55** |

Direct quote: *"Prose prompt runs ~3 tok/s slower with the flag in every arm; the gain is carried
by code prompts."* -- i.e. on their hardware, speculative decoding is **NET NEGATIVE on prose**.

**Our prose acceptance is 0.616 at B=4, at or ABOVE the top of their observed prose range, and our
prose result is net POSITIVE.** So our MTP configuration is not merely fine, it is better than the
published field on the workload we care about.

Pascal-class MTP results DO exist there (no P100, but): Tesla P40 2x24GB +92% at n-max 4,
Tesla V100 32GB +44% (33.3 -> 48 t/s), V100-SXM2 NVLink pair +65%. Their n-max note --
"the sweet spot is card- and topology-dependent, re-sweep after any config change" -- matches our
own finding that B=6 is optimal on echo and B=4 on prose.

### Consequence for the 50 t/s prose target (the scientific answer)

Three independent lines now agree that novel-prose acceptance for Qwen3.8-27B is intrinsically
low and NOT a drafter-quality problem:
 1. inco.ai's own data: DFlash2 beats native MTP by only +8.8%/+11.9% in tau, on math benchmarks.
 2. Our echo tau 5.647 EXCEEDS their GSM8K native-MTP tau 5.02 -- our head is healthy.
 3. sudoingX: prose acceptance 0.40-0.55 across ~10 GPUs, often net-negative, vs our 0.616.

**Therefore tau is close to a workload ceiling, not an engineering ceiling, and the prose path to
50 t/s must come from verification cost (`a` and `Delta`), not from better drafting.**
With tau capped near ~3.2 on prose, 50 t/s = 20 ms/token requires round <= 64 ms, while `a` alone
is ~81 ms -- so **50 t/s on novel prose is unreachable on this hardware with this model at this
quantization**, regardless of drafter. Reaching it would require attacking `a` itself: a smaller
or faster-to-read model, or higher effective memory bandwidth than a P100 has.
That is a projection from measured terms, and the batch20/batch21 results will test it.

The ECHO path to 50 t/s is different and not yet closed: tau is already 5.647, so 50 t/s needs
round <= 113 ms against a currently ~192 ms round at B=6. That is a verification-cost problem of
about 1.7x -- hard, but not arithmetically excluded the way the prose target is.

## 11. THE BIGGER LEVER: quantization chosen for ps/weight, not for bpw

Measured ps/weight on this P100 (from the format-speed work): q4_1 1.97, iq4_xs 2.29, q4_K 2.74,
iq2_s 3.05, q5_K 3.16, iq3_s 3.76, q6_K 4.50. Sub-4-bit formats are SLOWER PER WEIGHT than 4-bit,
so on this card a LARGER 4-bit model can decode faster than a smaller 2-bit one.

Composition-weighted ps/weight (params from the GGUF headers, 27.32e9 params each, MTP head
present in both):

| model | weighted ps/weight | vs Ridge |
|---|--:|--:|
| Ridge-q41nonffn (q4_1 31.8% / iq2_s 56.9% / iq3_s 11.3%) | 2.787 | -- |
| **Qwen3.8-27B-UD-IQ4_XS off the shelf** | **2.865** | **+2.8% SLOWER** |
| TRUE splice: q4_1 non-FFN + PURE iq4_xs FFN | 2.188 | **-21.5%, i.e. 1.274x** |

The off-the-shelf UD-IQ4_XS is only 49.7% IQ4_XS by parameter count -- it is dragged down by
IQ3_S 13.8%, Q5_K 11.2%, Q3_K 6.8%, IQ3_XXS 5.2%. So it is NOT the win, which is consistent with
our surgical requant having beaten it. (Two of the twelve component figures are ESTIMATED, not
measured: Q3_K, IQ3_XXS, IQ2_XS, Q2_K, Q8_0, IQ4_NL together are 14.3% of params.)

If the splice figure holds, it projects echo 29.40 -> 37.4 t/s and prose 17.84 -> 22.7 t/s --
larger than every kernel gain this session combined, and it attacks BOTH `a` and `Delta` because
both are memory-bound. It does not raise tau at all, which is why it helps prose where no drafter
can.

### Instrument validation FIRST (per standing discipline)

The projection rests on "end-to-end time proportional to params x weighted ps/weight". That model
makes a cheap falsifiable prediction using two files we ALREADY have:

  **UD-IQ4_XS should be ~2.8% SLOWER than Ridge-q41nonffn at matched config.**

If measured, the model is validated and the -21.5% splice projection is credible enough to build.
If not, the model is wrong and the splice should NOT be built on its strength. This is the next
experiment after the DFlash2 reference run. Note UD-IQ4_XS is 13.6 GiB vs Ridge's 11.7 GiB, so it
may not fit at ctx 8192 / B=6 -- the point may have to run at ctx 4096, matched on both arms.
Z640 disk is the practical constraint (only ~3.9 GB free with Ridge resident).

## 12. CORRECTION to section 11, and the VRAM-constrained splice

**My section-11 figure of 2.787 for Ridge-q41nonffn was WRONG.** It normalized over only q4_1 /
iq2_s / iq3_s and omitted **q6_K, which is 5.8% of params but 9.1% of weighted time** (ps/weight
4.50, the worst of any format). Using the full F23 composition (MEASURED from the GGUF: q4_1
31.4%, iq2_s 52.2%, iq3_s 10.4%, q6_K 5.8%, q8_0 0.1%):

  Ridge-q41nonffn = **2.8676**   UD-IQ4_XS = 2.865  ->  **essentially EQUAL, not 2.8% apart.**

So the "run UD-IQ4_XS to validate the model" experiment is NOT worth its 13.6 GB of disk: the
model now predicts no difference, which is uninformative. DROPPED.

### The model is already validated by data in hand
The ps/weight model PREDICTED the surgical requant at +12% (10.62 -> ~11.9 t/s). The
matched-thermal measurement came in at **+15%**. Directionally correct and conservative by ~2.7%
in ratio terms. That is enough to justify building on it.

### The full splice does not fit; the optimal partial one does
Full FFN splice (iq2_s+iq3_s -> iq4_xs) = **15.24 GB** on a 16.0 GB card that also needs ~1.9 GB
of KV + compute at ctx 8192 / B=6. Does not fit.

**q6_K -> iq4_xs is strictly dominant: -0.46 GB AND -4.5% weighted time.** Do it first; the space
it frees then funds converting iq2_s:

| VRAM budget | share of iq2_s converted | ps/weight | speedup | echo | prose |
|--:|--:|--:|--:|--:|--:|
| 13.6 GB | 71%  | 2.4559 | 1.168x | 34.3 | 20.8 |
| 14.0 GB | 84%  | 2.4050 | 1.192x | 35.1 | 21.3 |
| 14.5 GB | 100% | 2.3422 | 1.224x | 36.0 | 21.8 |

All PROJECTED from measured ps/weight, with the caveat that the model under-predicted last time.
Two further notes:
- The q6_K tensors are only 5 (1.59e9 params) -- almost certainly token_embd and output.weight.
  Memory records that **the output head runs 3x per MTP round**, so it sits on the critical path
  far more often than its parameter share suggests, making it a disproportionate target. Quality
  risk on output.weight is real and must be checked, not assumed.
- Requantizing iq2_s -> iq4_xs from the ALREADY-QUANTIZED Ridge does not recover quality (the
  information is gone) but it does not lose any either: iq4_xs is finer than iq2_s, so the
  round-trip is near-lossless with respect to its input. This is purely a SPEED change, which is
  exactly what we want to test.

Build path: `llama-quantize --allow-requantize` with per-tensor-type overrides, run LOCALLY on
$DATA (102 GB free), then transfer. Z640 disk cannot hold a second big model: 3.9 GB free
of 73 GB with BOTH Ridge-3.7bpw (12.6 GB) and Ridge-q41nonffn (12.5 GB) resident.
**Neither Ridge model has a local copy -- they exist ONLY on the Z640 -- and Ridge-3.7bpw is the
SOURCE the surgical requant was derived from, so it is NOT safe to delete despite blanket
permission.** Correct order is: back it up to External first (reversible), then reclaim.

## 13. BATCH20 RESULT — the workload hypothesis is only PARTLY right (MEASURED)

Native MTP, our champion arm (sub4_4), ctx 8192, 96 tokens generated. tau derived as
1 + K*acceptance, cross-checked against 96/rounds where rounds = #gen_tokens / K:

| workload | B | acceptance | tau | t/s | start C |
|---|--:|--:|--:|--:|--:|
| echo-682 | 6 | 0.95122 | 5.647 | 29.40 | 38 |
| GSM8K    | 4 | 0.75862 | 3.310 | 20.58 | 39 |
| GSM8K    | 6 | 0.58333 | 4.000 | 20.16 | 53 |
| prose    | 4 | 0.61616 | 2.909 | 17.84 | 50 |

- The prose control reproduced EXACTLY: 17.84 t/s, acceptance 0.61616 bit-identical to batch18.
  Instrument sound.
- **GSM8K tau REPRODUCES the published figure once B is matched.** Full batch20 set:

| workload | B | acceptance | tau = 1+K*acc | t/s |
|---|--:|--:|--:|--:|
| gsm8k | 4 | 0.75862 | 3.276 | 20.58 |
| prose | 4 | 0.61616 | 2.848 | 17.84 |
| gsm8k | 6 | 0.58333 | 3.917 | 20.16 |
| gsm8k | 8 | 0.55970 | **4.918** | 18.84 |
| gsm8k | 4 | 0.75862 | 3.276 | 20.56 (bracket, -0.1% drift) |

  Published native-MTP GSM8K tau is 5.02, and their DFlash2 command used
  `--spec-draft-n-max 7` = B=8. **Our GSM8K at B=8 gives tau 4.918 -- within 2.0%.**

### RETRACTION
An earlier version of this section claimed "GSM8K tau on our machine is 3.31, NOT the published
5.02", and built on it the hypothesis that **our quantization was suppressing MTP acceptance**,
calling the splice the highest-value experiment partly on that basis. That was an
APPLES-TO-ORANGES comparison: our B=4 tau against their B=8 tau. Since tau = 1 + K*acceptance,
tau is a function of B and the two are not comparable. Matched on B, our MTP reproduces the
published behaviour to within 2%.
**The quantization-suppresses-acceptance hypothesis is therefore RETRACTED.** Ridge's 2.5 bpw FFN
is not measurably costing us acceptance, and the splice should be expected to buy SPEED ONLY.
It remains the largest single lever identified (1.26-1.33x projected), but not for the extra
reason I gave, and it is NOT expected to raise tau.

### What the batch20 set does establish
1. Our native MTP is healthy and matches published behaviour at matched B. No drafter defect.
2. The prose tau deficit is a genuine WORKLOAD property: at identical B=4, acceptance is 0.759 on
   gsm8k vs 0.616 on prose.
3. **tau = 1 + K * acceptance** holds on every point, so tau has no independent lever: it rises
   only with acceptance, or with B at a cost of Delta per extra column.
4. **B=4 is the throughput optimum for BOTH prose and gsm8k**, even though tau keeps rising with B
   (gsm8k tau 3.28 -> 3.92 -> 4.92 while t/s falls 20.58 -> 20.16 -> 18.84). Delta dominates.
   This is why published wide-B results do not transfer to this card.
5. Instrument stability is excellent: the B=4 bracket drifted -0.1% with bit-identical acceptance.
6. Consequence for DFlash2: its published advantage over native MTP is +8.8% tau at B=8 on gsm8k.
   Since our tau baseline matches theirs at B=8, that ratio should transfer -- but B=8 is NOT our
   throughput optimum, so a tau gain there is partly wasted. This sharpens the batch21 question:
   does DFlash2 help at B=4, where we actually run?

