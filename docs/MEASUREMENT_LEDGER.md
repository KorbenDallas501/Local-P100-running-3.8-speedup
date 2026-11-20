# Ridge / P100 optimisation — measurement ledger

Target: **20 t/s at genuinely resident 128K**; stretch 21–22 t/s.
Opened 2026-08-31. Every row is tagged **MEASURED** (this ledger, with the command) or
**PROJECTED** (arithmetic, no run). Never promote a PROJECTED row without a command in the log.

Card state at open: **69 °C idle, no fan, 0 MiB used, 405 MHz, 35.6 W.**
Idle floor with no airflow ≈ 67–69 °C. Abort point 78 °C. Heating ≈ 0.8 °C/s at 145 W
⇒ **~11 s of sustained compute per run from idle.** Fan promised overnight; escalate when
idle temp drops (poll detects it).

## Thermal budget → what is runnable NOW vs after the fan

| experiment | compute time needed | runnable without fan? |
|---|---|---|
| #5 per-format MUL_MAT microbench | ~1–3 s per format | **YES** |
| #1 `-ot token_embd=CPU` VRAM+decode | load (cool) + ~2 s decode | **YES** |
| #8 sampler chain A/B | ~2 s decode ×2 | **YES** |
| #11 CUDA-graph capture instrumentation | ~2 s decode ×2 | **YES** |
| #4 requant A/B decode | ~2 s decode | **YES** (requant is CPU, off-box) |
| #2 resident 128K decode | 128K prefill @ ~100 t/s ≈ **21 min** | **NO — needs fan** |
| #3 clock headroom (1189→1328 MHz) | sustained load | **NO — needs fan** |

## Baseline (inherited, for reference — NOT re-measured yet)

| # | config | t/s | context | source |
|---|---|---|---|---|
| B1 | Ridge + MTP DRAFTN=1, 8K | 16.67 | 8K, empty | prior session, MEASURED |
| B2 | Ridge + MTP, 128K load | 16.28 | 128K **allocated, empty** | prior session, MEASURED, 16-tok sample |
| B3 | Ridge no spec | 10.62 | 8K | prior session, MEASURED |
| B4 | Ridge at genuinely resident 128K | **UNKNOWN** | — | never measured; predicted 12–13 (PROJECTED) |

## Findings established before any new run

| id | claim | status |
|---|---|---|
| F1 | sm_60 `mmvq.cu` = 81,883 XMAD, 0 IMAD, 0 IDP4A, 12,144 LDG; sm_86 same file = 11,840 IDP4A + 26,764 IMAD, 12,331 LDG | **MEASURED** (cuobjdump, this session) |
| F2 | ⇒ identical bytes moved, 2.1× the integer instructions, all 16-bit XMAD (Pascal has no native 32-bit IMAD) | **MEASURED** (follows from F1) |
| F3 | prior "dp4a costs only 1.07×" was measured on sm_86 in a bandwidth-bound regime — does not transfer to the ALU-bound P100 | **REASONED** — invalidates a prior SETTLED conclusion |
| F4 | q4_1 hits 318 GB/s-equiv on P100, exceeding the asserted "~200 GB/s compute ceiling" by 1.6× | **MEASURED** (prior session's own table, unfollowed) |
| F5 | prior ceiling calc scaled t1 but held t2+tm fixed, though all three are the same weights | **REASONED** — ceiling is an artifact |
| F6 | 8K decomposition (t1+t2+tm=120.7 ms, acc 0.694 → 14.03 t/s) contradicts reported 15.92 t/s by 13% | **MEASURED inconsistency** — unresolved |
| F7 | `TREE_MODE` + `parent_ids` + `persist_inter` already exist in `gated_delta_net.cu` — multi-branch speculation through DeltaNet is implemented, not impossible | **MEASURED** (source, this session) |
| F8 | q4_1 is imatrix-aware (`quantize_row_q4_1_impl(..., quant_weights)`, ggml-quants.c:2111) | **MEASURED** (source) |
| F9 | Ridge has untied embeddings ⇒ token_embd (1.27 B, Q6_K, ~1.04 GB) is gather-only at decode | **MEASURED** (GGUF inventory) |

## Run log

(appended below as runs complete)

---

## RUN 1 — per-format MUL_MAT mat-vec, P100, m=4096 n=1 k=14336  (**MEASURED** 2026-09-01 05:21)

Command (on Z640):
```
LD_LIBRARY_PATH=~/luceops:~/lucebin ~/luceops/test-backend-ops perf -b CUDA0 -o MUL_MAT \
  -p "type_a=(q4_0|q4_1),type_b=f32,m=4096,n=1,k=14336"
```
58,720,256 weights per run. ps/weight = us_per_run / 58.72e6 × 1e6.

| type | bpw | us/run | **ps/weight (measured)** | prior table | GB/s-equiv |
|---|---|---|---|---|---|
| q4_1 | 5.00 | 118.73 | **2.02** | 1.97 | 309 |
| q4_0 | 4.50 | 136.25 | **2.32** | *(absent)* | 243 |

**Instrument VALIDATED.** q4_1 replicates the prior table to 2.5% (2.02 vs 1.97) and its
GB/s-equiv to 3% (309 vs 318). The per-format table can therefore be trusted for *ranking*,
and now for magnitude too, at this shape.

**HYPOTHESIS FALSIFIED (mine, §3.3): q4_0 is NOT faster than q4_1.** It is 15% *worse* per
weight (2.32 vs 2.02) despite being 0.5 bpw smaller. So q4_1 remains the champion cheap format
and the §3.2 allocation does not improve. Cost to find out: 8 s of GPU time. q5_0/q5_1/iq4_nl
still unmeasured.

**Thermal calibration (MEASURED):** start 71 °C → peak **75 °C** for 2 perf cases (~8 s wall,
~4 s of 147 W load). Load spikes to 147 W and clocks *drop* 1189 → 1063 MHz under it.
⇒ **~2 °C per perf case; ~3 cases per burst from a 71 °C start.**
⇒ **Clock droop 1189→1063 MHz is now MEASURED under load** — relevant to item #3: the card is
already throttling below its 1189 idle clock, let alone the 1328 boost.

**Incidental, useful:** the build prints `ggml_cuda_graph_set_enabled: disabling CUDA graphs due
to GPU architecture`. That line is the capture indicator item #11 needs — if `LUCE_CUDA_GRAPHS=1`
suppresses it, graphs are genuinely engaged. The prior CUDA-graph test never checked this.

## RUN 2 — more formats, n=1, same shape (**MEASURED** 05:31)

| type | us/run | **ps/weight (measured)** | prior table | agreement |
|---|---|---|---|---|
| q4_1 | 118.73 | **2.02** | 1.97 | 2.5% |
| q4_0 | 136.25 | **2.32** | *(absent)* | — |
| iq4_xs | 137.71 | **2.35** | 2.29 | 2.4% |
| q6_K | 269.69 | **4.59** | 4.50 | 2.0% |

**The prior per-format table is now VALIDATED on 4 formats to within 2.5%.** Magnitudes, not just
ranking, can be trusted at this shape. Consequences for the §3.2 plan:
- **q4_1 (2.02) is confirmed the fastest format on this card**, 2.27× faster per weight than q6_K.
- **q6_K confirmed the worst (4.59).** lm_head + attn_output (2.13 B params) Q6_K→q4_1 saves
  2.57 ps/weight × 2.13e9 = **5.5 ms** — matches the 5.4 ms projection. This line of the plan holds.
- q4_0 (2.32) ≈ iq4_xs (2.35) in speed, but iq4_xs is 0.25 bpw smaller ⇒ **iq4_xs still preferred
  over q4_0**; q4_0 has no role.

**Thermal (MEASURED):** 2 cases = +4 °C (72→76). **~2 °C per perf case.** Budget from the 72 °C
floor is therefore **2–3 cases per burst.**

## FAN — no measurable effect (**MEASURED** 05:20–05:28)

Idle, 0 MiB loaded, 37 W: **72 °C flat for 8 minutes with the fan on**, vs 69–71 °C before it and
67–68 °C in the historical record. The fan is **not** lowering the idle floor, and idle equilibrium
is now 72 °C. Cooldown gating to 71 °C is unreachable — retuned to 72–73 °C.
*Operational consequence: burst budget did not improve. The blower shroud remains the real fix;
this fan does not substitute for it.*

## Judgment calls (user away, wheel taken)

- **Will not power the box off** except on a thermal emergency: the history's boot ritual needs the
  8-pin physically pulled, so a shutdown while the user is away would end all experiments until
  they are back at the machine. Guard + cooldown gating is the safer control.
- **Requant will be done without a fresh imatrix.** Generating one needs sustained GPU compute over
  a corpus, which the thermal budget forbids. Run #4 therefore tests the **speed** prediction only;
  quality is deferred to a separate perplexity run. Stated explicitly so it is not read as a
  quality result.
- **Crash trail** `~/p100_trail.log` logs timestamp + GPU uptime + temp every 15 s, so any crash is
  attributable. Boot was **2026-09-01 05:13:19**; zero Xids as of 05:28.
- Gotcha found: `pkill -f "[p]100_trail"` self-matches the ssh command line that contains the
  script path, killing the session (exit 255). Do not pkill a pattern that appears in your own argv.

## RUN 3 — Ridge tensor inventory, exact (**MEASURED** 05:35, header-only, no GPU)

`python3 ~/gguf_names.py Qwen3.8-27B-Ridge-3.7bpw.gguf` — vocab confirmed **248,320**
(1,271,398,400 / 5120), ffn_length 17408, native ctx 262144, **17 attention layers**
(attn_q/k/v/output) + **48 DeltaNet layers** (attn_qkv/attn_gate/ssm_out).

| tensor | n | params | type |
|---|---|---|---|
| ffn_down / ffn_gate / ffn_up | 65 each | 5,793,382,400 each | IQ2_S, IQ3_S, Q6_K |
| attn_qkv (DeltaNet) | 48 | 2,516,582,400 | Q4_K |
| attn_gate (DeltaNet) | 48 | 1,509,949,440 | Q4_K |
| ssm_out (DeltaNet) | 48 | 1,509,949,440 | Q4_K |
| **output.weight (lm_head)** | 1 | 1,271,398,400 | **Q6_K** |
| **token_embd.weight** | 1 | 1,271,398,400 | **Q6_K** |
| attn_q | 17 | 1,069,547,520 | Q5_K |
| attn_output | 17 | 534,773,760 | Q6_K |
| attn_k / attn_v | 17 each | 89,128,960 each | Q5_K |
| nextn.eh_proj (MTP head) | 1 | 52,428,800 | Q6_K |

### Item #1 saving is now EXACT (no GPU needed)
token_embd = 1,271,398,400 params × 6.5625 bpw / 8 = **1,043,059,200 B = 994.7 MiB.**
`-ot 'token_embd\.weight=CPU'` frees **994.7 MiB**, arithmetically certain, zero quality cost.
Still to confirm on GPU: that llama.cpp honours the override and decode stays correct.
(Prior plan's "token_embd Q6_K→Q4_K saves 0.328 GB" is dominated 3:1, as claimed.)

### RESIDENT 128K — the number that reframes the target (**PROJECTED**, arithmetic)
head_count_kv = 4, k-proj out 1024 ⇒ K+V = 2048 values/token/attention-layer.
f16 = 4096 B; turbo4 ≈ /3.8 ⇒ ~1078 B; × **17** attention layers = **18.3 KB/token**;
× 131072 = **2.40 GB of KV resident at 128K**.
Reading 2.40 GB/token at the ~170 GB/s this card achieves = **~14 ms/token**.

| state | round time | t/s | status |
|---|---|---|---|
| 128K allocated, **empty** (prior headline) | 109 ms | **16.28** | MEASURED (prior) |
| 128K **resident** | 109 + 14 = 123 ms | **~14.4** | **PROJECTED** |

**So the honest starting point against the new 20 t/s target is ~14.4 t/s, not 16.3.**

### Requant predicted saving, using MEASURED ps/weight
| tensor group | params | Δps/weight | Δt1 | Δsize |
|---|---|---|---|---|
| output.weight Q6_K→q4_1 | 1.2714e9 | 4.59→2.02 | **−3.27 ms** | −0.248 GB |
| attn_output Q6_K→q4_1 | 0.5348e9 | 4.59→2.02 | **−1.37 ms** | −0.105 GB |
| attn_qkv+gate+ssm_out Q4_K→q4_1 | 5.5365e9 | 2.74→2.02 | **−3.99 ms** | +0.346 GB |
| attn_q/k/v Q5_K→q4_1 | 1.2480e9 | 3.16→2.02 | **−1.42 ms** | −0.078 GB |
| **total** | | | **−10.05 ms** | **−0.085 GB** |

t1 89.2 → **79.2 ms predicted.** This is the falsifiable prediction of RUN 4.

### Roadmap to 20 t/s at RESIDENT 128K (**PROJECTED** — each step needs its own measurement)
| step | round | t/s |
|---|---|---|
| resident 128K today | 123 ms | 14.4 |
| + surgical requant (this run, no download) | ~110 ms | **16.1** |
| + full BF16 §3.2 allocation (−18.5 ms t1) | ~99 ms | **17.9** |
| + GP100 half2 mat-vec kernel (−10…25% t1) | ~88 ms | **~20.2** |

**Consequence for the target: the requant alone does NOT reach 20 t/s at resident 128K.**
Reaching 20 needs the BF16 allocation *and* item #10 (the fp16 kernel), or a reduction in the
14 ms KV term. Stretch 21–22 requires attacking KV as well. Stated now so the target is not
quietly missed later.

## RUN 4 — surgical requant (**IN PROGRESS** 05:37→)

Two tooling patches were needed and are committed on `sm60-p100-kernels`:

- **`dffc262d3` quantize: honour `--tensor-type` under the COPY ftype.** COPY refused to quantize
  anything, so changing one tensor meant re-encoding the whole model — lossy for the FFN we
  intend to leave alone. Also fixed a second bug found in my own first patch: under COPY the
  caller's ftype is `ALL_F32`, so `default_type` was F32 and the override block (gated on
  `ggml_is_quantized`) was skipped — it started converting `output.weight` to **f32, 994 MiB →
  4850 MiB**, which would have filled the 16 G disk. Caught at tensor 1/866; partial file removed.
- **`3e971f335` test-backend-ops: `GGML_PERF_MIN_US`.** The perf loop hardcoded a 1 s minimum per
  case ≈ 2 °C of budget. Now tunable ⇒ ~5× more cases per thermal burst.

Verification that the requant is doing the intended thing (**MEASURED**, from the log):
```
output.weight          - applying manual override: q6_K -> q4_1
blk.N.attn_gate.weight - applying manual override: q4_K -> q4_1
blk.N.attn_qkv.weight  - applying manual override: q4_K -> q4_1
blk.N.ssm_out.weight   - applying manual override: q4_K -> q4_1
blk.11.attn_output.weight - q6_K, converting to q4_1 .. 24.61 MiB -> 18.75 MiB
blk.19.ffn_up.weight   - type = iq2_s, size = 27.227 MiB      <-- copied, not re-encoded
```
**Item #1 cross-check, exact:** the log reports `output.weight` (same shape as token_embd) at
**994.63 MiB** — against my predicted 994.7 MiB for token_embd. **Item #1's saving is confirmed
to 0.01% without using the GPU at all.**

## THERMAL WALL — the binding constraint tonight (**MEASURED**)

| time | state | temp |
|---|---|---|
| 05:20–05:28 | idle, 0 MiB, 37 W, fan on | **72 °C** (equilibrium) |
| 05:31 | llama-server loading, 259 MiB, **49 W**, 1189 MHz | **76 → 79 °C** |
| 05:33–05:41 | idle, 0 MiB, 40 W, 405 MHz | **78 °C, not decaying** |

Two hard findings:
1. **Merely clocking up to 1189 MHz at 49 W with 259 MiB loaded pushed the card past 78 °C.** The
   card cannot hold even an idle-clocked llama-server. My `ot_test.sh` ran without the guard on the
   assumption that loading was thermally free — **that assumption was wrong** and is now corrected;
   every future script gets the guard.
2. **Heat does not leave.** After the load was killed the card sat at 78 °C for 8+ minutes at 40 W.
   The fan is not shedding accumulated heat, so the burst budget does not regenerate on a useful
   timescale. CPU load is *not* the cause — load average was 2.5 and the requant is I/O-bound.

**Consequence: GPU experiments are blocked at ~78 °C until the card decays below 73 °C, and it is
not decaying. The blower shroud is now the critical path for every remaining measured item**
(#1 GPU confirmation, #4 A/B decode, #8 sampler, #11 graphs, and above all #2 resident 128K).
Non-GPU work (requant, arithmetic, source patches) continues unaffected.

## RUN 4 RESULT — surgical requant artifact (**MEASURED** 05:43)

`Ridge-q41nonffn.gguf`, 351 s of CPU (I/O-bound, GPU untouched at 0 MiB throughout).

| | input | output | delta |
|---|---|---|---|
| bytes | 12,599,187,008 | **12,514,461,248** | **−84,725,760 (−84.7 MB)** |
| predicted delta | | | **−0.085 GB** |

**Size prediction exact.** Tensor verification:

| group | before | after |
|---|---|---|
| attn_qkv / attn_gate / ssm_out (48 each) | Q4_K | **Q4_1** |
| output.weight | Q6_K | **Q4_1** |
| attn_q / k / v / attn_output (17 each) | Q5_K / Q6_K | **Q4_1** |
| ffn_down/gate/up | IQ2_S 160 / IQ3_S 32 | **IQ2_S 160 / IQ3_S 32 — bit-copied** |
| token_embd | Q6_K | Q6_K (left for `-ot` CPU offload) |

Q4_1 = 213 tensors / **8,590,458,880 params** — exactly the intended target set
(1.2714 + 0.5348 + 5.5365 + 1.2480 = 8.5907e9). Q4_K and Q5_K eliminated.
**The artifact is correct. Its decode t/s is NOT yet measured** — see thermal wall.

**Falsifiable prediction for the A/B (no-spec, 8K, llama-bench, turbo4):**
baseline Ridge = 10.62 t/s (prior MEASURED) ⇒ requant should give **~11.9 t/s (+12%)**.
A result below ~11.2 falsifies the ps/weight→whole-model model.

## RUN 5 — GP100 fp16 probe: item #10 NOT supported (**MEASURED** 05:45)

`scripts/p100_fp16_probe.cu`, sm_60, same 4096×14336 shape, 300 iters:

| kernel | us/run | ps/weight |
|---|---|---|
| A int8 dot, sm_60 scalar dp4a fallback | 343.17 | 5.84 |
| B half2 / HFMA2 dequant-dot | **567.21** | 9.66 |
| **B/A** | **0.61× — SLOWER** | |
| llama.cpp production mmvq q4_1 (anchor) | **118.73** | 2.02 |

**Two honest readings, and the second matters more:**
1. The naive half2 formulation **loses**, 0.61×. Mechanism: unpacking nibbles to half needs a
   per-element `__int2half_rn` (I2F), and those conversions cost more than the HFMA2 saves.
   SASS confirms the intended difference was achieved (A: 41 XMAD / 0 HFMA2; B: 25 XMAD / 32 HFMA2).
2. **My baseline misses the production kernel by 2.9× (343 vs 118.73), so this probe is too weak to
   close the question either way.** Both kernels are one-warp-per-row with unvectorized loads.

**Verdict: item #10 downgraded to PLAUSIBLE-but-unsupported, NOT dead — and the specific next step
is now known:** a serious attempt must construct fp16 values from nibbles with the bitwise
"magic constant" trick (as Marlin/AWQ/FasterTransformer kernels do), avoiding I2F entirely. Testing
the I2F formulation was the wrong experiment; it cost 5 s of GPU time to learn that.

## THERMAL DRIFT — GPU work is now blocked (**MEASURED**)

Idle equilibrium over the session: **72 °C (05:20) → 78 °C (05:33) → 79 °C (05:46)**, at 40 W and
405 MHz with 0 MiB loaded. The accumulated heat does not leave; the fan does not shed it.
At 05:45 the guard killed a run **3 seconds in**, at 80 °C, because the card reaches 80 the moment
it clocks to 1189 MHz.

Card health after 33 min up and ~10 load events: **no Xid, no bus drop, boot time unchanged
(05:13:19), 16269 MiB still visible.** The card is not damaged; it is saturated.

**Lost to the wall:** the n=1 vs n=2 marginal-column test (the §4.2 decider) and the requant A/B.
Both are now armed to fire automatically: `~/queued_ab.sh` waits up to 4 h for ≤72 °C and then runs
the A/B with the guard on; if the card never cools, nothing runs and nothing is risked.

**GPU uptime accounting (for the crash question): booted 2026-09-01 05:13:19; still up at 05:46
(~33 min) with zero Xids.** `~/p100_trail.log` records temp + uptime every 15 s continuously.

---

# STATE OF THE TARGET — 20 t/s at genuinely resident 128K

| claim | value | status |
|---|---|---|
| Ridge, 128K allocated but empty | 16.28 t/s | MEASURED (prior) |
| Ridge, 128K **resident** | ~14.4 t/s | **PROJECTED** (2.40 GB KV / ~170 GB/s = ~14 ms/token) |
| + surgical requant (artifact exists, untested) | ~16.1 t/s | PROJECTED |
| + full BF16 §3.2 allocation | ~17.9 t/s | PROJECTED |
| + fp16 kernel | ~20.2 t/s | **SPECULATIVE — probe did not support it** |

**Blunt assessment: 20 t/s at resident 128K is not currently in hand, and the fp16 kernel that the
last step depends on just failed its first probe.** The realistic near-term ceiling is the requant
(≈16 t/s resident) plus the BF16 allocation (≈18 t/s resident). Reaching 20 needs either a
*working* fp16/bit-trick kernel or an attack on the 14 ms KV term — most plausibly a more
aggressive KV format than turbo4, or exploiting that only **17 of 65 layers** hold KV at all.

## Updated ranking after tonight

| Rank | Opportunity | Expected gain | Evidence | Status change |
|---|---|---|---|---|
| 1 | `-ot token_embd=CPU` — **994.7 MiB** freed | 0 t/s, +994.7 MiB | **PROVEN** arithmetically (994.63 MiB in the requant log) | ↑ confirmed exact |
| 2 | Surgical requant (artifact built) | predicted +12% | **STRONGLY SUPPORTED**; artifact verified, decode untested | ↑ built, A/B armed |
| 3 | **Blower shroud** | unblocks everything | **PROVEN** blocker: idle drifted 72→79 °C, guard kills 3 s in | ↑↑ now the critical path |
| 4 | Measure resident 128K | reframes target | needs ~21 min sustained prefill | blocked on shroud |
| 5 | q4_1 as the universal cheap format | −10 ms t1 | **MEASURED** 2.02 ps/weight, validated ±2.5% | ↑ confirmed |
| 6 | Attack the 14 ms KV term at 128K | up to +2–3 t/s | PLAUSIBLE, unexplored | ↑ new, now necessary for 20 |
| 7 | n=1 vs n=2 marginal column (§4.2 decider) | decides depth-2 | not measured — lost to the wall | queued |
| 8 | fp16/half2 kernel | +10–25% t1 | **SPECULATIVE**; naive I2F form measured 0.61× | ↓ downgraded, next form known |
| 9 | q4_0 as a cheaper q4_1 | — | **DEAD** — measured 2.32 vs 2.02 ps/weight | ↓ killed cheaply |
| 10 | Sampler chain, CUDA graphs | +3–5%, +2–5% | unproven | still queued |

## Next session — start here

1. **Fit the shroud.** Everything measured tonight says this is no longer a safety nicety; it is
   the gate on every remaining experiment. The fan does not substitute for it.
2. Read `~/queued_ab.log` — the requant A/B may have fired overnight if the card cooled.
   Prediction to check: **~11.9 t/s vs 10.62 baseline.**
3. Confirm `-ot 'token_embd\.weight=CPU'` on GPU (VRAM should drop 994.7 MiB; verify output sanity).
4. Run the marginal-column test: `GGML_PERF_MIN_US=150000 test-backend-ops perf -b CUDA0 -o MUL_MAT
   -p "type_a=(iq2_s|q4_1),type_b=f32,m=4096,n=[12],k=14336"` — ~0.6 s of load, decides depth-2.
5. Only then consider the BF16 pull (54 GB) — gate it on the A/B landing near +12%.

## Closing state 05:49

Cooling **stalled at 78 °C** (79→78 over 3 min, then flat). This corroborates the existing
an earlier note — *"battery fan off = 79 C idle equilibrium"* — so tonight's
78–79 °C floor is a **known, reproducible state of this card without the shroud**, not a new fault.
The fan the user added does not move it.

**Deliberate decision not to relax the thermal envelope while the user is away.** The queued A/B is
gated at ≤72 °C and will therefore probably not fire overnight. Raising the gate to ~76 °C with
LIMIT 82 would very likely let it run (P100 throttles ~85 °C, and `HW Thermal Slowdown` has been
0 µs for this card's whole history), but on hardware with a known marginal-contact/Xid-79 history I
am not widening safety margins unattended for a measurement that becomes trivially safe once the
shroud is on. To run it deliberately:
```
# on the Z640, user-supervised:
sed -i 's/-le 72/-le 76/; s/gt 73/gt 77/; s/LIMIT=80/LIMIT=82/' ~/queued_ab.sh && bash ~/queued_ab.sh
```

## One prior conclusion this session could NOT adjudicate

an earlier note records *"deltanet scans tokens SEQUENTIALLY, so each extra
verify-batch row costs 20-25 ms; DRAFTN=1 optimal by architecture."* My review argued the 20–25 ms
is instead **ALU-bound mmvq that does not amortise across columns** (F1/F2), which would mean
depth-2 becomes viable once kernels get cheaper — a materially different prescription.
**This remains a hypothesis. The deciding measurement (n=1 vs n=2 at fixed shape) was lost to the
thermal wall and is item 4 in the next-session list.** Until it runs, treat both the memory's
"architectural" claim and my "ALU-bound" claim as unresolved, and do not act on either.

## RUN 6 — cooling curve, definitive (**MEASURED** 05:47–05:53)

12 samples, 30 s apart, card fully idle (0 MiB, 405 MHz, **40.85 W constant**), fan on:
```
79 79 79 79 78 79 79 78 79 79 78 79   (C)
```
**Zero downward trend over 6 minutes. This is an equilibrium, not a decay.**
Confirmed again at 05:53 by the queued-A/B waiter: 8 consecutive 1-minute polls, all 78–79 °C.

### The mechanism (INFERRED, not measured)
The idle floor was **72 °C at 05:20** and is **78–79 °C at 05:53 at identical power (40.85 W)**.
Two different equilibria at the same wattage cannot both be die-level steady states, so the change
must be in what the heatsink is radiating *into*: **the chassis air has heated up.** ~10 load events
at 143–150 W dumped more into the case than the case can clear, and the GPU's equilibrium shifted
up with it.

**Operational consequence, and it is the main lesson of the session:** the constraint is not
"short bursts with short cooldowns". It is a **whole-machine budget of roughly 10 load events,
after which the box is done until the chassis cools** — tens of minutes to hours. Tonight's budget
was spent by 05:45. Planning future sessions around per-run cooldowns will therefore under-predict
the wall; plan around total events per session instead.

This also means the **shroud is an absolute prerequisite, not an optimisation**: without it there is
no duty cycle at all for the one experiment the target is defined against (resident 128K, ~21 min of
continuous prefill).

`~/queued_ab.sh` is left armed for 4 h. If real cooling is fitted overnight it fires by itself and
`~/queued_ab.log` will hold the requant A/B. Otherwise it logs "never cooled" and exits harmlessly.

---

## RUN 7 — n=1 vs n=2 marginal column (item #1 of the day plan)

**Date:** 2026-09-01 14:53 / 14:55. GPU up 34.5 ks (9.6 h) since boot 05:13:19. Zero Xid.
**Load consumed:** ~0.5 s + ~4 s = under 5 s total. Peak temp 80 C. Guard never fired.

Case list located in source, not guessed: `tests/test-backend-ops.cpp:9456`
`for (bs : {1,2,3,4,5,8,512}) for (type_a : all_types) test_mul_mat(type_a, F32, 4096, bs, 14336)`.
So `n` in the case string is the batch column count and the n=1/n=2 pair is native to perf mode.
Regex `type_a=q4_1,type_b=f32,m=4096,n=[12],k=14336` matches exactly 2 cases (n=512 excluded by the comma).

| case | us/run | run A | run B |
|---|---|---|---|
| q4_1 m=4096 n=1 k=14336 | 506.91 / 506.96 | 852 runs | 1704 runs |
| q4_1 m=4096 n=2 k=14336 | 561.72 / 561.89 | 426 runs | 1278 runs |

Reproducibility 0.05% across two independent bursts at different accumulation windows.

### F10 — MEASURED: the second column costs +10.8%, not +100%
54.8 us marginal on a 506.9 us base. The GEMV amortises the weight load across columns.

**Kills my own hypothesis** ("ALU-bound mmvq that does not amortise across columns", advanced in
the adversarial review as the rival to the DeltaNet-sequential explanation). It is falsified, and
falsified in the regime *most favourable* to it — see F11: SM clock is capped to 30% while memory
clock runs at full 715 MHz, i.e. maximally compute-starved. If extra columns were ALU-priced they
would have shown it here above all places.

**Therefore:** the 20-25 ms per extra verify row reported in
an earlier note cannot be coming from the matmuls. It is localised in
`gated_delta_net.cu`. "DRAFTN=1 optimal *by architecture*" is the wrong description; the correct
one is "the cost lives in one kernel, which already contains a TREE_MODE path". Still a hypothesis
until the DeltaNet kernel is timed directly at n_tokens=1 vs 2 — that is now the decisive test.

### F11 — MEASURED: the card is clock-capped to 405 MHz by SW thermal slowdown, under load
```
SW Thermal Slowdown : Active        HW Thermal Slowdown : 0 us   (counter N/A for SW)
Graphics/SM 405 MHz    Applications 1189 MHz    Max 1328 MHz
Memory 715 / 715 MHz  <-- NOT reduced
GPU Slowdown 82 C   GPU Shutdown 85 C   GPU Target Temperature: N/A (not tunable)
```
39 consecutive 100 ms clock samples across ~4 s of *sustained* load: all 405 MHz. The clock does
not ramp. 405/1189 = 34% of application clock, 30% of max.

Consequences, in order of importance:

1. **`HW Thermal Slowdown : 0 us EVER` is SUPERSEDED as evidence.** It is recorded in
   an earlier note as proof the card was never thermally distressed. It is
   worthless: SW thermal slowdown engages first and has **no counter** (`N/A`), so it was invisible
   to every check run to date.
2. **The 79-80 C "equilibrium" was never passive.** It is the driver regulating to a thermal target
   by cutting clocks. That is also why the fan appeared to do nothing to *temperature* — any
   headroom is spent on clocks, not on a lower reading. The fan may have been working all along.
3. **Absolute timings taken in this state are ~4.3x pessimistic** (506.9 vs the 118.73 us anchor for
   this exact shape). Ratios within a single burst remain valid.
4. Thresholds are 82/85, not the 85/90 I assumed when writing the guard. Previous LIMIT=78/80
   numbers were set on a wrong assumption; corrected to LIMIT=81.
5. Not tunable: `GPU Target Temperature : N/A`. **Only physical cooling restores the clock.**
   Item 13 of the day plan therefore resolves negatively and is closed.

### F12 — STRONGLY INFERRED: candidate reconciliation for F6 (the 13% arithmetic gap)
Trail log over 9.4 h: 2251 samples at 405 MHz, 10 at 1189, 3 at 1303. The full-clock samples sit at
74-79 C with 259 MiB resident — the server episodes. So the card *held* 1189 MHz earlier today and
fell to the 405 MHz cap only after prolonged heat-soak.

That gives a mechanism for F6, where the 8K component decomposition (t1 89.2 + t2 23.3 + tm 8.21 ms
-> 14.03 t/s) predicts *lower* than the reported end-to-end 15.92 t/s. If the component timings were
captured later/hotter than the end-to-end run, they were inflated by partial SW thermal slowdown,
depressing the predicted figure. Direction of error matches: throttled components -> too-slow
prediction.

**Falsify cheaply:** re-measure t1/t2/tm with a clock sampler attached, discard any sample where
clocks.sm < 1189, and re-derive. Requires the shroud. Until then F6 stays open, but the "the
decomposition model is wrong" reading is no longer the leading explanation.

### Judgment calls, RUN 7
- Enumerated the perf case list from **source** rather than probing the card with a loose regex.
  At 3 C from slowdown a regex that matched 50 cases instead of 2 was an unacceptable risk.
- Used `support` mode (~zero load, confirmed 0 C rise) to check output format before any perf run.
- Did **not** attempt to raise the thermal target or unlock clocks. It is not exposed on this card,
  and on a board with Xid 79 marginal-contact history I would not have done it if it were.
- Stopped GPU work at 80 C once F11 established that nothing further is learnable at 405 MHz.

---

## RUN 8 — GDN scan cost, column amortisation, CUDA-graph forensics (2026-09-01 15:0x-15:3x)

**Load consumed:** ~8 s total across 6 bursts. Peak 80 C. Guard never fired. Zero Xid.
**All GPU figures below taken at the 405 MHz SW-thermal cap (F11). Ratios valid, absolutes ~4.3x pessimistic.**

### Model geometry — MEASURED from the GGUF, not assumed
Dependency-free header parse (`/tmp/ggufkv.py`) of `Ridge-q41nonffn.gguf`:
```
ssm.state_size = 128   ssm.group_count = 16   ssm.time_step_rank = 48   ssm.inner_size = 6144
attention.key_length = 256  attention.value_length = 256  attention.head_count_kv = 4
block_count = 65   full_attention_interval = 4   nextn_predict_layers = 1
```
Loader mapping (`src/llama-model.cpp:492`): head_k_dim = head_v_dim = ssm_d_state; n_k_heads =
ssm_n_group; n_v_heads = ssm_dt_rank. So GDN = 16 k-heads / 48 v-heads / head dim 128, v_repeat 3.
Consistency check: 48 x 128 = 6144 = ssm.inner_size. OK.

**F13 — MEASURED: resident-128K KV is 2.41 GB, now derived rather than estimated.**
17 attention layers x 131072 tok x (4 kv-heads x 256 x 2 for K+V) x 2 B / 3.8 = **2.41 GB**.
Confirms the 2.40 GB figure the 20 t/s target was reframed against. `attention.key_length` is
**256, not 128** — the earlier "k-proj out 1024" note was right by coincidence (4 x 256).

**F14 — MEASURED: the 48 GDN layers contribute ZERO per-token KV growth.**
Recurrent state = 48 v-heads x 128 x 128 x 48 layers ~= 151 MB, constant in context length.
100% of KV growth is the 17 attention layers, and `full_attention_interval=4` identifies them
(il % 4). Windowed/selective KV (day-plan item 12) is therefore a **17-layer** problem, not 65.

### F15 — MEASURED: the DeltaNet scan is sequential AND cheap
New perf cases added at the true shape (`tests/test-backend-ops.cpp`, `make_test_cases_perf`).

| n_seq_tokens | us/run | marginal |
|---|---|---|
| 1 | 111.60 | - |
| 2 | 136.45 | +24.85 |
| 3 | 160.76 | +24.31 |
| 4 | 185.33 | +24.57 |
| 8 | 292.37 | +26.76 avg |

Linear, intercept ~87 us, slope **~24.6 us per extra token per layer**. So the scan IS sequential
in n_tokens -- that half of an earlier note is CONFIRMED.

But priced: 48 GDN layers x 24.6 us = **1.18 ms per extra token at 405 MHz, ~0.40 ms at 1189 MHz.**
Against a reported marginal verify cost of 20-25 ms, DeltaNet is **~2%**.

**F16 — the "20-25 ms per extra verify row" is REAL but MISATTRIBUTED.**
The mechanism is not the recurrent scan. See F17.

### F17 — MEASURED + STRONGLY INFERRED: the cost is mmvq launch geometry, at the 3rd column
q4_1 and iq2_s, m=4096 k=14336, same burst:

| n | q4_1 us | marginal | iq2_s us | marginal |
|---|---|---|---|---|
| 1 | 506.9 | - | 877.8 | - |
| 2 | 561.7 | **+54.8 (+10.8%)** | 885.2 | **+7.4 (+0.8%)** |
| 3 | 738.1 | +176.4 (+31%) | - | - |
| 4 | 952.7 | +214.6 | 1347.9 | +231/col |
| 8 | 1778.7 | +206.5/col | - | - |

Marginal is ~200 us/col for every step from n=2 onward. Only n=1 -> n=2 is anomalously cheap.

**Mechanism (source, `ggml/src/ggml-cuda/mmvq.cu`):** Pascal uses MMVQ_PARAMETERS_GENERIC, where
`calc_nwarps` = 4 for ncols_dst 1-4 (no change), but `calc_rows_per_block` returns **1 for
ncols_dst==1 and 2 for ncols_dst 2-8**. n=1 therefore runs at half the block's row capacity; the
second column consumes slack that n=1 wastes. From the third column on, the kernel is in its true
linear compute-bound regime.
Dispatch confirmed: `ggml-cuda.cu:2884` puts the mmvq branch *before* MMQ, and the gate is
`src1->ne[1] <= MMVQ_MAX_BATCH_SIZE` (8), so n=3 is still mmvq -- the discontinuity is internal,
not a kernel switch.

**Consequence for DRAFTN, which is the answer to the "architecture vs implementation" question:**
- DRAFTN=1 verifies **2** columns -> lands in the free slot -> genuinely cheap.
- DRAFTN=2 verifies **3** columns -> pays a full ~200 us/col, ~+31% of matmul time. Scaled over
  the model's matmuls (~70-80% of an 89 ms t1) that is **~+22 ms**, i.e. exactly the reported
  20-25 ms per extra row.
- **Verdict: "DRAFTN=1 optimal by architecture" is WRONG. It is optimal because of a launch-geometry
  step in one CUDA kernel.** This is an IMPLEMENTATION cost. It is the distinction the review asked
  to separate, and it now has a measured mechanism on both sides.

**Falsifiable prediction:** raising `calc_rows_per_block` (and/or nwarps) for ncols_dst==3..4 on
Pascal should flatten the n=2 -> n=3 step. If it does, DRAFTN=2 becomes viable and the acceptance
term (1+a+a^2 = 2.18 vs 1.69 at a=0.694) is worth ~+28% tokens per round. If it does not, the
200 us/col is genuine ALU/bandwidth cost and DRAFTN=1 stands -- but for the right reason.
Cost to test: one constexpr edit, a rebuild, and ~1 s of GPU. **Highest-value item in the queue.**

### F18 — the arithmetic gap (day-plan item 1) is narrowed, not closed
Inherited 8K decomposition: t1 89.2, t2 23.3, tm 8.21 ms, acceptance 0.694 -> 14.03 t/s vs a
reported 15.92 t/s (13% apart).

Measured contributions to the 23.3 ms second-column cost:
- DeltaNet: **0.40 ms** (F15) -- 1.7%
- matmul amortisation: q4_1 +10.8%, iq2_s +0.8%; Ridge is 52% IQ2_S by weight, so the
  weighted matmul marginal is ~5% of matmul time ~= **3-5 ms**
- **Unattributed: ~18 ms.**

Two candidate mechanisms are now EXCLUDED by measurement: DeltaNet (F15) and kernel-launch
overhead (F19, graphs give -1.5%). Remaining candidates, untested: lm_head (output.weight is
1.27 G params at Q6_K = 4.59 ps/weight ~= 5.8 ms/column at full clock), attention over 17 layers,
norms/RoPE/elementwise scaling linearly, sampler. **Next test: per-op profiling of a 2-column
decode.** Needs the shroud.
F12's partial-throttle explanation also remains live and would act in the right direction.

### F19 — MEASURED: CUDA graphs cost 1.1-1.8% on sm_60; capture status still UNKNOWN
Re-paired the Aug 31 logs correctly (I first mis-paired them and reported +1.8%; withdrawn):

| pair | GRAPHS=0 | GRAPHS=1 | delta |
|---|---|---|---|
| spec=none | 11.90 | 11.75 | -1.3% |
| draft-mtp N=1 | 17.43 | 17.23 | -1.1% |
| spec=none (verbosity 5) | 11.62 | 11.41 | -1.8% |

Three pairs, all negative. The inherited "slightly negative" was correct.

**Instrument invalidated:** I initially took "0 occurrences of the disabling message in the
GRAPHS=1 logs" as proof capture succeeded. The **control logs are also 0**, so debug logging was
never live -- `GGML_LOG_DEBUG` is a real call (not compiled out, `ggml-impl.h:122`) but
llama-server's log callback filters it and `--verbosity 5` did not lift it. The zero proves nothing.
So we still do not know whether the -1.5% is "captured and slower" or "never captured, paying only
attempt overhead". **Item 8 is NOT resolved.**

**Static findings (no GPU):**
- The pre-Ampere gate is explicitly a heuristic (`ggml-cuda.cu:4720` comment) and
  `LUCE_CUDA_GRAPHS=1` lifts it. Verified working: with it set, the disabling message disappears.
- `ggml_cuda_graph_check_compability` bails only on (a) split buffers and (b) `GGML_OP_MUL_MAT_ID`
  with non-quantized src0 / TQ4_1S / TQ3_1S / ne[2] > mmvq_mmid_max.
  **Qwen3.8-27B is dense (`qwen35`, not `qwen35moe`) and single-GPU, so neither applies.** The 27B
  decode graph should qualify.
- There is a **warmup state machine** (`ggml-cuda.cu:4795`): graphs engage only after 2 consecutive
  calls with no property change. `test-backend-ops` re-allocates per case, so it never warms up --
  which is why the harness shows capture=0 and why it is not a valid proxy for the model.

**Instrumentation added but NOT certified.** `LUCE_GRAPH_STATS=1` prints INFO-level counters
(capture/instantiate/update/launch/direct_eval). It fires, but reports `launch=1` with
`capture=0, instantiate=0`, which is internally inconsistent given the counters' placement
(verified by re-reading the patched region). Do not trust these numbers until that is explained on
the real model. Recorded as a known defect, not a result.

### Judgment calls, RUN 8
- Enumerated / added perf cases from source; never probed with a loose regex at 3 C of headroom.
- Did **not** attempt `nvidia-smi -lgc` to force 1189 MHz: it needs root (no passwordless sudo) and
  on a card with Xid 79 marginal-contact history I would not force clocks unattended even with it.
  HW slowdown (82 C) and shutdown (85 C) are hardware and would still protect, but the call is the
  user's. One-liner left below.
- Corrected my own guard limits (LIMIT 78/80 -> 81) after finding the real 82/85 thresholds.
- Caught and disclosed three instrument failures rather than reporting through them: the
  `~/lucebin` library-shadowing (old lib loaded, so the first graph test was meaningless), the
  debug-log-level control, and the inconsistent graph counters.
- Repeated a known mistake: declared the graph counters below their first use, exactly as with
  `perf_min_us` last night. Fixed by relocating; worth a lint rule if this recurs.

### For the user, when convenient
Force the application clock for a supervised measurement (needs root, ~1 s of load):
```
sudo nvidia-smi -lgc 1189,1189 && \
  LD_LIBRARY_PATH=$HOME/luceops2:$HOME/lucebin START_MAX=80 LIMIT=81 \
  bash ~/p100_burst.sh clkcheck env GGML_PERF_MIN_US=200000 \
  ~/luceops2/test-backend-ops perf -b CUDA0 -o MUL_MAT \
  -p 'type_a=q4_1,type_b=f32,m=4096,n=[12],k=14336' ; sudo nvidia-smi -rgc
```
Expect ~118 us at n=1 if the cap is the only thing holding performance down (it matched that
anchor when cool). This validates every absolute number taken today.

### F20 — CAVEAT on F17: the column-cost ratios are regime-dependent, and 405 MHz exaggerates them
At 405 MHz with memory at full 715 MHz, the kernel is deeply ALU-bound: a q4_1 4096x14336 weight
read is ~36.7 MB, which at 732 GB/s peak is ~50 us, yet n=1 measures 507 us -- 10x off peak.
At the 1189 MHz application clock the same case measured **118.73 us = 309 GB/s-equivalent, 42% of
peak**, i.e. materially closer to bandwidth-bound.

Extra columns are pure ALU work with no extra weight traffic. In a bandwidth-bound regime that work
is more hideable. **So the ~200 us/col marginal, and hence F17's ~+22 ms estimate for the 3rd
column, is very likely an OVERESTIMATE of the full-clock cost.** Direction of the error favours
DRAFTN=2 being cheaper than F17 concludes, not more expensive.

This does not disturb F15/F16 (DeltaNet is ~2% of the marginal under any regime) or the
architecture-vs-implementation verdict (the mechanism is a launch-geometry step either way). It does
mean the *magnitude* in F17 must be re-measured at 1189 MHz before anyone acts on it.
**Status: F17 mechanism = MEASURED. F17 magnitude = PLAUSIBLE, regime-limited.**

---

## RUN 9 — THE RESULT OF THE DAY: Pascal mmvq rows_per_block (commit b255c7bd2)

**Load consumed:** ~10 s across 9 bursts. Peak 80 C. Zero Xid. GPU up 10.2 h, boot 05:13:19.
All figures at the 405 MHz cap; see F20 for why the magnitudes are regime-limited.

### F21 — MEASURED: single-column decode was running at half the block's row capacity
`ggml/src/ggml-cuda/mmvq.cu`, `calc_rows_per_block`, table MMVQ_PARAMETERS_GENERIC:
`ncols_dst == 1` returns **1** row per block; `ncols_dst 2..8` return **2**.
Pascal (and every other NVIDIA arch) uses GENERIC. So the dominant decode shape got half the
rows per block that a 2-column shape gets.

Raising ncols_dst==1 to 2 rows per block, Pascal-scoped (new MMVQ_PARAMETERS_PASCAL, cc < TURING):

| format | baseline us | Pascal us | change |
|---|---|---|---|
| q4_1  |  506.91 | **376.71** | **-25.7%** |
| iq3_s | 1146.97 | **966.09** | **-15.8%** |
| iq2_s |  877.80 | **762.67** | **-13.1%** |

n=2 control, deliberately unchanged: q4_1 561.72 -> 561.64, iq2_s 885.23 -> 885.30,
iq3_s 1171.19 -> 1154.65. Confirms the edit is localised to ncols_dst==1.

**Correctness: `test-backend-ops test -o MUL_MAT`, 11 cases per format, 0 FAIL** for
q4_1 / iq2_s / iq3_s / q6_K / iq4_xs.

### F22 — MEASURED: gains above n=1 are type-dependent, so they were NOT taken
rows_per_block 2 -> 4 at ncols_dst 2..4:

| format | n=2 | n=3 | n=4 |
|---|---|---|---|
| q4_1  | -7.6% | -7.0% | -10.6% |
| iq3_s | -9.9% | - | - |
| iq2_s | **+10.1%** | +2.5% | -0.2% |

rows_per_block=8 is worse than 4 (q4_1 n=4: 882.78 vs 852.77), so 4 is a genuine optimum.
A **per-type Pascal table at ncols_dst==2** would capture q4_1/iq3_s while holding iq2_s at 2.
`calc_rows_per_block` currently takes no `type` argument, so that is a slightly larger change.
Projected value: ~3.4% of matmul time on the 2-column path -> ~+2.6% t/s. Queued, not done.

### F23 — Ridge weight distribution, MEASURED from the GGUF (866 tensors, 27.321 G params)
```
iq2_s  160 tensors  14,260,633,600  52.2%
q4_1   213 tensors   8,590,458,880  31.4%
iq3_s   32 tensors   2,852,126,720  10.4%
q6_K      5 tensors   1,591,214,080   5.8%
q8_0     96 tensors      23,592,960   0.1%
f32     360 tensors       2,671,616   0.0%
```
Confirms the inherited "52% IQ2_S". Also confirms the surgical requant landed: no Q4_K, no Q5_K.

### F24 — PROJECTED: +13% t/s on the no-spec decode path (NOT the champion path)
Time-weighting the F21 gains by F23 params and measured per-format ps/weight, with q6_K assumed
unimproved (conservative, not measured under the Pascal table):

**-15.3% matmul time on the 1-column path.** At matmul ~75% of decode -> ~-11.5% decode time
-> **~+13% t/s**, i.e. no-spec 11.90 -> ~13.4 t/s.

**Important limitation, stated plainly:** the champion config is `draft-mtp DRAFTN=1`, whose target
forward runs at **ncols_dst==2** -- exactly the case F22 left untouched. **So the champion (16.67
t/s) probably gains little or nothing from F21.** The measured win lands on the no-spec path.
Capturing the champion path requires the per-type n=2 table in F22 (~+2.6%).
Both are PROJECTIONS from single-shape microbenchmarks at 405 MHz. Neither is an end-to-end t/s
measurement, and F20 says the full-clock magnitudes are likely smaller.

**Falsify cheaply, in this order:**
1. `iq4_xs` at n=1, Pascal vs baseline. **The champion model is IQ4_XS, not Ridge, and this is the
   one format I did not get** -- the guard killed the run when idle crept to 80 C. ~0.4 s of GPU.
2. Same two cases at a forced 1189 MHz (needs root) to convert F21 into a full-clock number.
3. End-to-end `llama-server` no-spec decode, Pascal build vs baseline. Predict 11.90 -> ~13.4.
   If it comes in under ~12.5, the matmul-share model in F24 is wrong.

### Thermal wall reached
Idle crept 79 -> 80 C over ~10 s of cumulative load across the session. At 80 C idle even a 0.4 s
burst trips the 81 C guard: the `iq4_xs` run was killed mid-case with a 498-byte log. Box verified
healthy afterwards (alive, boot time unchanged, 0 MiB, no Xid).
**GPU work is finished until cooling is fitted.** I did not relax the guard to force the last run.

---

# THE QUEUE — ordered for the first minutes after cooling is fitted

Ordered by information per GPU-second, not by hoped-for upside. The first ~15 s of GPU time should
spend itself entirely on falsification, not on a requant or a long benchmark.

## Tier 0 — seconds, and they decide what everything else is worth (~20 s total)

| # | test | GPU s | decides | upside | risk | diff |
|---|---|---|---|---|---|---|
| 1 | `iq4_xs` n=1, Pascal vs baseline | 0.4 | whether F21 helps the **champion model at all** (champion is IQ4_XS, not Ridge) | gates F24 | none | trivial |
| 2 | Re-run F21 set at forced 1189 MHz (`sudo nvidia-smi -lgc 1189,1189`) | 1 | converts every 405 MHz ratio today into a full-clock number; validates/kills F20 | gates all magnitudes | needs root; HW slowdown 82 C still protects | trivial |
| 3 | q4_1/iq2_s n=1,2 to confirm the 118.73 us anchor returns | 0.5 | whether the SW cap was the *only* thing suppressing today's absolutes | measurement integrity | none | trivial |
| 4 | GDN sweep at 1189 MHz | 1 | the 0.40 ms/token figure in F15 | confirms DRAFTN attribution | none | trivial |
| 5 | Per-op profile of a **2-column** decode | ~30 | the **~18 ms unattributed** in F18 — the largest single hole in the model | unknown, possibly large | none | medium |

Tier 0 must complete before anything below is worth running.

## Tier 1 — the measured lead, end to end (~5 min)

| # | test | GPU s | decides | upside | risk | diff |
|---|---|---|---|---|---|---|
| 6 | end-to-end no-spec decode, Pascal build vs baseline | ~120 | F24's +13% projection | **+13% t/s** | none | trivial |
| 7 | per-type Pascal `rows_per_block` at ncols_dst==2 (add `type` param) | ~60 | whether the **champion** path can gain | +2.6% t/s | iq2_s regression if done blind | medium |
| 8 | requant A/B (`~/queued_ab.sh`, already armed) | ~120 | the ps/weight -> whole-model model; predict ~11.9 vs 10.62 | +12% | none | done, just run |
| 9 | `-ot 'token_embd\.weight=CPU'` | ~60 | frees a measured 994.7 MiB at zero quality cost | enables bigger ctx/quant | none | trivial |

## Tier 2 — the DRAFTN question, now properly posed (~5 min)

| # | test | GPU s | decides | upside | risk | diff |
|---|---|---|---|---|---|---|
| 10 | DRAFTN=2 with the F22 rows_per_block=4 at ncols 3 | ~120 | whether flattening the 3rd-column step makes DRAFTN=2 win | 1+a+a^2 = 2.18 vs 1.69 tokens/round | acceptance may fall with depth | medium |
| 11 | validate the `LUCE_GRAPH_STATS` counters on the real model, then re-run the graphs A/B | ~120 | whether the -1.5% is "captured and slower" or "never captured" | closes item 8 either way | none | low |

## Tier 3 — long-context, where the 20 t/s target actually lives (~30 min)

| # | test | GPU s | decides | upside | risk | diff |
|---|---|---|---|---|---|---|
| 12 | decode at genuinely resident 128K | ~1300 | **the number the target is defined against** (~14.4 t/s projected, never measured) | defines the ceiling | long prefill = long thermal load | trivial |
| 13 | windowed/selective KV on the **17** attention layers (F14) | - | whether the 2.41 GB / 14 ms KV term can be cut | large at 128K | changes model semantics | high |

## Do NOT do yet
- **The 54 GB BF16 pull.** Still gated on Tier 1 #8 landing near +12%. Nothing today changed that.
- Any broad A/B sweep. Tier 0 #1-#4 cost 3 s combined and re-price everything below them.
- Forcing clocks unattended. #2 is a supervised, one-shot, root-required step.

## Honest position on the 20 t/s target
Not in hand, and today moved the estimate **down**, not up:
- Resident 128K is ~14.4 t/s PROJECTED (2.41 GB KV, F13), against 16.28 measured empty-cache.
- The best measured lead (F21) lands on the **no-spec** path, not the champion path (F24).
- The champion's plausible gains today total ~+2.6% (F22), not +20%.
- Two of my own hypotheses died on measurement (ALU non-amortisation; launch overhead), and the
  fp16-kernel avenue failed its first probe last night.
What did improve is the *quality* of the map: the DRAFTN ceiling is now a located, one-kernel
implementation cost rather than an architectural verdict, and there is a real, correctness-verified
25.7% kernel win on the single-column path awaiting an end-to-end number.

---

# RUN 10 — 2026-09-01 (late) — Xid 79 recurrence + the n-column cliff relocated

## F25 — MEASURED (from the crash dump): the third Xid 79, and it happened at IDLE

At 17:47:38, ~12.5 h into the boot, the P100 fell off the bus with **nothing running**. My last
`nvidia-smi` poll closed at 17:46:52 — 46 s earlier — and no compute process had run for hours.
Kernel sequence: correctable Physical Layer errors (`RxErr (First)` / `Rollover` / `Timeout`),
then `Xid 79`, then `Uncorrectable (Fatal) ... [5] SDES (First)` + `CmpltTO` + `UnsupReq`, then
`Xid 154: GPU recovery action ... 0x1 (GPU Reset Required)`, then
`AER: Root Port link has been reset (0)` -> `AER: device recovery failed`.

Crash dump preserved: (not included) (471 KB), captured before recovery.

**Load and temperature are excluded as triggers for this event.** That is new. The two prior
events were at least adjacent to load; this one was not. It is the cleanest evidence yet that the
dropout is a physical/contact fault, not a thermal one.

**Recovery: PCI hot-remove + rescan FAILED.** After the rescan the device was gone from `lspci`
entirely and `/sys/bus/pci/devices/0000:02:00.0/` no longer existed. Only a POST brought it back.
Record this: for Xid 79 on this box, hot-remove/rescan is not a recovery path, it is a way to lose
the device faster. Reboot is the only route.

From the dump, root port `0000:00:02.0` (Port #3, the P100's port):
```
LnkCap:  Speed 8GT/s, Width x16      <- hardware is Gen3 x16 capable
LnkCap2: Supported Link Speeds: 2.5GT/s   <- but the port is PINNED TO GEN1
LnkCtl2: Target Link Speed: 2.5GT/s
UESta:   SDES+ CmpltTO+ UnsupReq+ ; AERCap First Error Pointer: 05  (05 = SDES)
```
Root port 1, for contrast, advertises `Supported Link Speeds: 2.5-8GT/s`. After the reboot
`nvidia-smi` reports `pcie.link.gen.current=1, width=16`, confirming it live.

**So the "pin slot 2 to Gen1" mitigation is already applied, and it did not prevent this.** That
candidate should move from "live, untried" to **TRIED AND FAILED**. Gen1 is the most
electrically forgiving speed available; dropping at Gen1 argues the contact fault is gross, not
marginal signal integrity. It also means the Gen1 pinning is currently costing link bandwidth
(model load / prefill H2D) for no demonstrated reliability benefit.

**Action, physical, costs no GPU time:** when the case is next open for the shroud, also
(a) reseat the card and support its far end, and (b) revert the slot to Auto/Gen3 in BIOS. Memory
records that BIOS writes over SSH fail (`hp-bioscfg` 0x4), so these are one trip together.
Note this cuts against the memory line "no brace needed while sideways" -- 12.5 h clean followed
by an idle drop is evidence against that, and it should be treated as superseded.

## F26 — MEASURED, and it SUPERSEDES F17: the marginal-column cliff is at n=4, not n=3

Re-analysis of `~/kernelbench_raw.txt`, a full pre-existing sweep on the box. **No new GPU time.**
`m=4096, k=14336`, us/run, and the marginal cost of each added column:

| format | n=1 | n=2 | n=3 | n=4 | n=5 | n1->2 | n2->3 | **n3->4** | n4->5 |
|---|---|---|---|---|---|---|---|---|---|
| q4_1   | 115.39 | 130.43 | 176.05 |  440.85 | 1161.52 | +15.0 |  +45.6 |  **+264.8** | +720.7 |
| iq4_xs | 134.51 | 150.38 | 223.49 |  470.12 |  569.74 | +15.9 |  +73.1 |  **+246.6** |  +99.6 |
| iq2_s  | 179.27 | 199.50 | 254.38 |  530.96 |  666.26 | +20.2 |  +54.9 |  **+276.6** | +135.3 |
| q6_K   | 264.01 | 274.71 | 404.20 | 1552.24 | 1786.10 | +10.7 | +129.5 | **+1148.0** | +233.9 |

**F17 said the step was at the 3rd column and was a launch-geometry change. The mechanism was
wrong.** Reading the tables directly: `calc_nwarps` returns 4 for ncols_dst 1..4 (dropping to 2
only at 5..8), and `calc_rows_per_block` returns 2 for ncols_dst 2..8 under GENERIC and 2 for
1..8 under the new PASCAL table. **Geometry is identical at n=2, n=3 and n=4.** So neither the
n=3 step nor the n=4 cliff is a launch-geometry effect. F17's conclusion that the DRAFTN ceiling
is "a one-kernel implementation cost" survives; its stated mechanism does not.

Dispatch is also excluded: `MMVQ_MAX_BATCH_SIZE == 8` and `use_mul_mat_vec_q` is tested before
`use_mul_mat_q` in `ggml_cuda_mul_mat`, so n=4 is still mmvq, not MMQ or cuBLAS.

**The consequence for the champion path is favourable.** DRAFTN=1 runs the target at **n=2**,
which is the *cheapest* marginal column in the whole curve (+11-15%). DRAFTN=2 would run at n=3,
costing +48.6% of matmul time on iq4_xs for one extra draft token -- moderate, not prohibitive.
DRAFTN=3 (n=4) is off the table. So the real question for DRAFTN=2 is whether 1+a+a^2 tokens per
round beats 1+a against a ~1.3x more expensive target forward, which is now an arithmetic
question against measured acceptance, not an architectural one.

## F27 — MEASURED: the sm_60 CUDA-graph disable is a soft heuristic, and the log proves it fires

`ggml-cuda.cu:4765` -- `if (!force_graphs && cc < GGML_CUDA_CC_AMPERE)` sets
`disable_due_to_gpu_arch` and logs at DEBUG. The reference sweep log contains
`ggml_cuda_graph_set_enabled: disabling CUDA graphs due to GPU architecture`, so **debug logging
was live in that run and the path is confirmed taken.** This retires the RUN 9 defect where a zero
message count proved nothing -- the control was simply the wrong log.

Combined with the three A/B pairs (-1.3 / -1.1 / -1.8%), the position on item 8 is now:
graphs are disabled on Pascal by an arch heuristic, not a capability gate; `LUCE_CUDA_GRAPHS=1`
lifts it; and when lifted they are consistently, slightly **negative** on sm_60. Treat CUDA graphs
as closed-unfavourable rather than unresolved. The `LUCE_GRAPH_STATS` counters remain uncertified
and are not needed for this conclusion.

## Instrument fix — thermal guard v2
`p100_guard.sh` (v1) killed only `llama-bench` and `llama-server`. **Kernel benchmarks run via
`test-backend-ops` were therefore never actually guarded** -- the guard process ran, logged
temperatures, and killed nothing. `p100_guard2.sh` adds `test-backend-ops` to the kill list, polls
at 0.5 s instead of 2 s, and treats a lost GPU reading as a kill condition rather than a reason to
exit the guard loop.

## F28 — MEASURED, and it RETRACTS F11: the P100 is NOT capped at 405 MHz

F11 claimed the card was pinned to 405 MHz by `SW Thermal Slowdown` under sustained load, on the
strength of 39/39 samples over 4 s. **That was wrong.** A 6 s load with 100 ms sampling from a
62 C start:

```
t=0.0s   62C   405 MHz   33 W    (idle)
t=0.3s   62C  1189 MHz   39 W    <- ramps in 300 ms
t=1.0s   63C  1139 MHz  153 W
t=2-6s   65-67C  1101-1126 MHz  ~147 W   SW Thermal Slowdown: Not Active throughout
t=6.0s   67C  1189 MHz   43 W    (load ends)
```

**405 MHz is the P100's normal IDLE clock.** Every burst I ran in RUN 7-9 was sub-second, which is
shorter than the ~300 ms ramp plus the sampling latency, so I was reading the idle clock and
calling it a thermal cap. `SW Thermal Slowdown` never asserted. Temperature rose only 5 C in 6 s.

**Consequences, and they are large:**
- The "absolutes are ~4.3x pessimistic, ratios valid" caveat in F11/F20 is **withdrawn**. The
  absolutes were not uniformly scaled by a fixed factor; they were taken at a clock that was
  ramping *during* the measurement, so both the absolutes and the ratios in RUN 7-9 are suspect.
- **F20's argument against F17 collapses with it** -- it was built on the 405 MHz premise.
- Sub-second microbenchmarks on this card are structurally invalid. Any burst must run >=3 s or it
  measures the ramp, not the kernel. This is now a protocol rule.
- Measured at full clock, q4_1 n=1 on the Pascal build is **98.23 us**, against the 115.39 us
  reference-sweep baseline -- a real gain, but nearer -15% than the -25.7% claimed in F21, and F21
  itself was measured under the broken protocol.

**F21/F22/F24 are therefore marked PROVISIONAL pending re-measurement under the >=3 s protocol.**
The Pascal `rows_per_block` change is still correctness-verified (0 FAIL) and still appears to be a
real gain; the *magnitude* is not established.

## F26 — DOWNGRADED to CONFOUNDED: the n=4 "cliff" is not a clean measurement

`test-backend-ops` iterates `for (bs : {1,2,3,4,5,8,512}) for (type_a : all_types)`, so **n is the
outer loop and increases monotonically with time-into-run.** The reference sweep's guard log shows
temperature climbing 55 C -> 79 C across that run. So in the F26 table, n is confounded with both
temperature and clock state.

It is worse than a simple confound: `temps_kbench.log` ends 17:25 while `kernelbench_raw.txt` was
written 17:32. Guard v1 hit its 78 C limit and exited, but it could only `pkill llama-bench` and
`llama-server`, **not `test-backend-ops`** -- so the final ~7 minutes of that sweep, which is
exactly where n=4/5/8 live, ran unguarded above 78 C.

The n=1 -> n=2 marginal (+11-15%) is early and cool and probably survives. **The n=4 cliff may be
substantially or entirely thermal.** F26's structural conclusions stand on source reading, not on
the timings, and those are unaffected:
- `calc_nwarps` = 4 for ncols 1..4; `calc_rows_per_block` = 2 for ncols 2..8. Geometry is
  identical at n=2, 3 and 4, so no launch-geometry step exists at n=3 **or** n=4.
- `MMVQ_MAX_BATCH_SIZE == 8` and `use_mul_mat_vec_q` precedes `use_mul_mat_q`, so n=4 is still
  mmvq. No dispatch change.
- **ptxas -v on every instantiation: 0 bytes spill stores, 0 bytes spill loads, all types, all
  ncols 1-8.** The register-spill hypothesis is FALSIFIED off-GPU.
- Occupancy moves little across the alleged cliff (iq4_xs 3 blocks/SM at both n=3 and n=4;
  q6_K 5 at both), so occupancy does not explain a 2-3.8x step either.

Register counts do climb steeply (iq4_xs 73/106/133/156 for n=1/2/3/4; iq3_s reaches 222 at n=8)
and a local-memory stack frame appears (8*ncols bytes = exactly `tmp[ncols][rows_per_block]`), but
with zero spills and near-flat occupancy neither is yet tied to a timing.

**Superseded: F17's "~200 us/col at the 3rd column" and the DRAFTN=2 arithmetic built on it.**
A clock-controlled per-n A/B is running now to replace the whole table.

## F29 — STRONGLY INFERRED (source): MTP pays a full draft-context re-eval every round

The ~18 ms hole in F18 was framed as "the marginal cost of a second column". Reading the spec
path, that framing is too narrow: **one MTP round is not one graph, it is up to three.**

- `server-context.cpp:344` `update_batch()` -- the verify batch is `sampled` + every drafted
  token, so DRAFTN=1 gives a 2-column target forward. (Confirms F24's premise.)
- `speculative.cpp:722` `llama_decode(ctx_dft, batch)` -- the draft head runs in its **own
  context**, so it is a separate graph, not extra columns on the target graph.
- `server-context.cpp:3148` carries an explicit upstream table and TODO
  (`TAG_SPEC_AVOID_DRAFT_REEVAL`):

  | spec type | need re-eval |
  | --- | --- |
  | draft model | no |
  | **MTP (std)** | **yes** |
  | MTP Gemma4 | no (KV cache is shared) |
  | Eagle3 | yes |

  With the comment: *"for now, always re-evaluate for simplicity"*.

So every MTP round additionally **re-evaluates the verify batch through the draft context**, plus
a `ckpt.load_dft()` state restore and a `common_context_seq_rm()`. None of that is a "second
column" cost and none of it was in the F18 accounting, which only ever compared 1-column against
2-column *target* matmul time. That is the most likely home for a large share of the missing
~18 ms.

**This is an implementation cost, not an architectural one** -- it is a known upstream TODO, and
the Gemma4 MTP variant already avoids it by sharing the KV cache. That makes it the first genuinely
promising lead for the champion path since the DRAFTN work, and it is worth far more than the
~2.6% the F22 per-type table was projected to give.

**Not yet measured.** `LUCE_OP_PROFILE=1` (commit `40d5831e6`) exists to measure exactly this: it
buckets MUL_MAT by src0 type and `src1->ne[1]`, so target columns (n=2..4) and draft-context work
(n=1, different shapes, and a much higher call count per round) separate in the table. The
instrument is built and deployed to `~/luceprof`; the run is staged as `~/opprofile.sh`.

---

# THE QUEUE — rev 2 (2026-09-01 night), after the F28 protocol break

Rev 1's ordering assumed the 405 MHz cap was real and the n=4 cliff was measured. Both are gone,
so the queue is re-cut. **Protocol rule now binding on every entry: no burst under 3 s, log
`clocks.sm` during the run, and use `p100_guard2.sh` (v1 cannot kill `test-backend-ops`).**
Idle floor is 68 C with no shroud, so cooldown gates must be >= 70 C or they deadlock.

## Tier 0 — running or ready now, no shroud needed

| # | test | state | decides |
|---|---|---|---|
| 1 | iq4_xs n=1..4, Pascal vs baseline, one 1.5-3 s burst per point from a fixed thermal start | **running** (`~/ncols_clean.sh`) | replaces the whole F26 table; n=1 done: **136.59 -> 111.90 us, -18.1%** | 
| 2 | `LUCE_OP_PROFILE=1` on the champion config | **staged** (`~/opprofile.sh`, lib in `~/luceprof`) | the ~18 ms (F18) and whether F29's draft-context re-eval is the bulk of it |
| 3 | re-run the fp16 half2 probe with the fixed warmup | **built** (`~/p100_fp16_probe`, commit `c0a511f0f`) | whether "half2 loses at 0.61x" was a clock artefact |

## Tier 1 — needs a longer run, defer until cooling is fitted
- End-to-end no-spec A/B of the Pascal build. F24's +13% is a projection off invalid
  microbenchmarks; it needs re-deriving from the Tier 0 #1 numbers before it is worth running.
- Per-type Pascal `rows_per_block` at ncols_dst==2 -- but **only if Tier 0 #1 shows iq4_xs gains at
  n=2**. F22's iq2_s regression was measured under the broken protocol and may not be real.
- `-ot 'token_embd\.weight=CPU'` (994.7 MiB, zero quality cost).
- Requant A/B (`~/queued_ab.sh`).

## Tier 2 — the real target
- Decode at genuinely resident 128K. Never measured. The 20 t/s target is defined against this.
- Windowed/selective KV on the 17 attention layers (F14).

## Do NOT do
- The 54 GB BF16 pull. Unchanged.
- Any conclusion drawn from a sub-second burst. RUN 7-9 are provisional; F11 and the fp16 probe
  verdict are retracted outright.
- Hot-remove/rescan as an Xid 79 recovery (F25) -- it destroys the device node.

## Hardware, and it now outranks the shroud
The third Xid 79 was **at idle**, and the slot is **already pinned to Gen1**. Cooling addresses the
68 C idle and nothing about the dropouts. The highest-value physical action is a **reseat plus a
support under the far end of the card**, with the slot reverted to Auto/Gen3 in BIOS on the same
trip (BIOS writes over SSH fail 0x4).

## Honest position on 20 t/s
Unchanged in substance and now on firmer ground in method. The best measured lead (-18.1% on
iq4_xs n=1) sits on the **no-spec** path. The champion runs at n=2, where nothing has yet been
shown to help. F29 is the first lead that targets the champion path directly, and it is an
upstream-acknowledged implementation cost rather than an architectural limit -- but it is
unmeasured. Resident 128K has still never been measured at all.

## F30 — MEASURED at full clock: the clean iq4_xs points, and what they say about F26

Protocol: one 1.5-3 s burst per point (>= 5x the 300 ms ramp), each from a controlled thermal
start, `p100_guard2.sh` armed. Champion format.

| point | build | start->end | us/run |
|---|---|---|---|
| iq4_xs n=1 | luceops2 (baseline) | 68 -> 73 C | 136.59 |
| iq4_xs n=1 | lucepascal          | 70 -> 74 C | **111.90** |
| iq4_xs n=2 | luceops2 (baseline) | 72 -> 76 C | 151.16 |

**Pascal `rows_per_block` at n=1 on the champion format: -18.1%.** The Pascal point started 2 C
*hotter* than its baseline, so if anything this understates the gain. This supersedes the -25.7%
q4_1 figure from F21, which was measured under the broken sub-second protocol.

**A useful cross-check on F26.** The clean full-clock numbers agree closely with the reference
sweep's early points -- n=1 136.59 vs 134.51, n=2 151.16 vs 150.38, both within ~1.5%. So the
reference sweep was *not* globally corrupted: its n=1 and n=2 points were taken while the card was
still cool and they are trustworthy. That localises the confound to the **late** points (n=4/5/8),
which is exactly where the alleged cliff sits and exactly where the guard had already exited.
The n=1 -> n=2 marginal of **+10.7%** (clean) vs +11.8% (reference) is therefore solid, and it is
the one number the champion path actually depends on.

**Still unmeasured, and it is the question that matters for the champion:** iq4_xs n=2 on the
Pascal build. Expected to be ~unchanged, since `calc_rows_per_block` returns 2 at ncols_dst==2
under both tables -- but expectation is not measurement, and this is the point that decides
whether F21 touches the champion at all. It is one 3 s burst.

**Thermal budget is the binding constraint, not GPU time.** With no shroud the heatsink soaks:
the idle floor rose from 68 C after the reboot to **76 C** after ~6 short bursts, at which point
the 72 C cooldown gate could no longer be met and the sweep deadlocked. Every burst costs ~4 C and
the card sheds it very slowly at ~40 W idle with no airflow. This, not the guard limit, is what
ends a session.

## F31 — STRONGLY INFERRED: the output head is run 3x per MTP round, and it is q5_K

Tensor-level read of the **champion** GGUF (`Qwen3.8-27B-UD-IQ4_XS.gguf`, 866 tensors, 27.321 G):

```
token_embd.weight              q3_K   1,271,398,400   [5120, 248320]
output.weight                  q5_K   1,271,398,400   [5120, 248320]
blk.64.nextn.eh_proj.weight    q6_K      52,428,800   [10240, 5120]
blk.64.ffn_{up,gate,down}      q6_K   3 x 89,128,960
blk.64.attn_output.weight      q6_K      31,457,280
```

Two things follow.

**1. "The champion is IQ4_XS" is only half true.** The UD mix is iq4_xs 49.7%, iq3_s 13.8%,
q5_K 11.2%, q4_K 8.2%, q3_K 6.8%, iq3_xxs 5.2%, iq2_s 2.3%, q6_K 1.6%. So the single-format
iq4_xs microbenchmark represents about half the model's weight-time, and the K-quants -- which
are the slow formats on sm_60 -- carry much of the rest. **Any whole-model projection built from
the iq4_xs number alone is wrong**, including F24's. This needs a weighted model.

**2. `blk.64` is the MTP head, and drafting runs the full output head.** The `nextn.*` tensors
(`eh_proj`, `enorm`, `hnorm`, `shared_head_norm`) are all on blk.64, and `shared_head_norm`
indicates the draft shares `output.weight`. Per DRAFTN=1 round the source path is:
- target verify forward, **n=2** -> `output.weight` on 2 columns;
- `speculative.cpp:722` draft decode with `logits=true` -> `output.weight` on **1 more column**;
- `speculative.cpp:603` catch-up decode with `logits=0` -> MTP head only, **no** output head.

So **three output-head columns per round against one for no-spec.** At 1.271 G params and a
q5_K cost between the measured q4_K and q6_K rates (q6_K n=1 was 264 us / 58.7 M = 4.50 ps/weight),
one column is order **4-6 ms**. Two extra columns is order **8-12 ms** -- which is the right size
to be the bulk of F18's unattributed ~18 ms, alongside the MTP head's own ~400 M of q6_K.

**This reframes the catch-up decode from F29.** The catch-up passes `logits=0`, so it does *not*
run the output head and is much cheaper than F29 implied. **F29's magnitude claim is softened:**
the mechanism is real and confirmed twice in source, but the expensive part of an MTP round is the
**output head being evaluated three times**, not the catch-up decode.

**Falsifying measurement:** `LUCE_OP_PROFILE` buckets by type, so `MUL_MAT[q5_K,n=1]` and
`MUL_MAT[q5_K,n=2]` will appear as separate rows with their own call counts. If output-head time
is not roughly 3 columns' worth per round, this is wrong.

**If confirmed, the lever is cheap:** `output.weight` at q5_K is 11.2% of the model and is
evaluated 3x per round. Requantising just that tensor to iq4_xs or q4_1 is a surgical change of
the kind already done for Ridge, and it would cut the single largest per-round term. Quality risk
on the output head is real and would need an eval, so this is a measure-first item.

**Separately, a free one:** `token_embd.weight` is q3_K, 1.271 G params (~546 MiB), and is only
ever a GET_ROWS lookup -- never a matmul. `-ot 'token_embd\.weight=CPU'` therefore costs no
compute time at all and buys context headroom on a card sitting at 15723/16384 MiB.

## F32 — MEASURED: the no-load thermal equilibrium is the real session limit

Tracked with zero GPU load after the sweep stopped: 76 C -> 76 -> 77 C, **rising**, at ~40 W idle.
The card does not shed heat-soak without airflow; it re-approaches the ~80 C equilibrium seen at
the end of the previous session.

Session thermal arc, all at idle unless noted: **54 C** right after the reboot -> 61 -> 68
(equilibrium reached) -> repeated 1.5-3 s bursts each adding ~4 C -> **77 C and climbing**, at
which point a 72 C cooldown gate is unsatisfiable and the sweep deadlocks.

**Practical rule:** the budget is not GPU-seconds, it is roughly **6-8 short bursts per boot**
from a cold start. Plan any experiment set to fit that, ordered so the most decisive point runs
first. The reboot itself is what buys a cold card (54 C), which is worth knowing given the Xid 79
recovery requires one anyway.

With the guard at 78 C and the card idling at 77 C, no further GPU work is possible this session.
The staged `~/opprofile.sh` run (F31's falsifying measurement) needs a cold start and did not run.

---

# RUN 11 — 2026-09-01 (evening) — the fan changes the regime

## F33 — MEASURED: the fan is transformative for IDLE, which is what gates burst work

Reboot 19:41:19 with the fan refitted. P100 enumerated on the first POST.

| | no fan (RUN 10) | **with fan** |
|---|---|---|
| idle temperature | 68 C fresh, drifting to **79 C** heat-soaked | **32-36 C** |
| idle power | ~40 W | **25.7 W** |
| headroom to the 78 C guard | 10 C when soaked | **~44 C** |

Idle power fell 40 -> 25.7 W purely from the lower die temperature (leakage current is strongly
temperature-dependent), which is an independent confirmation that the card really is cooler rather
than the sensor reading differently.

**This validates the user's proposal directly.** F32 identified *recovery between bursts*, not the
bursts themselves, as the binding constraint. A fan cannot fix sustained load -- long prefill still
died at ~36-45 s at both 150 W and 125 W -- but it fixes recovery completely, and recovery was the
thing that ended RUN 10. At ~0.8 C/s under ~147 W the usable burst goes from ~12 s to roughly
**45-55 s**, which covers the entire diagnostic queue including the op profile.

Still gated on the shroud: genuinely sustained work, i.e. the resident-128K decode that the 20 t/s
target is actually defined against.

## Instrument defects found and fixed (both mine, both in `opprofile.sh`)

1. **`curl /health` exits 0 on HTTP 503.** llama-server answers 503 `{"status":"loading model"}`
   until the weights are resident, so "the port is open" was being read as "the model is loaded".
   The first run therefore fired its completion request 5 s in, at 259 MiB of a 14 GB model, and
   reported a clean `OPPROF_COMPLETE` having measured nothing. Now checks `-w '%{http_code}'` == 200.
2. **Peak-temp extraction read the wrong field**, concatenating the timestamp into the temperature
   ("peak 19424634C"). The guard log is `HH:MM:SS <temp>, <power>, <clock>`, so it is field 2.
3. `pkill -f 'llama-server'` from an ssh one-liner **matches its own command line** and kills the
   session (exit 255). Use `'[l]lama-server'` -- the existing `p100_guard.sh` already documents
   this and I did not follow it.

A run that completes without error is not evidence it measured anything. Defect 1 produced a
success banner and an empty table.

## Blocker for the F31 requant experiment: disk, not thermals
`/` on the Z640 is **97% full, 2.6 GB free**. A requantised copy of the champion needs ~14 GB, and
there is no `llama-quantize` in `~/lucebin`. `~/models/qwen38-27b-ridge` holds 24 GB (Ridge-3.7bpw
plus the unmeasured Ridge-q41nonffn). The dev box has 102 GB free on `$DATA` and a gigabit
link, so the workable route is: pull the 14 GB model to the dev box, requant `output.weight` there
(`--tensor-type` + `COPY` are both present in `tools/quantize/quantize.cpp`), push it back --
**but the Z640 must free ~14 GB first.** Deleting anything under `~/models` is the user's call,
not mine.

## F34 — MEASURED: champion end-to-end, and the first real acceptance-rate number

Champion config (`run27b_mtp.sh` settings, `--spec-draft-n-max 3`), fan fitted, 36 C start,
Pascal-mmvq + profiler libs:

```
prompt eval  15 tokens   114.01 ms/tok    8.77 t/s
eval         96 tokens    78.42 ms/tok   12.75 t/s
draft acceptance rate = 0.45000  (54 accepted / 120 generated)
```

**12.75 t/s is NOT a clean throughput number** -- `LUCE_OP_PROFILE=1` was on, which records two
CUDA events per node and then does a host-side elapsed-time query per node per graph. On a
~1000-node graph that is thousands of extra API calls per token. Against the 14.23 t/s recorded in
the `run27b_mtp.sh` header, the profiler appears to cost ~10%. **Do not cite 12.75 as the
champion's rate**; re-measure with `PROF=0`.

**The acceptance rate is the genuinely new number, and it settles the DRAFTN question.**
With a = 0.45 pooled across draft positions, expected tokens per round:

| DRAFTN | target cols | tokens/round | vs DRAFTN=1 | matmul cost vs n=2 |
|---|---|---|---|---|
| 0 (no spec) | 1 | 1.00 | -31% | -10.7% |
| **1** | **2** | **1.45** | — | — |
| 2 | 3 | 1.65 | **+13.8%** | **+48.6%** |
| 3 | 4 | 1.74 | +20.0% | far worse |

(marginals from F30/F26: iq4_xs n=1->2 = +10.7% clean full-clock; n=2->3 = +48.6% from the
reference sweep's still-cool early points.)

**DRAFTN=2 buys 13.8% more tokens per round for ~48.6% more target matmul time. It loses.**
DRAFTN=1 buys 45% more tokens for 10.7% more time and wins comfortably. So **"DRAFTN=1 is
optimal" is re-confirmed -- but for a completely different reason than originally recorded.** It
is not "architectural, because the GDN scan is sequential" (F15/F16 killed that: the scan costs
~2% of the claimed figure). It is simply that the mmvq marginal-column curve is convex while the
acceptance geometric series is concave, and they cross between the 2nd and 3rd column.

**Caveat, stated plainly:** a = 0.45 is pooled over all draft positions, and first-position
acceptance is higher than third-position. A per-position breakdown would raise the DRAFTN=2
estimate somewhat. It would have to roughly triple the marginal benefit to change the verdict,
which is implausible, but the pooled figure is an approximation.

**Note the config tension:** `run27b_mtp.sh` -- the recorded champion -- passes
`--spec-draft-n-max 3`, i.e. it is *not* running DRAFTN=1. If the analysis above holds, the
champion is leaving throughput on the table by drafting deeper than optimal. **Testable in one
pulse:** same config with `--spec-draft-n-max 1`, PROF=0.

## F35 — MEASURED: the op profile, and it REFUTES F31

`LUCE_OP_PROFILE=1`, champion config, 144 graphs, 5896 ms accumulated. Top rows:

```
MUL_MAT[iq4_xs,n=4]   1646.19 ms  27.92%  calls=5908   0.27864 ms/call
MUL_MAT[iq3_s,n=4]     659.58 ms  11.19%  calls=1288   0.51210
MUL_MAT[q5_K,n=4]      541.31 ms   9.18%  calls=1427   0.37933
MUL_MAT[iq4_xs,n=11]   424.72 ms   7.20%  calls=211    2.01289
MUL_MAT[q4_K,n=4]      389.62 ms   6.61%  calls=1540   0.25300
MUL_MAT[q5_K,n=1]      259.68 ms   4.40%  calls=87     2.98482
MUL_MAT[iq3_xxs,n=4]   224.12 ms   3.80%  calls=476    0.47084
CPY                    150.67 ms   2.56%  calls=12048  0.01251
MUL_MAT[q6_K,n=1]      127.11 ms   2.16%  calls=498    0.25523
GATED_DELTA_NET         76.30 ms   1.29%  calls=1488   0.05128
FLASH_ATTN_EXT[kv=256]  40.16 ms   0.68%  calls=609    0.06594
```

**The single most important line is the `n=4`.** `--spec-draft-n-max 3` means 1 sampled + 3
drafted = **a 4-column target forward**. The champion has been running at n=4 the whole time --
the most expensive point on the marginal-column curve -- not at n=2 as F24/F29/F31 all assumed.

**Attribution by column count:**

| bucket | ms | share | what it is |
|---|---|---|---|
| MUL_MAT n=4 | ~3789 | **64%** | target verify forward |
| MUL_MAT n=11 | ~745 | 13% | prefill |
| MUL_MAT n=1 | ~387 | **6.6%** | **draft-side work** |
| MUL_MAT n=2 | ~110 | 1.9% | |
| non-MUL_MAT | ~865 | 15% | CPY, PAD, GET_ROWS, GDN, attn, ADD |

**F31 is refuted.** I predicted the output head running three times per round would be 8-12 ms and
"the bulk" of the missing time. Measured: one output-head column is **2.98 ms** (q5_K n=1,
87 calls), and *all* draft-side work totals **6.6%** of GPU time. The draft head is cheap. The
expense is the target forward at 4 columns.

**Two older claims now confirmed on the real model, not inferred:**
- `GATED_DELTA_NET` = **1.29%** (76.3 ms / 1488 calls). F15/F16 said the GDN scan was real but
  misattributed at ~2%; that is now measured end-to-end. The "20-25 ms per extra verify row from
  sequential deltanet" claim is dead.
- `FLASH_ATTN_EXT[kv=256]` = **0.68%**. Attention is negligible at short context.

**Instrument limitation:** bucketing by type alone cannot separate `output.weight` (q5_K,
248320 rows) from the q5_K layer tensors inside the n=4 target graph -- 1427 calls over 144 graphs
is ~10/graph, so it is dominated by layers. Adding `src0->ne[1]` to the key would isolate it.
Not needed for the conclusion above.

**Immediate consequence, now under test:** if the champion runs at n=4 and the marginal-column
curve is convex, then `--spec-draft-n-max 1` (2 columns) should beat it outright, exactly as
F34 predicted from a = 0.45. Running both at PROF=0 now.

## F36 — MEASURED: an idle llama-server pins the SM clock at 1189 MHz, and that breaks "keep it resident"

| state | temp | power | clocks.sm | util | VRAM |
|---|---|---|---|---|---|
| no process | 32-36 C | 25.7 W | 405 MHz | 0% | 0 MiB |
| **server loaded, idle** | **77 C** | **48.5 W** | **1189 MHz** | **0%** | 15723 MiB |
| server killed, +20 s | 77 C | 39.9 W | 405 MHz | 0% | 0 MiB |

**A live CUDA context holds the SM clock at its boost value even at 0% utilisation.** That is the
mechanism, not HBM refresh: killing the process drops the clock to 405 MHz instantly while the
temperature (thermal mass) lags. So an idle server burns ~23 W more than an idle card and parks it
at 77 C -- above the 78 C guard's useful range.

**This retracts the "keep llama-server resident and pulse at it" advice I gave earlier today.**
The reasoning was that model load is disk-bound and thermally cheap, which is true, but I did not
account for the idle context pinning the clock. Holding the model resident between pulses is
*worse* than reloading, because the card never cools while the server lives.

It also caused a second cooldown deadlock, the same class of bug as RUN 10's: a `COOL=55` gate is
unreachable when the resident-idle floor is 77 C. **Any cooldown gate must be set against the idle
floor of the state it will actually wait in** -- 32-36 C with no process, ~77 C with a server up.

**Correct pattern: load -> pulse immediately -> stop.** Measured on the first run: 36 C start ->
46 C loaded -> 49 C after a 96-token generation, and a separate 64-token pulse peaked at 58 C.
Loading is thermally cheap and the pulse right after it is the coolest one available. Repeated
pulses against a live server are what soak the card.

## F37 — MEASURED: correcting F33's burst estimate; the fan's soaked equilibrium is ~63-65 C, not 32

F33 quoted the fan's idle as 32-36 C and projected ~45-55 s usable bursts. **That was the
cold-boot figure and it is not the operating figure.** After several hours of work the case and
heatsink soak, and the no-load equilibrium settles much higher:

| state | idle temp | idle power |
|---|---|---|
| fan, cold boot (19:41) | 32-36 C | 25.7 W |
| **fan, soaked (21:27)** | **63-65 C** | 34 W |
| no fan, soaked (RUN 10) | 77-80 C | 40-41 W |

Cooling from soaked is **~0.1-0.4 C/min and decelerating** -- 71 -> 65 C took an hour with no
process running. So the fan is worth a real ~15 C off the soaked equilibrium (79 -> 64), which is
what matters, but it does not restore cold-boot conditions within a session.

**Corrected cost model, from measurements rather than extrapolation:**

| step | cost |
|---|---|
| model load (14 GB, disk-bound ~70 MB/s, ~4-5 min) | **+8-10 C** |
| 64-token pulse | **+3-6 C** |
| 96-token generation | +3 C (46 -> 49 C) |

**The load, not the generation, is the expensive step thermally.** That is the opposite of what I
assumed when I recommended keeping the server resident, and it is *also* not a reason to keep it
resident, because F36 showed an idle server pins the clock at 1189 MHz and parks the card at 77 C.
The load is expensive once; a live idle server is expensive continuously.

**Working envelope:** gate at 67 C, guard at 81 C (SW slowdown is 82, shutdown 85). From 67:
load -> ~75, pulse -> ~79. That fits, with the caveat that any point peaking above ~80 should be
flagged as possibly throttled.

**Two cooldown deadlocks in one session, from the same mistake:** a gate set against a remembered
idle floor rather than the current one (72 C against a 77 C resident floor; 45 C then 64 C against
a 63-65 C soaked floor). The gate must be derived from a fresh measurement of the state the script
will actually wait in, every time.

## F38 — MEASURED, and this is the session's result: `--spec-draft-n-max 1` beats the champion by ~26%

Interleaved A/B (3, 1, 3, 1) so thermal drift cannot fake the difference. `PROF=0`, 64-token
pulses, gate 67 C, guard 81 C, load->pulse->stop per F36.

| n-max | target cols | t/s | acceptance | loaded->end |
|---|---|---|---|---|
| 3 (champion) | 4 | 12.438 | 0.415 | 66 -> 71 C |
| **1** | **2** | **15.690** | **0.703** | 68 -> 73 C |
| 3 (repeat) | 4 | 12.365 | 0.415 | 69 -> 74 C |
| **1 (repeat)** | **2** | **15.625** | **0.703** | 68 -> 74 C |

Means: n-max 3 = **12.40 t/s**, n-max 1 = **15.66 t/s**. **+26.3%.**

**Both arms replicate to within 0.6%**, and acceptance is bit-identical across repeats
(0.41463 / 0.70270) because the pulses are deterministic (temperature=0, same prompt). The n=1 point ran *hotter* than
either n=3 point, so if anything this understates the gain.

**Both halves of the prediction confirmed, and one correction to it.** F34 predicted from the
convex marginal-column curve that deeper drafting loses; F35 showed the champion really was
running its target at n=4. Both hold. But the acceptance figures show my a = 0.45 was the wrong
input: **first-position acceptance is 0.703**, and 0.415 is the pooled average dragged down by
the 2nd and 3rd draft positions. So deeper drafting is penalised twice over -- it pays sharply
more per target forward *and* the extra draft positions are much less likely to be accepted than
the first. I flagged the pooled figure as an approximation in F34; it was, and correcting it
strengthens the conclusion rather than weakening it.

**APPLIED** to `~/run27b_mtp.sh` (backup at `~/run27b_mtp.sh.bak`):
`--spec-draft-n-max 3` -> `--spec-draft-n-max ${NMAX:-1}`, so the default is now 1 and
`NMAX=3 bash run27b_mtp.sh` still reproduces the old behaviour for comparison.
This costs nothing, risks nothing, and changes no output (speculative decoding is exact).

**Where this leaves the numbers.** 15.69 t/s was measured at 68-73 C on a soaked card with the
Pascal-mmvq libs, against 12.4 for the same build at n-max 3. The historical champion figure is
14.23 t/s (and 16.67 at 8K in memory) measured under different thermal conditions, so
**15.69 is not directly comparable to those** -- the honest claim is the *relative* +26%, measured
back-to-back. A clean absolute needs a cold card.

**"DRAFTN=1 optimal" was in memory all along -- but the recorded reason was wrong and the
shipped config contradicted it.** The reason is not that the GDN scan is sequential (F35 measured
GDN at 1.29%). It is that the mmvq marginal-column cost is convex while the acceptance series is
concave, and they cross between the 2nd and 3rd column. Meanwhile `run27b_mtp.sh` shipped
`n-max 3`, so the machine was never actually running the configuration the memory recommended.

## F39 — MEASURED: the Pascal mmvq change does NOTHING at n=2, so it does not touch the champion

The queued priority experiment. iq4_xs, m=4096 k=14336, **n=2 only**, interleaved
baseline/Pascal/baseline/Pascal, 3 s bursts at full clock, gate 70 C, guard 81 C.

| build | run 1 | run 2 | mean |
|---|---|---|---|
| luceops2 (baseline) | 148.41 us | 150.11 us | **149.26 us** |
| lucepascal (Pascal table) | 149.55 us | 150.58 us | **150.07 us** |

**Delta = +0.54%, i.e. Pascal is fractionally *slower*, and that is inside the noise:** the
within-build spread is 1.15% (baseline) and 0.69% (Pascal), both larger than the between-build
difference. Temperatures crept 65 -> 72 C across the four runs, which explains the mild slowdown
of the later run in *each* arm; interleaving cancels it.

**Verdict: no effect at n=2. |delta| < 1%.**

This is exactly what the source predicts, and the prediction was made before the run.
`calc_rows_per_block` differs between the tables at **`case 1` only** -- PASCAL returns 2 where
GENERIC returns 1 -- and every ncols_dst from 2 to 8 returns 2 in both. `calc_nwarps` places
PASCAL in the same branch as GENERIC, so it is identical throughout. The two builds are
functionally the same kernel at n=2. The measurement confirms the code reading rather than
surprising it, and it pins the noise floor of this harness at ~1%.

**Consequence, stated plainly: the -18.1% win from F30 does not reach the champion.**
With `--spec-draft-n-max 1` now the default (F38), the champion's target forward runs at **n=2**,
where the Pascal change is worth nothing. The F21/F30 kernel win applies only to the **n=1**
no-spec path. F24's projected "+13% t/s on the champion" is therefore dead in every form: it was
built on invalid microbenchmarks (F28), on an iq4_xs-only weighting of a model that is 49.7%
iq4_xs (F31), and now on a column count the champion does not use.

**So the +26.3% in F38 came entirely from the n-max change, and none of it from the kernel work.**

**The next candidate for the champion is now well-posed:** a per-type Pascal `rows_per_block` at
`ncols_dst == 2`. F22 measured rpb 2 -> 4 at n=2 as q4_1 -7.6%, iq3_s -9.9%, iq2_s **+10.1%**
(a regression) -- but that was under the broken sub-second protocol (F28), and **iq4_xs at n=2 was
never measured at all**. iq4_xs is 49.7% of the champion and n=2 is now its operating point, so
this is the one remaining kernel lever that lands where it matters. It needs a `type` parameter
threaded into `calc_rows_per_block`, then the same interleaved 3 s protocol used here.
Anything above ~1% would be real against this harness's measured noise floor.

## F40 — MEASURED: rows_per_block 4 at ncols_dst==2 gives iq4_xs -5.7%, on the champion's operating point

The lever F39 identified. Both arms built from **identical source**, differing only in the
constant `LUCE_PASCAL_RPB_N2` (2 vs 4), so no other build variation can contaminate the result.
`ggml_type` was threaded into `calc_rows_per_block` to make a per-type table expressible
(commit `481dc7183`). iq4_xs, m=4096 k=14336, n=2, interleaved, 3 s bursts, gate 70 C, guard 81 C.

**Correctness first:** `test-backend-ops test -o MUL_MAT -p type_a=iq4_xs` on the rpb=4 arm --
**FAIL=0, OK=13**. rpb=4 doubles the shared-memory footprint
(`tmp_shared[nwarps-1][ncols][rpb][warp]` = 3072 B vs 1536 B), so this was verified before timing.

| arm | pass 1 | pass 2 | mean |
|---|---|---|---|
| rpb=2 (control) | 149.82 us | 151.85 us | **150.84 us** |
| **rpb=4** | 141.81 us | 142.67 us | **142.24 us** |

**-5.7%** (pass1 -5.3%, pass2 -6.0%), against the ~1% noise floor F39 established on this exact
harness. The rpb=4 runs started at equal-or-higher temperature than their controls, and the
control itself drifted 149.82 -> 151.85 across passes, which is the thermal creep the interleaving
is there to cancel.

**Why this one matters when F39's did not.** With `--spec-draft-n-max 1` (F38) the champion's
target forward runs at **n=2**, and **iq4_xs is 49.7% of the champion's weights** (F31). This is
the first kernel change that lands on the path the champion actually executes.

**Expected end-to-end gain is modest, and I want to be honest about the size.** From the op
profile, iq4_xs MUL_MAT was ~28% of total GPU time. A 5.7% cut on ~28% is **~1.6% end-to-end** --
real, free, and correctness-clean, but not a headline. It only becomes interesting if the same
change helps the other quant types: all MUL_MAT was ~64% of GPU time, so a uniform 5.7% would be
~3.6%. **iq3_s (13.8%), q5_K (11.2%) and q4_K (8.2%) are being measured now.**

**Not shipped beyond iq4_xs.** The old rpb=4 figures for other types (q4_1 -7.6%, iq3_s -9.9%,
iq2_s **+10.1% regression**) come from the invalid sub-second protocol (F28) and cannot be
trusted in either direction. Types stay at 2 until measured under this protocol -- an iq2_s-style
regression on an unmeasured type would silently eat the iq4_xs gain.

---

# PROVENANCE — where the "16-17 t/s champion" figures actually come from (2026-09-01 audit)

Asked to pin the exact commit behind a "16-17 t/s at ~126K context" champion. **The premise needs
correcting first: there is no such single result.** There are two different records, on two
different models, and neither is 126K of *resident* context.

## Record A — 16.28 t/s at n_ctx 131072 (this is the "128K" figure)
Primary evidence: `<user>@<host>:~/longctx_server.log`, mtime **2026-08-31 06:23:42**.
```
prompt eval time = 4791.58 ms /  407 tokens ( 11.77 ms per token,  84.94 tokens per second)
       eval time =  982.85 ms /   16 tokens ( 61.43 ms per token,  16.28 tokens per second)
draft acceptance rate = 1.00000 (8 accepted / 8 generated)
```
Config, read from the server's own startup lines (not reconstructed):
```
model      $HOME/models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf   <-- RIDGE, NOT IQ4_XS
n_ctx      131072            (slot: "new slot, n_ctx = 131072")
MTP        n_max=1, n_min=0, p_min=0.00, n_embd=5120, backend_sampling=1      <-- DRAFTN = 1
draft KV   cache_k=turbo4, cache_v=turbo4
window     "MTP KV window = 1024 tokens"                                       <-- LUCE_MTP_WINDOW=1024
target KV  --cache-type-k/v turbo4, but the log warns:
           "auto-asymmetric: GQA ratio 6:1 (n_head=24, n_head_kv=4) -- upgrading K from turbo4
            to q8_0 to prevent quality degradation"                            <-- K was q8_0, NOT turbo4
-b/-ub     "DEPRECATED: argument '-b' specified multiple times" => 512 from longctx_test.sh
           overridden by EXTRA "-b 256 -ub 256"
```
Driver: `~/longctx_test.sh` with `MODEL=Ridge CTX=131072 CTXCP=1 BIN=$HOME/lucewin
EXTRA="-b 256 -ub 256 -ctkd turbo4 -ctvd turbo4" LUCE_MTP_WINDOW=1024 DRAFTN=1`.
The env-var values are confirmed by the log; the exact shell line is **not** recoverable --
`~/.bash_history` is only 54 lines and does not contain it.

**Critical caveat: this is 128K ALLOCATED, not 128K RESIDENT.** The prompt was 407 tokens and the
decode sample was 16 tokens. It measures decode with an essentially empty KV cache in a
131072-token allocation. **Decode at genuinely resident ~128K has still never been measured**
(projected ~14.4 t/s, F13). So "16-17 t/s at ~126K context" overstates what was established.

## Record B — 16.67 t/s at 8K on UD-IQ4_XS
A different run, different model, different context: `HISTORY_PAGE.md:503` -- "IQ4_XS @8K:
**16.67 t/s** at depth 1 (77.5% acceptance) vs 14.14 at depth 3", plus 16.23 t/s at 16K and
17.11 t/s at 16K on a copy task (100% acceptance). These are the IQ4_XS numbers; the 128K number
is Ridge. Conflating them produces the non-existent "16-17 t/s at 126K on the champion".

## The commit: BEST EVIDENCE e56e140b8, but NOT PROVABLE
Binary used: **`~/lucewin/llama-server`**, mtime 2026-08-31 05:38:32 -- 45 minutes before the run,
and the only build in existence at 06:23 that contains the feature the log used
(`lucewin2` 15:33 and `lucewin3` 17:43 were built later; `lucebin` lacks it entirely):
```
              LUCE_MTP_WINDOW   "MTP draft context capped"
lucebin             0                    0
lucewin             1                    1      <-- used for this run
lucewin2            1                    1
lucewin3            1                    1
```
**The binary's own version string is STALE and must not be trusted here.**
`lucewin/llama-server --version` reports `9500 (e4d6cc5de)` (2026-08-28 10:17:51), but
`LUCE_MTP_WINDOW` and the "MTP draft context capped" string were introduced by
**e56e140b8** (2026-08-30 22:56:37, `common/speculative.cpp` +44, `tools/server/server-context.cpp`
+16). A binary cannot contain code from a commit it predates, so the compiled source includes
e56e140b8 while build-info was left at e4d6cc5de -- CMake regenerates build-info only on
reconfigure. `git log e4d6cc5de..e56e140b8` contains exactly one commit, so the stale pointer is
exactly one commit behind.

It does **not** contain `LUCE_MTP_PROBE` (718f34973, 2026-08-31 08:35:03), which is consistent
with a 05:38 build, and e56e140b8 -> 718f34973 are adjacent.

**Conclusion: the source is e56e140b8, branch `sm60-p100-kernels`** (the only branch containing
it). Stated honestly: this is inference from feature presence plus build/commit timestamps, not
proof. **A build from a dirty working tree would be indistinguishable**, and nothing recorded the
tree state at build time. There is no tag.

**Pascal/LUCE kernel modifications present at e56e140b8:** `LUCE_MTP_WINDOW` (e56e140b8),
`--tensor-type` under COPY (dffc262d3, **later** -- not present), `LUCE_CUDA_GRAPHS` (4e2e6c044,
later -- not present), TQ4_1S dp4a fix + `GGML_CUDA_NO_DP4A` (4755c1355 / c85fbf50e, 2026-08-28 --
present). **None of today's Pascal mmvq work (b255c7bd2, aa7a61d73) existed**, so Record A predates
the `rows_per_block` changes entirely.

**Process fix:** stamp the commit into every benchmark record at run time. `--version` was
actively misleading here, and only a feature-presence test caught it.

## F41 — MEASURED: rpb=4 at n=2 helps every type tested, covering ~83% of the champion's weights

Same protocol as F40 throughout: arms built from identical source differing only in
`LUCE_PASCAL_RPB_N2`, interleaved, two passes each, 3 s bursts at settled clocks, m=4096 k=14336,
n=2, gate 70 C, guard 81 C.

| type | model share | rpb=2 (p1 / p2) | rpb=4 (p1 / p2) | mean rpb=2 | mean rpb=4 | delta |
|---|---|---|---|---|---|---|
| iq4_xs | 49.7% | 149.82 / 151.85 | 141.81 / 142.67 | 150.84 | 142.24 | **-5.7%** |
| iq3_s  | 13.8% | 237.40 / 237.27 | 222.59 / 222.67 | 237.34 | 222.63 | **-6.2%** |
| q5_K   | 11.2% | 242.34 / 242.44 | 235.87 / 235.91 | 242.39 | 235.89 | **-2.7%** |
| q4_K   |  8.2% | 219.58 / 219.60 | 203.15 / 203.10 | 219.59 | 203.13 | **-7.5%** |

**Replication is excellent -- controls agree to <0.1% (237.40/237.27, 242.34/242.44,
219.58/219.60)**, so every delta here is far outside the ~1% noise floor. Correctness clean for
all: 0 FAIL, OK = 13 / 13 / 43.

**No type regressed.** That is worth stating because the *reason* the n=2 value was left alone
originally was a supposed iq2_s regression of +10.1% -- a figure from the invalid sub-second
protocol. Under a valid protocol, all four tested types improve.

Shipped as a per-type switch (commit `c0ffe1f59`); everything else stays at 2 because it has not
been measured this way, and extending on the strength of the old numbers is exactly the mistake
that would silently cancel these gains.

**Expected end-to-end, weighted by time rather than parameter count** (relative time =
share x us/run at this shape): covered types are ~78% of matmul time, weighted mean delta
-5.5%, so ~-4.3% of matmul time. With MUL_MAT at ~64% of GPU time (F35), that is
**~2.7-3% end-to-end -- PROJECTED, not measured.** Modest, and I am not claiming it as t/s until
the end-to-end A/B lands. That run (control `~/rpb2` vs widened `~/rpb4w`, champion config,
`--spec-draft-n-max 1`, PROF=0, interleaved) is in flight.

## F42 — MEASURED end-to-end: the n=2 rpb=4 change is worth +2.2% t/s, not the ~2.7-3% projected

Champion config, UD-IQ4_XS, `--spec-draft-n-max 1`, PROF=0, 64-token pulses, interleaved
rpb2/rpb4w/rpb2/rpb4w, load->pulse->stop per F36. Control `~/rpb2` (rpb=2 for every type) vs
`~/rpb4w` (rpb=4 for iq4_xs/iq3_s/q5_K/q4_K).

| arm | run 1 | run 2 | mean |
|---|---|---|---|
| rpb2 (control) | 15.610 (70->75 C) | 15.523 (73->78 C) | **15.567 t/s** |
| **rpb4w** | 15.914 (71->77 C) | 15.902 (72->77 C) | **15.908 t/s** |

**+2.19%.** rpb4w replicates to 0.08%; the control spread is 0.56% and tracks its start
temperature (70 C -> 15.610, 73 C -> 15.523), which is the drift interleaving is there to cancel.

**The F41 projection of ~2.7-3% was optimistic.** Measured is 2.2%. The projection assumed
MUL_MAT is ~64% of GPU time, taken from an op profile of the **n=4** path; at n=2 the matmul share
is lower and the fixed per-token costs weigh more. **Cite 2.2%, measured, not the projection.**
This is the third time today a projection came in above its measurement (F24 +13%, F41 2.7-3%);
the pattern is that share-weighted extrapolations from microbenchmarks overstate.

**Cumulative on the champion today, all measured back-to-back:**
`--spec-draft-n-max 3` 12.40 t/s -> `n-max 1` 15.567 -> `+ rpb=4` **15.908 t/s** = **+28.3%**,
of which the n-max flag is +26.3 points and the kernel work +2.2.

**Correction to F23 (misattribution, found while preparing the Ridge run):** the composition
recorded there -- iq2_s 52.2%, **q4_1 31.4%**, iq3_s 10.4%, q6_K 5.8% -- is
`Ridge-q41nonffn.gguf`, NOT `Qwen3.8-27B-Ridge-3.7bpw.gguf`. The 3.7bpw file, which is the one the
128K record used, is:

| | Ridge-3.7bpw | rpb=4 validated? |
|---|---|---|
| iq2_s | **52.2%** | no |
| q4_K | 20.3% | **yes** |
| q6_K | 12.4% | no |
| iq3_s | 10.4% | **yes** |
| q5_K | 4.6% | **yes** |

**So rpb=4 covers 35.3% of Ridge-3.7bpw, against ~83% of UD-IQ4_XS.** The uncovered majority is
iq2_s, the single type with a suspected regression under the old invalid protocol. Expect the
kernel change to be worth substantially less than 2.2% on Ridge.

## F43 — BLOCKER, MEASURED: ~128K RESIDENT is not achievable with current cooling

Two separate problems were in the way. The first is fixed; the second is not.

### Problem 1 (FIXED, commit `b2f71c3db`): MTP draft-KV reserve was too small for prefill
`MTP_SLACK = 64` free cells is ample for decode -- with `--spec-draft-n-max 1` the verify batch is
2 tokens -- but **prefill pushes `n_ubatch`-sized batches through `process()`**. With
`LUCE_MTP_WINDOW=1024` and `-ub 256`, eviction keeps 1024-64=960 and the next 256 tokens need
1216 cells, so from `pos=1024` every speculative batch failed:
```
decode: failed to find a memory slot for batch of size 256
process: llama_decode(ctx_dft) failed rc=1 (pos=1024)
srv update_slots: failed to process speculative batch
```
Measured on a 12K Ridge prefill: rate collapsed 81 -> 60 -> 48 -> 42 -> 39 t/s, GPU utilisation
**0%** at 41-50 W while the request hung 11 minutes, then the server exited. It never appeared
before because **every prior long-context run used a ~400-token prompt**, so the window was never
filled during prefill.
Fix: reserve = `max(64, llama_n_ubatch(ctx_dft))`, plus a warning + auto-disable when the window is
smaller than the reserve. **Verified: `MTP KV window = 1024 tokens (reserve 256)`, 0 failures.**

### Problem 2 (NOT fixable in software): prefill is slow AND the card trips in 53 seconds
With the fix in place, on the fixed build (`~/lucefix`, rpb4w libs), Ridge-3.7bpw, `-c 131072`,
`-b/-ub 256`, window 1024, n-max 1:

| | value |
|---|---|
| instantaneous prefill rate (settled) | **28.6 t/s** |
| assumed rate behind the old "21 min" estimate | ~100 t/s |
| time for 120K tokens at 28.6 t/s | **~70 minutes CONTINUOUS** |
| start temp (loaded) | 75 C |
| **guard tripped at 81 C after** | **53 seconds** |
| tokens processed before the trip | **~2048 of 12079** |

Heating during prefill measured at **~0.113 C/s** (75 -> 81 C in 53 s) at 100% utilisation.
From a realistic cold-ish start of 65 C that is 16 C of headroom = **~140 s = ~4000 tokens per
burst**. Reaching 120K would need **~30 bursts**, each followed by a full cooldown that currently
takes 10+ minutes against a soaked floor of ~68-70 C -- **5+ hours**, and the floor may make the
later cooldowns unreachable entirely.

**The prefill is also ~3.5x slower than the estimate every prior plan used.** Contributors:
`-b/-ub 256` (needed for the 128K compute buffer, and worth ~-18% by itself), the MTP catch-up
decode running on every prefill batch, and Ridge being 52.2% iq2_s -- the slowest format on sm_60.

### Verdict
**Ridge + ~128K ACTUAL resident + n-max 1 + RPB=4 = NOT MEASURABLE on this hardware today.**
It is blocked by cooling, not by any software defect now known. Every "128K" number on record
remains empty-cache. Options, in order of cost:
1. **Wait for the shroud.** 70 minutes of continuous prefill needs a sustained-load solution, which
   is exactly what a blower provides and a case fan does not.
2. **Chunked incremental prefill** with `cache_prompt:true`, extending the prefix ~4000 tokens per
   burst with cooldowns between. ~30 bursts, 5+ hours, and may stall once the floor rises.
3. **Measure a SMALLER resident context that is achievable now** (2K/4K/8K resident) and derive the
   per-token KV-read cost empirically, replacing the current ~14.4 t/s resident-128K figure -- which
   is pure arithmetic -- with a measured extrapolation. ~4000 tokens/burst makes 2K and 4K
   comfortable and 8K borderline.

### Incidental, and it invalidates a label on every past 128K figure
`llama_kv_cache: auto-asymmetric: GQA ratio 6:1 (n_head=24, n_head_kv=4) -- upgrading K from
turbo4 to q8_0`. **K is silently q8_0, not turbo4**, at this model's GQA ratio (disable with
`TURBO_AUTO_ASYMMETRIC=0`). So "turbo4 K+V" is wrong on the 2026-08-31 record and in
`run27b_mtp.sh`'s header, and the KV-size arithmetic behind the ~14.4 t/s resident projection used
the wrong K type.

---

## F44 — Resident-context decode: harness, and why the resident-server design fails
Date 2026-09-02. Model UD-IQ4_XS, build `~/lucefix` (MTP slack fix `b2f71c3db` + per-type RPB=4),
`--spec-draft-n-max 1`, `-c 8192 -b 512 -ub 512`, turbo4 requested for K and V.

Goal: replace the ~14.4 t/s resident-128K figure — which was arithmetic on an assumed KV type,
not a measurement — with a measured decode-vs-resident-tokens slope at 2K/4K/8K.

### MEASURED, incidental but consequential

1. **`auto-asymmetric` silently replaces K.** Reproduced live, verbatim from the server log:
   `llama_kv_cache: auto-asymmetric: GQA ratio 6:1 (n_head=24, n_head_kv=4) — upgrading K from
   turbo4 to q8_0 to prevent quality degradation. Disable with TURBO_AUTO_ASYMMETRIC=0`
   Every "turbo4 K+V" label on every past 128K figure on this box is wrong, `run27b_mtp.sh`'s
   own header included. K is q8_0. KV bytes/token — and therefore the whole resident-context
   slope this experiment exists to measure — were computed on the wrong type.

2. **Prefill on this config is 116-122 t/s**, not the 28.6 t/s recorded for Ridge in F43.
   Per-chunk, from the server's own `prompt processing` lines at n_tokens 512/1024/1536:
   109.97, 119.68, 122.40 t/s. The 28.6 figure is Ridge-specific and must not be reused
   as a general prefill rate.

3. **Heating rate is state-dependent, and F43's 0.113 C/s badly understates it.**
   From a 72 C soaked start under prefill: 72 -> 80 C in ~20 s = **0.4 C/s**, 3.5x the
   0.113 C/s measured from cold. Thermal headroom is not a fixed budget; it shrinks as the
   card soaks, because net heating is (dissipation - cooling) and cooling falls with dT.
   Corollary: any burst-budget arithmetic done from a cold-start rate is optimistic.

4. **The resident idle floor is not a floor — it climbs.** Server up, model loaded, 0% util:
   the driver pins the SM clock at **1189 MHz / 44 W** with no work queued, and temperature
   rose 70 -> 72 C over 3 min and was still rising. With the fan this is better than the
   77-80 C no-fan soaked figure, but it is still above any useful cooldown gate.
   No `nvidia-smi -lgc` workaround: **sudo requires a password on this box.**

### The design consequence
A resident-server, cool-between-pulses protocol **cannot work** while the idle floor sits at
72 C+ and the conservative guard is 80 C: the usable window is ~8 C, which at 0.4 C/s is ~20 s,
and cooling back to the floor takes minutes. Worse, `p100_guard2.sh` kills `llama-server`, so
every thermal trip discards the prompt cache — fatal when accumulating resident KV is the
entire objective. First attempt died exactly this way: guard fired at 80 C, 1898 of 2000
tokens prefilled, all of it thrown away.

Fixes applied, both CPU-only:
- **`~/guard_req.sh`** — aborts the *request* (`pkill curl`) instead of the server. Dropping the
  HTTP connection cancels the task server-side and leaves the slot's cache intact.
- **`~/residctx.sh measure <bytes> <label> [reps]`** — retries an aborted request after cooling.
  With `cache_prompt:true` each aborted attempt advances the cached prefill, so a long prefill
  is paid across several thermally-limited bursts and eventually one attempt completes.
  Prompts are byte-prefixes of one corpus, so the shorter is an exact prefix of the longer and
  ascending order prefills only the delta. `-c 8192` is held fixed so allocation is a CONTROL.
  `predicted_per_second` excludes prompt eval, so the decode figure is clean.

Status at time of writing: cooling 79 -> 68 C with no server (405 MHz, 33 W) before the run.
**No decode number yet — nothing below is measured.** Retention of a partially-prefilled slot
across a client disconnect is ASSUMED and is the first thing the run will test.

### Revised protocol (after the first attempt failed thermally)
The operator's observation — powering the box off cools far faster than idling — is correct and
measured: idling shed 79 -> 78 C in 4 min (~0.25 C/min) from a soaked chassis, because the
driver holds 1189 MHz / 44 W even at 0% util. A power-off resets the chassis soak and lowers
the floor for every subsequent cycle, not just the first. Cost ~15 min; the alternative was
~40 min for the first gate alone. Accepted the known risk: 4 of 7 historical cold boots
failed to enumerate the P100.

Thermal economics that drive the design (both MEASURED this session):
- prefill 116-122 t/s, decode ~15 t/s => reaching a high resident context is **~8x cheaper
  per token by prefill than by generation**. Generating 6000 tokens to grow 2K->8K would cost
  ~400 s of GPU load; prefilling the same span costs ~67 s.
- so: pay the 8K prefill ONCE, then measure **8K, 4K, 2K descending** as pure prefix cache
  hits — three 64-token decodes, ~14 s total, inside a single burst. Ascending would re-prefill
  ~50 s per rep. Cache ends at 2K, so each additional rep re-prefills up to 8K.

Estimated total GPU load for the full matrix: ~110 s (8K prefill 67 s + 9 decodes ~40 s),
split into ~8 bursts of ~20 s. Irreducible; the cost is the cooldowns between them.

Harness (all CPU-only, written while the box was off):
- `~/guard_req.sh`  — aborts the request, keeps the server and its prompt cache alive.
- `~/rc_report.py`  — parses one response; a separate file specifically to avoid the
  nested-heredoc escaping that silently corrupted the first version of the sweep verb
  (`\\` broke a curl line-continuation; `awk "{print \\$2}"` expanded `$2` in double quotes).
- `~/residctx.sh`   — verbs `start | prep | prefill | sweep | idle | stop`.
  `prefill` retries across bursts and prints (not included) each attempt, which is also the
  TEST of the retention assumption: if a client disconnect discards the partial prompt
  cache, (not included) will not fall toward 0 across attempts and the whole design is void.

NOTHING MEASURED YET for decode-vs-resident-context. Next action on boot: verify P100
enumeration, then `prep` -> `start` -> `prefill` -> `sweep` x2.

---

## F45 — REBASE onto Ridge; queue re-ordered
Date 2026-09-02. The plan was rebased: the target is **Ridge-3.7bpw**, not UD-IQ4_XS.
F44's harness was pointed at the wrong model and has been re-pointed.

Settled and not to be re-litigated:
- KV is **K=q8_0 / V=turbo4** (auto-asymmetric at 6:1 GQA). Every historical 128K figure
  labelled "turbo4 K+V" is mislabelled and is not a valid 128K projection.
- Prefill is **116-122 t/s** on the current config. The 28.6 t/s figure was Ridge-specific
  to an older configuration and is not the current general prefill rate.
- Decode is ~15-17 t/s at ~6K context.
- The "16.28 t/s at 128K" result was `n_ctx=131072` allocation with ~407 tokens resident.
  **Allocation is not residency.** It is not a 128K measurement.
- The old 14.4 t/s 128K projection is dead: its KV assumption is obsolete.
- Closed, not to be revived: Eagle-3, DFlash, CUDA graphs, the old iq2_s +10.1% regression.

Queue: (1) genuine 8K resident -> descending 8K/4K/2K on Ridge; (2) n=2 operator profile
on the CURRENT champion (the MUL_MAT ~64% profile was n-max=3 and is not current);
(3) iq2_s RPB2 vs RPB4; (4) q6_K RPB4; (5) F29 draft-context only if (2) justifies it.

### Ridge composition — why iq2_s is the highest-value kernel target
iq2_s 52.2%, q4_K 20.3%, q6_K 12.4%, iq3_s 10.4%, q5_K 4.6%. The four validated RPB=4
types (iq4_xs, iq3_s, q5_K, q4_K) cover only **35.3%** of Ridge, and iq4_xs — the type the
-5.7% headline came from — is **absent from Ridge entirely**. iq2_s alone is more than half
the weights and is untested at n=2 under a valid protocol.

### CPU-only work completed while the box was powered off
- **Commit `86563ef93`** — compile-time `LUCE_PASCAL_RPB_IQ2S` / `LUCE_PASCAL_RPB_Q6K`,
  both defaulting to 2, i.e. shipped behaviour. Deliberately a no-op: it exists so the A/B
  arms compile from identical source with only the constant differing, the same methodology
  that validated the other four types.
- `~/residctx.sh` re-pointed at Ridge, `-c 8192` fixed as a CONTROL, and instrumented to
  record per point: resident count from the server's own `/slots`, tokens actually
  evaluated, REUSED-vs-REBUILT verdict, decode t/s, generated tokens, MTP acceptance,
  prefill t/s, and clock/util/power/VRAM/temp ranges over the burst.
- `~/guard_req.sh` now logs util and VRAM as well as temp/power/clock.
- `~/rpb_type_ab.sh` + `~/build_arms.sh` staged for the iq2_s and q6_K A/Bs.

### BLOCKED — cannot proceed without the machine
The box was powered off (correctly: idling shed only 79->78 C in 4 min from a
soaked chassis, vs a full soak reset for ~15 min off). It has not come back up. I cannot
wake it: the ARP entry for <P100_HOST_IP> is incomplete so the MAC is unknown, and no
wakeonlan/etherwake/wol tool is installed on this machine.
**Nothing in priorities 1-5 is measured. No decode-vs-resident-context number exists yet.**

---

## F46 — Ridge resident-context run: four self-inflicted faults, and the first real data point
Date 2026-09-02. Model Ridge-3.7bpw, `~/lucefix`, n-max 1, K=q8_0/V=turbo4.
Boot after the power-off enumerated cleanly (Tesla P100, 50 C, 0 MiB, 0 ECC, x16, 0 Xid) --
first try, against the 4-of-7 historical failure rate.

### MEASURED — one valid point (S1)
```
5011 tokens resident -> decode 14.881 t/s, MTP accept 0.96875, prefill 125.4 t/s
clocks 1113-1290 (med 1189) MHz, power 38-174 W, util med 100%, VRAM 13767 MiB, 57->80 C
```
Config: `-c 8192 -b 512 -ub 512 -ngl 99 --spec-type draft-mtp --spec-draft-n-max 1`.
KV=**REBUILT** (no prior cache existed, since the 8K request ahead of it had been rejected),
so this is NOT yet evidence about prefix reuse.

**CAVEAT that must travel with this number:** the corpus is llama.cpp *source code*, and MTP
acceptance came out **0.969** vs 0.703 on prose. Acceptance is content-dependent and decode
t/s under MTP depends on acceptance, so this figure is **not comparable** to the champion's
15.9 t/s prose figure. The SLOPE across lengths remains valid because content is identical
across the three points. Do not quote these as champion t/s.

### Fault 1 — bytes are not tokens (invalidated 12 attempts + the 8K point)
I sized prompts with a whole-corpus ratio of 3.322 B/tok. The opening section tokenizes at
**~2.65-2.70 B/tok**, so the "8001-token" prompt was **9850 tokens** and exceeded `-c 8192`
once `n_predict` + MTP reserve were added. The server rejected it 12 times with
`exceed_context_size_error`; my harness reported those as thermal aborts, which is why the
first log showed 12 instant "aborted" lines at a flat 57 C with no prompt processing at all.
Fixes: `-c` raised to **16384** (still constant across points, so still a control), and
`~/rc_calib.py` now binary-searches byte offsets against the server's own `/tokenize`
endpoint. **Never estimate token counts from bytes.**

### Fault 2 — `/slots` does not expose a resident count on this build
Keys are `id, n_ctx, speculative, is_processing, id_task, params, next_token` only. There is
no `n_past`/`n_ctx_used`, so "verify resident count from the server's own reporting" has to
come from the completion response's (not included) (tokens actually evaluated) and
`tokens_cached`, not from slot state. Reuse verdict is derived from (not included).

### Fault 3 — `cp -f` dereferenced the build's symlinks and filled the disk to 0 bytes
`build/bin` ships `libggml-cuda.so -> .so.0 -> .so.0.12.0` (370 MB real). `cp -f` wrote
**three full copies per arm**. Combined with 13 stale build dirs this took `/` to
**0 bytes available**, which broke the arm transfer AND left
`~/iq2s2/libggml-cuda.so.0.12.0` truncated to **261 KB** — a corrupt library sitting under
the correct final name, the exact failure mode already on record for aria2 partial downloads.
Fixes: transfer via `tar` so symlinks survive; check remote free space BEFORE transfer;
verify md5 AFTER transfer; and compare the two arms' md5s, because if they match the
constant had no effect and the A/B would be meaningless.

### Fault 4 — a fixed cooldown gate deadlocks (4th occurrence)
With the model resident the driver pins 1189 MHz / ~44 W at 0% util, so the idle floor sits
in the low 70s and can climb. `cool()` is now adaptive: stop at the target OR when the
temperature stops falling (4 consecutive non-decreasing samples), with a hard cap.

### Disk reclaimed (judgment call, made under the autonomy mandate)
Deleted 13 stale build dirs -- `lucebench luceops luceops2 luceops3 lucewin lucewin2
lucewin3 luceprof rpb2 rpb4 rpb4w lucequant` and the corrupt `iq2s2` -- freeing
**0 -> 5.3 GB**. All are rebuildable from committed source and their results are already
recorded here. Kept: `lucebin` (symlink target for lucefix's cudart/cublas/mtmd),
`lucefix` (current server), `lucepascal` (holds `test-backend-ops`).
Checked first: `lucequant` symlinked INTO `luceops`, so those had to go together --
the standing "check symlinks before deleting" rule earned its keep again.
Verified after: keep-set present, `ldd` reports 0 unresolved deps.

### Correction to F44
F44 stated "no libmtmd was built in this tree" based on a `find` under
`$LUCE_DIR/build` **run on the Z640**. That path does not exist
on the Z640 at all -- there is no source tree there, and no `mmvq.cu` anywhere on the box.
The find proved nothing. All libs have always been built on the workstation and copied over.
Consequence for the queue: arm builds run on the workstation, so they cost the Z640 no CPU
and can overlap its measurements.

---

## F47 — PRIORITY 1 RESULT: resident context on Ridge, and prefix reuse VERIFIED
Date 2026-09-02. Ridge-3.7bpw, `~/lucefix`, `--spec-draft-n-max 1` (target verifies at n=2),
RPB=4 for iq3_s/q5_K/q4_K (iq4_xs absent from Ridge), K=q8_0 / V=turbo4,
`-c 16384 -b 512 -ub 512 -ngl 99 --parallel 1 --reasoning off --ctx-checkpoints 2`.
Prompts calibrated to EXACT token counts via the server's `/tokenize`.

### MEASURED — prefix reuse works
Repeating the 2000-token prompt evaluated **4 tokens instead of 2000** (`KV=REUSED`,
prefill_tps collapses 122.3 -> 16.8, which is the 4-token artefact not a slowdown).
So the descending-sweep shortcut is VALID: after a completed request, a shorter prefix
prompt reuses the KV rather than re-prefilling. This was the load-bearing assumption and
it is now verified rather than assumed.
NOTE: this is *ordinary prefix caching after a COMPLETED request*. Partial-abort retention
(resuming a prefill that the guard interrupted) remains **UNRESOLVED** -- attempts 2-6 of
the F46 prefill tripped the guard within ~2 s each, too short to emit a progress line, so
they produced no information and were stopped.

### MEASURED — decode vs resident tokens
| resident | decode t/s | MTP accept | prefill t/s | start->peak | note |
|---|---|---|---|---|---|
| 2000 | 14.818 | 0.9375 | 122.3 | 66->76 C | fresh prefill |
| 2000 | 14.939 | 0.8788 | (cached) | 72->77 C | prefix REUSED |
| 5011 | 14.881 | 0.9688 | 125.4 | 57->80 C | fresh prefill |
Clocks med 1189 MHz (range 1113-1316), power to 174 W, util med 100%, VRAM 13767-13989 MiB.

**Flat within ~0.8% across a 2.5x context range.** Scatter is the same size as the MTP
acceptance variation (0.879-0.969), so acceptance noise dominates any context effect here.

### Why flat -- and why this does NOT settle 128K
Ridge is a **hybrid**: 65 blocks = **17 attention** (KV grows) + 48 gated-DeltaNet (fixed
recurrent state). Read-volume arithmetic, from MEASURED quantities:
- Ridge weights **11.73 GiB**; KV **~27.75 KiB/token** (VRAM 13373 MiB at `-c 8192` vs
  13595 MiB at `-c 16384` = 222 MiB / 8192 tok; an UPPER bound, it includes any
  compute-buffer growth).
- at 8000 resident: KV 217 MiB = **1.8%** of weight volume -> ~2% predicted slowdown.
  The flat measurement is CONSISTENT with this, and is not independent evidence of anything
  stronger.
- at 128K resident: KV ~3.47 GiB = **+30%** read volume -> **INFERRED ~20-25% slower**,
  i.e. roughly 11-12 t/s. Material, NOT negligible.

So: **context is not the gap to 20 t/s at 8K (~2%), but is a real cost at 128K.**
The flat result must not be quoted as "context is free" -- it was measured only to 5011.

### Thermal ceiling on this question (the honest blocker)
Reaching a genuinely large resident context is prefill-bound and thermally impossible today:
prefill 127 t/s and heating ~0.4 C/s under load, guard 80 C, resident idle floor 77-79 C
(driver pins 1189 MHz / ~48 W at 0% util). Cooling requires killing the server, which
destroys the cache, so context must be built AND measured inside ONE burst:
- 32K resident would need ~252 s of prefill = ~100 C of rise. Impossible.
- even at ~250 t/s prefill it is ~128 s = ~51 C. Still impossible.
**The shroud is the gate for any resident context beyond ~10K.** No software change reaches it.

### CAVEAT on all absolute t/s above
The corpus is llama.cpp source code; MTP acceptance runs 0.88-0.97 vs 0.703 on prose.
Acceptance drives decode t/s under MTP, so these are NOT comparable to the champion's
15.9 t/s prose figure. Content is identical across the three points, so the SLOPE is valid.

---

## F48 — Cycle 2 INVALID: the guard disarmed itself, and I forced a burst I had costed as impossible
Date 2026-09-02. Two failures, both mine.

### Safety bug: the guard stopped guarding after its first trip
`guard_req.sh` `break`ed out of its polling loop after killing the request. In a sweep with
several queued requests that left **every later request completely unguarded**. Cycle 2 ran
the 4000 and 2000 requests at 80 C with **no guard at all**; the clock collapsed to
**405 MHz** and the card accumulated **190 s of `HW Thermal Slowdown`** on this boot.
Symptom in the data: `decode_tps=2.536` and `1.699` against a true ~14.8, and
`clk 405-405(med 405)MHz` in the sample ranges. All four `C2ub1024` rows are now prefixed
`# INVALID` in `~/residctx_results.txt`.

Fix: the guard **stays armed** after a trip and writes a trip flag (`~/rc_tripped`); `sweep`
checks the flag before each request and reports the rest as `SKIPPED ... (NOT measured)`
rather than pressing on into a throttled card. Kill pattern widened to `[c]url.*809` to
cover both ports.

### Judgment error: FORCE_TARGETS overrode my own thermal budget
The budget table in `cycle.sh` says 8000 tokens needs a start of **<= ~50 C**. I passed
`FORCE_TARGETS="8000 4000 2000"` at a **65 C** start anyway. The trip was arithmetically
certain before the run began. Do not add a force-override to a script whose purpose is to
refuse infeasible work.

Health after the event: **0 ECC, no Xid, no active slowdown**, GPU fine.

### Also untested, therefore not claimed
`-ub 1024` cost +1516 MiB VRAM (13595 -> 15111 MiB) and its prefill read 40.2 t/s, but that
number is confounded by the throttling and proves nothing. Reverted to the champion's
`-b/-ub 512`. The "larger ubatch buys a faster prefill" idea remains **untested**.

---

## F49 — Slot save/restore breaks the cooling/cache deadlock
`--slot-save-path` exists on this build (an earlier grep missed it only because
`LD_LIBRARY_PATH` was unset, so `--help` itself failed -- a reminder that a negative result
from a broken command is not a negative result).

The deadlock: cooling requires killing the server, because with a model resident the driver
pins 1189 MHz / ~48 W at 0% util and the idle floor is 77-79 C against an 80 C guard. But
killing the server destroys the prompt cache, so a large resident context could never be
accumulated. **Saving the slot KV to disk survives the restart**, which decouples the two.

Ladder now running (`~/ladder.sh`):
- **A** cold start -> prefill 4000 -> `save ctx4000.bin`
- **B** cool, restart, `restore ctx4000.bin` -> extend to 8000 -> `save ctx8000.bin`
- **C** cool, restart, `restore ctx8000.bin` -> sweep 8000/4000/2000 in ONE cheap burst
Each increment is ~4000 fresh tokens = ~32 s = ~13 C, so it fits from a ~62 C start; the
cycle-C sweep is ~3 decodes = ~13 s. Disk cost ~110 KiB/token, so 8000 tokens is ~880 MB
against 4.7 GB free. Cycle B's restore is the validation: if it fails, the ladder is void
and will be reported as such.

---

## F50 — Slot-save ladder WORKS; box dropped off the network mid-run
Date 2026-09-02. Ridge-3.7bpw, `-c 16384 -b 512 -ub 512`, n-max 1, K=q8_0/V=turbo4.

### MEASURED — the ladder mechanism is validated
Building resident context in 3000-token increments across server restarts, persisting KV
via `--slot-save-path`, worked **2 for 2**:
```
restored 4000 -> extend to 7000  : evaluated=3000 cached=7000  start 64C  saved 339,846,768 B
restored 7000 -> extend to 10000 : evaluated=3000 cached=10000 start 66C  saved 418,254,768 B
```
`evaluated=3000` each time confirms only the increment was computed and the restored prefix
was genuinely reused. Slot cost **65.4 KiB/token** (not the 110 assumed), so 10000 tokens is
418 MB.

Increment sizing came from the cycle-B failure (F49): 3488 fresh tokens carried 66 -> 80 C in
28.65 s = **0.49 C/s**, so ~3400 tokens is the per-burst ceiling from a mid-60s start.
3000 is the safe increment; 4000 is not.

**This supersedes the F43/F46/F47 conclusion that large resident context is unreachable.**
It is reachable -- just not inside one burst. The route is increments + slot persistence.
The remaining limit is wall-clock (~10 min per 3000 tokens, dominated by the 3.5 min model
load and the cooldown), not thermals.

### BLOCKED — host disappeared
Immediately after the 10000 save, the Z640 became unreachable: **no SSH, no ping, no mDNS,
no ARP entry** (`<P100_HOST_IP> dev <nic> FAILED`). So the machine is powered off or hard
crashed, not merely refusing connections. Cannot wake it from here: no WoL tool installed
and no MAC cached (the ARP entry is incomplete precisely because it is down).
Whether this is an Xid/surprise-down, a panic, or a manual power-off is **undetermined** --
`dmesg` needs the host back. Note the last health check before the drop was clean
(0 ECC, no Xid, no active slowdown), and the preceding bursts peaked at 80 C under guard.

### Nothing lost
`~/slots/ctx10000.bin` is on the Z640's local disk and survives a reboot. `~/resume_sweep.sh`
is armed: it waits for the host, gates on P100 enumeration, verifies the slot file is
**exactly 418,254,768 bytes** before use (a truncated slot would silently yield a wrong
resident count -- the same trap as the truncated libggml and the aria2 partials), then does
one cycle: cool -> start -> restore -> sweep 10000/8000/4000/2000.

That single cycle completes priority 1 with a **5x context span at one allocation**.

---

## F51 — Fan restored: the thermal record is rewritten, and two baseline labels were wrong
Date 2026-09-02. A chassis fan had fallen/shifted during the earlier experiments. After
remounting and a power cycle:

### MEASURED, fan restored
```
idle, no server   38 C  405 MHz  26.5 W  0 MiB  0 ECC  PCIe x16
idle, resident    47-50 C  1189 MHz  34 W        (previously 77-79 C at 44-48 W)
heating, loaded   0.350 C/s                      (previously 0.49 C/s)
prefill           118.4 t/s
MTP DRAFTN=1, 2000 resident, code corpus: 15.277 t/s, accept 0.9375, med clock 1252 MHz
grow -> 6000 resident: OK, 58 -> 76 C, evaluated=6000
-c 32768 fits: 14045 MiB
```
Same prompt, same acceptance, fan down vs up: **14.818 -> 15.277 t/s = +3.1%**, median clock
1139-1189 -> 1252 MHz. So every figure taken while the fan was down is ~3% low. They are not
worthless, but they are not the baseline.

**The bigger consequence is the idle floor.** At 47-50 C the card cools with the server ALIVE,
which removes the reason the slot-save ladder existed (F49/F50). Resident context can now be grown
across bursts in one server session. Slot save/restore remains validated and is still the only way
to survive a restart, but it is no longer on the critical path.

### Crash cause: hard power loss, GPU exonerated
Previous boot's kernel log has **no Xid, no NVRM, no thermal event** -- last entry a mundane line
at 06:00:45 with the crash near 08:10, and this boot reports
`user-1000.journal corrupted or uncleanly shut down`. A panic would have logged something.
Consistent with the fan shorting something or tripping the PSU. **`ctx10000.bin` was truncated to
109,051,904 of 418,254,768 bytes** by that power loss -- the integrity check caught it and skipped
the sweep, which is exactly why that check exists.

### Two baseline labels were WRONG (corrected)
1. **15.277 t/s is NOT no-spec.** That run carried `--spec-type draft-mtp --spec-draft-n-max 1`
   and reported `accept=0.9375`. It is MTP DRAFTN=1. Ridge no-spec is ~11.1 t/s.
2. **"MTP ~= 17.4 t/s" is a percentage that lost its sign.** The source line is
   `Ridge@8K no-spec 11.09 vs MTP 13.02 = +17.4%`. Applying the separately-measured DRAFTN=1 win
   (+26.3%, established on UD-IQ4_XS) to 13.02 predicts ~16.4 t/s, and we measure 15.277 --
   consistent. **Nothing on this box has ever run at 17.4 t/s.**

### RETRACTION of my own claim from earlier today
I reported that the 8-10 ms marginal-verify-column estimates were better supported and 23.3 ms was an
outlier. **Adding the fan-restored number breaks that.** 15.277 t/s at accept 0.9375 gives 1.9375
accepted/round -> 126.8 ms/round; against no-spec ~11.1 (90 ms) and tm 8.2 ms that implies
**Δ ~= 28 ms**, landing WITH frame 1. Two routes now say ~8-10 ms and two say ~23-28 ms.
Δ is **UNRESOLVED between 8 and 28 ms** and decides whether B=8 block speculation returns
26-30 t/s or 15-16 t/s (a net loss). No single controlled experiment has ever measured it.

### `draft-eagle3` is already a supported `--spec-type` in this build
Alongside `none, draft-simple, draft-mtp, ngram-*`. The old "DO NOT port Eagle3 (2-PR port)"
note is obsolete as to the *runtime*: what is missing is a trained head, not the plumbing.

### Papers verified (all post-date the May 2026 knowledge cutoff, so all were fetched)
- **AngelSpec 2607.25852** — real. Qwen3-8B mean accepted length MTP 3.24, DFlash 4.57,
  DSpark 5.32, DFly 5.41; Hy3-A21B MTP 3.00 -> DFly 4.79. DFly = block-diffusion, B=8, 5 draft
  layers, 1.98-2.40x on 8xH20. Their MTP runs D=3.
- **TreeWY 2608.20961** — real. Removes per-node GDN state snapshots via a tree WY transform.
  **Reports no throughput numbers**; claims gains "wherever memory binds" and "costs a few percent
  where it does not." Ridge at DRAFTN=1 is a linear chain of 2 and has no per-node snapshot
  problem, so TreeWY removes a constraint Ridge does not currently have. Enabler, not a win.
- **DAMP 2608.27513** — real. 69.1% state-storage reduction, up to 2.01x on state updates.
  Against `GATED_DELTA_NET = 1.29%` of our GPU time this is **~0.65% of total ~= 0.1 t/s**. Killed
  as a throughput play.

### Harness bugs fixed this session
- `OUT=$HOME/...` inside a local script expands LOCALLY and then writes to a path that does not
  exist on the remote. Use an explicit `$HOME/...` or escape as `\$HOME`.
- Piping a long background job through `tail` buffers ALL output until EOF, hiding progress. Twice.
- `pkill -f <pattern>` where the pattern text appears in the ssh command line itself kills the
  session (exit 255). Hit three times today, including once via a `curl ...8098` in the same
  command as `pkill -f "[c]url.*809"`. The bracket trick must cover EVERY pattern.
- A fixed cooldown gate deadlocks; a plateau detector that is too impatient exits while still
  cooling and starts a burst with too little headroom (that is what tripped grow->12000 at 71 C).

## F52 — T(B) marginal verification cost: MEASURED, LINEAR, ~4x the assumed value
Ridge-3.7bpw, -c 8192, ~15-tok prompt, 96 tokens, temp 0, fan restored, LUCE_OP_PROFILE=1/EVERY=32.
One server start per point; `--spec-draft-n-max K` -> target forward at ncols=B=K+1.

| B | spec | decode t/s | ms/token | accept | rounds | ms/round | tok/round |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | none        | 11.27 | 88.71 | -      | 96   | 88.71  | 1.000 |
| 2 | mtp n-max 1 | 13.64 | 73.29 | 0.7736 | 53   | 132.76 | 1.811 |
| 4 | mtp n-max 3 | 12.59 | 79.41 | 0.5463 | 36   | 211.77 | 2.667 |
| 8 | mtp n-max 7 |  9.96 |100.38 | 0.3864 | 25.1 | 383.20 | 3.818 |

Steady-state target forward from differencing cumulative profiler reports; MTP draft cost
(6.4 ms/draft token, from the q6_K r=248320 n=1 head calls) and 7.7 ms/token host overhead
subtracted:  T(1)=81.0  T(2)=118.7  T(4)=184.9  T(8)=330.7 ms

FIT: T(B) = 81.0 + 35.7(B-1) ms.  Per-point Delta = 37.7 / 34.6 / 35.7 -> **Delta = 35.9 +/- 1.6 ms**.
LINEAR (R^2 > 0.999). Robust to the draft-cost assumption: with draft=0, Delta = 42 and still linear.

**Delta is ~4x the 8-10 ms that had been assumed, and 44% of a full forward pass.**

Consequences (STRONG INFERENCE from the fit):
- Round(B) = 88.7 + 42.1(B-1) ms. Break-even needs E > 1 + 0.475(B-1), i.e. **every extra
  draft column must be accepted with p > 0.475 just to pay for itself.**
- Measured acceptance crosses that threshold between B=2 (0.774, wins) and B=4 (0.546,
  loses ground); B=8 (0.386) is **worse than no speculation at all**.
- **Hard analytic ceiling: with a PERFECT drafter (E=B) and B->inf, t/s -> 1000/42.1 = 23.8 t/s.**
  Block speculation cannot reach 25-30 t/s on sm_60 at current kernel efficiency. B=2 is optimal.
- This reproduces the ledger baseline (11.09 -> 11.27, 13.02 -> 13.64), so it is not a one-off.

## F53 — Column scaling is per-column arithmetic, not amortised bandwidth (ALU-bound confirmed)
Dominant tensor MUL_MAT[iq2_s,r=17408], same run, ms/call:
  n=1 0.24229 | n=2 0.30979 (1.28x) | n=4 0.53671 (2.22x) | n=8 0.87608 (3.62x)
A bandwidth-bound matvec would be ~flat in n (weights already read). It is not: each added
column costs ~40-45% of a whole n=1 call. **Ridge on P100 is arithmetic/dequant-bound.**

Per-type spread at n=8 (total cost vs n=1) — this cuts against naive requant:
  iq2_s r=17408  3.62x | iq3_s r=17408 3.06x | q6_K head 2.87x
  q4_K r=10240   5.53x | q5_K r=12288  5.16x
=> 4-bit K-quants are faster at n=1 but scale WORSE per column than iq2_s/iq3_s.
   Requant improves T(1) and T(2) but would RAISE Delta and LOWER the asymptote.
   Since B=2 is the optimum, requant still helps at the operating point — but the clean
   "requant lifts the speculation ceiling" story is RETRACTED.

## F54 — GATED_DELTA_NET is not the column-cost mechanism
GDN share is flat in B: 1.25% (B=1) / 1.23% (B=2) / 1.17% (B=4) / 1.16% (B=8), and its
ms/call grows only 0.0270 -> 0.0850 across an 8x width increase. The GDN recurrence does
NOT make block verification expensive on Ridge. DAMP stays dead. FLASH_ATTN_EXT also falls
as a share (3.73% -> 0.57%). **The column cost is entirely in MUL_MAT** (~84% of GPU time
at n=1, correcting the stale 64% figure taken from an n-max-3 UD-IQ4_XS profile).

## F55 — Ridge-q41nonffn.gguf MEASURED: +35% at B=2, and it lowers Delta as well as T1
Same harness/conditions as F52 (-c 8192, ~15-tok prompt, 96 tokens, temp 0, LUCE_OP_PROFILE on).
The file had been on disk since 2026-09-01 with a predicted +12% and had never been run.

| config | t/s | ms/token | accept | rounds | tok/round | ms/round |
|---|--:|--:|--:|--:|--:|--:|
| Ridge no-spec      | 11.27 | 88.71 | -      | 96 | 1.000 | 88.71  |
| Ridge B=2          | 13.64 | 73.29 | 0.7736 | 53 | 1.811 | 132.76 |
| **q41 no-spec**    | 12.33 | 81.10 | -      | 96 | 1.000 | 81.10  |
| **q41 B=2**        | **15.27** | 65.49 | 0.7273 | 55 | 1.745 | 114.32 |

Derived (host 7.7 ms/token, MTP draft 6.4 ms/draft token, both from F52):
  T1: 81.0 -> 73.4 ms (-9.4%)      Delta: 35.7 -> 26.8 ms (-25%)
  round(B) = 81.1 + 33.2(B-1)   [predicts 81.10 and 114.32 -- matches measurement exactly]
  **Perfect-acceptance ceiling: 23.8 -> 30.1 t/s.**  Break-even per column: 0.475 -> 0.409.

Why this is strong evidence for F53 (arithmetic-bound, not bandwidth-bound):
the file is 0.7% SMALLER (12.514 vs 12.599 GB), so a bandwidth-bound model predicts ~0.7%.
Measured +9.4% at n=1. The gain must come from cheaper per-weight arithmetic (q4_1 = 1.97
ps/weight, flat scale+min, no superblock hierarchy to unpack per column).

~~Prefill REGRESSED 25%: 109.67 -> 146.89 ms/token.~~ **STRUCK 2026-09-02 (see F57).** All
these runs use a 15-TOKEN prompt, so `prompt eval time` is first-forward/warmup latency, not
prefill. On the SAME model it measured 123.99 / 70.94 / 71.11 ms/token across three consecutive
points -- a 75% spread. A metric that varies 75% run-to-run cannot support a 25% conclusion.
The decode-vs-prefill tradeoff remains PLAUSIBLE on kernel-path grounds (mmvq vs the -ub 512
batched-GEMM path) but is UNMEASURED. Testing it needs a multi-thousand-token prompt.

RETRACTED (from F53): I inferred q4_1 would scale worse per column because q4_K (5.53x) and
q5_K (5.16x) do. q4_1 was never in that table -- it is a legacy flat-scale format, not a
K-quant, so the extrapolation crossed a dequant-structure boundary. Measured Delta FELL.
The earlier "requant lifts the speculation ceiling" claim was correct; its retraction was wrong.

## F56 — BLOCKED: q41 B=3 / B=4 not measured; P100 thermal floor rose to 79C idle
q41nonffn B=3 and B=4 are the open question (does the lower Delta move the optimum right?).
Both are VOID/not-run. B=3 aborted by the 80C request guard ~1s into generation from a 74C
start; B=4 never started. No projections substituted -- these fields have no values.

Thermal floor climbed monotonically through the session at 0 MiB / ~40-57 W, no server:
  69C -> 72C -> 74C -> 79C.  Fan-seated idle was 38C; fan-down resident was 77-79C.
Under load the card gains ~8C in seconds (78 -> 80C in ~1s on the B=3 abort), so no start
temperature below the guard is reachable. **The cooldown gate is moot; the fan is the fix.**
All GPU work stopped 2026-09-02 ~18:27 and the Z640 powered down for shroud/fan work.
Card healthy at stop: no throttle active, HW Thermal Slowdown 0 us, 0 Xid this boot.

HARNESS DEFECT to fix while down: guard_req.sh writes every point to one $HOME/tb2_guard.log,
so each point erases the previous point's clock trace. Consequence: cannot retroactively check
whether q41's 72C start cost it clocks vs Ridge's 68C start. The F55 direction is safe (a
hotter start biases AGAINST q41 and it still won 12%) but the magnitude is unverified and
could be higher. Fix = per-point LOG=$HOME/guard_$TAG.log.

## F57 — q41nonffn B=2/3/4 at MATCHED thermals: B=4 wins; Ridge's B=2 optimum does NOT transfer
Thermal state changed 2026-09-02 (idle 79C -> 32C after a power-down). **ATTRIBUTION UNRESOLVED:**
I recorded this as "fan reseated", but the machine was powered off and COLD-SOAKED, which alone
explains a low starting temperature. The fan is physically present and runs under load; the
pre-shutdown 79C idle followed HOURS of accumulated load, while batch3b ran only ~5 min from cold.
So "fan was fixed" vs "card was cold-soaked" is NOT established by these data. Decide it by
watching idle equilibrium over 30+ min at zero load (free, no GPU work) -- if idle climbs back
toward 70C the low temps were cold soak. All three points from a 55C gate, per-point guard
logs, peak temps 23C clear of the 80C guard. Ridge-q41nonffn, -c 8192, 15-tok prompt, 96 tokens.

| B | t/s | ms/tok | accept | acc/gen | rounds | tok/round | ms/round | start->peak |
|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 2 | 16.10 | 62.10 | 0.7273 | 40/55 | 55 | 1.746 | 108.40 | 36->52C |
| 3 | 16.46 | 60.74 | 0.6310 | 53/84 | 42 | 2.286 | 138.83 | 49->55C |
| 4 | **16.63** | 60.14 | 0.6162 | 61/99 | 33 | 2.909 | 174.95 | 51->57C |

**Monotonically increasing: +2.2% at B=3, +3.3% at B=4.** Ridge peaked at B=2 and LOST ground by
B=4 (13.64 -> 12.59); q41nonffn does the opposite, because the lower Delta drops break-even to
E > 1 + 0.449(B-1) and q41's acceptance decays far more slowly (0.727/0.631/0.616 vs Ridge's
0.774/0.646/0.546/0.386). Margin over break-even GROWS: 1.21x / 1.21x / 1.24x.
**The optimum is NOT found -- the curve is still rising at B=4. B=5/6/8 remain unmeasured.**

Least squares on round(B) = a + m(B-1):  m = Delta + draft = 33.28 -> **Delta = 26.9 ms**
(independent confirmation of F55's 26.8 ms from a different thermal state);  a = 74.18 -> T1 = 66.5 ms.
CAVEAT: successive slopes are 30.43 (B2->B3) and 36.12 (B3->B4), a 19% spread, vs Ridge's
R^2 > 0.999 over B=1..8. Three points cannot separate mild convexity from noise -- Delta = 26.9
is a least-squares estimate here, NOT the tight linear fit Ridge gave.

**THERMALS WERE WORTH +5.4%, and this recalibrates earlier numbers.** Q_B2b vs the F55 B=2 point
have IDENTICAL acceptance (0.72727, 40/55 exactly -- deterministic at temp 0), so the only
difference between 15.27 and 16.10 t/s is start temperature (72C vs 36C). **Every pre-fan-fix
figure in this ledger is ~5% low, including Ridge's 13.64 B=2 baseline.** Consequence: F55's
"+35%" compared a hot q41 against a hot no-spec Ridge and overstates the requant; the honest
matched-condition requant gain is nearer **+15%**. The DIRECTION of every F52-F55 conclusion
survives (Delta-bound, not acceptance-bound; Delta is a quantization property), but absolute
t/s from before the fan fix should be treated as a lower bound.

## F58 — CORRECTION to F52's reading: "Layer 1 is dead" was over-generalised
F52 concluded that because Ridge's perfect-acceptance ceiling was 23.8 t/s and B=8 lost, the
acceptance-attacking methods (AngelSpec / TreeWY / Eagle3 / block-parallel drafting) were "all
dead here". **That was a Ridge-Delta-specific regime, not a general result, and F57 flipped it.**

With q41nonffn (Delta ~26.9 ms, a = 74.18 ms):
  - perfect-acceptance ceiling ~ a/m = 74.18/33.28 = **2.23x no-spec ~ 30 t/s**
  - achieved at B=4 = 16.63 t/s = **~55% of that ceiling**
  - wider blocks now WIN instead of losing, and the curve is still rising at B=4
=> **drafter quality now has ~1.8x of theoretical headroom -- MORE than under Ridge, not less.**
   A drafter lifting accepted-tokens-per-round at B=4 from the measured 2.909 to ~3.5 would give
   ~20 t/s with verification cost unchanged.

What survives from F52: Layer 2 (cheaper verification) is still the right FIRST move, because it
raises the cap every speculative method operates under, and it is the one we have demonstrated.
What is retracted: the claim that acceptance is not worth attacking. It is -- now more than before.
Still true: do NOT port a decoder off a paper's headline multiplier, because published speculative
gains assume verification is nearly free, which sm_60 does not provide.

## F59 — KV / TURBO_AUTO_ASYMMETRIC: changes ACCEPTANCE, not verification cost
Ridge-3.7bpw, B=2 (--spec-type draft-mtp --spec-draft-n-max 1), -b/-ub 512, --ctx-checkpoints 2,
-ngl 99 --parallel 1, LD_LIBRARY_PATH=$HOME/iq2s2:$HOME/lucefix, LUCE_OP_PROFILE=1 EVERY=32,
guard LIMIT=80 (per-point logs). 15-tok prompt, 96 tokens, temp 0. No throttle, no abort, no Xid.

| point | ASYM | ctx | VRAM MiB | t/s | ms/tok | accept | acc/gen | rounds | tok/round | ms/round | start->peak |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| KV-1 | 0   |  8192 | 13505 | 14.51 | 68.92 | 0.8431 | 43/51 | 51 | 1.882 | 129.73 | 52->63C |
| KV-2 | 0   | 32768 | 13973 | 14.43 | 69.30 | 0.8431 | 43/51 | 51 | 1.882 | 130.45 | 59->?    |
| KV-3 | def | 32768 | 14045 | 13.89 | 72.01 | 0.7736 | 41/53 | 53 | 1.811 | 130.43 | 61->?    |

**THE decomposition (MEASURED): ms/round at 32K is 130.45 (ASYM=0) vs 130.43 (default) --
IDENTICAL to within 0.02 ms. So the KV type does NOT change target verification cost at all.**
The entire +3.9% throughput advantage of ASYM=0 comes from ACCEPTANCE (1.882 vs 1.811 tok/round,
+3.9% exactly). Forcing K to turbo4 instead of q8_0 changes the target's NUMERICS so the MTP head
agrees with it more often (0.8431 vs 0.7736).

**This is NOT a free win and must not be shipped as one.** Acceptance rising because K became MORE
lossy suggests the target got less discriminative -- the agreement improved because the thing being
agreed with changed. Output quality is UNVALIDATED (no perplexity run). Treat as a quality/speed
trade of unknown sign, not a speedup. It is also a clean caution for the whole project: acceptance
is only meaningful relative to a fixed target.

Other measured facts:
- **Decode is flat in resident context**: 14.51 @8K vs 14.43 @32K (-0.6%). Consistent with F52's
  reasoning that KV is a small share of read volume (only 17 of 65 blocks grow KV).
- **KV cost with ASYM=0 = 19.5 KiB/token** measured from the 8K->32K VRAM delta
  (468 MiB / 24576 tok). NOT derived from bit-width arithmetic.
- ASYM=0 saves only **72 MiB at 32K** vs default (13973 vs 14045) = ~2.25 KiB/token difference,
  much less than a naive q8_0-vs-turbo4 K calculation predicts (~4.25 KiB/tok). Mechanism unexplained.
- 128K feasibility remains **UNTESTED**: at 19.5 KiB/tok, 128K KV ~ 2.44 GiB on 11.73 GiB of
  weights ~ 14.2 GiB before compute buffers. Marginal in 16 GB. Per the standing rule, do NOT
  treat an n_ctx allocation as resident context -- this needs a real long-prompt test.
- ASYM=def bytes/token could not be computed from this batch (only one context measured);
  batch5's Ridge B=2 asym=def ctx8192 control supplies the second point.

## F60 — q41nonffn B-sweep COMPLETE: peak is B=4; the depth-1 MTP head SATURATES at ~1.9 accepted/round
Ridge-q41nonffn.gguf, -c 8192, -b/-ub 512, --ctx-checkpoints 2, -ngl 99 --parallel 1,
--cache-type-k/v turbo4, --spec-type draft-mtp --spec-draft-n-max K (B=K+1), 15-tok prompt,
96 tokens, temp 0. libs $HOME/iq2s2:$HOME/lucefix. Guard LIMIT=80, per-point logs. No abort, no Xid.

| B | K | t/s | ms/tok | accept | acc/gen | rounds | tok/round | ms/round | accepted/round | start->peak |
|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|---|
| 2 | 1 | 16.10 | 62.10 | 0.7273 | 40/55  | 55 | 1.746 | 108.40 | 0.727 | 36->52C |
| 3 | 2 | 16.46 | 60.74 | 0.6310 | 53/84  | 42 | 2.286 | 138.83 | 1.262 | 49->55C |
| 4 | 3 | **16.63** | 60.14 | 0.6162 | 61/99  | 33 | 2.909 | 174.95 | 1.848 | 51->57C |
| 5 | 4 | 14.22 | 70.30 | 0.4317 | 60/139 | 35 | 2.743 | 192.83 | 1.714 | 63->70C |
| 6 | 5 | 12.74 | 78.52 | 0.3653 | 61/167 | 33 | 2.909 | 228.43 | 1.848 | 65->72C |
| 8 | 7 |  9.89 |101.08 | 0.2793 | 62/222 | 32 | 3.000 | 303.24 | 1.938 | 67->75C |

**PEAK = B=4 at 16.63 t/s, located empirically. B=8 LOSES decisively** (9.89 t/s, below q41's own
no-spec ~13.5 at matched thermals) -- so speculation at B=8 is actively harmful.

**The B>=5 losses are TEMPERATURE-INDEPENDENT and provably so.** Acceptance at temp 0 is
deterministic -- a pure function of (model, prompt, B) -- and it is independent of clocks. At B=5,
tok/round FELL to 2.743 from B=4's 2.909 while ms/round ROSE to 192.83 from 174.95: fewer tokens
for more time is a strict loss no thermal correction can reverse. At B=6, tok/round is IDENTICAL to
B=4 (2.909) for 31% more round time. So the ranking holds despite B=5/6/8 starting 12-16C hotter
than B=2/3/4 (the plateau detector released above the 55C gate once the card had heat-soaked).

**MECHANISM (STRONG INFERENCE): the native MTP head has depth 1, so drafting K tokens runs it
autoregressively K times on its own predictions, and error compounds.** Total accepted tokens are
essentially CONSTANT across the sweep -- 61 / 60 / 61 / 62 -- while proposals grow 99 -> 139 -> 167
-> 222. **The drafter delivers ~1.85-1.94 accepted tokens per round and nothing more, at any width.**
Beyond B=4 every added column is pure verification cost with zero return. This quantifies the
Layer-1 ceiling far more sharply than the acceptance rate does: the rate falls because the
NUMERATOR IS PINNED, not because quality decays smoothly.

**Strategic consequence, and it links the two layers:** q41nonffn (Layer 2) made wide blocks
economically profitable, and the depth-1 drafter (Layer 1) is now the thing preventing us from
exploiting them. A drafter that sustains acceptance at B=4-8 -- a genuine multi-step head
(Eagle-3-style) or a small draft model -- is the next real lever. See F58.

## F61 — KV bytes/token measured both ways; ASYM=0 is what makes 128K arithmetically fit
Two contexts per policy, same model (Ridge-3.7bpw), same B=2, VRAM from nvidia-smi after health OK:

| ASYM | ctx 8192 | ctx 32768 | delta | **KiB/token** |
|---|--:|--:|--:|--:|
| default (K=q8_0, V=turbo4) | 13373 MiB | 14045 MiB | 672 MiB | **28.0** |
| 0 (K=turbo4, V=turbo4)     | 13505 MiB | 13973 MiB | 468 MiB | **19.5** |

The 28.0 KiB/token figure independently confirms the 27.75 previously on record for K=q8_0/V=turbo4.
**ASYM=0 saves 8.5 KiB/token (-30%) but costs ~200 MiB of FIXED allocation** (fitted intercepts
13349 vs 13149 MiB) -- which is why it uses MORE memory at 8K and LESS at 32K. **Break-even ~24K
tokens.** Earlier I read the 32K pair alone (72 MiB apart) as "the saving is tiny"; that was
reading a crossover point, and the two-context slopes correct it.

Linear extrapolation to 128K (**INFERENCE, and deliberately not treated as resident context**):
  default: 13149 + 3500 = **16649 MiB -> does NOT fit** in 16384 MiB
  ASYM=0 : 13349 + 2438 = **15787 MiB -> fits, ~600 MiB headroom**
=> **ASYM=0 is the setting that makes 128K feasible at all.** UNTESTED: this is a 2-point
extrapolation, compute buffers may grow with context, and per the standing rule an n_ctx
allocation is not resident context. A real long-prompt test is required before claiming 128K works.
Also unresolved: ASYM=0's acceptance/quality trade (F59) has NOT been validated by perplexity,
so 128K-via-ASYM=0 currently buys context at an unmeasured quality cost.

Decode is flat in context either way: 14.51 @8K vs 14.43 @32K (ASYM=0).
CONTROL (Ridge B=2 asym=def, ctx8192, start 68C): **13.61 t/s**, acc 0.77358 (41/53), peak 78C --
reproduces the pre-reboot 13.64 to 0.2%, validating the harness across a reboot and a rebuild.

## F62 — Thermal bracket: B=4 peak CONFIRMED under matched-hot conditions; penalty is 4.2%
B=2/3/4 ran at 36-51C starts (fresh cold boot); B=5/6/8 at 63-67C once the card heat-soaked.
Re-ran B=4 in the hot regime as a bracket so the sweep compares like with like.

| point | start->peak | t/s | acceptance | acc/gen |
|---|---|--:|--:|---|
| q41 B=4 cool | 51->57C | 16.63 | 0.61616 | 61/99 |
| q41 B=4 HOT  | 70->77C | 15.93 | 0.61616 | 61/99 |

**Acceptance is bit-identical (0.61616, 61/99) — so the 4.2% gap is purely clock/thermal.**
Third independent instance of this pattern (see F57: 5.4% at 72C vs 36C, also bit-identical
acceptance). Acceptance at temp 0 is a deterministic instrument; throughput is not.

**Matched-hot ranking: B=4 15.93 > B=5 14.22 > B=6 12.74 > B=8 9.89.**
=> **B=4 is the true peak, not an artifact of B=2/3/4 having run cold.** F60's conclusion stands
on matched conditions, in addition to the temperature-independent tok/round argument.

Working thermal calibration for this workload (INFERENCE, workload-specific, NOT a universal
constant): ~4-5% throughput per ~20C of start temperature, over the 36-77C range.

## F63 — Thermal state is OPERATOR-DEPENDENT, not a machine property (cooling is a desk fan on an open case)
2026-09-02: the P100's cooling is **a desk fan placed on the open chassis**, not a shroud
or a seated internal fan. This retroactively explains the entire thermal record and **kills my
"fan reseated" framing** (F57) as well as the cold-soak alternative -- the true variable is where
the household fan happens to be pointed and whether it is on:
  - idle floor drifting 69 -> 72 -> 74 -> 79C across one long session
  - 32C immediately after a power-down (cold-soaked card AND fan blowing on a cold heatsink)
  - equilibrium ~63-65C in the partial idle-drift log below

Partial idle-drift (ZERO load, 0 MiB, 405 MHz, 34-38 W), aborted at minute 22 as not worth
finishing once the cooling mechanism was known:
  min  0: 76C (server still resident)   min 10: 68C
  min  1: 74C                           min 15: 66C
  min  5: 71C                           min 20: 65C
  min 22: 65C  -- rate decayed 2.0 -> 0.67 -> 0.30 -> ~0.0 C/min, asymptote ~63-65C

**Consequence for methodology, and it is the important part: temperature is an UNCONTROLLED
variable on this rig.** It is not reproducible across sessions, so:
  - NEVER compare t/s across thermal states, and never "correct" with a multiplier (F57/F62)
  - ALWAYS log start and peak temp per point, with per-point guard logs
  - when a sweep spans thermal states, **re-measure a bracket point** in the later regime (F62)
  - a cooldown gate cannot guarantee a start temperature (the plateau hatch released at 63-74C
    against a 55C target), so treat the gate as best-effort and the logged temps as the truth
The 80C request-abort guard stays exactly as is; it is the only hard protection and it worked
every time it fired.

## F64 — All four n-gram drafters are INERT on a short prompt: the map indexes the PROMPT ONLY
Ridge-q41nonffn, -c 8192, 15-tok prompt, 96 tokens, temp 0, --spec-draft-n-max 3 unless noted.
Per-point guard logs. No CUDA fault, no abort, no Xid in ANY run.

| config | start->peak | t/s | #gen drafts / #calls | tau |
|---|---|--:|--:|--:|
| draft-mtp B=4 (REFERENCE) | 42->57C | **17.77** | 33/33 | 2.909 |
| ngram-map-k defaults      | 54->60C | 14.04 | **0/94** | 1.000 |
| ngram-map-k4v defaults    | 56->63C | 14.02 | **0/94** | 1.000 |
| ngram-mod (n_match=24)    | 58->65C | 13.94 | **0/94** | 1.000 |
| ngram-cache (no cachefile)| 60->67C | 13.86 | **0/94** | 1.000 |
| map-k size_n=6            | 62->69C | 13.78 | **0/94** | 1.000 |
| map-k size_n=4            | 64->71C | 13.73 | **0/94** | 1.000 |
| map-k size_n=3            | 66->73C | 13.64 | **0/94** | 1.000 |

**ROOT CAUSE, from source not inference** (`common/ngram-map.cpp`, `common/speculative.cpp`):
`common_ngram_map_begin(map, tokens)` is called ONCE per request with the **prompt**, and
`common_ngram_map_accept()` only updates value STATISTICS for the draft just used -- it never
adds new n-grams. **The map therefore indexes the prompt and NEVER the generated text.** A
15-token prompt gives an essentially empty index, so key length is irrelevant: sweeping size_n
12 -> 6 -> 4 -> 3 produced identically zero drafts. **This is NOT evidence that n-gram drafting
fails on Ridge -- it is evidence the index was empty.** Every run degenerated to no-spec.
size_n=2 and the cascade points were cancelled as uninformative for the same reason.

Also settled: the recorded "ngram-simple CUDA-faults at ggml-cuda.cu:107 on sm_60" does NOT
generalise -- four sibling impls started and ran cleanly. And `ggml-cuda.cu:107` is inside
`ggml_cuda_error()`'s `GGML_ABORT`, i.e. the generic error REPORTER, not a fault site; the real
cause was on the un-recorded preceding "CUDA error:" line. Source confirms all four n-gram
drafters are pure CPU (they take `const llama_batch & /*batch*/` and ignore it; no llama_decode,
no ctx_dft/ctx_tgt, no ggml/cuda references) => zero GPU work and zero VRAM for drafting.

## F65 — Three by-products worth more than the failed experiment
1. **q41 no-spec floor, well replicated: 13.64-14.04 t/s** across 8 degenerate runs. Against the
   same-session MTP B=4 of 17.77, **speculation is worth +26.6%** -- which retires the old
   "~17-20% by any method" doctrine (that came from Ridge's pre-q41 economics).
2. **Thermal calibration curve, from identical computations at different start temps** (each
   inert run is the same no-spec work): 54/56/58/60/62/64/66C -> 14.04/14.02/13.94/13.86/13.78/
   13.73/13.64 t/s. Monotonic, **-4.4% per 20C**. Independently reproduces the 4-5%/20C figure
   from the bit-identical-acceptance bracket pairs (F57/F62) by a completely different route.
3. **MTP draft cost MEASURED, correcting my model:** the built-in `statistics` line reports
   `dur(b,g,a) = 0.004, 385.346, 0.053 ms` over 33 rounds = **3.89 ms per draft token**, i.e.
   11.68 ms/round = only **7% of the 163.66 ms round**. I had been using 6.4 ms/token, inferred
   from the RIDGE profile's q6_K 248320 output head -- but in q41 that head is already q4_1, so
   I was double-counting a cost the requant had removed. Consequence: "make drafting free" is
   worth **+7.7%**, not the +13% I claimed. Zero-draft thresholds at B=4 (round 151.98 ms):
   **tau > 2.701 beats 17.77 t/s; tau > 3.040 gives 20 t/s** (vs MTP's measured tau 2.909).

## F66 — Champion figure restated with its thermal condition
**Ridge-q41nonffn + native MTP B=4 = 17.77 t/s from a 42C start (peak 57C)**, acceptance
0.61616 (61/99, bit-identical to the 16.63 run), 33 rounds, tau 2.909, 163.66 ms/round.
The standing 16.63 figure was the same work from a 51C start. Both peaked at 57C, so what
differs is thermal HEADROOM during the run, not peak -- start temperature is the variable to
report. Cite 17.77 only with "42C start"; never compare it to a warm-start number unbracketed.

## F67 — MTP hits tau = 4.000 (the ARITHMETIC MAX for B=4) on an echo workload: 22.18 t/s
Ridge-q41nonffn, -c 8192, B=4 (--spec-draft-n-max 3), temp 0. Prompt = ~2166-token yaml config
listing (40 upstream blocks) with "reproduce it exactly, changing only port 8080 -> 9090", so the
output must echo the input. Same binary/libs/flags as F66's prose reference; ONLY the prompt differs.

| | prose prompt (15 tok) | **echo prompt (2166 tok)** |
|---|--:|--:|
| t/s | 17.77 | **22.18** |
| ms/token | 56.26 | 45.09 |
| acceptance | 0.6162 (61/99) | **0.9722 (70/72)** |
| rounds | 33 | 24 |
| **tau (tokens/round)** | 2.909 | **4.000** |
| ms/round | 163.66 | 180.35 |
| accepted draft/round | 1.848 | 2.917 (of max 3) |
| start->peak | 42->57C | 67->78C |

**tau = 4.000 IS THE CEILING AT B=4** (3 drafted + 1 bonus). The drafter is essentially perfect
here. And this is a HOT run (67C start vs the reference's 42C); at -4.4%/20C a cold-start run of
the same config projects to **~23.5 t/s**.

**RETRACTION of F60's mechanism claim.** F60 concluded "the depth-1 MTP head delivers ~1.85-1.94
accepted tokens per round at ANY width" and attributed it to autoregressive error compounding
inside the draft block. **That is workload-specific, not architectural.** Changing only the prompt
took accepted-draft/round from 1.848 to 2.917 and acceptance from 0.616 to 0.972. The constant
"61 accepted regardless of width" in F60 should have been read as a property of that prompt --
on near-deterministic continuations there is nothing for the depth-1 head to compound errors on.
F60's *measurements* stand; its generalisation does not.

**Consequence for the Layer-1 framing:** acceptance is WORKLOAD-DETERMINED. Layer 1 is the
bottleneck on novel prose (0.616) and is NOT the bottleneck on echo-heavy work (0.972, already at
the width ceiling). So "find a better drafter" is the right goal only for the prose regime; for
code edits / document reproduction / RAG quoting / agentic transcripts the existing MTP head is
already near-perfect and the binding limit is BLOCK WIDTH B, which is a verification-cost question
(Layer 2) -- see batch10, which sweeps B=6/8/12/16 on this workload.

**PREFILL, measured at last (retires the struck F55 claim's open question):** 2166 tokens at
**121.22 t/s / 8.25 ms per token**, in line with the 116-127 t/s already on record. The struck
"25% prefill regression" was measured on 15-token prompts (warmup latency); this is a real
multi-thousand-token prefill number for q41nonffn. A matched Ridge-3.7bpw prefill run is still
NOT measured, so the q41-vs-Ridge prefill comparison remains open.

MTP draft cost reconfirmed: 283.986 ms / 24 rounds = **3.94 ms per draft token** (F65: 3.89).

## F68 — AUTONOMOUS POWER CYCLE now available; and the 71-74C "floor" was HEAT-SOAK, not the fan's limit
**Wake-on-LAN WORKS.** The Z640's onboard NIC is Intel e1000e, MAC `<P100_MAC>` (confirmed
from a crashdump kernel log, not guessed). A magic packet broadcast to <BROADCAST>:9 from
<DEV_HOST_IP> woke it; ping answered in **~80 s** and the P100 enumerated first try. No WoL tool
is installed -- the packet is 6xFF + MAC x16 sent as a plain UDP broadcast from python.
**Shutdown works too:** `sudo poweroff` (sudo requires a password on this box). So a full **run -> power off -> wake -> run cold** cycle is
available without a human at the machine.

**Cooling calibration (power OFF, desk fan left running):**
| off duration | GPU temp at boot | source |
|---|--:|---|
| ~5 min  | **59C** | measured 2026-09-02 21:59 |
| 10-15 min | **~40C** | operator experience; consistent with a 41C boot measured 15:00 |
| long/overnight | 32C | measured earlier 2026-09-02 |

**This resolves the ambiguity in F63.** A 40-minute ZERO-LOAD idle drift with the machine ON
asymptoted at **63-65C**, but **five minutes powered OFF reached 59C** and 10-15 min reaches ~40C.
Powering off removes stored heat from the heatsink mass that idling cannot, so the 69->79C floor
observed during the long session was **heat-soak, not the fan's ceiling**. Note the temperature
CANNOT be polled while off (no nvidia-smi), so the curve above is sampled at boot, one point per
power cycle -- do not expect a continuous trace.

**Operational consequence:** any configuration needing >~11 s of sustained compute (no-spec, or a
non-drafting n-gram run, both ~13 s) voided at the 80C guard from a ~70C start. From ~40C those
become measurable. Schedule anything slow, or any A/B that needs matched thermal conditions,
immediately after a 12-15 min power-off rather than trying to cool with the machine running.

## F69 — CALCULATED: the quantization path is worth ~3-4x the kernel path; and a SPLICE makes it possible
Composition of q41nonffn (from the requant record: non-FFN -> q4_1 = 213 tensors / 8.59e9 params;
FFN bit-copied as IQ2_S 160 + IQ3_S 32 tensors), against MEASURED per-format ps/weight on this P100:

  q4_1  8.59B (31.8%) @ 1.97      iq2_s 15.35B (56.9%) @ 3.05      iq3_s 3.06B (11.3%) @ 3.76
  baseline blended = 2.787 ps/weight, ~10.7 GiB of weights

| option | blended | matvec | weights GiB | fits 16 GiB | est end-to-end |
|---|--:|--:|--:|---|--:|
| baseline (as shipped)          | 2.787 | 1.000 | 10.69 | yes  | 1.000 |
| IQ3_S -> q4_1                  | 2.584 | 1.078 | 11.25 | yes  | 1.065 |
| IQ3_S -> IQ4_XS                | 2.620 | 1.064 | 10.98 | yes  | 1.053 |
| **IQ2_S -> IQ4_XS**            | 2.355 | 1.184 | 13.82 | yes  | **1.150** |
| all -> IQ4_XS                  | 2.290 | 1.217 | 13.36 | yes  | 1.176 |
| **q4_1 non-FFN + IQ4_XS FFN**  | 2.188 | 1.274 | 14.11 | tight| **~1.21** |
| all -> q4_1                    | 1.970 | 1.415 | 15.72 | **NO** | 1.327 |

CALCULATED (Amdahl on the measured ~84% MUL_MAT share), NOT measured. VRAM totals add KV at 8K
(19.5 KiB/tok measured = 0.19 GiB) + ~1.2 GiB compute buffers; `token_embd.weight` Q6_K = 1.04 GiB
is offloadable to CPU (it is a gather, not a matmul) and buys back headroom for the tight options.

**REPRIORITISATION: P3 now outranks P1/P2 on expected gain.** RPB is ~5% (n=2 study measured
-2.7% to -6.2% per type); requant is ~15-21%. Together 1.05 x 1.21 = 1.27x => echo 22.94 ->
**~29.1 t/s**, with Delta 26.9 -> ~21 ms, m ~25 ms, and the perfect-acceptance asymptote rising
32.5 -> ~40 t/s. This is the FIRST costed path that actually reaches the 30 t/s target.

**The obstacle, and the way round it.** A normal requant needs bf16 source weights (~54 GB, not
available anywhere on this rig). Re-encoding IQ2_S -> IQ4_XS from the EXISTING file would add
3.1 GiB and a second rounding for **zero quality recovery** -- precision IQ2_S discarded cannot be
restored -- so that route is speed-for-size at slightly WORSE quality.
**Instead: tensor-level SPLICE.** Take q41nonffn's non-FFN q4_1 tensors + the FFN IQ4_XS tensors
from `~/models/iq4xs/Qwen3.8-27B-UD-IQ4_XS.gguf`, which are genuine IQ4_XS derived from the
original weights => **better quality than q41nonffn AND faster**. Same arch, same tensor names.
Note UD-IQ4_XS is a MIXED UD quant (iq4_xs only 49.7% of it), so the FFN tensors must be
inspected per-tensor, not assumed uniform.

**Blocker + safe resolution:** Z640 has 4.6 GB free, output is ~14 GiB. Copy UD-IQ4_XS here first
($DATA has 102 GB; ~2 min over gigabit), verify checksum, THEN reclaim its 13.6 GB on
the Z640 -- nothing irreversible, a verified local copy exists. Splice locally, push back.
GATE BEFORE ANY BYTES MOVE: verify tensor names AND shapes match between the two GGUFs.

## F70 — Echo wide-B: PEAK AT B=6 = 27.73 t/s; marginal column cost is CONVEX; B=12 dies of VRAM
Ridge-q41nonffn, echo-682 workload (682-tok prompt the output must echo), -c 8192, temp 0,
cold-cycled start (13 min power-off -> 39C). All VALID points peaked <=64C, far from the guard.

| B | t/s | ms/tok | acceptance | acc/gen | rounds | tau | ms/round | draft ms/round | start->peak |
|--:|--:|--:|--:|---|--:|--:|--:|--:|---|
| 4 | 23.70 | 42.19 | 0.9722 | 70/72 | 24 | 4.000 | 168.75 | 11.28 | 39->56C |
| 6 | **27.73** | 36.07 | 0.9512 | 78/82 | 17 | 5.647 | 203.67 | 18.99 | 52->60C |
| 8 | 26.29 | 38.04 | 0.9111 | 82/90 | 13 | 7.385 | 280.94 | 26.52 | 56->63C |
| 12 | VOID | | | | | | | | 59->64C |

**PEAK = B=6 at 27.73 t/s (+17.0% over B=4).** Draft cost is flat at 3.76-3.80 ms/draft token
across all widths, so the shape is entirely verification + acceptance.

**The marginal column cost is CONVEX, and this corrects two of my own numbers:**
  B=4 -> B=6 : (203.67-168.75)/2 = **17.46 ms/column**
  B=6 -> B=8 : (280.94-203.67)/2 = **38.64 ms/column**  (2.2x jump)
I had been carrying m ~ 30.8 ms from the q41 PROSE fit -- three points with a 19% slope spread
that I flagged as weak and then used as a constant across a different workload anyway. On seeing
B=6 I over-corrected to 17.46 and projected 28-30 t/s at B=8-12. Both were wrong: **neither value
is a global constant; m is convex in B and workload/context dependent.** Refit per workload from
>=2 measured widths; never carry m across workloads.

**B=12 is VOID with a NON-thermal cause: CUDA error at ggml-cuda.cu:107 during decode, peak only
64C.** The real error string never reached the log (only the `cmd_child_to_router:error:` final
line survived), but ready VRAM was **14787 MiB** (vs 13591 at B=4, so +1196 MiB of wider
draft/verify buffers) plus two ~151 MiB context checkpoints ~ 15.1 GiB of 16 GiB before
decode-time allocation. **Probable OOM => B<=8 is the measurable range at ctx 8192.** Costs
nothing scientifically since the peak is B=6. Not pursued further.

**HYPOTHESIS worth one cheap test:** this may also explain the historical "ngram-simple
CUDA-faults at ggml-cuda.cu:107 on sm_60" note. `ngram-map`'s `size_m` default is **48**, so a
FIRING n-gram drafter would submit a ~49-column verification batch -- far wider than the B=12 that
just died. If both are VRAM exhaustion from wide verification batches, the historical fault was
never an n-gram bug, it was WIDTH. Testable by capping --spec-draft-n-max low while n-gram drafts.

**Distance to target from the peak:** 30 t/s = 33.33 ms/token; measured 36.07. Closing it needs
round 203.67 -> 188.2 ms at unchanged tau = **-7.6%**. That is far inside the estimated range of
work already queued (RPB kernel ~5%, splice ~15-21%), so 30 t/s no longer needs a new idea --
it needs one of the queued changes to validate.

## F71 — The wide-B block is thermally validated by its own bracket
Opening and closing B=4 points on either side of the B=6/8/12 sweep, acceptance BIT-IDENTICAL:
  R1 (opening): 23.70 t/s, 0.97222 (70/72), start 39C peak 56C
  R5 (closing): 23.52 t/s, 0.97222 (70/72), start 60C peak 68C
**Drift across the entire block = -0.76%.** The B=6 peak is +17.0% over B=4, so thermal drift is
~22x too small to explain it. **F70's peak is a real width effect, not an artifact.** This is the
bracketing discipline working as intended -- and it is why the earlier practice of comparing
against a stored number from a different thermal state was unsafe.
Draft cost across the block: 3.76 / 3.80 / 3.79 / 3.84 ms per draft token -- flat, so the entire
B-curve shape comes from verification cost and acceptance, not from drafting.

## F72 — B=6 peak REPRODUCED at 27.73/27.14 t/s; B=5 and B=7 bracket it; plus a concurrency error of mine
Reproducibility (echo-682, MTP B=6, acceptance BIT-IDENTICAL 0.95122 / 78-of-82 in both):
  R2: **27.73 t/s**, 203.67 ms/round, start 52C peak 60C
  P1: **27.14 t/s**, 208.08 ms/round, start 63C peak 66C
-2.1% between them, and the -4.4%/20C calibration predicts -2.4% for an 11C hotter start. So the
champion is REPRODUCED and the gap is thermal, not noise. **B=6 = 27.73 t/s is a real champion.**

Peak bracket:
  B=5: 26.21 t/s, acc 0.96154 (75/78), tau 4.800, 183.14 ms/round, start 68C peak 75C
  B=6: 27.14 t/s @63C  (and 27.73 @52C)
  B=7: VOID -- contaminated, re-running
Normalising B=5 from 68C to B=6's 63C start (-0.22%/C) gives ~26.50 vs 27.14, so **B=6 wins by
~2.4% at matched temperature** (CALCULATED normalisation, flagged as an estimate). Peak holds at 6.

**MY ERROR -- concurrency, and it cost one point.** I chained batch14 on `BATCH12_WIDEB_COMPLETE`,
a **mid-batch** marker, while batch12 still had its kernel-micro and n-gram-probe phases queued.
Two drivers then raced: both launch llama-server on port 8097, and `tb6_remote.sh` kills all
llama-servers on entry, so each tore down the other's run. P1/P2 finished BEFORE the overlap and
are valid; P3 (B=7) overlapped and is VOID. **Rule adopted: chain only on a driver's FINAL
completion marker, and never let two drivers exist at once -- batch15 runs every remaining phase
sequentially inside ONE driver.** The failure was silent in the sense that P3 would have produced
a plausible-looking number; it was caught by checking `ps` for concurrent drivers, not by the data.

Draft cost stays flat across the bracket: 3.85 (B=5), 3.78-3.80 (B=6) ms per draft token.

## F73 — RPB=4 at ncols 3..8 is NOT uniformly good: iq2_s REGRESSES +27% at n=5. Patch as written is unsafe.
Settled-clock A/B, arms rpbn4_2 (control, == HEAD behaviour) vs rpbn4_4, differing ONLY in
LUCE_PASCAL_RPB_N4. Shape m=4096 k=14336 (the only MUL_MAT shapes test-backend-ops generates).
>=3 s per case, both arms measured BACK-TO-BACK within each group so thermal drift cancels.
Correctness gated first: q4_1 FAIL=0/OK=45, iq2_s FAIL=0/OK=13, iq3_s FAIL=0/OK=13.

| type  | n | control us (x2) | candidate us (x2) | delta |
|---|--:|---|---|--:|
| q4_1  | 4 | 216.43 / 217.25 | 185.11 / 186.11 | **-14.40%** |
| q4_1  | 5 | 252.09 / 254.77 | 221.46 / 221.51 | **-12.61%** |
| q4_1  | 8 | 385.62 / 388.05 | 343.32 / 344.82 | **-11.06%** |
| iq2_s | 4 | 291.98 / 292.58 | 269.65 / 272.45 | -7.26% |
| iq2_s | 5 | 338.65 / 341.27 | 431.10 / 434.46 | **+27.30% REGRESSION** |
| iq2_s | 8 | 524.81 / 527.27 | 450.28 / 452.95 | -14.15% |
| iq3_s | 4 | 326.42 / 328.09 | 301.46 / 303.22 | -7.61% |
(iq3_s n=5 and n=8 not captured -- run stopped on thermal grounds, see below.)

**q4_1 gains double digits at EVERY width** (-11 to -14%), far more than the -2.7..-6.2% the n=2
study suggested -- the untuned `case 3..8: return 2` path was leaving a lot on the table.
**But iq2_s is NON-MONOTONIC with a pathological spike at exactly n=5** (+27.30%, replication
<1% in both arms, measured twice). Composition-weighted blend (q4_1 31.8%, iq2_s 56.9%):
  n=4: **-8.7% gain** | n=5: **+11.5% LOSS** | n=8: **-11.6% gain**

**CONSEQUENCE: my patch as committed is UNSAFE and must not ship.** A single constant across
ncols 3..8 for seven types would impose the n=5 regression on 57% of the model. This also
explains why upstream ships a flat `return 2` there -- the per-n behaviour is erratic enough that
a conservative uniform value avoids the pathological cases. The correct design is **per-type AND
per-n**: q4_1 -> 4 throughout; iq2_s -> 4 at n=4 and n=8, 2 at n=5.

**AND THE CHAMPION'S WIDTH IS UNMEASURABLE HERE.** B=6 runs the target forward at n=6, but
test-backend-ops only generates n in {1,2,3,4,5,8,512} -- there is no n=6 case. With iq2_s
flipping -7% -> +27% -> -14% across n=4/5/8, n=6 cannot be predicted. **Only an end-to-end A/B at
B=6 can decide it.** Note also the n=8 blend (-11.6%) suggests a tuned B=8 might reach ~28.7 t/s,
ABOVE untuned B=6's 27.73 -- i.e. the kernel change may MOVE the optimal block width, so the
end-to-end test must cover B=6 AND B=8, not one width.

**MY ERROR (thermal safety):** rewriting the harness as micro2 I dropped the `guard_req.sh`
invocation the first version had, so this ran UNGUARDED. Caught it at 71->78C with two groups
left, each adding ~8C across its four bursts and the cooldown unable to reach 62C from a
heat-soaked start -- trajectory ~80-83C against an established 84C failure point. Stopped the run
and power-cycled. The guard must be armed in EVERY harness that puts sustained load on the card.

## F74 — KERNEL CHANGE VALIDATED END-TO-END: 27.74 -> 28.94 t/s at B=6 (new champion)
Cold cycle to 42C, then interleaved control/candidate at B=6. Arms differ ONLY in
LUCE_PASCAL_RPB_N4 (2 vs 4); both built from identical source.

| point | arm | t/s | ms/round | acceptance | tau | start->peak |
|---|---|--:|--:|--:|--:|---|
| K1 | rpbn4_2 (control, == HEAD) | 27.74 | 203.56 | 0.95122 (78/82) | 5.647 | 42->58C |
| K2 | rpbn4_4 (candidate)        | **28.94** | 195.13 | 0.95122 (78/82) | 5.647 | 55->62C |

**+4.3% raw, and the candidate ran 13C HOTTER, so at matched temperature the gain is ~+7.2%
(CALCULATED via the -0.22%/C figure). A cold candidate run projects to ~29.8 t/s.**

Why this is trustworthy:
- **Acceptance is BIT-IDENTICAL** (0.95122, 78 of 82, tau 5.647 in both). Speculative decoding is
  exact, so a verification-kernel change CANNOT alter output -- there is no quality question here.
- **The control arm is a verified no-op:** K1 at 27.74 reproduces the champion measured three
  times on the ORIGINAL build (27.73 / 27.14 / 27.70). So the patched source at constant=2 behaves
  exactly like HEAD, and the K1-vs-K2 difference is attributable to the constant alone.
- Interleaved, cold-started, single driver, guard armed.

**n=6 behaves like n=4 and n=8 (gain), NOT like n=5 (+27% regression).** The microbenchmark could
not have told us this -- test-backend-ops has no n=6 case -- so the end-to-end test was necessary,
not confirmatory. F73's warning stands nonetheless: **the shipped patch must be per-type/per-n
with iq2_s pinned to 2 at n=5**, because a single constant would impose that regression wherever
n=5 occurs.

**Distance to the 30 t/s target: 3.5%** (34.55 ms/token vs 33.33 required). A cold reproduction
may reach it outright. Per the reproducibility rule, 28.94 is a CANDIDATE champion until repeated.

## F75 — Kernel A/B complete at both widths: ~7-8% consistently; B=6 stays optimal; micro overstates by ~1.7x
| B | control rpbn4_2 | candidate rpbn4_4 | raw | thermally matched (est) |
|--:|--:|--:|--:|--:|
| 6 | 27.74 (42->58C) | **28.94** (55->62C) | +4.3% | ~+7.2% |
| 8 | 26.02 (58->65C) | 27.92 (61->68C) | +7.3% | ~+8.0% |
Acceptance bit-identical within each pair (0.95122 at B=6, 0.91111 at B=8), so the change is pure
verification cost and cannot affect output.

**The optimum did NOT move.** I flagged that a kernel change might shift the best width toward
B=8 (the n=8 micro blend was the largest). Normalising K4 from 61C to K2's 55C start gives 28.29
vs 28.94, so **B=6 remains optimal with the tuned kernel.** Tested rather than assumed.

**MICROBENCHMARK-TO-END-TO-END CALIBRATION (new, useful):** I predicted, before K4 ran, that the
n=8 blend of -11.6% would give ~29.4 t/s. Measured round reduction was **-6.8%** (283.76 ->
264.47 ms) for 27.92 t/s. So **microbenchmark deltas overstate end-to-end gain by ~1.7x** on this
model -- Amdahl dilution against attention/draft/host overhead, plus the benchmark shape
(4096x14336) not being Ridge's dominant matvec (17408x5120, which test-backend-ops cannot
generate). **Divide micro deltas by ~1.7 before believing them end-to-end.** The same ratio holds
at B=6: micro blend for n=6 is unmeasurable, but the -4.1% round reduction against a q4_1/iq2_s
blend of roughly -8 to -11% is the same factor.

CHAMPION (candidate, pending cold reproduction): **28.94 t/s, echo-682, MTP B=6, tuned RPB.**
Session progression on this workload: 22.18 -> 22.94 -> 23.70 -> 27.73 -> **28.94 t/s**.

## F76 — ECHO B=6 candidate, COLD reproduction of the 28.94 champion
MEASURED 2026-09-03 00:2x UTC, batch18 R1, arm rpbn4_4, Ridge-q41nonffn, ctx 8192, K=5 (B=6).
- **29.37 t/s** (34.04 ms/tok), start 39 C, peak 55 C, VRAM 13889 MiB, acceptance 0.95122 (78/82).
- Prior champion 28.94 t/s started at 55 C. Temp calibration (-4.4%/20 C) predicts ~29.9 cold;
  measured 29.37 is 1.8% under that prediction and 1.5% over the raw prior number.
- VERDICT: **28.94 REPRODUCES**. Champion promoted from CANDIDATE to CONFIRMED.
  Best echo figure of record is now 29.37 t/s cold / 28.94 t/s warm-start.
- Acceptance IS bit-identical to the warm champion (both 78 accepted / 82 generated, per
  EXPERIMENT_LOG row K2-echo-B6-cnd), so this pair also isolates temperature: +1.5% for a 16 C
  cooler start, against +3.5% predicted by the -4.4%/20 C calibration. Reproduction confirmed;
  the calibration slightly OVER-predicts here, which is the safe direction for champion claims.

## F77 — PROSE A/B, control vs candidate arm, matched widths
MEASURED 2026-09-03, batch17 (control, rpbn4_2) + batch18 (candidate, rpbn4_4). Prose prompt,
96 tokens generated, ctx 8192. Acceptance is bit-identical between arms at every width.
| B | control t/s | cand t/s | raw d | acceptance |
|--:|--:|--:|--:|--:|
| 4 | 17.79 (39C) / 17.51 (58C) | 17.84 (51C) | +1.1% vs control mean | 0.61616 |
| 5 | 15.55 (52C) | **13.10 (54C)** | **-15.8%** | 0.43165 |
| 6 | 13.85 (55C) | 14.35 (57C) | +3.6% | 0.36527 |
- Drift bracket: B=4 open 17.79 -> close 17.51 = -1.6% across the sweep, so the +1.1% at B=4 is
  at the edge of noise; the -15.8% at B=5 is 10x the drift and bit-identical in acceptance.
- **The n=5 iq2_s regression predicted from the microbenchmark (+27%) IS REAL END-TO-END.** First
  case this session where a micro prediction was confirmed by an independent end-to-end run.
- Prose optimum is **B=4 (17.84 t/s)**, not B=6 as on echo. Prose champion effectively unchanged
  from 17.77; the kernel buys ~nothing at B=4.
- HYPOTHESIS (untested): prose spends more rounds at low acceptance, so per-round fixed cost `a`
  dominates and the marginal column cost the kernel improves is a smaller share of the total.
- CONSEQUENCE: the single-constant patch must NOT ship. Confirmed per-type/per-n requirement.

## F78 — SHIPPING A/B: which types need rows_per_block 4 at n>=3?
MEASURED 2026-09-03, batch19, 7 points, Ridge-q41nonffn, ctx 8192. Arms differ ONLY in
LUCE_PASCAL_RPB_N4_SUB4 (2 vs 4); LUCE_PASCAL_RPB_N4=4 in both. Acceptance bit-identical within
every pair. Temp-matched at the calibrated -4.4%/20 C, normalized to a 38 C start.
| point | sub4_4 raw | sub4_2 raw | sub4_4 matched | sub4_2 matched | iq2_s@4 worth |
|---|--:|--:|--:|--:|--:|
| echo  B=6 | 29.40 (38C) | 28.42 (50C) | 29.40 | 29.17 | **+0.8%** |
| prose B=5 | 13.08 (54C) | 15.85 (57C) | 13.54 | 16.51 | **-18.0%** |
| prose B=6 | 14.28 (59C) | 14.01 (61C) | 14.94 | 14.72 | **+1.5%** |
- **P1 = 29.40 t/s reproduces the 29.37 champion within 0.1% on a refactored source**, so the
  per-type split is behaviour-preserving. Third independent sighting of this figure.
- **The B=5 regression VANISHES when iq2_s is pinned to 2**: 13.08 -> 15.85 raw (+21.2%), and the
  faster arm ran 3 C HOTTER, so the true effect is larger. It also beats the all-2 control from
  batch17 (15.55), i.e. the shippable patch recovers full B=5 performance rather than merely
  avoiding the loss.
- **q4_1 carries essentially the entire kernel win.** The sub-4-bit types buy +0.8%/+1.5% at n=6
  and cost -18% at n=5.
- CORRECTION to an earlier reading of this pair: the iq2_s n=6 benefit was first quoted as +3.3%
  from raw values 12 C apart. Temperature-matched it is +0.8%. That changes the shipping decision
  from "iq2_s helps" to "iq2_s is noise at n=6 and catastrophic at n=5".
- SHIPPED as commit c2fba60ba: LUCE_PASCAL_RPB_N4 default 2 -> 4, SUB4 stays 2.

## F79 — sm_60 admitted to the FP16 fast path (upstream issue #25593) — NOT YET MEASURED HERE
Our tree gates FAST_FP16 on `cc != 610` at three sites (common.cuh:235 compile-time,
fast_fp16_available, fast_fp16_hardware_available), so sm_60 IS in the fast path. Upstream issue
#25593 reports this silently truncates FP32 math to FP16 on P100: median KL divergence 0.002298 vs
an FP32 reference and 1-in-29 flipped top tokens, on 4x P100 / CUDA 12.4. The SAME report measures
token-gen +1.5% (F16 KV) / +1.4% (Q8_0 KV) WITH the carve-out, i.e. plausibly quality AND speed.
Merged in llama-cpp-turboquant v0.3.0 (PR #212) and buun-llama-cpp (PR #80); open upstream as
bug-unconfirmed. GP100 genuinely has 2:1 FP16 (sm_61 is 1/64), so this is a precision policy
choice, not a capability correction.
Added as knob LUCE_SM60_NO_FAST_FP16 (commit 857a667d6), default 0 = no-op. Arm `fp16fix` built
against HEAD defaults. **Changes numerics -> acceptance will NOT be bit-identical between arms**,
so the acceptance-identity thermal isolation does not apply to this A/B, and RPB optima may need
re-validation on top of it. All our figures to date were measured WITH the fast path active.

## F80 — WORKLOAD CLASS: native MTP on GSM8K vs prose (batch20)
MEASURED 2026-09-03, arm sub4_4, ctx 8192, 96 tokens, Ridge-q41nonffn. tau = 1 + K*acceptance.
| point | workload | B | acceptance | tau | t/s | start C |
|---|---|--:|--:|--:|--:|--:|
| G1 | gsm8k | 4 | 0.75862 | 3.276 | 20.58 | 39 |
| G2 | prose | 4 | 0.61616 | 2.848 | 17.84 | 50 |
| G3 | gsm8k | 6 | 0.58333 | 3.917 | 20.16 | 53 |
| G4 | gsm8k | 8 | 0.55970 | 4.918 | 18.84 | ~56 |
| G5 | gsm8k | 4 | 0.75862 | 3.276 | 20.56 | ~58 |
- **G5 vs G1 bracket: -0.1% drift, acceptance bit-identical.** Best instrument stability recorded.
- G2 reproduces the prose champion EXACTLY (17.84 t/s, acc 0.61616) for the third time.
- **Our gsm8k tau at B=8 is 4.918, within 2.0% of the published native-MTP GSM8K tau of 5.02**
  (their DFlash2 run used --spec-draft-n-max 7 = B=8). The published figure DOES reproduce here.
- RETRACTED: an intermediate reading compared our B=4 tau (3.28) against their B=8 tau (5.02) and
  concluded our quantization was suppressing acceptance. tau = 1 + K*acceptance is a function of
  B; the comparison was invalid. **No evidence our 2.5 bpw FFN costs acceptance.** The splice
  should be expected to buy speed only.
- Workload effect is real but modest: at matched B=4, gsm8k acceptance 0.759 vs prose 0.616.
- **B=4 is the throughput optimum for BOTH gsm8k and prose**: tau rises monotonically with B
  (3.28 -> 3.92 -> 4.92) while t/s FALLS (20.58 -> 20.16 -> 18.84). Delta dominates the tau gain.
  This is the core reason published wide-B speculative results do not transfer to a P100.

## F81 — batch21 VOID: DFlash2 reference run produced NO data (harness packaging fault)
2026-09-03. All 7 points failed identically before loading the model:
`llama-server: error while loading shared libraries: libmtmd.so.0: cannot open shared object file`
CAUSE, three separable faults, all mine:
1. The payload tar globbed `libggml*.so* libllama*.so*`, which omits **libmtmd.so.0** (llama-server
   links the multimodal lib). build/bin has 3 libmtmd entries; the complete set is 27 files.
2. tb9_remote.sh's startup-error patterns did not include the dynamic-linker message, so instead of
   failing fast each point burned its full health-check timeout (200 x 3 s = 600 s), wasting a
   whole cold cycle -- ~50 minutes of machine time producing nothing.
3. No smoke test proved the transferred binary could execute before a cold cycle was spent on it.
FIXES: payload repackaged from `ls *.so*` (27 entries, ldd clean); tb9 now matches
"error while loading shared libraries|cannot open shared object|symbol lookup error"; the batch24
chain runs `llama-server --version` on the Z640 and ABORTS if it cannot start.
LESSON (generalizes): verify a transferred artifact by EXECUTING it, not by checksumming it. The
md5 check passed on every file that was shipped -- it cannot detect a file that was never shipped.
No measurement conclusions are affected; this batch contributed no numbers.

## F82 — sm_60 FP16 carve-out A/B (batch22): +1.6% on echo, UNRESOLVED on prose
MEASURED 2026-09-03. Arms differ ONLY in LUCE_SM60_NO_FAST_FP16 (control = sub4_2, both
N4=4/SUB4=2). ctx 8192, Ridge-q41nonffn.
Echo control bracket E1 28.53 @41C -> E5 27.69 @62C = -2.9% over 21C, giving an EMPIRICAL drift
slope of -0.0400 t/s per C (the generic -4.4%/20C calibration would have predicted -4.6%, so it
over-predicts here; prefer the in-batch bracket).

**ECHO B=6 — acceptance BIT-IDENTICAL in both arms (0.95122, 78/82):**
| arm | t/s | start C |
|---|--:|--:|
| control interpolated to 54C | 28.01 | 54 |
| fp16fix measured | **28.46** | 54 |
=> **fp16fix is +1.6%**, independently matching the external 4xP100 report of +1.4 to +1.5%.
Notable: the carve-out changes numerics, yet on echo it flipped ZERO accept/reject decisions --
echo drafts are predictable enough to be insensitive to the precision change.

**PROSE B=4 — acceptance CHANGED, result NOT attributable:**
| arm | t/s | acceptance | tau | start C |
|---|--:|--:|--:|--:|
| control | 17.89 | 0.61616 (61/99) | 2.848 | 58 |
| fp16fix | 16.79 | 0.56190 (59/105) | 2.686 | 60 |
t/s -6.1% raw at only 2 C apart, acceptance -8.8%.
**Cannot be attributed.** A 96-token prose generation is only ~33 rounds, so a handful of flipped
accept/reject decisions moves acceptance by several points. Across arms with DIFFERENT numerics,
one sample cannot separate a systematic effect from sampling variance. (Within a fixed binary our
prose runs are deterministic and reproduce exactly -- 17.84/17.89 four times -- but that
determinism does not transfer across a numerics change.)

DECISION: do NOT ship LUCE_SM60_NO_FAST_FP16 by default yet. It is a confirmed +1.6% where
acceptance is controlled, and it carries an independent quality argument (upstream #25593: 1-in-29
flipped top tokens), but its prose effect is unresolved and possibly negative. Resolve with a
MULTI-PROMPT prose panel before shipping.
METHOD NOTE (generalizes): single-prompt prose measurement is adequate for kernel changes that
preserve numerics, and INADEQUATE for anything that alters them. Any future numerics-changing
arm needs a multi-prompt panel.

## F83 — splice sizing, MEASURED by llama-quantize --dry-run (not arithmetic)
2026-09-03. Source Ridge-q41nonffn = 11924.23 MiB (3.66 BPW), 866 tensors.
| recipe (all include token_embd -> iq4_xs, ssm_* pinned q8_0) | quant size | BPW |
|---|--:|--:|
| all FFN (gate+up+down) -> iq4_xs | **14872.09 MiB** | 4.57 |
| all FFN -> q4_1 | **16425.99 MiB** | 5.04 |

CORRECTION to my own constant: **q4_1 is 5.0 bpw, not 4.5.** A q4_1 block is 32 weights = 16 B of
quants + fp16 `d` + fp16 `m` = 20 B. q4_0 is the 4.5 bpw format. Every size projection I made
using 4.5 for q4_1 was too small; the dry-run delta between the two recipes is 0.762 bits/param,
which matches 5.0 - 4.25 exactly.

VRAM ceiling, from MEASURED occupancy: total 16276 MiB, used 13889 at B=6/ctx8192 with weights
11924 => overhead (KV + compute + ctx) = **1965 MiB**, so max weights ~14161 MiB.
**Neither full recipe fits**: all-iq4_xs needs 14872 + 1965 = 16837 > 16276. Disk was also
binding (14214 MB free), so stale arm dirs are being reclaimed.

CHOSEN RECIPE (batch25): `ffn_gate` + `ffn_up` -> iq4_xs (58% of FFN by params), `ffn_down` left
at iq2_s/iq3_s, token_embd -> iq4_xs, ssm_alpha/ssm_beta/ssm_conv1d pinned q8_0 so the base ftype
cannot silently downgrade the 48 Gated-DeltaNet layers.
  projected weights 13644 MiB (517 MiB spare), ps/weight 2.7884 -> 2.4240 = **1.150x**
  => PROJECTED echo 33.8, prose 20.5 t/s. Revised down from the earlier 1.26-1.33x claims, which
  assumed recipes that do not fit in 16 GB.

## F84 — METHOD FIX, and it revises F82: compare ROUND TIME when numerics change
Round time = 1000 * tau / t_per_s. It is the pure verification-cost measure and the quantity the
ps/weight model actually predicts. t/s conflates verification cost with a sampling-sensitive
acceptance draw, so t/s is only a valid A/B metric when acceptance is IDENTICAL between arms.

Re-analysing the batch22 fp16 carve-out with round time:
| point | t/s | tau | round ms |
|---|--:|--:|--:|
| echo B=6 control (interp to 54C) | 28.01 | 5.647 | 201.6 |
| echo B=6 fp16fix | 28.46 | 5.647 | 198.4 |
| prose B=4 control | 17.89 | 2.848 | 159.2 |
| prose B=4 fp16fix | 16.79 | 2.686 | 160.0 |

  echo  round time **-1.6%** (acceptance identical, so equal to the t/s delta)
  prose round time **+0.5%** -- against **-6.1% in t/s**

**REVISES F82's prose reading.** The prose t/s loss was ENTIRELY an acceptance artifact; the
verification kernel is essentially unchanged there (+0.5%, inside noise). So the carve-out is a
~1% verification-speed change overall (-1.6% echo, +0.5% prose), NOT a -6% prose regression.
That removes the speed objection to shipping it; what remains is only the question of whether the
acceptance draw is systematically worse on prose, which still needs a multi-prompt panel because
one 96-token generation is ~33 rounds.

APPLIES TO batch25: the splice changes model weights, so acceptance may shift and its prose t/s
would be unattributable. **Round time is the valid comparison for the splice**, and it is also
exactly what the 1.150x ps/weight projection predicts. Judge the splice on round time first, then
report t/s separately as the practical outcome.

## F85 — DFLASH2 vs NATIVE MTP on a single P100 (batch24) — THE ANSWER
MEASURED 2026-09-03. Upstream master binary (67a17c17c, NO Pascal mmvq table), q8_0 KV, ctx 8192,
Ridge-q41nonffn target, drafter Qwen3.8-27B-DFlash2-Q4_K_M (1.14 GB, resident).
Drafter initialised correctly: `block_size=8, mask_token_id=248070, n_extract=5,
sample_from_anchor=true`. tau is llama.cpp's reported "mean len" = acceptance length = our tau.
VRAM: drafter costs +1200 MiB (12389 -> 13589 at B=4; 12987 -> 14187 at B=8). Fits at ctx 8192.

| workload | B | drafter | acceptance | tau | t/s | round ms |
|---|--:|---|--:|--:|--:|--:|
| prose | 4 | MTP     | 0.59804 | 2.79 | 18.88 | 147.8 |
| prose | 4 | DFlash2 | 0.57692 | 2.71 | 16.88 | 160.5 |
| prose | 8 | MTP     | 0.35484 | 3.36 | 12.78 | 262.9 |
| prose | 8 | DFlash2 | 0.24280 | 2.64 |  9.89 | 266.9 |
| gsm8k | 8 | MTP     | 0.67544 | 5.53 | 20.45 | 270.4 |
| gsm8k | 8 | DFlash2 | 0.82653 | **6.79** | **24.26** | 279.9 |

| comparison | tau | t/s | round time |
|---|--:|--:|--:|
| prose B=4 | -2.9% | -10.6% | +8.6% |
| prose B=8 | -21.4% | -22.6% | +1.5% |
| **gsm8k B=8** | **+22.8%** | **+18.6%** | +3.5% |

Drift bracket D1 -> D7: 18.88 @42C -> 18.18 @68C = -3.7% over 26C, so the signs above are safe.

### VERDICT
**DFlash2 does NOT help novel prose. At B=4 its tau is 2.71 vs native MTP's 2.79 -- no
improvement at all -- and the 2B drafter forward costs +8.6% in round time, so prose loses 10.6%.**
On gsm8k at B=8 it is a LARGE genuine win: tau 6.79 vs 5.53 (+22.8%), net +18.6% t/s, and that
6.79 exceeds the published DFlash2 acceptance length of 5.46 (different prompt/config).
The drafter's price is +1.5 to +8.6% of round time; the question is only whether tau pays for it.

This closes the question the whole investigation was for: **the block-diffusion drafter cannot
predict novel prose any better than the MTP head we already have.** It is strong independent
confirmation of the workload-ceiling conclusion (F80), from a completely different drafting
algorithm. Prose tau is a property of the text, not of the drafter.
DFlash2 IS worth keeping in mind for structured/math/agentic workloads (+18.6%), which is a real
result for a different use case than ours.

## F86 — UNEXPECTED: upstream master is FASTER than our tuned fork on prose
Same batch, D1 vs our champion. Both native MTP, prose, B=4, ctx 8192, same target GGUF.
| binary | KV | tau | t/s | start C | round ms |
|---|---|--:|--:|--:|--:|
| ours (c2fba60ba, Pascal table + all RPB work) | turbo4 | 2.848 | 17.89 | 58 | 159.2 |
| ours, same, cooler point | turbo4 | 2.848 | 17.84 | 50 | 159.6 |
| **upstream master 67a17c17c (NO Pascal table)** | q8_0 | 2.79 | **18.88** | 42 | **147.8** |
Temperature-matching ours to 42C via the measured -0.0400 t/s/C slope gives ours ~18.53 t/s =
153.7 ms round, so **master is ~3.8% FASTER in round time despite lacking every Pascal win we
measured.** Three candidate causes, not yet separated:
 (a) turbo4 KV costs more than it saves at ctx 8192 (decompression work per token),
 (b) upstream kernel/scheduling improvements since our merge base exceed our Pascal gains,
 (c) q8_0 vs turbo4 changes the KV read volume.
DECIDING EXPERIMENT: run OUR binary with q8_0 KV, prose B=4, cold. That isolates (a)+(c) from (b)
in one point. If turbo4 is the cost, we get a free ~4% by switching KV type at 8K context.
If master is genuinely faster, the right move is to port our Pascal table ONTO master -- which
would also bring DFlash2 and DSpark along for free.

## F87 -- SPLICE1 VOID: my recipe let the base ftype rewrite ffn_down (2026-09-03)

batch25 built `Ridge-splice1.gguf` and every splice point OOMed. Cause is entirely mine, not the
hardware: `llama-quantize ... q4_1 16` applies the BASE ftype to every tensor not matched by a
`--tensor-type` pattern. I passed patterns for ffn_gate/ffn_up/ssm_* only, so **ffn_down (5.70e9
params sitting at iq2_s/iq3_s) was requantized up to q4_1 @5.0 bpw.**

  measured result: 3.66 BPW / 11924.23 MiB  ->  4.73 BPW / 15390.05 MiB   (+3466 MiB)
  VRAM ceiling for weights at B=6/ctx8192: ~14161 MiB
  failure mode: `cudaMalloc failed: out of memory` on a 505.00 MiB compute buffer, then SIGSEGV

Three of six points (S2/S4/S5, all splice) produced no data. Splice1 deleted; 15.6 GB reclaimed.

**Lesson (generalises the batch21 one).** batch21 taught: verify a transferred artifact by
EXECUTING it, not by checksumming it. F87 adds: verify a GENERATED artifact by reading its
contents back, not by trusting the flags you passed. The requant log printed the 4.73 BPW figure
and the script's own size gate (`>12e9`) passed it, because the gate only tested for truncation.

### What DID come out of batch25 (the champion controls ran)

| pt | workload | B | start C | acceptance | tau | t/s | round ms |
|----|----------|--:|--------:|-----------:|----:|----:|---------:|
| S1 | echo  | 6 | 43 | 0.95122 (78/82) | 5.756 | 28.48 | 202.1 |
| S3 | prose | 4 | 63 | 0.61616 (61/99) | 2.848 | 17.42 | 163.5 |

S1 acceptance is the fourth bit-identical sighting of 78/82; S3 acceptance is bit-identical to
batch22 E3 (0.61616) at the same B. The champion is reproducing exactly. S3 at 63 C corrects to
~18.26 t/s at 42 C on the -0.0400 t/s/C slope -- consistent with the 17.89/17.84 pair.

S6 (echo closing bracket) was ABORTED by the thermal guard at exactly 80 C after five points with
no power cycle between them. Guard behaved correctly, `dmesg | grep -ci xid` = 0. Method fix for
batch26: three points per cold cycle, never six.

## F88 -- turbo4 K is silently upgraded to q8_0, so F86 has only ONE variable left (2026-09-03)

Visible in every batch25 server log:

    llama_kv_cache: auto-asymmetric: GQA ratio 6:1 (n_head=24, n_head_kv=4)
      -- upgrading K from turbo4 to q8_0 to prevent quality degradation.
      Disable with TURBO_AUTO_ASYMMETRIC=0

So `--cache-type-k turbo4` on Ridge has never actually been turbo4. Our fork and upstream master
both run **K = q8_0**; the only KV difference between the F86 arms is **V** (turbo4 vs q8_0).
That shrinks candidate causes (a) and (c) for master's 3.8% round-time lead to a single V-cache
read-volume term, and makes cause (b) -- genuine upstream kernel/scheduling improvements -- the
leading explanation. Raises the value of porting the Pascal mmvq table ONTO master.

## F89 -- SPLICE2: sized from the measured per-tensor table (2026-09-03, queued)

Built `gguf_tensors.py` (parses the GGUF header directly; no python-gguf on the Z640) and dumped
the champion's true composition. Corrects the percentages used in F23:

| type | cnt | params | MiB | % size |
|---|--:|--:|--:|--:|
| q4_1  | 213 | 8.5905e9  | 5120.3 | 43.33% |
| iq2_s | 160 | 14.2606e9 | 4250.0 | 35.96% |
| q6_K  |   5 | 1.5912e9  | 1244.8 | 10.53% |
| iq3_s |  32 | 2.8521e9  | 1168.8 |  9.89% |
| q8_0  |  96 | 0.0236e9  |   23.9 |  0.20% |
| f32   | 360 | 0.0027e9  |   10.2 |  0.09% |

Layer map: blk 0-3 and 60-63 all-iq3_s; blk 4-11 ffn_down iq3_s, gate/up iq2_s; blk 12-59 all
iq2_s; **blk.64 (the MTP head) is q6_K** and must be pinned -- acceptance depends on it.
Matmul-weighted ps/weight recomputed from this table = **2.7893** (confirms the 2.7884 figure).

Enumerated size/speed options against the 14161 MiB ceiling (11924 baseline, 2237 slack):

| option | change | MiB | fits | ps/weight | speedup |
|---|---|--:|:--:|--:|--:|
| A | gate+up->iq4_xs, embd->iq4_xs | 13656 | yes | 2.4980 | 1.117x |
| B | A + all ffn_down->iq4_xs | 14549 | **NO** | 2.2126 | 1.261x |
| D | A + iq3_s->iq4_xs | 13932 | tight | 2.3372 | 1.193x |
| **D+** | **D + output.weight->iq4_xs** | **13819** | **yes, 342 MiB spare** | **2.3526** | **1.1857x** |

Option B is the 1.261x figure the earlier stack projection used -- **it does not fit**, and that
projection must be revised down. Shipping D+ (4.211 bpw). At MUL_MAT = 84% of round time the
end-to-end prediction is **1.152x**: prose 17.89 -> ~20.6, echo 29.4 -> ~33.9. PROJECTED.

Recipe is a `--tensor-type-file` with an explicit line for **all 506 quantized tensors**, generated
from the champion file's own types, so nothing is left to the base ftype's mixture heuristics.
batch26 verifies composition by reading the OUTPUT file back before it spends a cold cycle.

## F90 -- F89's D+ recipe is UNBUILDABLE; two bugs, and the real menu (2026-09-03)

batch26 failed in 4 seconds, before spending a cold cycle -- the readback gate worked. Two bugs:

**(1) `--tensor-type` names are `std::regex` + `regex_search`, not exact names.**
`src/llama-quant.cpp:188-199` compiles each pattern with `std::regex`; matching is a SEARCH, so my
line `output.weight=iq4_xs` also matched `blk.11.attn_output.weight` (and the `.` matched any char).
Visible in the log: `blk.11.attn_output.weight - applying manual override: q4_1 -> iq4_xs`.
All 506 lines are now anchored `^name$` with dots escaped.

**(2) IQ2_S REQUIRES AN IMATRIX and we have none.** `tensor_requires_imatrix()`,
`src/llama-quant.cpp:822-841`: IQ3_XXS/IQ2_XXS/IQ2_XS/IQ2_S/IQ1_M/IQ1_S always return true;
**Q2_K returns true only for a Q2_K_S file**; everything else false. token_embd and output.weight
are exempt entirely (line 823). Consequence: **"leave ffn_down alone" is not expressible.** Every
tensor in the file is requantized, so the 48 iq2_s ffn_down layers must be re-emitted as SOME
imatrix-free type. Q2_K (2.5625 bpw) is the only option at or below 2.6 bpw.

This kills F89's Option D+ and every option that keeps iq2_s. Revised menu, **sizes MEASURED by
`--dry-run`, not computed**:

| recipe | composition | quant size | BPW | ps/weight | speedup | headroom |
|---|---|--:|--:|--:|--:|--:|
| **E** | uniform iq4_xs; blk.64 MTP + nextn.eh_proj pinned q6_K; ssm_alpha/beta q8_0 | **13950.70 MiB** | 4.28 | 2.3178 | **1.2034x** | ~270-360 MiB |
| **F** | E but ffn_down blk 12-59 -> q2_K | **13121.95 MiB** | 4.03 | ~2.4344 | ~1.146x | ~1100 MiB |

E is the faster recipe and is uniform (one FFN mmvq type, so the SUB4 knob stops mattering), but
its headroom is inside the error bar of my 14161 MiB ceiling estimate. F is certain to fit but its
q2_K ps/weight is UNMEASURED and its ffn_down quality is worse than the imatrix-quantized iq2_s it
replaces. batch27 therefore builds E and **load-tests it warm** -- no cold cycle at risk -- and
falls back to F only if the compute buffer will not allocate. Disk holds one 14.6 GB file at a time.

## F91 -- ROOFLINE: we are at 21% of memory bandwidth, so we are ARITHMETIC-bound (2026-09-03)

Prompted by an external result (Mads Henrichsen / syv.ai, Qwen3.8 on a single RTX 3090 24GB,
250 W cap, 46 t/s no-spec -> 133 t/s single-user -> 381 t/s with prompt-lookup drafting). The
decisive datum in it is not a speedup: a stranger's 4090 with NO power cap ran his identical stack
**1.9% faster**, tracking that card's ~8% bandwidth advantage. He is pinned to the roofline --
15 GB at ~900 GB/s = 61 passes/s, and his 46 t/s no-spec figure is 75% of it.

Same arithmetic on ours:

    weights           11924.23 MiB = 12.504 GB
    P100 16GB HBM2    ~732 GB/s
    one weight pass   12.504 / 732 = 17.1 ms   <- bandwidth roofline
    MEASURED B=1 round intercept        81.0 ms
    ratio                               4.74x off roofline  (21% of peak bandwidth)

**We are not memory-bound. We are arithmetic/kernel-bound on dequantization.** ~64 ms of the 81 ms
intercept is instruction cost, not bytes. This is independently consistent with two of our own
measurements that otherwise look paradoxical: q4_1 at **5.0** bpw is faster per weight (1.97) than
iq2_s at **2.5** bpw (3.05), and the sm_60 mmvq path is 81,883 XMAD.

Consequences, and they are large:
- **Further bpw reduction is worth nothing to us.** Every remaining bpw-minimising idea is dead.
  The splice going UP in bit-width (3.66 -> 4.28) is the correct direction, not a compromise.
- Rank candidate optimisations by whether they cut INSTRUCTIONS or raise SM OCCUPANCY, never by
  bytes moved. Judge every external result by which regime it came from first.
- His single largest lever (+~200 t/s: int8 activations onto the integer tensor cores) aims at
  exactly our bottleneck class and is **unavailable** -- sm_60 has no int8 tensor cores and no
  `__dp4a` (sm_61+) -- and it is a batch-64 compute-bound win while we run --parallel 1.

Transferable from his nine, ranked: (1) split-KV/flash-decoding attention for the verify step --
his standard kernel parallelised only for a single query, so a B-query verify ran on 24 of 82 SMs
and a replacement kernel more than halved that layer; note sm_60 gets no FA path at all, so our
attention is explicit KQ/KQV **MUL_MAT** and therefore sits INSIDE the 84% MUL_MAT share, though
bounded by only 17 of 65 layers being attention. (2) reduced draft vocabulary -- lossless w.r.t.
the target distribution, no special hardware, 97.12% vs 92% coverage worth ~10%, and it attacks a
cost we already measured (the output head runs 3x per MTP round on 1.271e9 params). (3) GDN
recurrent state fp32->fp16 -- irrelevant for concurrency at --parallel 1 but ~75 MB of VRAM
HEADROOM, which is precisely what gates our faster splice recipe. (4) prompt-lookup drafting --
llama.cpp's prompt-only n-gram map already IS this.

## F92 -- quality gate added: acceptance identity cannot detect fluent garbage (2026-09-03)

His int8 path "benchmarked beautifully and served garbage" for an hour because a kernel read its
scaling factors as unsigned while half were negative. Our gate has the same hole: we verify
acceptance bit-identity across runs (78/82 four times, 0.61616 twice) but have **never checked the
generated text**, and acceptance identity would not catch a quantization that emits fluent nonsense.
This matters now because the splice rewrites every tensor in the file.

`qgate.py` (deployed to the Z640) reads the `tb4_resp_<TAG>.json` the harness already saves and
judges the TEXT: printable ratio, English stopword rate, unique-word ratio, longest character run,
and for the GSM8K point a **semantic** check -- Natalia sold 48 in April and half as many in May,
so the completion must contain 72. No speed number from the splice is reportable until this passes.

## F93 -- the matmul-dispatch explanations for F86 are ELIMINATED in source (2026-09-03)

The four subagents assigned to the F86 question and the external verification-optimization sweep
all died on an org monthly spend limit, so I did the source work directly and cheaply. Result: the
obvious candidates are dead, and one candidate survives.

**Both trees have all four matmul kernels** -- mmvq.cu, mmq.cu, mmvf.cu, mmf.cu. Our fork is not
an ancient base; it already carries the mul_mat_vec_f fusion. CONFIRMED-IN-SOURCE.

**MMQ never runs on our card, in either tree.** `ggml_cuda_should_use_mmq()`
(master mmq.cu:259-320): when `highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A` -- true for sm_60,
dp4a is sm_61+ -- it returns `cc >= GGML_CUDA_CC_PASCAL && n_experts > 0`. Ridge is DENSE, so
n_experts = 0 and **MMQ is unreachable for us**. Any idea that routes through MMQ is inapplicable.

**MMVQ does run at B=4..8 in master**, so a ported Pascal table would take effect:
`MMVQ_MAX_BATCH_SIZE = 8` (mmvq.cuh:3) and `ggml_cuda_should_use_mmvq()` (mmvq.cu:289-328) special-
cases only Ada, Blackwell, DGX Spark and CDNA; Pascal falls through to `ne11 <= 8`.

**Fusion is not the cause.** `ggml_cuda_should_fuse_mul_mat_vec_q()` is explicitly disabled on our
arch in BOTH trees -- `if (cc <= GGML_CUDA_CC_PASCAL) return false;` with the comment "fusion is not
universally faster on Pascal" (master ggml-cuda.cu:1803, fork ggml-cuda.cu:2896). And both fusion
predicates bail unless `dst->ne[1] == 1`, so at B=4 fusion is off regardless.

**CUDA graphs are not the cause.** Both trees have structurally identical graph capture/replay
(`use_cuda_graph`, `ggml_cuda_graph_update_required`, `ggml_cuda_graph_evaluate_and_capture`,
`disable_due_to_gpu_arch`). The fork carries one EXTRA disable -- split buffers -- which cannot
fire on a single GPU.

**And master's rows-per-block is WORSE than ours at B=4, not better.** master's
`calc_rows_per_block(ncols_dst, table_id, small_k, nwarps)` (mmvq.cu:524-542) has no `type`
argument and no Pascal table: Pascal maps to MMVQ_PARAMETERS_GENERIC, which returns **2** for
ncols_dst 2..8 and 1 for ncols_dst 1. Our fork returns **4** for q4_1/iq4_xs/q4_K/q5_K/q6_K at
n>=3 and 2 at n=1. So on the mmvq axis our fork should be FASTER at B=4. The anomaly is real and
is not coming from here.

**Surviving candidate: the turbo4 V cache, which is unique to our fork.** K is auto-upgraded to
q8_0 in both trees (F88), so V is the only remaining difference, and a per-layer per-token V
dequant that master does not perform is the one mechanism our fork has and master does not.
batch28 P2 measures exactly this: same binary, prose B=4, V=turbo4 vs V=q8_0, champion-bracketed.
If V=q8_0 recovers ~147.8 ms, the 3.8% is ours to reclaim with a flag. If it does not, master is
genuinely faster for reasons outside the matmul path and the Pascal table should be ported ONTO
master -- which would bring DFlash2 and DSpark along.

## F94 -- recipe E BUILDS and INITIALISES but OOMs during prefill (2026-09-03)

Build was exact. Readback matched the `--dry-run` projection to the megabyte -- 13950.70 MiB,
4.28 BPW, iq4_xs 406 tensors / 97.96%, q6_K 4 / 1.79% (the blk.64 MTP head still pinned), q8_0 96,
f32 360. So the tensor-type-file method is now validated end to end.

Echo B=6 / ctx 8192 / -ub 512:

    == u_e6_ch  (champion, 40 C)  28.58 t/s  78/82  tau 5.756  round 201.4 ms
    == u_e6_sp  (splice,   52 C)  ready VRAM=16267 MiB ... then:
         srv main: server is listening on http://127.0.0.1:8097
         ggml-cuda.cu:107: CUDA error
         CUDA error: out of memory
    == u_e6_chc (champion, 59 C)  27.81 t/s  78/82  tau 5.756  round 207.0 ms

**The warm load test was not sufficient, and this sharpens the F87 lesson.** `server is listening`
proves the init-time buffers allocated -- 16267 of 16384 MiB, only **117 MiB free**, itself tighter
than my 270-360 MiB estimate -- but it does NOT prove the prefill graph allocates. The echo prompt
is 682 tokens of YAML, so its prefill runs the -ub 512 path and needs more than 117 MiB. A load
test must RUN A COMPLETION, not merely reach listening. Fixed for the next round.

So recipe E does not fit at B=6 / ctx 8192 / **-ub 512**. It is not yet known whether it fits at
-ub 256 or 128, which is precisely what batch28 P1 was queued to answer -- that probe is now on the
critical path rather than a side inquiry. Decode runs at batch B, not n_ubatch, so a smaller -ub
should cost PREFILL throughput only. Any -ub comparison must re-run the CHAMPION at the same -ub;
splice@256 vs champion@512 would be confounded. tb11_remote.sh adds the -ub parameter.

Prose B=4 has a 15-token prompt and a narrower verify batch, so cycle 2 may still return the prose
A/B -- the more important of the two workloads.

Champion bracket this cycle: 28.58 @ 40 C -> 27.81 @ 59 C = -0.77 t/s over 19 C = **-0.0405 t/s/C**,
independently reproducing the -0.0400 slope from batch22 to within 1.3%. Acceptance 78/82 both
ends: fifth and sixth bit-identical sightings.

## F95 -- qgate false-positived on the echo workload (2026-09-03)

First run of the new quality gate FAILED both echo points: stopword rate 0.000, 18 words. Cause is
the gate, not the model -- the echo prompt is a YAML configuration file and the completion copies
it verbatim, so English function words are absent BY DESIGN. The prose-likeness heuristic is simply
wrong for a copy task. Gate patched to drop the stopword test for `echo` and both points now PASS.
Worth recording as a gate-calibration failure rather than a model failure: a gate that cries wolf
on the champion would have been discarded within a day, taking the real check with it.

## F96 -- recipe E does not fit at -ub 512 at EITHER B, and the missing 505 MiB is the prefill graph

| point | model | B | start C | ready VRAM | free | outcome |
|---|---|--:|--:|--:|--:|---|
| u_e6_ch  | champion | 6 | 40 | 13889 | 2495 | 28.58 t/s, 78/82, round 201.4 ms |
| u_e6_sp  | SPLICE   | 6 | 52 | **16267** | **117** | listened, then `CUDA error: out of memory` |
| u_e6_chc | champion | 6 | 59 | 13889 | 2495 | 27.81 t/s, 78/82, round 207.0 ms |
| u_p4_ch  | champion | 4 | 41 | 13591 | 2793 | **18.12 t/s, 0.61616, tau 2.848, round 157.2 ms** |
| u_p4_sp  | SPLICE   | 4 | 53 | **15969** | **415** | listened, then `CUDA error: out of memory` |

So it is not a B=6-only problem: even a 15-token prose prompt at B=4 with 415 MiB free dies. The
culprit is the ~505 MiB compute buffer batch25 already named -- `graph_reserve: failed to allocate
compute buffers`, `allocating 505.00 MiB` -- and it is allocated LAZILY at the first decode, not at
init. That is why "server is listening" is worthless as a fit test: at init the splice looked fine
with 117 MiB (B=6) and 415 MiB (B=4) spare.

Empirical ceiling, derived rather than assumed: splice ready 15969 + ~505 = 16474 > 16384, i.e.
**short by ~90 MiB at B=4** and by ~390 at B=6. Also note the splice's ready-time overhead is
2008 MiB against the champion's 1667 at the same B, so overhead is not weight-independent and my
14161 MiB ceiling was itself too generous.

Two ways out, and they differ in what they cost:
- **-ub 256** halves the prefill graph (~505 -> ~253 MiB), freeing ~252 MiB. Costs PREFILL
  throughput only, since decode runs at batch B. Free, reversible, and testable -- batch28 P1 now
  runs a real 682-token completion (patched) so it can actually answer this.
- **a smaller recipe**: ffn_down blk 12-35 -> q2_K saves ~430 MiB (13520 MiB total) and keeps three
  quarters of the iq4_xs win. Held in reserve; batch29 names it explicitly if no -ub works.

batch29 is queued and ADAPTIVE: it reads batch28 P1, picks the largest -ub at which the splice
actually RAN, and re-runs prose+echo with the CHAMPION AT THE SAME -ub. A splice@256 vs
champion@512 comparison would be confounded and is exactly the mistake to avoid here.

Champion prose at 41 C = 18.12 t/s / round 157.2 ms is now the cleanest cold prose control we have
(previous: 17.89 at 159.2 ms, 17.42 at 163.5 ms warm). Acceptance 0.61616 for the seventh time.

## F97 -- the thermal slope is PROPORTIONAL, not absolute; and F86's gap is 5.9%, not 3.8%

Two brackets from batch27, same protocol, different workloads:

    echo  B=6:  28.58 @ 40 C -> 27.81 @ 59 C   -0.77 over 19 C = -0.0405 t/s/C = -0.142 %/C
    prose B=4:  18.12 @ 41 C -> 17.70 @ 59 C   -0.42 over 18 C = -0.0233 t/s/C = -0.129 %/C

The ABSOLUTE slopes differ by 74%; the PROPORTIONAL slopes agree to within 10%. So thermal drift is
a percentage of throughput, not a fixed t/s offset, and **the -0.0400 t/s/C constant is only valid
near the echo operating point.** Applying it to prose over-corrects by ~70%. Use
**-0.13 %/C (range -0.129 to -0.142)** and scale by the arm's own t/s. Every prose figure I
temperature-corrected with -0.0400 needs revisiting; the echo figures are unaffected.

**Which forces a correction to F86.** I had corrected our fork's prose 17.89 up to ~18.53 with the
absolute slope and called master 3.8% faster. That correction was too aggressive -- and it is now
unnecessary, because batch27 gives a **correction-free comparison at the same temperature**:

| arm | binary | KV | start C | acceptance | tau | t/s | round ms |
|---|---|---|--:|--:|--:|--:|--:|
| F86 | upstream master | K q8_0 / V q8_0 | 42 | 0.59804 | 2.794 | 18.88 | **148.0** |
| U4  | our fork | K q8_0 / V turbo4 | 41 | 0.61616 | 2.848 | 18.12 | **157.2** |

**Master is 5.9% faster per verification round, not 3.8%**, measured cold at matched temperature
with no correction applied. Acceptance differs between the arms (0.598 vs 0.616) because the KV
numerics differ, which is exactly why round time and not t/s is the comparison metric here.

The gap is therefore larger than I reported and still unexplained by anything in the matmul
dispatch (F93). The surviving candidate remains our turbo4 V cache, and batch28 P2 measures it
directly. Note the two arms above differ in V type as well as binary, so P2 -- same binary, V the
only variable -- is the experiment that separates them.

Champion prose acceptance 0.61616 for the eighth consecutive sighting (U4 and U6).

## F98 -- DFlash2 external evidence: our prose result is CORROBORATED, but our B=8 arm was misconfigured twice (2026-09-03)

External sweep (llama.cpp PR #27342 and #25173, inco.ai blog, alphaxiv 2602.06036, prismix/lukaLLM,
HF model cards). Three separate conclusions, and they do not all point the same way.

**(1) Every headline number is against NO SPECULATION.** inco.ai 2.7-3.4x, DFlash paper "6x
lossless", prismix 2.26x, DimInfer DSpark 1.69-2.51x, PR #27342 1.81x -- all baselined on
autoregressive decoding. We already bank +17.4% from native MTP, so none of that was ever
available to us. The only honest DFlash2-vs-MTP figures that exist anywhere:
  H200 BF16 n=7   MTP 90.01 -> DFlash2 104.83 t/s   +16.5%   (MEASURED, PR author)
  Strix Halo      code +12.6%, prose +3% ("inside noise")    (MEASURED, Zeldatales)
  RTX 3090        overall acceptance 0.6317 -> 0.6153        (MEASURED, treo) = DFlash2 WORSE
inco.ai's own like-for-like table: Qwen3.8-27B mean acceptance length MTP 4.28 -> DFlash2 4.80,
i.e. **+12%, not 3x**. And DSpark is BELOW MTP there (3.62).

**(2) Our prose conclusion is independently reproduced.** treo's 11-category panel on a 3090 has
DFlash2 winning coding (+0.064) and losing humanities (-0.047), qa (-0.055), multilingual (-0.246),
flat on writing and roleplay. lukaLLM measured acceptance 60-81% on repetitive/code contexts vs
**32% on creative prose**. No published source shows DFlash2 beating native MTP on prose by a
margin clearing noise on a consumer GPU. The single prose-favourable datapoint is H200/BF16/n=7 --
the configuration our hardware cannot afford.

**(3) But the B=8 arm had a SECOND defect, bigger than p_min: n_max should be 5, not 7.**
Five independent rigs converge on it -- Zeldatales' 7-point deterministic sweep, svgop ("5 peak,
6 erratic, 7 -5%"), lukaLLM ("n_max=7 leaves ~11% on the table"), treo ("maxn=4, 7 was slower"),
and the PR author's own video ("peak around 5"). The drafter's metadata is block_size=8/n_extract=5.
So our B=8 arm paid **3 extra verify columns for lattice slots the drafter was never trained to
fill**. On flat-verify Blackwell that costs 5-11%; on our linear T(B)=81.0+35.7(B-1) it costs far
more. **The 0.243-acceptance / 9.89 t/s prose figure is probably a measurement of over-drafting,
not of DFlash2.** F85's B=8 rows are hereby marked SUSPECT pending re-measurement at B=6.

## F99 -- p_min: the two published negatives DO NOT APPLY to us, and MTP honours it too

The operator's question -- if it drafts 7 regardless, we pay for the rejected ones, so truncation should
pay -- is mechanically right, and the saving is bigger than it first looks: DFlash2 is block
diffusion, so the DRAFTER cost is one decode regardless of proposal count. **The entire saving
lands on the target's verify columns**, which is our expensive linear term (~26.9-35.7 ms each).

Both published p_min results are negative, and both explicitly blame the same thing:
  lukaLLM (Blackwell): "draft width falls but acceptance climbs equally... +/-0.8% below a 4.1%
    noise floor."
  svgop (RTX 5090): "confidence truncation measured strictly WORSE (76 -> 45 t/s) -- **verify
    batches are flat**, so wasted tail tokens cost ~nothing while truncated good drafts force more
    passes. Including it as an option for **CPU-bound lanes only**."

**Their premise is false on sm_60.** We measured T(B) = 81.0 + 35.7(B-1): at B=4, 44% of the round
is width cost. Flat-verify hardware gets tail tokens nearly free; we pay full price for every one.
We are precisely the "CPU-bound lane" svgop describes, and nobody has published this measurement.

**And the bigger find: native MTP honours p_min too** (common/speculative.cpp:1680, inside
`common_speculative_impl_draft_mtp`). Default is 0.0 (common/common.h:330), so **our champion has
never had adaptive draft width enabled either** -- every prose figure we own (18.12 / 17.89 / 17.84)
is fixed-width, proposing all K tokens regardless of confidence. This is a flag, not a port: no
build, no VRAM, no rebuild. If adaptive width wins, the CHAMPION banks it, not DFlash2.

batch30 rewritten accordingly: cycle A is MTP p_min 0/0.3 at B=4, MTP p_min 0.3 at B=6, and DFlash2
at **B=6** (n_max 5) p_min 0/0.3, all on prose_p1; cycle B carries MTP p_min 0 vs 0.3 across the
remaining three panel prompts. DSpark deferred -- `DimInfer/Qwen3.8-27B-Dspark-v1` Q8_0 (1.985 GB,
block_size 15, recommended n_max 4) is the fetch if we want it, but inco.ai's own table puts DSpark
below MTP on the 27B, and the Z640 is offline.

## F100 -- RETRACTION of F93: MMQ on sm_60 is a POLICY gate, not a hardware limit. This is the
##          highest-value finding of the project so far. (2026-09-03)

In F93 I wrote "MMQ is unreachable for us, full stop" and used it to eliminate a whole class of
optimisation. **That was wrong.** I read the gate correctly and then mis-stated what it means.
Verified in source myself, all five points, in $LLAMA_DIR @ 67a17c17c:

1. `mmq.cu:314-317` -- the gate is `return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;`, sitting
   under the comment **"TODO: check if cards older than pascal might benefit from this as well"**.
   It is an untested-arch policy line with an open TODO, not a capability check.
2. **`mmq-config-pascal-older.cuh` EXISTS IN TREE** (25,123 bytes), `#include`d at `mmq.cuh:216`
   and selected at `:254`/`:277` via `ggml_cuda_mmq_get_config_pascal_older()`. Upstream ships
   MMQ configs tuned for exactly our arch class.
3. `ggml_cuda_dp4a` (`common.cuh:733-737`) has a scalar int8 fallback under
   `__CUDA_ARCH__ < DP4A`, so the MMQ path RUNS on sm_60.
4. The one real capability check -- `smpbo < 48*1024` (`mmq.cu:303-308`) -- **passes** on P100
   (sharedMemPerBlockOptin = 49152, test is strict `<`).
5. `GGML_CUDA_FORCE_MMQ` is checked at `:320`, i.e. AFTER the dp4a branch returns, so it never
   reaches us. The branch itself must be edited. (Our own fork is stricter still: bare
   `if (highest_compiled_arch(cc) < DP4A) return false;` with no MoE exception, and it has NO
   Pascal MMQ config files at all -- so this experiment belongs on master, not our fork.)

**Why it attacks precisely our bottleneck.** MMVQ re-runs the quant decode inside `vec_dot`
**once per output column**; MMQ stages a dequantized weight tile in shared memory **once** and
reuses it across all ne11 columns. With no native dp4a our per-column decode is XMAD-emulated
(F: sm_60 mmvq is 81,883 XMAD). The wasted work therefore scales with verification width -- which
is literally the `35.7*(B-1)` term. This is an INSTRUCTION cut, the category F91 says to rank first.

**Independent measurement, same model and engine** (hanxiao, Qwen3.8-27B UD-Q4_K_XL, llama.cpp,
MTP spec decode, NVIDIA L4 sm_89) -- MEASURED:

    MMVQ (stock)   intercept 66.9 ms   marginal 9.30 ms/token
    MMQ (patched)  intercept 76.6 ms   marginal 1.98 ms/token   (+9.7 ms admission fee)
    end-to-end 24.14 -> 28.05 t/s = +16.2%, from a one-line change in should_use_mmvq

Normalised marginal/intercept -- the arch-independent read:

    L4 sm_89 MMQ      0.030
    L4 sm_89 MMVQ     0.139
    RTX 3090 MMVQ     0.25-0.29   (thc1006)
    **P100 sm_60 MMVQ 0.44**      (ours: 35.7/81.0)

We are the worst case in the published set, for exactly the reason MMQ fixes. So the RELATIVE
prize on sm_60 should be LARGER than L4's +16.2%, not smaller.

CALCULATED bound (not measured): if MMQ merely halves our marginal, T(4) = 188.1 -> 134.6 ms
(1.40x) and prose goes ~18.1 -> ~25 t/s at unchanged tau -- **through the 23.8 t/s "perfect
drafter" ceiling**, because that ceiling assumed T(B) fixed. The ceiling was never a property of
the hardware; it was a property of the kernel we happened to be using.

Note the admission fee: +9.7 ms on L4. If Pascal's is similar the crossover may want to be B>=4
or B>=5 rather than B>=3, and it should be swept PER QUANT TYPE (hanxiao routes Q6_K and IQ4_XS
back to MMVQ for +1.4%) -- consistent with our own RPB finding that the right answer is per-type
and per-n, never one constant.

**Implemented**: `LUCE_MMQ_DENSE_MIN_NE11` in master's `mmq.cu` -- 0 = upstream behaviour,
N>0 = take MMQ for dense mul_mat at ne11 >= N. Env-overridable so the crossover sweeps without a
rebuild. Building now.

## F101 -- three more from the same sweep, verified in source

**(a) The B>=3 FA path dequantizes the ENTIRE K and V cache to f16 every round.**
`fattn.cu:663` -- Pascal falls through every tensor-core branch to `BEST_FATTN_KERNEL_TILE`.
`fattn.cu:684-688` -- TILE sets `need_f16_K = need_f16_V = true`. `fattn-common.cuh:1026` then
runs `to_fp16(K_data, K_f16, ggml_nelements(K), main_stream)` and the same for V. VEC
(`fattn.cu:690-691`) converts only from F32, i.e. **never for our q8_0 KV**. And VEC is chosen
only at `Q->ne[1] <= 2` (`fattn.cu:655-661`).
So: **at B<=2 this cost is ZERO; at B>=3 we convert the whole q8_0 K and V cache to f16 across all
17 attention layers, every verification round**, plus f16 shadow allocations in the KV buffer.
This is a concrete candidate mechanism for our own MEASURED "n-max 1 beats n-max 3 by +26%"
(an earlier note), which we had attributed to drafter economics. Two cheap probes:
`-ctk f16 -ctv f16` at B=4, and `-fa off`. THEORETICAL until measured; the code path is certain.

**(b) CUDA graphs are hard-disabled on Pascal.** `ggml-cuda.cu:4403-4408`:
`if (cc < GGML_CUDA_CC_VOLTA) { graph->disable_due_to_gpu_arch = true; }` -- unconditional. So
every kernel in a 65-layer graph is launched individually every round; at ~10^3 launches x 3-8 us
this is a real slice of the **81 ms intercept**, the only term neither MMQ nor a better drafter
touches. No issue or PR documents an actual Pascal graph bug; capture/launch has been supported
since CUDA 10. Reads as an untested-arch cutoff. `ggml_cuda_graph_check_compability` only bails on
MUL_MAT_ID needing a sync -- we are dense, so that never fires. Caveat: warmup needs two
consecutive calls with unchanged node properties, so a p_min that varies B per round would keep
re-capturing. **p_min and CUDA graphs may be in direct tension -- measure them separately.**

**(c) FR-Spec draft-vocab trim has a native-MTP implementation already.** llama.cpp issue #25187,
branch `frspec-mtp-vocab-trim`, reuses the existing `d2t` tensor infrastructure from EAGLE-3 plus
`ggml_set_rows`; no sampler changes. Reported elsewhere, not measured here (llama.cpp issue #25187, RTX 5090 Laptop): draft LM-head compute -84.9%,
end-to-end +1.4% (public corpus) / +3.1% (code), byte-identical output. For us the prize is bigger:
`output.weight` is 1.271e9 params ~8% of the model and runs **3x per round**; trimming the two
DRAFT passes (the target must keep the full vocab for exactness) to ~32k of 248,320 rows removes
an estimated **11-12 ms/round ~ 6% at B=4**, more at low B. Build the frequency table from the
model's OWN generations (97.5% coverage vs 92% for generic web text); VocabTrim (arXiv 2603.05210)
puts the optimum near 94% coverage, not 99%.

**Ranking correction.** The syv.ai split-KV port is now DEAD: llama.cpp's tile kernel already does
occupancy-driven split-KV with a combine pass (`fattn-common.cuh:1167-1200`, `parallel_blocks`
raised by a wave-filling search targeting >=95% occupancy) on Pascal today. Mads's patch fixes a
vLLM/FA2 defect with no llama.cpp analogue, and is Triton + bf16-only. Attention is also bounded at
17/65 layers and single-digit percent of round time at our ~5k resident context. **Do not port it.**
New order: (1) MMQ dense gate, (2) FA f16-conversion probe, (3) CUDA graphs on Pascal,
(4) GPU-side sampling check, (5) FR-Spec vocab trim, (6) GDN fp16 state (VRAM enabler only).

## F102 -- round-time model REFIT; the 0.44 ratio in F100 was stale (2026-09-03)

F100 quoted our marginal/intercept as 0.44 from T(B) = 81.0 + 35.7(B-1). **That predates the
surgical requant.** Refitting on the current champion from batch27's two clean cold points --
prose B=4 = 157.2 ms and echo B=6 = 201.4 ms, same binary, same model, both bracketed:

    T(B) ~ 90.9 + 22.1*(B-1) ms      marginal/intercept = 0.243

So we sit with the RTX 3090 (0.25-0.29), not alone at the bottom. The requant already banked part
of the prize MMQ targets, and F100's "we are the worst case in the published set, so the relative
prize should be LARGER" is **withdrawn**. The mechanism argument stands; the magnitude claim does not.

PROJECTED MMQ outcome, tau held fixed, two bracketing assumptions about Pascal's admission fee:

| scenario | prose B=4 | echo B=6 |
|---|--:|--:|
| now (MEASURED) | 18.12 | 28.58 |
| conservative: marginal /2, intercept +10% | 21.4 (+18%) | 37.1 (+30%) |
| L4-like: marginal /4.7, intercept +14.5% | 24.1 (+33%) | 45.1 (+58%) |

Why not L4's ratio: MMQ's shared-memory staging still runs the dot product, and with no native
dp4a each int8x4 is four scalar ops (common.cuh:733-737) rather than one instruction. The tile
amortisation is real; the arithmetic underneath is not.

**Structural consequence, and it reframes the target.** Echo plausibly reaches the high 30s-mid 40s
-- the first credible path to 50 t/s on either workload. Prose lands ~21-24 and then becomes
**drafter-bound again**: with even a FREE verify slope (intercept only, ~104 ms) tau 2.85 caps prose
at 27.4 t/s. The 23.8 t/s "perfect drafter ceiling" was never hardware -- it assumed T(B) fixed.
After MMQ the binding constraint on prose flips back to acceptance, which is what p_min addresses.

## F103 -- overnight chain armed (2026-09-03). Markers verified to align.

    batch28  V-cache decider (turbo4 vs q8_0 V, prose B=4)          RUNNING
    batch29  splice recipe E A/B at -ub 256 (champion at same -ub)  waits BATCH28_COMPLETE
    batch30  p_min: MTP sweep on OUR fork + DFlash2 at n_max 5      waits BATCH29_COMPLETE|ABORT
    batch31  MMQ dense gate sweep (LUCE_MMQ_DENSE_MIN_NE11)         waits BATCH30_COMPLETE|ABORT
    batch32  FA f16-conversion probe + CUDA graphs on Pascal        waits BATCH31_COMPLETE|ABORT

Guards now standing on every build-shipping batch, one per lesson learned the hard way:
  F87  read the GENERATED artifact back (gguf_tensors.py composition readback)
  F94  a load test must RUN A COMPLETION, not just reach "listening"
  F96  the buffer that kills you may be allocated lazily at first DECODE
  NEW  wait for `Built target llama-server`, then prove the patch SYMBOL is present in the
       shipped lib both locally and AFTER transfer -- a smoke test proves the binary RUNS, not
       that it is the binary I meant. Without this, batch31 would have tarred a stale but working
       llama-server (build was still linking) and measured stock master while reporting MMQ.

DFlash2 width: n_max 5 (B=6) per the drafter's own metadata (block_size=8, n_extract=5) and five
independent rigs. Nothing contradicts it. p_min and CUDA graphs are never combined -- graph capture
needs two consecutive decodes with unchanged shapes, and variable draft width forces a re-capture.
MMQ changes FP reduction order, so it is judged on TASK ACCURACY (the gsm8k arm and qgate), never
on byte-parity.

## F104 -- p_min WORKS ON PROSE: +7.2% on the champion, from a flag (2026-09-03)

BATCH30A, our own fork (lucefix + sub4_2, turbo4 KV), prose_p1 192 tokens, cold, bracketed.
Rounds derived as `rounds = n_out - n_accepted` (every round emits the target's own token plus
its accepted drafts), so tau = n_out/rounds and mean width = n_gen_draft/rounds.

| arm | degC | t/s | rounds | tau | mean width | round ms | t/s @43C |
|---|--:|--:|--:|--:|--:|--:|--:|
| A1 p_min=0   B=4 (control) | 43 | 17.51 | 70 | 2.743 | 2.96 | 156.6 | 17.51 |
| A2 p_min=0.3 B=4 | 55 | 16.77 | 74 | 2.595 | 2.81 | 154.7 | 17.04 |
| **A3 p_min=0.6 B=4** | 59 | **18.38** | 82 | 2.341 | **1.67** | **127.4** | **18.77** |
| A4 p_min=0.3 B=6 | 62 | 15.06 | 67 | 2.866 | 4.40 | 190.3 | 15.44 |
| A5 p_min=0   B=4 (close) | 64 | 17.02 | 70 | 2.743 | 161.2 | 2.96 | 17.50 |

**The mechanism is confirmed exactly as predicted.** At p_min=0.6 the drafter proposes a mean of
**1.67 tokens instead of 3.00**, and verification round time falls **156.6 -> 127.4 ms = -18.6%**.
tau falls too (2.743 -> 2.341, -14.7%) because some good drafts are truncated with the bad, but the
round-time saving is larger, so the net is positive: **+7.2% temperature-corrected (17.51 -> 18.77)**.

The operator's reasoning was right, and the saving lands where predicted -- on the target's verify
columns, not on the drafter.

Method notes:
- The control bracket corrects to **17.51 and 17.50** at 43 C from 43 C and 64 C. The -0.133 %/C
  proportional slope (F97) reproduces to 0.06%. That is the tightest validation of the thermal
  correction we have, and it is what makes a +7.2% claim credible at all across a 16 C spread.
- Acceptance at p_min=0 is 0.58937 at both ends of the bracket -- bit-identical, ninth sighting.
- **p_min=0.3 does essentially nothing** (width 3.00 -> 2.81, +0.2% inside noise). The threshold
  has to bite hard enough to actually truncate; 0.6 is where it does. Sweep higher next.
- **B=6 with p_min is WORSE than B=4** (15.44 vs 17.51 corrected): width still averages 4.40, so
  the extra columns are not being truncated away. Wide-then-truncate loses to narrow.

Why this works for us when it failed on a 5090 and on Blackwell (F99): both published negatives
blame flat verify batches. Ours are linear at 22.1 ms/column, so a truncated column is real money.
**This is now a measured result on linear-T(B) hardware, and it is worth reporting upstream** --
svgop's comment explicitly asks for data from "CPU-bound lanes".

Shippable as-is: it is a flag on the champion binary, no build, no VRAM change. Pending the
remaining panel prompts (BATCH30B) and an echo check that it does not hurt the other workload.

## F105 -- batch29 splice arms VOID: -ub never reached the server (2026-09-03)

Fifth iteration of the same class of error, and the first one where the guard I had already
written did not fire because the failure was upstream of it.

`batch29` wrote its remote driver with `ssh $H "cat > file <<EOF ... EOF"` -- an UNQUOTED heredoc
evaluated by the REMOTE shell at write time. `\$UB` survived the local double quotes, and the
remote then expanded it while UB was unset in that scope, yielding an **empty** 12th argument.
`tb11`'s `UB=${12:-512}` fires on empty as well as unset, so every arm silently ran at **-ub 512**.

Proof, three ways: the driver's own header prints `at -ub  ` with nothing between; the driver text
reads `... turbo4 turbo4 ` with a trailing space; and splice ready VRAM was **15969 MiB (B=4) /
16267 (B=6)**, byte-identical to batch27's -ub 512 runs, then OOMed identically.

The champion arms are unaffected and reproduce: prose 18.10 @41 C / 17.68 @59 C (acceptance
0.61616 twice more), echo 27.70 @64 C (78/82). Only the splice arms are void.

**Fix (batch33):** build the driver LOCALLY and scp it -- the runcycle() pattern batch26/30/31/32
already use -- so there is no remote heredoc and nothing can be re-expanded. Plus two new guards:
assert the numeric -ub is present in the driver TEXT before spending a cold cycle, and count the
matching lines again after transfer. **Generalised rule: never let a remote shell expand a
variable that only exists locally; write the artifact locally and ship the bytes.**

## F106 -- the cost model is indexed on MEAN DRAFT WIDTH, not nominal B (2026-09-03)

Building the scenario calculator exposed a 26% discrepancy: `T(B) = 90.9 + 22.1(B-1)` predicted
14.89 t/s for the p_min=0.6 prose point, but we MEASURED 18.77. The model was wrong, not the
measurement.

**Cause: with p_min the drafter proposes fewer tokens than n_max, so the verification cost follows
the width ACTUALLY proposed, not the width requested.** At p_min=0.6 the mean width is 1.67, so
the round costs what a ~2.7-wide round costs, not what a 4-wide round costs.

Refit on mean width `w = n_gen_draft / rounds`:

    T(w) = 89.6 + 22.64*w  ms

    prose p_min=0     w=2.96  measured 156.6  model 156.6   +0.0%
    prose p_min=0.6   w=1.67  measured 127.4  model 127.4   +0.0%
    echo  B=6         w=4.56  measured 186.6  model 192.7   +3.3%   [HELD OUT of the fit]

The echo point was not used to fit and lands within 3.3% across a completely different workload,
prompt length and tau. **This is now the validated cost model** and it supersedes every earlier
B-indexed form (81.0+35.7(B-1), then 90.9+22.1(B-1)).

Two consequences:
- **p_min's whole mechanism is now explicit**: it does not make rounds cheaper by magic, it lowers
  `w`. Anything else that lowers mean width buys exactly the same thing at the same rate.
- **Every future arm must report mean width**, and any A/B where width differs must be compared on
  round time, never t/s.

**Also corrected: tau.** `tau = n_out/(n_out - n_accepted)` is exact -- every round emits the
target's own token plus its accepted drafts. The `1 + K*acceptance` form assumes every round
proposes exactly K and is therefore wrong whenever width varies or the last round is clipped by
n_predict. Echo: exact tau **5.333**, the 1+K*acc form said 5.756 (+7.9% too high). All earlier
tau figures computed the 1+K*acc way are overstated by roughly that much; round times derived from
them are correspondingly overstated.

## F107 -- scenario calculation: what each term can and cannot buy (SCENARIO, not measurement)

Using T(w) = 89.6 + 22.64*w and the measured tau/w of each arm. **These are arithmetic
consequences of the model, not predictions of any experiment.**

PROSE at the current best (tau 2.341, w 1.67, MEASURED 18.77 t/s; need 2.66x):

    d = 0   (perfect marginal kernel)      26.1 t/s   1.39x
    d -50%                                 21.6 t/s   1.15x
    a -50%                                 28.3 t/s   1.51x
    a = 17.1 (weight-pass roofline)        42.6 t/s   2.27x
    a -50% AND d -75%                      43.2 t/s   2.30x
    a = 17.1 AND d = 0 (absolute floor)   136.9 t/s   7.29x

ECHO (tau 5.333, w 4.56, MEASURED 28.58; need 1.75x):

    d = 0                                  63.3 t/s   2.22x  ** target reachable **
    d -75%                                 48.6 t/s   1.70x  (just short)
    a -25% AND d -50%                      46.6 t/s   1.63x
    a -50% AND d -75%                      78.8 t/s   2.76x  ** target reachable **

**The strategic split is now quantitative.** Echo reaches 50 on marginal cost alone if d falls far
enough -- MMQ is the whole question for echo. Prose does NOT: even with d = 0 exactly, prose tops
out at **26.1 t/s**. Prose needs the INTERCEPT. Dropping a to the 17.1 ms weight-pass roofline
alone would give 42.6 t/s, and that is the single largest lever available to prose by a wide margin.

tau required for prose 50 t/s: **6.37 at today's a and d**; **2.71** if a halves and d falls 75%.
That second number is BELOW the tau we already have (2.341 is close, and p_min=0's 2.743 exceeds
it) -- so if the intercept and marginal both collapse, prose 50 becomes a tau problem we may
already have solved. That is the first arithmetic in this project showing prose 50 is not
structurally impossible; it requires roughly a 2x intercept cut AND a 4x marginal cut together.

**Which makes the ~72.5 ms of non-weight-movement intercept (89.6 measured minus 17.1 roofline)
the single most valuable unknown in the project**, ahead of MMQ. R50C partitions it.

## F108 -- THE INTERCEPT IS EXECUTION, MEASURED. sm% = 99%. (2026-09-04)

`nvidia-smi dmon -s u` sampled over decode-only windows (a throwaway 8-token prefill first, then
512 tokens = 28-42 samples per point), identical x50 binary, MMQ off / graphs off / p_min 0 / q8_0 KV:

| point | temp | ms/token | t/s | **mean sm%** | samples |
|---|---|--:|--:|--:|--:|
| nospec (w=0) | 48->66 | **80.52** | 12.42 | **99.0** | 42 |
| b2 | 62->74 | 52.62 | 19.00 | **99.2** | 28 |
| b4 | 70->79 | 65.07 | 15.37 | **99.1** | 34 |
| ~~b8~~ | 78->82 | ~~440.96~~ | ~~2.27~~ | 99.9 | 225 | **VOID -- see F109** |

**The GPU is 99% busy at every width. There is no idle to reclaim.** Of the ~80 ms forward pass,
~79.7 ms is the GPU executing kernels and ~0.8 ms is gaps. This is a DIRECT measurement of what
R50C could only infer from a null result: CUDA graphs gave +0.2% because launch overhead does not
exist here to begin with.

Partition against the roofline:

    forward pass (MEASURED, hot)        80.5 ms
      GPU executing         99.0%       79.7 ms
      idle / launch gaps     1.0%        0.8 ms
    weight-movement roofline            17.1 ms
    => arithmetic beyond memory traffic 62.6 ms   <-- the entire remaining prize

**Also: `a` is 80.5 ms, not 89.6.** This is the direct w=0 measurement, and it is the HOT number
(48->66 C), so cold it is lower still. The 89.6 was extrapolated from two points at w=1.67 and
w=2.96 and overshot by ~11%; the earlier 90.2 ms cross-check was taken at 8K resident context where
attention costs more, while this sits at ~500 tokens. Refit through the measured w=0 point gives
**d ~ 26-28 ms**, i.e. the line is steeper and the intercept lower than F106 had it. Rerun the
T(w) fit with w=0 anchored once the acceptance line is available again.

**Caveat on sm%:** it measures OCCUPIED vs IDLE, not efficiency. 99% is consistent with many
under-occupied kernels running back to back. It rules out gaps; it does not rule out bad kernels.
That distinction is now exactly what Nsight Compute would resolve, and it has finally earned the
disk space -- pointed at named kernels, not speculatively.

## F109 -- b8 VOID: thermally/power capped, and the probe had no guard (2026-09-04)

b8 reported 440.96 ms/token (2.27 t/s), 6.8x slower than b4, over a 225 s run starting at 78 C and
ending at **82 C**. Not a measurement -- a throttle event:

    SW Power Capping        267,195,622 us  (267 s, per-boot; b8 ran 225 s)
    HW Thermal Slowdown      87,639,625 us  (88 s)

**My error:** `probe_one.sh` never armed `guard_req.sh`, so nothing aborted the request at the
80 C limit every other harness enforces. The four points also ran back-to-back with no cooling
between them, so each started hotter than the last (48 -> 62 -> 70 -> 78 C). 82 C is above our own
80 C guard limit and within 2 C of the 84 C at which this card has previously died (Xid 79).
No Xid this time and the card idles at 81 C; powered off to cool.

Two fixes before any further probe: **arm the guard in probe_one.sh**, and **cap a cold cycle at
three points**, which is the rule batch26 onward already follows and which I dropped when writing
a "quick" probe. b2 and b4 also inherit thermal penalty from running third and fourth in sequence,
so their absolute numbers are depressed -- only nospec (first, coldest) and the sm% figures, which
are temperature-independent, should be quoted.

## F110 -- KERNEL ATTRIBUTION WITHOUT A PROFILER. It IS dequantization, and it is quantified. (2026-09-04)

**Nsight Compute cannot profile this GPU.** It requires compute capability >= 7.0 (Volta+); the
P100 is 6.0. Not a disk problem -- the architecture is unsupported. The Pascal-era tools were
nvprof/nvvp, deprecated in CUDA 11 and REMOVED in CUDA 12; this box runs driver 580 (CUDA 13 era),
has no toolkit, no CUPTI, and 1603 MB free. Abandoned as a dead end rather than spending hours
finding that out.

**Substitute: `test-backend-ops perf`** -- already in the tree, the same tool used for the Pascal
RPB tuning, 1.2 MB, runs on the real silicon and reports us/run per (type, shape, n).

MEASURED, MUL_MAT m=4096 k=14336 n=1, single P100, ~66 C:

| type | bpw | us/run | MB moved | achieved GB/s | % of 732 peak | ps/weight |
|---|--:|--:|--:|--:|--:|--:|
| q4_1   | 5.00 | 115.64 | 36.7 | 317.4 | 43.4% | 1.97 |
| iq4_xs | 4.25 | 134.51 | 31.2 | 231.9 | 31.7% | 2.29 |
| iq2_s  | 2.50 | 179.78 | 18.4 | 102.1 | **13.9%** | 3.06 |
| q8_0   | 8.50 | 206.28 | 62.4 | 302.5 | 41.3% | 3.51 |
| iq3_s  | 3.44 | 222.22 | 25.2 | 113.5 | **15.5%** | 3.78 |
| q6_K   | 6.56 | 266.09 | 48.2 | 181.0 | 24.7% | 4.53 |
| **f16**| 16.0 | 275.66 | 117.4 | **426.0** | **58.2%** | 4.69 |

**This is the proof, and it is unambiguous: achieved bandwidth falls monotonically with unpack
complexity.** f16 does NO dequantization and reaches 426 GB/s. q4_1 (shift+scale+min) reaches 317.
The i-quants with codebook lookups reach 102-113 -- a quarter of what the same memory system
delivers when nothing has to be unpacked. The time is in the unpacking, not the bytes. Note iq2_s
moves HALF the bytes of q4_1 and takes 55% LONGER.

**Roofline corrected.** I have been quoting 17.1 ms from the 732 GB/s spec sheet. f16 shows the
access pattern only sustains **426 GB/s**, so the honest roofline is 12.504 GB / 426 = **29.4 ms**,
and the dequant/instruction overhead is **51.1 ms**, not 62.6.

### Bottom-up reconstruction of the 80.52 ms forward pass

Champion matmul inventory x measured ps/weight (token_embd excluded -- it is a lookup):

| type | params | ps/wt | ms | % of forward pass |
|---|--:|--:|--:|--:|
| **iq2_s** | 14.26e9 | 3.06 | **43.64** | **54.2%** |
| q4_1 | 8.59e9 | 1.97 | 16.92 | 21.0% |
| iq3_s | 2.85e9 | 3.78 | 10.78 | 13.4% |
| q6_K | 0.32e9 | 4.53 | 1.45 | 1.8% |
| q8_0 | 0.02e9 | 3.51 | 0.08 | 0.1% |
| **MUL_MAT total** | | | **72.87** | **90.5%** |
| residual (attention, norms, GDN scan, sampling) | | | 7.65 | 9.5% |

Reconstruction lands within 9.5% of the measured 80.52 ms using only per-kernel micro-benchmarks
and the tensor inventory. **MUL_MAT is 90.5% of the forward pass and iq2_s ALONE is 54% of it.**
That also retires the earlier "MUL_MAT 84%" estimate and confirms attention is a single-digit
share -- so all attention-kernel work (split-KV etc.) was correctly deprioritised.

### The answer to "what should we do", and it is already built

Recipe E converts ffn iq2_s+iq3_s -> iq4_xs and attn q4_1 -> iq4_xs:

    ffn iq2_s+iq3_s -> iq4_xs   saves 15.23 ms
    attn q4_1 -> iq4_xs         costs  2.75 ms   (q4_1 is FASTER than iq4_xs -- this is the price of fitting)
    NET                                12.48 ms  =>  forward pass 68.0 ms
    prose p_min0.6 PROJECTED     18.89 -> 22.35 t/s (+18%)

`Ridge-splice.gguf` is already on disk. Its A/B (batch33) is the outstanding experiment, and this
analysis says it is the highest-value one available -- not the marginal-cost play I nearly dropped.

### CEILING, and it is the important number

PROJECTED, holding tau and the residual fixed, d scaled proportionally to a:

| scenario | a (ms) | w=0 t/s | prose | echo |
|---|--:|--:|--:|--:|
| today | 80.5 | 12.42 | 18.9 | 26.8 |
| recipe E (built) | 68.0 | 14.71 | 22.4 | 31.7 |
| all-q4_1 (fastest REAL type, does not fit VRAM) | 59.0 | 16.96 | 25.8 | 36.6 |
| **perfect zero-unpack 4.25-bpw kernel** | **40.1** | 24.92 | **37.9** | **53.7** |

**Even a hypothetical PERFECT dequantization-free 4-bit kernel -- faster than anything that exists
-- leaves prose at ~38 t/s.** Echo crosses 50 (53.7). So on this hardware, with this model:
**echo 50 t/s is reachable in principle via kernel work; prose 50 t/s is NOT reachable through the
verification path at any achievable kernel efficiency.** Prose 50 additionally requires tau, and
the tau needed at a=40.1/d=13.0 is 3.10 -- above the 2.341 we run and just below the 3.36 best ever
measured. So prose 50 requires BOTH a near-perfect kernel AND a drafter better than any tested.

## F111 -- RPB IS worth 12-19% and it was NOT in the 80.52 ms baseline. Master-vs-fork, interleaved. (2026-09-04)

Fable flagged that the 80.52 ms w=0 baseline might not have our RPB tuning active. **It did not.**
That baseline ran `$HOME/x50/llama-server` = upstream master, which has NO Pascal mmvq table at
all: `calc_rows_per_block(ncols_dst, table_id, ...)` takes no `type` argument, Pascal falls to
MMVQ_PARAMETERS_GENERIC, and returns **1** at ncols_dst==1 (mmvq.cu:528). Our fork registers
MMVQ_PARAMETERS_PASCAL (mmvq.cu:83,103) and returns **2** (mmvq.cu:433). So every number in F110 --
the ps/weight table, the 90.5% MUL_MAT attribution, the recipe E projection, the ceiling table --
describes MASTER's kernel and is retained only as a historical upstream result.

**Protocol:** interleaved master/fork/fork/master per type, one shape (m=4096 n=1 k=14336, the only
case both binaries generate identically), temp read before and after every case, cold start at 42 C.
Interleaving makes thermal drift cancel WITHIN each pair instead of being corrected afterwards.

| type | master us | fork us | change | fork GB/s | % of 732 peak |
|---|--:|--:|--:|--:|--:|
| q4_1   | 113.47 | 95.33  | **-16.0%** | 385.0 | 52.6% |
| iq4_xs | 130.00 | 105.67 | **-18.7%** | 295.2 | 40.3% |
| iq2_s  | 174.78 | 152.15 | **-12.9%** | 120.6 | 16.5% |
| iq3_s  | 214.92 | 187.63 | **-12.7%** | 134.5 | 18.4% |
| q6_K   | 260.89 | 226.11 | **-13.3%** | 213.0 | 29.1% |
| q8_0   | 202.94 | 178.67 | **-12.0%** | 349.2 | 47.7% |
| **f16**| 269.69 | 269.27 | **-0.2%**  | 436.1 | 59.6% |

**f16 is the internal control and it validates the experiment.** f16 does not use the quantized
mmvq path, so RPB cannot reach it -- and it moves -0.15%, the noise floor. Every quantized type
gains 12-19%; the one type the mechanism cannot touch gains nothing. Within-pair replication:
fork q4_1 0.09%, master q4_1 0.37%.

**On the retired figures:** -25.7 / -15.8 / -13.1 stay INVALID (clock-ramp artifacts). But iq4_xs
measures **-18.7%** here against the fork source comment's **-18.1%**, taken under the correct >=3 s
protocol -- it reproduces almost exactly. The protocol distinction was real, not pedantry.

## F112 -- fork w=0 MEASURED 72.75 ms. Prediction 3.1% optimistic, and the miss is the interesting part.

| arm | binary | ms/token | t/s | sm% | temps |
|---|---|--:|--:|--:|---|
| master w=0 | x50 | 80.52 | 12.42 | 99.0 | 48->66 C |
| **fork w=0** | lucefix+sub4_2 | **72.75** | **13.75** | 98.8 | 45->69 C |

Both are the FIRST point of their own cold cycle at comparable thermals, so this is a fair
comparison: **fork is 9.6% faster at w=0** (+10.7% t/s). sm% 98.8 reconfirms F108 -- GPU-bound.

But the kernel sweep said MUL_MAT -14.0%, which with a constant residual predicts 70.55 ms.
Measured 72.75. **The prediction was 3.1% optimistic, and backing out the residual says why:**

    master  80.52 - 71.28 predicted MUL_MAT =  9.24 ms residual
    fork    72.75 - 61.31 predicted MUL_MAT = 11.44 ms residual
    => the FORK carries 2.20 ms MORE non-matmul cost per forward pass

**This is very likely the F86/F97 anomaly, finally localised.** The fork WINS at w=0 (72.75 vs
80.52) but LOSES at B=4 (master 19.14 temp-corrected vs fork 18.12). A fixed +2.2 ms per round
cannot produce that reversal -- but a per-COLUMN cost can, and turbo4's V-cache dequant is read
per token position, i.e. it scales with width. Prediction: the fork's `d` is worse than master's
by roughly the same mechanism, and the crossover sits between w=0 and w=3.
**Decisive test: fork vs master `d`, i.e. the same interleaved treatment at B=4 and B=6.**

**Recipe E on the MEASURED fork baseline:** net -13.76 ms (saves 11.29 iq2_s + 3.98 iq3_s, costs
1.51 attn off q4_1) => forward pass **58.99 ms, w=0 16.95 t/s**. The absolute saving is LARGER on
the fork than on master because iq4_xs benefits more from RPB (-18.7%) than iq2_s does (-12.9%).

## F113 -- cycle design flaw: two of three points were skipped as too hot

`masterw0` and `forkb4` both SKIPPED at 67 C / 66 C against a 60 C start threshold. A 512-token
no-spec decode is 37 s of sustained load and raises the card **24 C** (45->69). So three points per
cold cycle is arithmetically impossible with a cold-start rule -- the first point alone consumes
the whole thermal budget. F109 told me to cap at three; the real cap for THIS workload is **one**.

Fixes for the next cycle: either one heavy point per cold cycle, or drop n_predict to 192 (~14 s,
proportionally less heating) and keep two. The cooldown itself is validated -- 480 s took 59 C to
45 C, inside the spread the old 800 s produced.

## F114 -- PHASE BISECTION: the 12.3 ms gap did NOT reproduce (2026-09-04)

Three arms, cold (idle 50 C), np=32, identical LUCE_PHASE patch on both trees.
Arms: fork_w1 / master_w1 / fork_w0. Preflight asserted the LOADED libggml-cuda carries
LUCE_PHASE (sym=4) on every arm -- see F113 note on the sub4_2 shadow.

REAL wall time (the size check, run before interpretation):
  fork_w1   63.89 ms/tok  15.65 t/s
  master_w1 62.03 ms/tok  16.12 t/s   -> fork only 2.9% slower
  fork_w0   79.88 ms/tok  12.52 t/s
Real delta over 32 tokens = +59.5 ms. The 12.3 ms/round gap requires 209-394 ms.
MEASURED IS 3.5-6.6x TOO SMALL. **The gap did not reproduce under this configuration.**

INSTRUMENT FLAW, self-inflicted and now on record:
I justified the method with "the distortion is symmetric because the identical patch runs on
both trees". That is wrong at run level. Sync cost scales with OP COUNT; the fork runs more ops:
  inflation fork_w1 1.90x, master_w1 1.48x, fork_w0 1.37x
The instrument penalises the fork MORE. Therefore the headline instrumented finding
(MUL_MAT +44% per-op, 82% of the instrumented delta) is CONTAMINATED and is NOT promoted.

ROBUST (direction survives the +/-2000 dump quantisation):
  fork/master ops-per-token ratio in [1.063, 1.200] -- fork does 6-20% MORE ops.
  Excess is non-MUL_MAT: CPY 2.1x ops / 2.7x time; PAD and GATED_DELTA_NET appear in fork's
  top-10 while both sit below master's 10th place (>=4.4x and >=3.4x).
  Consistent with the fork's TREE_MODE GDN port and its store_inter_state writes.

GDN NOT REOPENED: at most 29.9 ms instrumented vs a 59.5 ms real delta, and instrumented ms
cannot be converted to real ms because the sync removed the overlap. That conversion was
forbidden in advance in PHASE_PREREG.md.

WHY IT MAY NOT REPRODUCE -- the method may be blind to the target:
if the gap is LOST KERNEL OVERLAP (fork's extra CPY/PAD/GDN serialising against MUL_MAT), then
synchronising every op makes both trees fully serial and ERASES the effect. A serialising
profiler cannot observe an overlap-structure gap. This should have been reasoned out before
the cycle was spent. Other live differences: np=32 vs 192, instrumentation active.

STATUS per the pre-committed null: the 12.3 ms first-column asymmetry is OPEN, UNEXPLAINED,
TRACKED. Seven candidates now dead. Effort reverts to directive 5 (q4_1-first FFN allocation
at real shapes), the top GPU-time priority and the only shippable item remaining.

Also fixed this cycle (each had silently voided the previous night's run):
  1. libcudart missing from phasefork/phasemaster -> symlinked (disk was at 100%).
  2. LD_LIBRARY_PATH put sub4_2 FIRST; sub4_2 has libggml-cuda.so.0 with 0 LUCE_PHASE symbols.
     Would have run clean and emitted a fork breakdown with the instrument ABSENT.
  3. Dump trigger calls % 40000 never reached at np=8/32; atexit is an empty lambda, no fallback
     dump. Fixed via PHEVERY env (no rebuild). Caught by a warm preflight, not by a cold cycle.

## F115 -- Directive 5 recipe validated; imatrix blocker solved; ordering found WRONG (2026-09-04)

DIRECTIVE 5 SIZING (llama-quantize --dry-run, per F83 precedent -- dry runs cost ~0.3 s, no disk):
  champion                                    11924.23 MiB (3.66 BPW)
  all 192 FFN -> iq4_xs                       15069    MiB  -- OVER budget, NOT buildable
  partial (32 iq3_s + 99 iq2_s), keep IQ2_S   13975.52 MiB (4.29 BPW)
  same but keeping Q2_K instead of IQ2_S      14016.03 MiB (4.30 BPW)
  budget 13000 -> 80/192 upgraded             13135.48 MiB (4.03 BPW)   <- VRAM-safe candidate
  budget 12600 -> 59/192 upgraded             12772.91 MiB (3.92 BPW)

IMATRIX BLOCKER SOLVED: the champion carries imatrix METADATA inherited from a cloud workspace
(/workspace/qwen38-exl3/work/ridge.imatrix, 497 entries, 80 chunks) but the FILE was never on this
box. The newer binary enforces the check; build 10775 did not. Diagnosed by dry-run A/B:
the culprit is NOT IQ4_XS -- it is the 61 tensors we KEEP at IQ2_S. llama-quantize always
requantises, so "leave alone" is inexpressible (F87), and IQ2_S is imatrix-mandatory.
FIX: keep them as Q2_K (imatrix-free). Warning disappears. Costs +40.5 MiB.

**MY UPGRADE ORDERING IS PROBABLY WRONG.** I ranked upgrades by cheapest SIZE increase (all 32
iq3_s first, then iq2_s). The correct criterion for "maximise speed under a VRAM budget" is
TIME SAVED PER BYTE ADDED. Bytes/weight added: iq2_s->iq4_xs 0.21875, iq3_s->iq4_xs 0.10156.
  upgrade iq2_s FIRST iff (ps_iq2s - 2.35) > 2.154 * (ps_iq3s - 2.35)
Under every scenario where iq2_s is materially slower (and it is the pathological kernel: 780 SASS
instrs vs 318, 152 regs vs 88, 20% vs 36% occupancy) iq2_s wins. The ttf as generated is likely a
SUBOPTIMAL allocation at identical VRAM cost. DO NOT BUILD IT until ps_iq2s/ps_iq3s are measured.

GATING MEASUREMENT (serves BOTH tracks): ps/weight for iq2_s, iq3_s, q2_K at real Ridge shapes.
The measured table (F: RUN 2) covers only q4_1 2.02 / q4_0 2.32 / iq4_xs 2.35 / q6_K 4.59.
Must be run GUARDED, <=3 cases per burst (+2 C per case), from a genuine cold start.

## F116 -- THERMAL INCIDENT, my error (2026-09-04)
I launched a FULL `test-backend-ops perf -o MUL_MAT` sweep (all shapes, all types) with NO thermal
guard armed. Card reached **84 C** -- the Xid 79 death threshold (80 C = 12 h clean, 84 C = dead).
Killed on detection. Established discipline is 2-3 perf cases per burst; the full suite is hundreds.
Post-kill the card sat at 83 C and cooled only ~1 C in 2 minutes: idle equilibrium with the fan
situation is ~79-80 C, i.e. there is NO passive headroom. GPU work now requires a real cool_cycle.
This is the same class as F109 (b8 thermally void) and it is now twice.

## F117 -- IDEA B (bounded KV): quantified from EXISTING data, no new measurement
Derived from R50C's two "ready VRAM" figures (q8_0 KV 12389 MiB vs f16 KV 12629 MiB, delta 240 MiB):
  KV @8192 ctx, q8_0 ~= 272 MiB  = **1.7% of the 16269 MiB card**;  f16 ~= 512 MiB
  = 34.0 KiB/token. ctx 32768 -> 1088 MiB; ctx 131072 -> 4352 MiB.
REASON IT IS SO SMALL: Ridge is a HYBRID -- only 17 of 65 blocks are attention with growing KV;
the other 48 are Gated DeltaNet with CONSTANT-SIZE recurrent state. The architecture already
implements the spirit of "bounded KV" for 74% of its blocks.
CONCLUSION: at our 8K benchmark context, bounded-KV/KV-Direct can free at most ~272 MiB against an
11924 MiB model -- it cannot buy a materially better precision allocation. It only becomes material
at >=32K, and 128K resident is already established as thermally impossible. **LOW priority for the
50 t/s target**, on measured grounds rather than opinion. Reconstruction cost is the second strike:
rebuilding evicted KV costs prefill, measured at 28.6 t/s.

## F118 -- ps/weight MEASURED for iq2_s / iq3_s / q2_K (2026-09-04, guarded, cold 44 C start)

Shape m=4096 k=14336 -- the ONLY broadly-typed perf shape in the suite (test-backend-ops.cpp:10649,
`for bs in {1,2,3,4,5,8,512} for type_a in all_types`). NOTE: `-p` FILTERS existing cases, it does
not create shapes; a first attempt at real Ridge FFN shapes (17408x5120) returned 12x NO_RESULT in
11 s at 44->44 C -- no data, no thermal cost. Real-shape measurement needs a source change.
q4_1 is carried as the control: 1.947 here vs 2.02 in the validated table (3.6%).

| type   | bpw    | ps/w n=1 | ps/w n=4 |
|--------|--------|----------|----------|
| q4_1   | 5.0000 |  1.947   |  3.481   |
| iq4_xs | 4.2500 |  2.216   |  3.834   |
| q2_K   | 2.5625 |  2.805   |  **5.923** |
| iq2_s  | 2.5000 |  2.973   |  4.677   |
| iq3_s  | 3.4375 |  **3.660** |  5.191   |

**SURPRISE 1 -- iq3_s is DOMINATED on both axes.** It is slower than iq2_s at n=1 AND n=4 while
being 0.94 bpw LARGER. There is no budget at which iq3_s is the right choice over iq2_s. Any
iq3_s in an allocation is a free win to remove.

**SURPRISE 2 -- q2_K REVERSES with width.** At n=1 it beats iq2_s (2.805 vs 2.973); at n=4 it is
the worst of every type measured (5.923 vs 4.677, **+26.6%**). Our champion's measured optimum is
B=4, so n=4 is the operating point, not n=1. A single-width benchmark would have chosen wrong.

**MY PREDICTION IN F115 IS FALSIFIED.** I argued the directive-5 ordering (all iq3_s first) was
"probably wrong" because the correct criterion is time-saved-per-byte and I assumed iq2_s was the
slow kernel (780 SASS instrs, 152 regs, 20% occupancy). Measurement: iq3_s is the slow one. The
criterion was right; the assumed input was wrong. Ordering test at BOTH widths returns iq3_s FIRST.
The original ttf ordering stands. Instruction-count/occupancy did NOT predict ranking.

**THE Q2_K WORKAROUND IS DEAD AT B=4.** Costed per tensor-equivalent at n=4:
  upgrade iq2_s->iq4_xs SAVES 49.51 us ; forced iq2_s->q2_K COSTS 73.13 us
  upgrade 99 / keep 61 as Q2_K -> net  -440.6 us (marginal gain, and that variant is VRAM-risky)
  upgrade 80 / keep 80 as Q2_K -> net +1889.6 us (**net LOSS** at the VRAM-safe budget)
So directive 5 AS A REQUANT is not viable: the imatrix-free keep-format eats the gain.

## F119 -- the fix is a TRUE tensor splice, not a requant
Requant forces every kept tensor to be re-encoded, which is what drags Q2_K in. But the iq4_xs FFN
tensors ALREADY EXIST in Qwen3.8-27B-UD-IQ4_XS.gguf, and the iq2_s ones already exist in the
champion. A byte-level GGUF splice copies each tensor from whichever file already holds it:
  * no requantisation at all -> **no imatrix needed** (kills the F115 blocker outright)
  * kept tensors stay bit-identical iq2_s -> **no Q2_K penalty**
  * upgraded tensors come from the ORIGINAL-weight IQ4_XS quantisation, not upquantised from
    iq2_s -> strictly better quality than Ridge-splice, which requantised FROM the champion (F115)
This is exactly the "assemble tensors from different source GGUFs" mechanism of Idea A.
PROJECTED (labelled, not measured): per-tensor matmul saving iq2_s->iq4_xs 18.0%, iq3_s->iq4_xs
26.1% at n=4. End-to-end is bounded by FFN's share of MUL_MAT and MUL_MAT's 87.7% share of decode;
a rough band for upgrading 80/192 tensors is ~4-6% end-to-end. NOT a measured figure.

## F120 -- CORRECTION to F118, and it is worse than the peer flagged (2026-09-04)

Fable's audit says the iq3_s result does not contradict the SASS/register analysis. Checking our
own ledger, the reality is stronger: **line 3776 already contains a DIRECTLY MEASURED per-format
table** with iq2_s 3.06 and iq3_s 3.78 ps/weight. iq3_s being slower than iq2_s was not merely
predictable -- it was already measured, in this file, and I did not read it before declaring a
surprise and before writing the F115 prediction that assumed the opposite.

  cross-session agreement (old table vs F118 n=1): q4_1 1.97/1.947, iq4_xs 2.29/2.216,
  iq2_s 3.06/2.973, iq3_s 3.78/3.660 -- all within ~3%.

RETRACT from F118: "SURPRISE 1" (iq3_s slower than iq2_s). It reproduces a known result.
RETRACT from F118: the framing that instruction-count/occupancy "did NOT predict ranking" --
  that conclusion was drawn against an assumption, not against the recorded data.
KEEP from F118: the n=4 column (new), the q2_K width reversal (new and decisive), the q4_1
  control, and the Q2_K-workaround costing. Those stand.
UNVERIFIED: the peer's specific "iq3_s 163 regs vs iq2_s 152, occupancy 19% vs 20% at ncols_dst=4".
  Our ledger records iq3_s registers only at n=8 (222). Not adopted as evidence.

PROCESS: this is the project's "read the history first" rule. The relevant measurement already
existed; I re-derived it at the cost of a cold cycle and then mislabelled it as a new finding.
Before any future "surprise", grep the ledger for the quantity first.

## F121 -- C.3 PASSES: champion and IQ4_XS share upstream weights (2026-09-04)

Fable's C.3 is the load-bearing splice hazard: are the two GGUFs quantised from the IDENTICAL
upstream weights, or merely the same architecture? Metadata cannot answer it (two fine-tunes of
one architecture carry identical hyperparameters). Test used instead: **f32 tensors are stored
UNQUANTISED in both files, so if the upstream weights match they must be bit-identical.**

  metadata: 46 vs 47 keys, only 2 differences, both imatrix provenance
    champion 497 entries / 80 chunks   vs   IQ4_XS 496 entries / 1251 chunks
    (+ general.base_model.count on the IQ4_XS side). Every qwen35.* hyperparameter matches
    exactly: block_count 65, ctx 262144, n_embd 5120, ffn 17408, heads 24/4, ssm.state_size 128,
    group_count 16, nextn_predict_layers 1, full_attention_interval 4. Tensor shapes match.

  f32 tensor comparison: **347 bit-identical, 13 differing, 0 missing.**
  All 13 are `ssm_a` -- and only 13 of the 48 that exist, which already argues against a
  systematic converter transform. Magnitudes:
    blk.0.ssm_a  max|diff| 0.0        (identical)
    blk.12.ssm_a max|diff| 5.96e-08   max rel 1.01e-07
    blk.1.ssm_a  max|diff| 1.91e-06   max rel 7.87e-08   (on a value of -35.251217)
  Max relative difference ~1.0e-7 = float32 epsilon (1.19e-7). This is LAST-BIT ROUNDING.

  MECHANISM: ssm_a is DERIVED, not stored verbatim (A = -exp(A_log)); the arithmetic can land one
  ULP apart depending on whether the converter accumulated in f32 or f64. The 347 verbatim norm
  tensors, which undergo no arithmetic, are bit-identical -- that is the actual evidence.

**VERDICT: same upstream weights. The splice is a precision-map experiment, not model-mixing.**

TOOL FLAW, recorded because it nearly became a false report: f32cmp.py tested BYTE equality and
printed "DIVERGENT -- NOT the same underlying weights". Byte-exactness is correct for verbatim
tensors and WRONG for derived ones. A verdict line must not outrun the test's validity domain.
Any future use must classify derived tensors (ssm_a) and compare them by tolerance (<=1e-6 rel),
not bytes.

REMAINING SPLICE HAZARDS from the audit, still open: verify raw TYPE-ID not the "IQ4_XS" display
label (repacked variants share names, not byte layouts); rebuild offsets/alignment from scratch
rather than copying data sections; single metadata block from the champion; per-tensor hash
self-test; check the fork's GDN/MTP paths for tensor-type special-casing.

## F122 -- allocation objective refined per audit item A/D; projections reconciled (2026-09-04)

precmap.py changed on three points from the audit:
  * **MTP head (blk.64) EXCLUDED from the objective entirely**, not warned about. Its precision
    acts on draft ACCEPTANCE (tau), not per-op latency; a cheaper head that accepts worse can
    RAISE wall-clock. Ranking it by time-saved-per-byte is a category error. Now an assert.
  * **All 213 q4_1 tensors EXCLUDED.** q4_1 is both faster AND higher precision than iq4_xs
    (3.481 vs 3.834 ps/w at n=4; 5.0 vs 4.25 bpw). No axis on which the swap helps. Now an assert.
  * Cost data required only for PARTICIPATING types. q6_K/q8_0 are constant offsets (every q6_K
    tensor is on blk.64 or is token_embd), so demanding their ps/weight blocked the optimiser on
    data it does not use. Objective now runs over the MUTABLE portion only.
Operating point is **n=4**, not n=1: the champion's optimum is B=4 and q2_K reverses between
widths (F118), so an n=1 table would rank wrongly.

| budget | swaps | mutable time | x trunk matmul | end-to-end (x87.7% MUL_MAT) |
|---|---|---|---|---|
| 13000 |  80 (48 iq2_s + 32 iq3_s) |  -9.17% | -6.71% | **-5.88%** |
| 13400 | 102 (70 + 32)             | -11.20% | -8.19% | **-7.19%** |
| 13950 | 131 (99 + 32)             | -13.88% | -10.15% | **-8.91%** |

mutable (iq2_s+iq3_s) = 73.2% of trunk matmul time. All 32 iq3_s swap in EVERY budget, which is
the ratio criterion agreeing with the F118 measurement that iq3_s is dominated on both axes.

RECONCILIATION with the peer's independent estimate: it computed ~9-11% end-to-end for a FULL
iq2_s+iq3_s swap. Our 13950 case is 131/192 tensors and lands at 8.91% -- just under that ceiling,
which is the expected relationship. Two methods, different arithmetic, consistent answer.

**ALL OF THIS IS PROJECTED AND PROBE-SHAPE-DERIVED.** Single shape (m=4096 k=14336), extrapolated
to real FFN tensors of a different geometry. The honest band remains 0% to ~11%. Audit item 1
(real-shape measurement) is a PREREQUISITE for trusting any of it, including the reconciliation.
Gate is held: no splice implementation until item 1 lands.

## F123 -- independent-assessment claim CHECKED; concern upheld, number wrong; splice unblocks RPB

A fresh Fable 5.1 instance (no prior context) claimed "the 10.7% RPB win was measured at n=1, but
n=1 is the DRAFT regime, not the VERIFICATION regime."

**THE NUMBER IS MISATTRIBUTED.** Ledger 1069/1229/1233: 10.7% is the n=1->n=2 MARGINAL COST of
adding a draft column for iq4_xs -- "DRAFTN=1 buys 45% more tokens for 10.7% more time." It is a
cost, not a speedup, and not an RPB figure. Doubly wrong: 10.7% is itself an n=1->n=2 transition,
not an n=1 measurement.

**THE CONCERN IS CORRECT, and provable more directly from F21 than the instance argued.** The
shipped RPB edit raises calc_rows_per_block for `ncols_dst == 1` ONLY. F21's own control line:
"n=2 control, deliberately unchanged: q4_1 561.72 -> 561.64 ... the edit is localised to
ncols_dst==1." So "we made the cheap part faster and left the expensive part unchanged" is an
accurate description of the shipped change. Right conclusion, wrong evidence.

**ITS #1 EXPERIMENT ALREADY EXISTS -- F22.** rows_per_block 2->4 at ncols_dst 2..4:
  q4_1 -7.6/-7.0/-10.6% (n=2/3/4); iq3_s -9.9% (n=2); **iq2_s +10.1% (n=2)**, +2.5%, -0.2%.
Not taken because type-dependent and iq2_s REGRESSES. Matches the ncols 3..8 sweep (q4_1
-11..-14% everywhere, iq2_s +27% at n=5). Conclusion stands: per-type AND per-n, no blanket change.

**NEW, and neither Fable instance had it: THE SPLICE UNBLOCKS THE QUEUED RPB GAIN.** F22 could not
take rows_per_block=4 at ncols_dst==2 solely because iq2_s regresses there. The splice's entire
effect is to remove iq2_s from the FFN. Remove it and the blocker for a per-type Pascal table at
ncols_dst==2 largely goes too -- a gain F22 already projected at ~3.4% of matmul time -> **~+2.6%
t/s, "Queued, not done."** Splice and RPB-at-verification-width are COMPLEMENTARY, not competing;
the splice is a precondition for a measured-and-shelved gain.
  CAVEAT: F22's table covers q4_1/iq3_s/iq2_s but NOT iq4_xs at ncols 2..4. iq4_xs's RPB behaviour
  at the verification width is UNMEASURED and must go into the real-shape run.

ALSO: F21's -25.7/-15.8/-13.1% were taken at the 405 MHz clock cap and are already RETIRED.
ALSO: "22 t/s" is not a claimed baseline -- ledger line 3 is a TARGET ("20 t/s resident 128K,
stretch 21-22"). Measured champion is 17.77 t/s @42 C. The config-trail argument stands regardless.
ALSO: adaptive B is partly explored -- p_min already does adaptive draft truncation, +7.2% measured.

OPEN QUESTION WORTH KEEPING (raised by the fresh instance, genuinely new): **is B=4 still the
optimum under the fork's current kernel costs?** If RPB shifted the relative cost of small-vs-large
batch verification, the tau-maximising B may have moved since that result was established.

## F124 -- answers to the two baseline questions, and a CORRECTION to my own F123 (2026-09-04)

**Q: is F22's +2.6% projection clock-cap contaminated? -- YES. CONFIRMED.**
F22 sits inside the RUN 9 block whose header states "All figures at the 405 MHz cap; see F20 for
why the magnitudes are regime-limited." F21's magnitudes were RETIRED for exactly that defect;
F22 shares it. **So the ~3.4%-of-matmul / +2.6% t/s projection is derived from clock-capped
numbers and must be re-validated thermally-uncapped before it is relied on.**
This WEAKENS my own F123 claim that "the splice unblocks a queued +2.6% gain": the unblock
argument stands structurally (iq2_s is the blocker; the splice removes it), but the SIZE of the
prize is not established. Structural claim survives, magnitude does not.

**Q: is 17.77 pre- or post-p_min? -- PRE.** F64's reference row runs p_min at its 0.0 default.
BUT the right stacking base is NOT 17.77. F104 carries its OWN control measured in-session:
  A1 p_min=0   B=4 control  43 C  **17.51** t/s
  A3 p_min=0.6 B=4          59 C  18.38 raw -> **18.77 @43C normalised**  = **+7.2%**
So the measured p_min arm is **18.77 t/s**, not 17.77 x 1.072 = 19.05. Applying the percentage to
a different experiment's control double-counts across a 1.5% inter-session spread (17.51 vs 17.77).
Quote 18.77, the within-experiment normalised value.

**CORRECTION TO MY OWN PRIOR REPLY -- 22.18 t/s IS A REAL LOGGED MEASUREMENT.**
I told the peer that "22 t/s was never claimed as a baseline; line 3 is a target." That is WRONG.
**F67 measured 22.18 t/s**, fully logged: Ridge-q41nonffn, -c 8192, B=4, temp 0, same binary/libs/
flags as the F66 prose reference, ONLY the prompt differing -- a ~2166-token yaml echo task.
tau = **4.000, the arithmetic maximum for B=4**; acceptance 0.9722 (70/72); 24 rounds; 67->78 C.
It is not an anecdote and it does have a config trail. What it IS: an **ECHO-workload** number,
not comparable to the 17.77 PROSE number. The correct objection is workload non-comparability,
NOT missing provenance. I asserted the latter without reading F67. Same failure as F120.

**INTERACTION WARNING for anyone stacking the shippable flags:** p_min=0.6 and n-max=1 are BOTH
width-reducing (p_min drops mean width 2.96 -> 1.67; n-max 1 caps B at 2). They attack the same
quantity and must NOT be assumed additive. Any combined figure needs its own measurement.

## F125 -- p_min SHIPPED and REPRODUCED (2026-09-04)

`run_ridge_mtp.sh` rewritten to F104 arm A3 verbatim. The old file was BROKEN, not merely stale:
it pointed at `Qwen3.8-27B-Ridge-3.7bpw.gguf` (no longer exists -- that name predates the surgical
requant; the champion Ridge-q41nonffn is 3.66 BPW, i.e. the same thing renamed) and used `lucebin`,
which is the MASTER kernels, not the fork's. Old file kept as run_ridge_mtp.sh.bak-20260904.

Shipped config: champion model, binary `lucefix/llama-server` with LD_LIBRARY_PATH
`sub4_2:lucefix` (sub4_2 supplies the FORK's ggml-cuda -- this split is load-bearing),
turbo4 K+V, --spec-type draft-mtp --spec-draft-n-max 3, **--spec-draft-p-min 0.6**, -c 8192,
-b 512 -ub 512, --ctx-checkpoints 2, --reasoning off.

FLAG VERIFIED ACTIVE from the server's own log, not merely accepted on the command line:
  `common_speculative_impl_draft_mtp: - n_max=3, n_min=0, p_min=0.60, n_embd=5120, backend_sampling=1`
Load VRAM 13591 MiB.

MEASURED, prose_p1 192 tokens, temp 0, seed 5:
| arm | start C | raw t/s | @43C (est) |
|---|--:|--:|--:|
| F104 A1 control p_min=0   | 43 | 17.51 | 17.51 |
| F104 A3        p_min=0.6  | 59 | 18.38 | 18.76 |
| **TODAY shipped p_min=0.6** | 51 | **18.70** | **18.89** |

Reproduces F104 A3 to **+0.70%** normalised; +7.91% over the control. The raw pair (18.70 from
51 C vs 18.38 from 59 C) shows the same direction with NO correction applied, so the conclusion
does not rest on the normalisation. Normalisation uses the documented proportional slope
(~-0.13%/degC, F97) and is an ESTIMATE.

**WORKLOAD SCOPE -- do not merge with 22.18.** This is PROSE. F67's 22.18 t/s is the ECHO
workload and is already SATURATED (tau = 4.000 = arithmetic max at B=4). p_min does not move
22.18; the two numbers are different workloads with different baselines. Conflating them is the
error corrected in F124.

## F126 -- THE "GATING EXPERIMENT" IS ALREADY IN THE FORK'S SOURCE (2026-09-04)

Fable ranked "iq4_xs at rows_per_block 2->4, ncols_dst 2..5" as THE gating experiment for the
whole precision-map direction, and I agreed and planned a cold cycle for it. Reading
`luce/ggml/src/ggml-cuda/mmvq.cu` instead: **it has already been done, and it is in the shipped
binary.** From the source comments at calc_rows_per_block (line 416ff), ncols_dst==2, m=4096
k=14336, interleaved 3 s bursts at FULL clock, replication < 0.1%:

    iq4_xs  150.84 -> 142.24  **-5.7%**   (49.7% of the champion's weights)
    iq3_s   237.34 -> 222.63  -6.2%
    q5_K    242.39 -> 235.89  -2.7%
    q4_K    219.59 -> 203.13  -7.5%
  Correctness: test-backend-ops test -o MUL_MAT, FAIL=0 OK=13.

**iq4_xs IMPROVES at the verification width. It does NOT regress like iq2_s.** That is exactly the
cell Fable named as gating, with the falsification it stated ("if iq4_xs regresses, the splice
relocates the blocker to a bigger tensor") already resolved in the favourable direction.

**AND THE n=5 REGRESSION IS ALREADY HANDLED IN SOURCE.** ncols_dst 3..8 splits the types: iq2_s
and iq3_s get their own knob `LUCE_PASCAL_RPB_N4_SUB4` (default **2**), everything else gets
`LUCE_PASCAL_RPB_N4` (default 4). The comment records the end-to-end confirmation:
  "+27% (WORSE) on iq2_s at n=5 in the microbenchmark, and that prediction was then CONFIRMED
   end-to-end: prose at B=5 lost 15.8% (15.55 -> 13.10 t/s, acceptance bit-identical) while the
   same binary GAINED at B=6 and on echo."

CONSEQUENCE FOR FABLE'S "SECOND INDEPENDENT LEG" FOR THE SPLICE: its argument was that raise-B
pushes verification into n=5 where iq2_s regresses 27%, so removing iq2_s prevents that. **But the
regression is ALREADY avoided** -- not by removing iq2_s, but by holding sub-4-bit types at
rows_per_block 2 while letting the other types take 4. The per-type table solves it without
touching the model. So the second leg is largely spent; the splice's case reverts to Track 2
(the quant swap on its own merits: iq4_xs is faster per weight than iq2_s/iq3_s at n=4).

ALSO CONFIRMED FROM SOURCE, independently of my F124 finding: the clock-ramp contamination is
already documented in-tree -- "The older per-type figures (q4_1 -7.6%, iq3_s -9.9%, iq2_s +10.1%)
came from sub-second bursts that timed the P100's 405 MHz -> 1189 MHz clock ramp rather than the
kernel, so they are unusable in either direction." My F124 rediscovered this from the RUN 9 header.
Third time this session that reading the existing record first would have saved the work (cf F120,
F124). The source comments are part of the record and I was not grepping them.

NET: Fable's #2 does not need a cold cycle for the iq4_xs cell. What remains genuinely unmeasured
is (a) rows_per_block 2 vs 4 for iq4_xs at ncols_dst 3..5 specifically -- the source measured
ncols_dst==2 and then set N4=4 for iq4_xs at 3..8 without a per-width table for it, and (b) whether
B=4 is still optimal now that p_min=0.6 cuts mean draft width to 1.67.

## F127 -- p_min AXIS CLOSED: 0.6 is a genuine local optimum (2026-09-04)

5 arms, prose_p1, cold 47 C start, 0.6 bracketed FIRST and LAST. Flag asserted from the server's
own log on every arm ("requested / log says") -- not trusted from the command line.

| p_min | start C | raw t/s | @47C (est) | vs 0.6 |
|---|--:|--:|--:|--:|
| **0.6 (open)** | 47 | **18.56** | **18.56** | -- |
| 0.5  | 60 | 17.32 | 17.61 | **-5.10%** |
| 0.7  | 63 | 17.38 | 17.74 | **-4.41%** |
| 0.75 | 67 | 16.35 | 16.78 | **-9.62%** |
| 0.6 (close) | 71 | 17.85 | 18.41 | -0.8% |

BRACKET CHECK (gate set BEFORE the run): open 18.56 @47 C vs close 17.85 @71 C = -3.83% raw; the
proportional slope explains -3.03% over 24 C, leaving a **-0.82% residual, inside the ~1.5% noise
floor. Bracket HOLDS, interior points VALID.**

**0.6 is a genuine LOCAL OPTIMUM, not best-of-three.** The curve is sharply peaked and
non-monotonic on both sides: 0.5 and 0.7 are each ~4-5% worse, 0.75 is ~10% worse, and F104's 0.3
was worse than no p_min at all. **THE p_min AXIS IS CLOSED -- no further gain available from it.**
A warm probe at 0.85 (17.80 from 58 C) had already pointed the same way and correctly retargeted
the arms from {0.75,0.85,0.95} into {0.5,0.7,0.75}.

**BASELINE CORRECTION.** Three independent 0.6 measurements today: 18.70 (51 C), 18.56 (47 C),
17.85 (71 C) -> normalised to 47 C: 18.79 / 18.56 / 18.42, a ~2% spread. **The honest champion
figure is ~18.6 +/- 0.2 t/s, not 18.70 as a point value.** F125's 18.70 was a single sample and
should not be quoted as THE baseline. Distance to the 20 t/s target is therefore ~+7.5%, not +7.0%.

## F128 -- B=3 BEATS B=4 UNDER p_min: +13.2% within-session (2026-09-04, needs replication)

B sweep at p_min=0.6, prose_p1, cold 52 C. n-max asserted from the server log on every arm.

| arm | start C | raw t/s | @47C (est) |
|---|--:|--:|--:|
| n-max 3 (B=4, current) | 52 | 17.13 | 17.24 |
| **n-max 2 (B=3)** | 64 | **19.09** | **19.51** |
| n-max 4 (B=5) | 68 | 15.96 | 16.40 |
| n-max 5 (B=6) | 72 | 15.28 | 15.78 |

**Monotonic decline with width: B=3 > B=4 > B=5 > B=6.** B=3 vs B=4 within this session = **+13.2%**.
MECHANISM, predicted in advance: p_min=0.6 cuts MEAN draft width to 1.67 (F104), so verifying at
B=4 pays for ~2.3 columns the drafter rarely proposes. B=3 matches the width actually produced.
A warm probe had already excluded n-max 1 (B=2): 16.43 from 72 C -- too narrow, over-narrowing on
top of p_min. So the optimum is interior at B=3.

**TWO VALIDITY PROBLEMS -- 19.51 IS NOT YET A BASELINE:**
1. The closing-bracket arm SKIPPED at 76 C, so this session has NO thermal validity check.
2. **Same-config disagreement ACROSS sessions:** n-max 3 / p_min 0.6 measured 18.56 @47 C in the
   F127 sweep but 17.13 @52 C (= 17.24 @47 C) here -- **-7.1% for an identical configuration.**
   That is session-to-session variance beyond the thermal slope, and it is larger than several
   effects we have treated as real. Absolute values across sessions are therefore NOT comparable
   at better than ~7%; only WITHIN-session interleaved comparisons are.

What survives: the ORDERING (interleaved, one thermal state, monotonic) is well supported.
The MAGNITUDE and the absolute 19.51 are not. Decisive replication launched: B=3 and B=4
interleaved 2:3:2:3 from cold, giving two of each under equal thermal treatment.

**IMPLICATION IF IT REPLICATES:** B=3 is a launcher flag (`NMAX=2`), no build, no requant -- and
it would put prose at or near the 20 t/s target on its own. This also retires the assumption that
B=4 is the champion's operating point: that was established pre-p_min, and p_min moved the optimum.
Every ps/weight table indexed at "n=4 because that is the operating point" (F118, F122, the
real-shape sweep as designed) would need re-indexing to n=3.

## F129 -- B=3 REPLICATES and is SHIPPED; my +13.2% and my -7.1% variance claim BOTH RETRACTED

Interleaved 2:3:2:3, prose_p1, cold 51 C, all four arms completed (no skips), n-max asserted from
the server log each arm.

| arm | start C | raw t/s | @47C (est) |
|---|--:|--:|--:|
| n-max 2 (B=3) | 52 | 19.28 | 19.41 |
| n-max 3 (B=4) | 63 | 18.25 | 18.63 |
| n-max 2 (B=3) | 67 | 18.91 | 19.40 |
| n-max 3 (B=4) | 70 | 17.88 | 18.41 |

  B=3 pair spread **0.02%**, mean **19.40**.  B=4 pair spread 1.15%, mean 18.52.
  **B=3 is +4.76% over B=4. Direction REPLICATES.**

**RETRACTION 1 -- F128's +13.2% is WRONG. The true gain is ~4.8%.** The outlier was the first
sweep's B=4 arm (17.24 @47 C), not B=3. B=4 now reads 18.52, consistent with F127's independent
18.56 (-0.22%). I attributed the discrepancy to the wrong arm.

**RETRACTION 2 -- the "-7.1% cross-session instrument variance" (F128) is largely RETRACTED.**
Within-session replication is 0.02% (B=3) and 1.15% (B=4); cross-session B=4 agrees to -0.22%.
So the -7.1% was ONE anomalous arm, NOT a systemic 7% noise floor. This matters because I had
told the peer that -7.1% "bounds what any single-session percentage can claim" and that it was
"larger than several effects we have been treating as real" -- which then put the -5.7% iq4_xs
gating cell and the -2.7% q5_K cell under suspicion. **That suspicion is withdrawn.** Working
noise floor is ~1-1.5% within session, consistent with the harness's long-standing figure.
Lesson: one discrepant arm is not an instrument characterisation. I generalised from n=1.

**SHIPPED:** run_ridge_mtp.sh n-max default 3 -> 2. Flags only, no build, no requant.

**PROSE BASELINE NOW: 19.40 t/s @47 C** (from 17.51 pre-p_min = **+10.8% total**, both flags).
Distance to the 20 t/s target: **+3.07%**.

**CONSEQUENCE -- the n=4 indexing IS wrong.** B=4 has not been the operating point since p_min
shipped. Every table indexed at "n=4 because that is the operating point" needs re-indexing to
n=3: F118's ps/weight table, F122's allocation and the 0-11% splice band, and the real-shape
sweep as built. Fable's guidance (in a consultation) is to re-index FULLY to n=3 rather than split
n=3/n=4 or drop a type -- "getting the width right dominates getting full type coverage at a
width you don't use." Adopted.
  Its refinement accepted too: the q2_K reversal spans a 3-step gap (n=1->n=4) whereas n=3 vs n=4
  is ONE step, so a ranking flip is less likely than I implied -- but the sweep is being re-indexed
  regardless, since it is the load-bearing table for the allocation.

**ANSWER to Fable's challenge on the -5.7% iq4_xs cell:** it was NOT a cross-session absolute.
The source comment states "Measured per type, two interleaved passes each, replication < 0.1%"
at full clock in >=3 s bursts, 0 FAIL. Interleaved and bracketed within one session -- the
discipline it asked for was already applied. Same for the q5_K -2.7% cell in the same table.

## F130 -- REAL-SHAPE ps/weight at n=3: the splice case STRENGTHENS; aspect ratio is a NEW axis

8 cases, cold 51 C, temps 51->58 C only (real shapes ran COOLER than the probe-shape budget
predicted). q4_1 bracketed first and last: 253.45 / 254.73 us = **0.50% spread**. Interleaved
across both shapes. Binary: master + real shapes compiled in (test-backend-ops.cpp, new case block
for m=17408 k=5120 and m=5120 k=17408 at bs 1..6). 89.13M weights per case, vs 58.72M on the probe.

| type | ffn_gate/up (17408x5120) | ffn_down (5120x17408) | probe shape @n=4 |
|---|--:|--:|--:|
| q4_1   | **2.851** | -- | 3.481 |
| iq4_xs | 3.412 | **3.016** | 3.834 |
| iq2_s  | 4.333 | 3.907 | 4.677 |
| iq3_s  | 4.859 | 4.465 | 5.191 |

**THE SPLICE CASE STRENGTHENS ON REAL GEOMETRY:**
  ffn_gate/up: iq2_s->iq4_xs **-21.3%**,  iq3_s->iq4_xs **-29.8%**
  ffn_down:    iq2_s->iq4_xs **-22.8%**,  iq3_s->iq4_xs **-32.4%**
The probe shape at n=4 predicted only -18.0% for iq2_s->iq4_xs. Real shapes deliver -21 to -23%.
So the single-probe-shape extrapolation was CONSERVATIVE, not optimistic -- the opposite of what
I and both Fable instances feared. Type ORDERING is unchanged from the probe
(q4_1 < iq4_xs < iq2_s < iq3_s), so Fable's 1-step-vs-3-step argument held: no ranking flip
between n=3 and n=4.

**NEW AXIS -- ASPECT RATIO, invisible on a single probe shape:**
  iq2_s  gate 4.333 vs down 3.907  = down **-9.8%**
  iq4_xs gate 3.412 vs down 3.016  = down **-11.6%**
  iq3_s  gate 4.859 vs down 4.465  = down **-8.1%**
ffn_down (tall-thin, m=5120 k=17408) is consistently FASTER PER WEIGHT than ffn_gate/up
(wide-flat, m=17408 k=5120) for the SAME type and the SAME weight count. This changes which
TENSORS to prioritise, not merely which types -- the precision map should be indexed by
(type, shape-class), not type alone. F122's allocation used one cost per type and is therefore
under-specified. Mechanism not established: candidate is rows-per-block/occupancy interacting
with m, since the grid is sized by m/rpb.

STATUS: this was the prerequisite Fable and I both ranked #1 for trusting any splice projection.
It is now measured. The allocation must be re-derived at these numbers (per shape class), which
is analysis-only, no GPU.

## F131 -- allocation RE-DERIVED at real-shape n=3 costs; one self-correction

Ranked by time-saved-per-byte, indexed by (type, shape-class), real-shape n=3 costs (F130):
  iq3_s down x16  ratio 14.96 | iq3_s gate x16  14.94 | iq2_s gate x112  4.41 | iq2_s down x48  4.27

**SELF-CORRECTION:** I first wrote "a type-only table cannot express this." That OVERSTATES it.
Within-type shape-class differences are 0.1% and 3.3%; BETWEEN types the ratio gap is 3.4x. Type
DOMINATES and shape-class does not reorder the groups -- so the type-only ordering (all iq3_s,
then all iq2_s) is essentially correct, which independently re-validates the original F115
ordering I had earlier predicted was wrong (F118 retraction). Shape-class matters for predicting
ABSOLUTE time, not for choosing upgrade order.

| budget MiB | upgraded | mutable time | end-to-end (proj) | projected t/s from 19.40 |
|---|--:|--:|--:|--:|
| 13000 |  80/192 | -11.01% | **-7.24%** | **20.80** |
| 13400 | 102/192 | -13.48% | -8.86% | 21.12 |
| 13950 | 131/192 | -16.73% | -11.00% | 21.53 |

All three CLEAR the 20 t/s target ON PAPER. **This is PROJECTED**, from kernel microbenchmark time
to wall clock, using MUL_MAT's 87.7% share (F114). It is NOT a result and must not be quoted as
one. Prior projections tonight have erred in both directions (the probe shape was conservative by
3-5 points on the splice; my B-sweep magnitude was optimistic by 8 points).

The band is now real-shape and real-width, so the honest splice range tightens from "0 to ~11%"
to roughly **-7% to -11% end-to-end, conditional on the splice being buildable at VRAM.**

## F132 -- p_min RE-TUNE under B=3: 0.6 HOLDS; baseline consolidated at 19.43 t/s

Fable's top pick, and a gap in my own reasoning: F124 recorded MY warning that p_min and n-max are
both width-reducing and must not be assumed independent, then F129 shipped B=3 without re-testing
p_min against it. p_min had been tuned under B=4 and B moved underneath it.

4 arms, n-max 2 throughout, cold 44 C, 0.6 bracketed first and last. [INTERLEAVED, BRACKETED]
| p_min | start C | raw | @47C |
|---|--:|--:|--:|
| 0.6 (open)  | 44 | 19.49 | 19.41 |
| 0.45 | 57 | 17.43 | 17.66 (**-9.2%**) |
| 0.7  | 61 | 17.52 | 17.84 (**-8.3%**) |
| 0.6 (close) | 65 | 19.04 | 19.49 |
Bracket spread **0.37%** -- holds. **p_min=0.6 REMAINS the joint optimum under B=3; staleness
~0%.** Fable predicted 0-2% and was right. The peak stays sharp: +-0.1 in p_min costs 8-9%.

**CONSOLIDATED BASELINE -- 4 independent B=3/p_min=0.6 arms today: 19.41, 19.40, 19.41, 19.49
=> 19.43 t/s @47 C, spread 0.44%.** This is a properly replicated figure, not a single sample.
Distance to the 20 t/s target: **+2.95%**. Total from the 17.51 pre-p_min control: **+11.0%**,
all from two launcher flags, no build, no model change.

## F133 -- shape-class COVERAGE CONFIRMED COMPLETE (Fable's pre-build check)
All 192 splice candidates are exactly three families and NO attention projections:
  ffn_down  iq2_s 48 + iq3_s 16   (17408x5120)
  ffn_gate  iq2_s 56 + iq3_s  8   (5120x17408)
  ffn_up    iq2_s 56 + iq3_s  8   (5120x17408)
ffn_gate and ffn_up share geometry, so the two shapes measured in F130 cover **100% of the
candidate set**. Nothing sits unprobed; the tier curve's absolute-time and VRAM-per-tensor inputs
are complete.

**ADOPTED from Fable (d), free:** ffn_down shows the LARGER relative gain in both type pairs
(-22.8%/-32.4% vs -21.3%/-29.8%), so within a fixed VRAM budget prefer ffn_down FIRST. The
re-derived F131 ranking already does this (iq3_s/down ratio 14.96 ranks first), but it now has an
explicit rationale rather than falling out of the arithmetic.

## F134 -- MUL_MAT share re-measured on the CURRENT fork; Fable's mechanism CONFIRMED but its
##          algebra is INVERTED, so the projection is unchanged

LUCE_PHASE on phasefork (sym=4 asserted), champion model, p_min 0.6, cold 49 C, 48 tokens.
Instrument serialises every op so ABSOLUTE totals are inflated (1.59x at B=3, 1.31x at B=6) and
are NOT comparable to unpatched timings; SHARES within one arm are what this measures.

| | MUL_MAT | share | other | MUL_MAT/token | other/token |
|---|--:|--:|--:|--:|--:|
| B=3 (shipped) | 4107.3 ms | **86.5%** | 640.7 ms | 85.57 ms | 13.35 ms |
| B=6 | 3212.4 ms | 82.0% | 706.0 ms | 66.92 ms | 14.71 ms |

**MECHANISM CONFIRMED:** between B=3 and B=6, MUL_MAT/token moves **-21.8%** while other/token
moves only **+10.2%** -- i.e. non-MUL_MAT time is much closer to FIXED-ABSOLUTE than proportional
to MUL_MAT. That is exactly the property Fable proposed, and it matches the MMQ finding
(469-483 us near-constant across quant types).

**BUT THE ALGEBRAIC CONSEQUENCE IS THE OPPOSITE OF WHAT IT STATED.** With total = MM + other,
share s, and MUL_MAT shrinking by fraction f:
    other FIXED         => new = MM(1-f) + other  => delta/total = **-f*s**
    other PROPORTIONAL  => new = (MM+other)(1-f)  => delta/total = -f
**Multiplying by the share IS the fixed-other form -- the conservative one.** Fable said
share-multiplication "assumes the non-MUL_MAT part is proportional to wall time"; that is
inverted. Assuming proportional would REMOVE the multiplier and predict MORE gain, not less.
F131 already used delta = f x (mutable/trunk) x share, so the confirmed fixed-absolute behaviour
is precisely what our projection assumes. **No correction to the projection follows from the
mechanism.** Its instinct to check was still worth the cycle -- the share value needed refreshing
and the fixed-vs-proportional question is now measured rather than assumed.

**The only real correction is the share value, and it is small:** historical 87.7% (F114, master
at B=1) -> **86.5% measured on the current fork at B=3**.
| budget | was | now | projected t/s from 19.43 |
|---|--:|--:|--:|
| 13000 | 7.24% | **7.14%** | **20.82** |
| 13400 | 8.86% | 8.74% | 21.13 |
| 13950 | 11.00% | 10.85% | 21.54 |

NOTE the B=6 share (82.0%) is lower, so at wider B the non-MUL_MAT floor is a bigger fraction --
consistent with why widening B loses on prose. Independent support for the F129 B=3 result from a
completely different instrument.

## F135 -- THE DONOR IS A MIXED UD QUANT, NOT IQ4_XS. Splice re-scoped. (2026-09-04)

The splice tool's raw TYPE-ID check (Fable's audit item, made fatal) fired on the first dry run:
  ERROR blk.0.ffn_gate.weight: donor raw type-id is 17 (IQ2_XS), not IQ4_XS(23)
  ERROR blk.1.ffn_up.weight:   donor raw type-id is 11 (Q3_K),   not IQ4_XS(23)
  ... 52 of 80 failed. Nothing written.

`Qwen3.8-27B-UD-IQ4_XS.gguf` is an **Unsloth Dynamic** quant: only **50.69% iq4_xs** by size, with
q5_K 14.7%, iq3_s 11.4%, q4_K 8.9%, q3_K 5.6%, iq3_xxs 3.8%, q6_K 2.5%, iq2_s 1.4%, iq2_xs 0.4%.
The filename advertises one type; the file is a per-tensor mixture. **This is exactly the hazard
the audit named -- "verify raw type-id not the display label" -- and the check earned its place.**

**AND IT WAS ALREADY IN MEMORY:** an earlier note records "Champion GGUF is a MIXED
UD quant, not pure IQ4_XS; iq4_xs only 49.7%". I had this and did not apply it. **FOURTH instance
today** of a known fact re-derived at cost (cf F120, F124, F126). The pattern is now specific:
I grep the LEDGER but not the MEMORY FILES or source comments. All three are the record.

**FEASIBLE SET: 105 of 192 candidates have a genuine iq4_xs donor.**
  by (champion type, class): iq2_s/gate 63, iq2_s/down 33, iq3_s/down 6, iq3_s/gate 3
The other 87 would have to take iq3_s(37), iq3_xxs(14), q5_K(14), iq2_s(7), q4_K(7), q3_K(6),
iq2_xs(2) -- several of which are the champion's own type or WORSE, and none measured at real
shapes. Note only 9 of 32 iq3_s candidates are upgradable, and iq3_s was the top-ranked group.

**RE-PROJECTED on the feasible set** (round-time saving -> t/s as 19.43/(1-r), MUL_MAT share 86.5%):
| budget | upgraded | MiB | round time | projected t/s |
|---|--:|--:|--:|--:|
| 13000 | 68 | 12992.7 | +5.31% | **20.52** |
| 13400 | 89 | 13383.2 | +6.80% | 20.85 |
| 13950 | 105 | 13680.7 | +7.92% | 21.10 |
vs the infeasible plan's 20.82 at 13000. **Still clears 20 t/s at every tier**, and the smallest
tier alone (68 tensors, 12993 MiB, barely above the champion's 11818) is projected 20.52.

**A SIGN ERROR I MADE AND CAUGHT:** first pass computed t/s as 19.43*(1+r) with r negative,
giving 18.40 -- i.e. reporting a time SAVING as a throughput LOSS. Correct form is 19.43/(1-r).
F131/F134 used the magnitude and were unaffected.

**OPEN OPPORTUNITY, not yet costed:** the donor's q5_K/q4_K tensors may still beat the champion's
iq2_s at real shapes -- the splice does not have to be iq4_xs-only. q5_K, q4_K, q3_K, iq3_xxs are
all UNMEASURED at real shapes. Adding them to a future real-shape sweep could extend the feasible
set well beyond 105.

## F136 -- SPLICE MEASURED: it LOSES 5.4%. Projection FALSIFIED. Root cause = the AGGREGATE
##          BANDWIDTH TERM our whole precision-map framework omitted. (2026-09-04)

**2026-09-11 audit correction — see F153.** The claimed 426 GB/s ceiling is exceeded by a
direct triad measurement (approximately 541–547 GB/s). The 266 GB/s figure combines splice
size with champion timing and is not a hardware-counter reading. The saturation mechanism
asserted below is **UNVERIFIED**; the historical splice timings remain recorded as measured.

Interleaved S:C:S:C, cold 51 C, shipped config (B=3, p_min 0.6). Quality gate PASSED on every arm
(17x23 -> 391 correct, so the splice is a valid model).

| arm | start C | raw t/s | @47C | VRAM |
|---|--:|--:|--:|--:|
| splice s1 | 51 | 18.14 | **18.23** | 14577 MiB |
| champion c1 | 64 | 18.85 | **19.27** | 13441 MiB |
| splice s2 | 72 | 15.45 | 15.95 (SUSPECT, hot start) | 14577 MiB |
| champion c2 | -- | SKIPPED 78 C | | |

**splice vs champion = -5.4%.** Projection said **+5.31% -> 20.52 t/s**. Measured **18.23**.
The splice arms disagree by 12.5% (s2 started at 72 C); s1 is the usable one and c2 was lost to
the guard, so this needs a cleaner replication -- but the DIRECTION is unambiguous and the
mechanism below predicts the magnitude to within 0.1 points, which is far better corroboration
than a second arm.

**ROOT CAUSE -- the term the per-tensor metric structurally cannot see:**
  champion 12.514 GB/token -> floor 29.4 ms at the measured 426 GB/s achievable
  splice   13.705 GB/token -> floor 32.2 ms          => **+2.8 ms/token**
  champion round 51.5 ms + 2.8 = 54.3 ms => 18.43 t/s = **-5.2% PREDICTED FROM BYTES ALONE**
  **MEASURED -5.4%. The bandwidth term alone explains essentially the entire result.**

Why ps/weight misled us even though it IS byte-inclusive: each microbench case streams ONE
89M-weight tensor in isolation, where it is UNPACK-bound and nowhere near the bus limit --
  iq2_s 27.8 MB in 386 us = 72 GB/s ; iq4_xs 47.3 MB in 304 us = 155 GB/s (vs 426 achievable).
But the FULL model streams 13.7 GB/token and achieves 266 GB/s = **63% of achievable**, i.e.
**in aggregate the model IS bandwidth-bound.** Kernels overlap and collectively saturate the bus,
so a per-kernel unpack win does NOT survive if it adds bytes. iq4_xs is 4.25 bpw vs iq2_s 2.5 --
**70% more bytes** on every upgraded tensor.

**THE FRAMEWORK ERROR:** F122/F131 minimised sum(params x ps_per_weight) subject to a VRAM cap,
treating VRAM purely as a capacity constraint. But on this card **bytes are also a TIME cost**.
The correct objective is closer to
    minimise  max( unpack_time , total_bytes / BW_achievable )
and under it, moving UP in bit-width is nearly always wrong while the model is bandwidth-bound.
Every projection in F122/F131/F135 (-5.3% to -11%, 20.52-21.54 t/s) is therefore VOID.

**COROLLARY -- the direction of the whole precision map INVERTS.** If we are bandwidth-bound at
63% of achievable, the lever is FEWER BYTES, not more: going DOWN in bit-width on the least
quality-critical tensors should now be tested. That also retro-explains why the champion
(3.66 BPW) beats the all-iq4_xs Ridge-splice (4.28 BPW) and why F135's "still clears 20 at every
tier" was backwards -- the tiers were ranked by ADDED bytes, i.e. by increasing harm.

**WHAT SURVIVES:** the splice TOOL is validated (all audit checks fired correctly, self-test
passed, quality gate passed, the type-id check caught a 65%-wrong donor). The MECHANISM works;
the OBJECTIVE was wrong. And the real-shape ps/weight table (F130) is still correct for what it
measures -- isolated per-kernel cost -- it just is not the whole-model objective.

## F137 -- CONSEQUENCE OF F136: no byte lever exists, and tau is the ONLY remaining axis

Byte pools in the champion, checked for a "go DOWN in bpw" lever now that fewer bytes = less time:
| pool | params | bpw | MiB |
|---|--:|--:|--:|
| q4_1, 213 non-FFN | 8.59 G | 5.0000 | 5120.3 |
| iq2_s, FFN | 14.26 G | 2.5000 | 4250.0 |
| q6_K, 4x blk.64 + token_embd | 1.59 G | 6.5625 | 1244.8 |
| iq3_s, FFN | 2.85 G | 3.4375 | 1168.8 |

  * q6_K is the fattest format but is NOT a lever: 4 of its 5 tensors are blk.64 (the MTP head,
    where precision acts on ACCEPTANCE) and the 5th is token_embd, a GET_ROWS lookup reading ONE
    row per token -- it contributes ~nothing to streamed bandwidth.
  * q4_1 -> q4_0: saves 537 MiB = 1.26 ms/token, costs +0.30 ps/w x 8.59 G = +2.6 ms. NET WORSE.
  * q4_1 -> iq4_xs: saves 768 MiB = 1.80 ms, costs +0.56 ps/w x 8.59 G = +4.8 ms. NET WORSE.

**=> NO byte lever is available without quality loss. The champion is already near-optimal on the
bandwidth objective -- which retro-explains why it beat every fatter variant we ever built
(Ridge-splice 4.28 BPW, v15/v18, etc.). Those were not lost to kernel details; they were lost to
bytes.**

**THE ONLY REMAINING AXIS IS tau.** Each target forward reads the full ~12.5 GB regardless of
draft width, so t/s = tau / time_per_forward, and time_per_forward is bandwidth-floored.
  measured tau 2.341 (B=4, p_min 0.6) vs the B=3 arithmetic maximum of 3.000 => **+28.2% = ~24.9 t/s**
That is almost exactly the 25 t/s stretch target, and it is the door Fable independently
identified. It requires a BETTER DRAFT MECHANISM -- structural R&D, not a flag or a sweep.
Do NOT treat "tau is uncharacterised" as permission to assume the headroom is reachable.

INDEPENDENT CORROBORATION of a structural ceiling (Fable): the dual-P100 community reports
reach ~20 t/s on TWO cards with plain UD-Q4_K_M/Q5_K_XL configs -- barely above our 19.43 on ONE.
If doubling the hardware on this architecture family buys ~nothing, that points at a structural
limit rather than an unexploited tuning axis. Three independent lines (splice bytes, V-cache
bounds, hardware scaling) now agree.

## F138 -- V-cache decider RUNNING, with my own prediction REVERSED by F136
Queued as batch28 P2 long ago, never reported. K is already q8_0 (runtime auto-upgrade on the
6:1 GQA), so only V is in question; both arms pin K=q8_0 so the comparison is V-only.
**I queued this arguing q8_0 V would WIN (unpack cheaper than turbo4). F136 reverses that:**
if the model is aggregate-bandwidth-bound, turbo4 V = fewer bytes = FASTER, and q8_0 should LOSE.
Running it regardless -- it closes an open experiment cheaply, and a decided loss is still a
decided question. Fable flagged the same sign risk independently, plus two arguments I did not
have: (1) the unpack-dominates finding is specific to CODEBOOK i-quants, and nothing
characterises turbo4's per-element cost; (2) KV reads scale with context and sweep the whole
history every layer every step -- a structurally different access pattern from fixed-size weight
reads. Its ceiling note: attention/KV is only ~9.5-13.5% of time, V a subset, so the effect is
bounded either way. Also worth recording: the runtime's K-but-not-V asymmetry is plausibly
CORRECT, since K errors distort the softmax multiplicatively while V errors are additive.

## F139 -- V-CACHE DECIDED: turbo4 V WINS by 6.1%. Open experiment closed. (2026-09-04)

batch28 P2, queued long ago and never reported, finally answered. K pinned q8_0 on both arms
(it auto-upgrades anyway), so this is a V-only comparison. Interleaved t:q:t:q, cold 50 C.

| arm | start C | raw t/s | @47C | VRAM |
|---|--:|--:|--:|--:|
| turbo4 V | 50 | 19.31 | 19.39 | 13441 MiB |
| q8_0 V | 62 | 18.02 | 18.37 | 13509 MiB |
| turbo4 V | 66 | 18.93 | 19.40 | 13441 MiB |
| q8_0 V | 69 | 17.67 | 18.18 | 13509 MiB |

turbo4 pair spread **0.06%** (mean 19.39); q8_0 pair spread 1.07% (mean 18.27).
**turbo4 V is +6.1% better. The shipped config was already correct.**
My ORIGINAL prediction (q8_0 would win, since unpack dominates bytes) was WRONG. The prediction
I REVERSED after F136 (turbo4 = fewer bytes = faster) is confirmed. Fable's independent sign
warning was right, on the grounds that the unpack-dominates result is specific to codebook
i-quants and nothing characterises turbo4's per-element cost.

**BUT THE MAGNITUDE REFUTES THE SIMPLE BYTE STORY, AND THIS BOUNDS F136's SCOPE.**
The VRAM difference is only 68 MiB = 0.16 ms/token = ~0.3% at 426 GB/s. Measured is **6.1%,
20x larger than bytes predict.** So this is NOT the F136 aggregate-bandwidth mechanism -- it is
the KV/attention KERNEL PATH: turbo4's dequant inside flash-attention is evidently cheaper or
better specialised than q8_0's. Cause unidentified; the decision is settled regardless.

**SCOPE CORRECTION TO F136:** the bandwidth floor predicted the SPLICE to 0.1 points (-5.2% vs
-5.4%) but predicts the V-cache result badly (-0.3% vs -6.1%). **F136 governs the WEIGHT-STREAM
path, not the KV path.** Do not generalise it to all of decode. Two distinct regimes coexist:
weights are aggregate-bandwidth-bound, KV/attention is kernel-bound.

INSTRUMENT NOTE: the tau readout returned acc=None/None -- the /completion response carries no
draft_n_accepted/draft_n fields on this build. tau must come from the SERVER LOG's own draft
accounting (as F104 did via rounds = n_out - n_accepted), not from the JSON response. Fix before
the tau diagnostic.

**CONSOLIDATED PROSE BASELINE, now 6 independent arms of the shipped config:**
19.41, 19.40, 19.41, 19.49, 19.39, 19.40 => **19.42 t/s @47 C, spread 0.52%.** Rock solid.
Distance to 20 t/s: **+3.0%.** No remaining flag lever: p_min closed (F127/F132), B closed
(F129), V-cache closed (F139), splice negative (F136), byte levers exhausted (F137).

## F140 -- TAU DIAGNOSTIC: acceptance is ALREADY 87%. The drafter is NOT the weak link.

From the server's own draft accounting (shipped config, B=3, p_min 0.6, prose_p1 192 tokens):
    draft acceptance rate = 0.86992 (107 accepted / 123 generated)
    #calls(b,g,a) = 1, 84, 72 ; #gen drafts 72, #acc drafts 68 ; dur(g) = 614.7 ms
  rounds = 192 - 107 = 85  =>  **tau = 2.259** (B=3 arithmetic max = 3.000)
  mean tokens PROPOSED per round = **1.447** of a possible 2.000
  drafter cost = 614.7 ms of ~9960 ms = **6.2% of wall time**

**THIS OVERTURNS THE "BUILD A BETTER DRAFTER" PREMISE (mine and Fable's).** Acceptance is already
87%. Fable's (b) diagnostic was designed to split "real entropy" from "drafter weakness" -- but
the accounting shows the drafter is not the binding constraint at all. The constraint is that
**p_min=0.6 truncates it to proposing 1.45 of 2 available columns.**

  if it proposed the full 2.000 at the same 87% acceptance:
    accepted/round 1.740 -> tau 2.740 -> **+21.3% = 23.56 t/s**

**But that width is not free, and we measured that directly:** p_min 0 and 0.45 are both far worse
(F127/F132), and B=4 is worse than B=3 (F129). So proposing more costs more than it earns TODAY.
Consequence: verification time is **NOT flat in width**. At 63% of achievable bandwidth the model
is part bandwidth-bound and part compute-bound, and the compute part scales with width. This also
bounds F136 further -- another regime, alongside F139's KV finding.

**THE SYNTHESIS -- tau and the RPB kernel work are the SAME problem:**
tau is capped by the COST OF VERIFICATION WIDTH, not by drafter accuracy. Lower the per-column
verification cost and the measured optima for p_min and B both shift WIDER, which raises tau --
a COMPOUNDING gain, not an additive one. The arithmetic ceiling if width became cheap is ~23.6 t/s
at B=3, more at B=4.
  F126: iq4_xs rpb 2->4 at ncols_dst==2 = -5.7% (measured, in-tree). But the FFN is ~52% iq2_s and
  iq2_s REGRESSES (+27% at n=5, confirmed end-to-end), so the per-type table pins sub-4-bit at 2.
  **=> THE ONE REMAINING LEVER IS A FASTER iq2_s KERNEL AT ncols_dst 3..5.**
That is real kernel work (the 780-vs-318 SASS / 152-vs-88 register / 20%-vs-36% occupancy problem
on the type that is half the model), and it is the only identified path that moves tau. The
four-arm ablation queued long ago -- Arm B `__constant__` for iq2s_grid, Arm C --maxrregcount
95-123, Arm D gather-elimination -- targets exactly this, and LUCE_MMQ_OCC is already built and
never run.

**ADOPTED from Fable (a):** RPB is REINTERPRETED not retracted -- it changes no bytes, so under a
pure bandwidth floor it should not have moved wall clock; the reconciling mechanism is that slow
unpack leaves the bus IDLE and faster unpack keeps it busier. **The correct whole-model diagnostic
is therefore "% of achievable bandwidth realized" = 63% today**, which both replaces "lower
ps/weight" as the target metric and bounds the remaining upside at closing 63% -> 100%.
Its point (a)(2) is accepted: the B=3 result and the splice failure are plausibly the same
mechanism seen from the tau side. Its (a)(3) is accepted: the -5.7% iq4_xs cell inherits the
isolated-methodology suspicion and must not be reused as if whole-model-validated.

## F141 -- "two forwards per round" INFERENCE RETRACTED before it became a finding

I inferred from LUCE_PHASE op counts (MUL_MAT n=14349, 21.2 rounds => 675/round) against a
hand-counted 361 matmuls-per-forward that the system runs **1.87 forward-equivalents per round**,
i.e. nearly two full target forwards -- which would have been the largest inefficiency found all
session. **It is an artefact of my undercount.** Every quantized tensor is a matmul operand, so
the right denominator is ~506 (the ttf line count / 501 by type tally), not 361. At 506 the figure
is **1.33 forward-equivalents per round = ONE target forward plus a cheap draft**, which is the
normal structure of speculative decoding.

INDEPENDENT CORROBORATION THAT WAS IN THE SAME LOG LINE: `dur(b,g,a) = 0.003, 614.692, 0.091 ms`
over 84 draft calls = **7.3 ms per draft call**, versus ~50 ms for a full target forward. The MTP
draft is genuinely just the small head. I should have read dur(g) before inferring from op counts.
Also confirmed from source: every llama_decode in common/speculative.cpp runs on ctx_dft, and the
MTP draft context carries only the head path.

So Fable's (c) reduces to the LM-head question alone: output.weight is q4_1, [5120 x 248320],
1.272 G params = **758 MiB**; if it is read 3x per round rather than batched once, that is
+1516 MiB/round = ~3.0% of round time. Real but small, and still unverified.

**RUNNING TALLY OF MY OWN RETRACTIONS TODAY: five.** (1) B=3 +13.2% -> +4.8%; (2) -7.1% instrument
variance -> ~1-1.5%; (3) sign error reporting a time saving as a throughput loss; (4) "type-only
table cannot express shape-class" -> type dominates; (5) this one. Every single one was caught by
checking against data already in hand rather than by a new measurement. The failure mode is
consistent: I compute a striking number and reason forward from it before validating the inputs.

## F142 -- iq2_s rpb=4 at ncols_dst=3 ALSO REGRESSES (-3.1%). Lever DEAD. (2026-09-04)
## [AMENDED F147] -3.1% at n=3 stands. The "general across n=3..5" claim is WITHDRAWN:
## n=4 was only ever measured in the clock-cap-contaminated sweep. Read F147 first.

Interleaved r:s:r:s, cold 49 C, shipped config otherwise. Quality gate passed all arms (391).
  rpb=4: 18.80 / 18.87 @47C  spread 0.39%  mean **18.84**
  rpb=2: 19.52 / 19.35 @47C  spread 0.88%  mean **19.43**  (= shipped)
**rpb=4 is -3.1%.** So the iq2_s wide-n regression is GENERAL across n=3..5, not an n=5 artefact.
The shipped `LUCE_PASCAL_RPB_N4_SUB4=2` is correct at our actual operating width. Build verified
before the run (flag present in flags.make, mmvq.cu recompiled, own dir, deps checked, warm-
validated coherent). **This was the cheapest route to cheaper verification width, and it is dead.**

## F143 -- ACCEPTANCE DEGRADES WITH WIDTH: the tau ceiling is ~21.4 t/s, not 23.6

Reconstructed from draft accounting ALREADY IN LOGS ON DISK -- no new cycle needed (Fable's
suggestion; the p_min and B sweeps had recorded it all along).

| mean width | acceptance | tau | B | p_min | t/s |
|--:|--:|--:|--:|--:|--:|
| 0.796 | 0.878 | 1.699 | 2 | 0.60 | 16.43 |
| 0.990 | **0.969** | 1.959 | 4 | 0.85 | 17.80 |
| 1.184 | 0.810 | 1.959 | 3 | 0.70 | 17.52 |
| 1.222 | 0.769 | 1.939 | 4 | 0.75 | 16.35 |
| **1.447** | **0.870** | **2.259** | **3** | **0.60** | **19.49 (SHIPPED)** |
| 1.494 | 0.774 | 2.157 | 4 | 0.70 | 17.38 |
| 1.648 | 0.673 | 2.110 | 3 | 0.45 | 17.43 |
| 1.671 | 0.803 | 2.341 | 4 | 0.60 | 18.56 |
| 2.338 | **0.639** | **2.494** | 4 | 0.50 | 17.32 |

**CORRECTED (peer-flagged, verified): DO NOT record this as a curve.** I originally wrote
"acceptance falls monotonically as proposed width rises" citing 1.447 -> 1.671 -> 2.338. That is a
SELECTED SUBSEQUENCE, not the series: the full set has **3 violations in 8 steps**
(0.796->0.990 rises 0.878->0.969; 1.222->1.447 rises 0.769->0.870; 1.648->1.671 rises 0.673->0.803).
It is also CONFOUNDED and not merely noisy: 1.648 (B=3, p_min 0.45, acc 0.673) vs 1.671 (B=4,
p_min 0.60, acc 0.803) are near-identical in width but 13 points apart in acceptance, with BOTH
other variables differing. **"Mean proposed width" may not be a sufficient statistic** -- B and
p_min are plausibly separate causal levers, and collapsing them into one derived scalar discarded
real structure. **Recorded status: insufficient, non-orthogonal data. The only defensible claim is
the single decisive point below, NOT a relationship.** Characterising it properly would need a
small orthogonal grid (2-3 x 2-3) -- worth noting for any future drafter effort, not worth a cycle.

**WHAT ACTUALLY FALSIFIES 23.56 -- ONE MEASURED POINT, no curve required:**
  **width 2.338 -> acceptance 0.639, tau 2.494, and only 17.32 t/s.**
That is a DIRECT OBSERVATION IN the target regime, not an extrapolation into it. The downward
drift in the table is NOT what falsified 23.56 and must not be cited as if it were -- otherwise
someone later re-derives false confidence from a trend that is not there. 87% was only ever
observed at width 1.447.

**F140's 23.56 t/s FIGURE IS FALSIFIED.** It plugged acceptance measured at a truncated width of
1.447 into an unobserved 2.000-column regime -- structurally the same error that cost the splice
(isolated measurement extrapolated to a different regime). At width 2.0 the measured trend gives
~0.68-0.72, so accepted/round ~1.40 and tau ~2.40, not 2.740.

**THE REAL tau CEILING:** highest tau ever observed is **2.494** (B=4, p_min 0.5, width 2.338) and
it delivers only **17.32 t/s** -- worse than shipped, because the wider round costs more time than
the extra tokens earn. Even if verification width were FREE, tau caps near 2.49, giving
**~21.4 t/s**, not 23.6. And width is not free: F142 just measured -3.1% for the one lever that
would have cheapened it.

**CONCLUSION: the shipped config sits at a genuine joint optimum in (B, p_min), and the tau axis
is now measured rather than assumed.** Every identified lever is closed:
  p_min (F127/F132) | B (F129) | V-cache (F139) | splice/bytes (F136/F137) | iq2_s rpb (F142)
  | MMQ (F100) | CUDA graphs | n-gram drafters (F64) | GDN | tau (F143)
**Practical ceiling on this hardware with known mechanisms: ~19.4 measured, ~21.4 arithmetic if a
cheaper wide-verification kernel existed. 25 t/s is not reachable.**

ADOPTED -- Fable's general pre-build REGIME TEST, now standing policy:
  compute the pure byte-delta prediction (bytes / achievable BW) and compare to a cheap
  interleaved measurement of the isolated effect BEFORE any build.
    measured ~= byte prediction (within 1-1.5%)  -> bandwidth-bound; only fewer bytes help
    measured >> byte prediction (V-cache: 20x)   -> kernel/latency-bound; bytes UNDERSTATE upside
    isolated microbench, no whole-model check    -> regime UNKNOWN, treat as unvalidated
The splice fell into the third case; the -5.7% iq4_xs cell still sits there.

## F144 -- THREADS: default 8 is already near-optimal. No gain. (warm probe, 2026-09-04)
Our launcher never set -t, so the default was unverified. It is **8** (n_threads = n_threads_batch
= 8 on the 16-thread E5-2630 v3). Warm probe, acceptance BIT-IDENTICAL (0.86992) in both arms:
  -t 4  : 18.71 from 69 C -> ~19.25 @47C   (vs 19.42 baseline: within noise)
  -t 16 : 17.24 from 74 C -> ~17.84 @47C   (**clearly worse, -8%**)
So the default is fine and over-threading hurts -- consistent with the Qwopus note that -t 8 was
catastrophic there and 4 correct, i.e. the optimum is low but 8 is not past it here. **No cold
cycle spent: expected value too low.** Warm arms, labelled as such.

## F145 -- sm_60 FP16 CARVE-OUT: REJECTED. Slower AND wrong answers. (2026-09-04)
## [AMENDED F147] RE-TAGGED build-fault-suspected. The BINARY is rejected; the FIX is
## NOT disproven. Speed number not load-bearing. Read F147 first.

`LUCE_SM60_NO_FAST_FP16=1` vs shipped default 0. Interleaved, cold 51 C. Flag verified in
flags.make, mmvq/common recompiled, own dir, deps checked.

| arm | start C | raw t/s | @47C | 17x23 |
|---|--:|--:|--:|--:|
| fp16fix f1 | 51 | 17.17 | 17.26 | **411 (WRONG)** |
| control s1 | 63 | 19.09 | **19.49** | 391 correct |
| fp16fix f2 | 68 | 16.78 | 17.24 | **411 (WRONG)** |
| control s2 | -- | TERMINATED by the 78 C guard | | |

fp16fix pair spread 0.12%, mean **17.25** vs control **19.49** => **-11.5%**, a large regression,
not the +1.5% expected. Only ONE clean control arm (s2 lost to the guard), but the fp16fix pair is
internally tight and the gap is 10x any plausible single-arm error.

**AND IT FAILS THE QUALITY GATE.** 17 x 23 = 391. Control returns 391; **fp16fix returns 411 on
both arms.** temp=0, so this is deterministic -- the carve-out perturbs numerics enough to flip
the sampled token and produce a WRONG arithmetic answer.

**REJECTED on both speed and correctness. Not shipped.** This CONTRADICTS both the theory behind
the knob (keeping FP32 at 32 bits should improve precision, not degrade output) and F82's +1.6% on
echo with bit-identical acceptance. Either the effect is workload-specific (echo drafts are
predictable enough to be insensitive; prose is not) or my rebuild differs from the 2026-09-03
fp16fix binary -- which I deleted this morning and therefore cannot diff against. **That deletion
now has a real cost: an unreproducible prior result.**
Recorded as a caution: upstream issue #25593's fix is NOT a free win on this model/workload, and
F82's +1.6% should not be cited as a general result.

## F146 -- LM-HEAD QUESTION CLOSED FROM SOURCE: no redundant reads. (2026-09-04, no cycle spent)

Fable's (c) asked whether output.weight (q4_1, [5120 x 248320], 1.272 G params = **758 MiB**) is
recomputed once per candidate position rather than batched -- which would be pure waste, an
implementation inefficiency rather than a physics tradeoff, and therefore free money if real.

**It is already batched, and the gather happens FIRST.** In `src/models/qwen35.cpp` (the dense
graph Ridge uses):
    212  if (!cparams.embeddings_nextn_masked && inp_out_ids) {
    213      cur = ggml_get_rows(ctx0, cur, inp_out_ids);      <-- select output positions FIRST
    220  cur = build_lora_mm(model.output, cur, model.output_s);  <-- ONE matmul, n = #positions
and identically for the MTP head at 631/634. Same structure in qwen35moe.cpp (237/244).

So only the needed rows reach the head, and the head runs ONCE per graph with n = number of output
positions (3 at B=3), NOT three separate full-weight reads. **No redundancy exists; the ~3.0%
I sized is not available.** The memory note "output head runs 3x per MTP round" refers to three
GRAPH invocations per round (target verify + MTP draft steps), each computing the head once for
its own positions -- structurally necessary, not waste.

**Resolved by reading source, zero GPU time.** This was the last untested candidate.

## F147 -- FABLE CLOSURE REVIEW: 2 reopened, 1 accepted, 1 noted. NOT a confirmation
## of "maximum reached". (2026-09-04, in a consultation, no cycle spent)

Consult sent: F142/F143(corrected)/F144/F145/F146 + the three unrun items + the
alternative-serving-stack check. Question asked verbatim: is 19.42 t/s at or near the
practical maximum, and is any closure too thinly evidenced. ANSWER: NO CONFIRMATION.

REOPENED 1 -- F145 is "BUILD FAILED, ROOT CAUSE NOT ISOLATED", not "fix rejected".
  Rejecting THIS BINARY from shipping: unconditional yes, 411 for 17x23 at temp=0.
  Concluding the FIX is bad: NOT SUPPORTED. The result is wrong in magnitude (-11.5%
  vs +1.5%), wrong in sign, AND wrong in correctness. The third is the tell: a flag
  whose PURPOSE is fixing silent FP32 truncation should move answers TOWARD the
  reference, not to a different wrong one. Plus: original binary deleted (no diff), and
  RPB / V-cache defaults / splice-adjacent code ALL changed since the +1.6% echo test.
  Reads as stale/incomplete recompile, bad object-file interaction, or build fault.
  PROTOCOL: rebuild clean, verify artifact (checksum, flag actually compiled in, no
  stale linked objects), run the QUALITY GATE FIRST before timing anything. If a clean
  rebuild passes and lands near +1.5%, the current F145 entry should not stand as a
  finding about the fix at all.
  => F145 RE-TAGGED: build-fault suspected. Speed number NOT load-bearing.

REOPENED 2 -- F142 number stands, GENERALISATION DOES NOT.
  -3.1% at n=3 is trusted as fact. But "general across n=3..5" rests on two clean points
  bracketing an n=4 measured ONLY in the sweep retracted for 405 MHz clock-cap
  contamination. Same shape of error that already burned this session, and it runs
  against our OWN repeated finding that per-type behaviour across n is not smooth.
  => F142 SOFTENED IN THE RECORD: "confirmed at n=3 and n=5; n=4 unconfirmed under
  clean conditions." OPEN: F142's own clock state was never explicitly stated as
  uncapped (it was stated for other results that night). Unresolved.

ACCEPTED -- F146. Direct code-level fact, not a benchmark inference. Untouched.

NOTED, NOT FIXED -- F144. Warm probe + single arm are the two conditions behind two of
tonight's five retractions, so -8% / "within noise" are NOT settled. But nothing
proposes moving off the default the weak test already favours, so it changes no
decision. Fable explicitly does not want a cycle spent here.

THE THREE UNRUN ITEMS, RANKED BY FABLE -- and its pick is NOT a kernel item:
  1. ORTHOGONAL B x p_min GRID -- PRIORITISE. Lives entirely in drafting economics,
     which the bytes/bandwidth work never touched. Patches a KNOWN hole: the
     1.648-vs-1.671 confound proves the informal search that produced the shipped
     0.6/B=3 NEVER SEPARATED THE TWO VARIABLES. An untested (B, p_min) pair could
     plausibly beat 19.42 by a point or two. No build, existing sweep infra. The ONLY
     one of the three that closes a known evidentiary gap rather than speculating on a
     new mechanism.
  2. iq2_s ABLATION -- one cheap arm, bounded low-single-digit expectation. Mechanism
     genuinely distinct from rpb (constant-memory placement / register forcing target
     codebook-lookup LATENCY, not work redistribution) and that is exactly the class
     that can still move a bandwidth-bound system. But F142 just showed an
     occupancy-style lever losing for iq2_s at the real width => lower the prior, do
     not skip the test.
  3. LUCE_MMQ_OCC -- effectively foreclosed. MMQ's 2.6-3.3x deficit is fixed
     per-launch/dynamic-smem overhead; occupancy would have to close most of it. Run
     only to close the ledger item, no expectation.

TWO AXES NOBODY HAS TOUCHED (both new, neither mine nor previously logged):
  A. QUALITY VALIDATION BEYOND THE SINGLE ARITHMETIC PROBE. Every closure tonight was
     checked against exactly ONE deterministic fact (17x23 -> 391). That probe just
     earned its keep by catching F145 -- but it is one narrow capability, and the
     ORIGINAL OBJECTIVE paired throughput with "maintain acceptable model quality".
     Nothing has validated the shipped config for coherence or instruction-following.
     => "quality is fine" is UNVERIFIED beyond that probe, not established.
  B. THE FULL SERVING PATH. Everything measured is at the llama-server/harness level,
     but the deployed architecture puts a front-end in front. "End-to-end" was the word
     used at the very start; per-token / per-request overhead there is uncharacterised.

STATUS: 19.42 t/s stands as the measured shipped figure. "Maximum reached" is NOT
confirmed by this review. Whether to spend one more cold cycle is an open decision.

## F148 -- ORTHOGONAL B x p_min GRID: the shipped cell IS the joint optimum. Hole CLOSED.
## (2026-09-05, full 3x2 factorial, 7/7 arms valid, no skips, no guard trips)

Fable ranked this above both kernel items because the 1.648-vs-1.671 confound proved the
informal sequential search never separated the two variables -- "the true joint optimum was
approximated, not found". Run from a 37C cold start (coldest of the project), which bought
enough thermal budget for the whole factorial in ONE burst -- so every cell shares a boot,
a binary and a model load.

RAW, then temperature-corrected to 37C at the F97 proportional slope (-0.13%/C):

  arm  nmax p_min  T0   raw    corr
  a1   2    0.60   37  19.70  19.70
  a2   1    0.60   51  17.76  18.08
  a3   3    0.60   55  18.55  18.98
  a4   1    0.45   59  17.68  18.19
  a5   2    0.45   63  17.25  17.83
  a6   3    0.45   66  17.61  18.27
  a7   2    0.60   70  18.76  19.56

INSTRUMENT VALIDATED BY ITS OWN BRACKET: a1 and a7 are the SAME CELL 33C apart. Raw they
differ by -4.8%; corrected they differ by -0.7%, inside the known ~1-1.5% floor. The
thermal correction is doing real work and is not fitted to this data (slope predates it).
Cross-check: a1 corrected to the 47C reference = 19.44, vs the independently established
19.42 baseline. The instrument agrees with six prior arms.

CORRECTED GRID (t/s):
             p=0.60   p=0.45
    n-max=1   18.08    18.19
    n-max=2   19.63    17.83   <-- SHIPPED
    n-max=3   18.98    18.27

RESULT 1 -- THE SHIPPED CELL WINS, by 3.3% over the next best (n=3, p=0.6). Fable's
"plausibly a point or two" upside DOES NOT EXIST. The informal search landed on the right
cell; it just had not earned the right to say so. Closed as CONFIRMED-OPTIMAL, not as
improved. This is a negative result and is recorded as one.

RESULT 2 -- THE VARIABLES DO INTERACT, so the grid was necessary even though it moved
nothing. At p=0.60 the n-curve is PEAKED at 2 (18.08 / 19.63 / 18.98). At p=0.45 it is
FLAT and slightly rising (18.19 / 17.83 / 18.27) -- a different SHAPE, not a shifted level.
Non-separable, exactly as Fable argued. A sequential search on a non-separable pair is only
correct by luck; this one was lucky.

RESULT 3 -- A SHIPPED ASSERTION OF MINE, MEASURED FOR THE FIRST TIME AND WRONG IN ITS
REASONING. run_ridge_mtp.sh carries the header warning "DO NOT COMBINE p_min WITH n-max 1.
Both are width-reducing." That was INFERENCE, never measured. Measured: (1, 0.60)=18.08 vs
(1, 0.45)=18.19 -- a 0.6% difference, i.e. p_min barely matters at n-max 1. There is no
compounding harm. n-max 1 is simply a worse cell on its own (-8%), which is the F-record's
own +26% n-max-1 claim finally contradicted under clean, temperature-corrected, same-boot
conditions. The WARNING'S CONCLUSION (do not ship that combination) survives; its stated
MECHANISM does not. Header to be corrected -- a warning that is right for the wrong reason
will mislead the next person to touch it.

STATUS: p_min/n-max axis CLOSED on a full factorial rather than on a sequential search.

## F149 -- FP16 CARVE-OUT RE-RUN CLEAN: F145's CONCLUSION STANDS, its METHOD did not.
## Fable's reopen was correct to demand this; the answer is unchanged. (2026-09-05)

Fable reopened F145 as "build failed, root cause not isolated". It was right that the METHOD
was invalid, and I found a worse fault than it suspected before spending a watt:

  F145's arms were NOT a flag A/B AT ALL. Arm A = $HOME/fp16fix (own single lib dir, tree
  HEAD). Arm B = $HOME/sub4_2:$HOME/lucefix (TWO-dir split, tree from a day earlier). Two
  binaries, two lib stacks, two tree states. And the arm never set the knob -- because the
  knob is COMPILE-time (common.cuh excludes __CUDA_ARCH__ 600 from FAST_FP16_AVAILABLE),
  not env. Nothing in F145 isolated the flag.

MY OWN CONTAMINATION THEORY ALSO FAILED, and is withdrawn here rather than left standing.
I proposed the leftover LUCE_PHASE patch (43 uncommitted lines in ggml-cuda.cu) as the
culprit. The flag=OFF control built from that same dirty tree came out 370,622,400 bytes vs
sub4_2's 370,611,432 -- an 11 KB difference. The tree barely moved, the patch is env-gated
and dormant, and it cannot account for -11%. I also floated fp16fix's 5.9 MB SHRINK as
evidence of a bad build; reading common.cuh shows the carve-out REMOVES compiled code paths
for arch 600, so a smaller library is the flag working as designed. Both inferences wrong,
both caught before they became findings.

CLEAN A/B: same tree, same commit, same dirty state, same compiler invocation, single-dir
LD_LIBRARY_PATH on BOTH arms, differing ONLY in -DLUCE_SM60_NO_FAST_FP16. Provenance
md5-verified: box fp16fix == local build/bin (flag=1); box fp16ctl == local build-fp16ctl
(flag=0), checked at both ends after transfer.

POOLED, temperature-corrected to 42C (F97 slope), across TWO sessions and TWO different
control lib stacks:
  carve-out ON   17.37 / 17.35 / 17.49  -> mean 17.40, spread 0.8%
  carve-out OFF  19.61 / 19.58          -> mean 19.60, spread 0.2%
  DELTA -11.2%

So the flag genuinely costs ~11%, and F145's number was right for the wrong reasons.
NOTE ON TAGGING: the clean burst is SINGLE-ARM per build (c2/f2 both SKIPPED at 75/74C --
the quality gates added heat the budget did not have). It is load-bearing only because it
REPRODUCES two prior independent arms; on its own it would not qualify.

MECHANISM, and the part neither Fable nor I weighted: the in-tree comment says it outright --
GP100 has GENUINE 2:1 FP16, unlike sm_61 at 1/64, so this is "a precision policy choice, not
a hardware-capability correction". Carving sm_60 out of the fast path removes a REAL 2x
arithmetic path on this specific card. A slowdown is the mechanistically EXPECTED result.
Upstream's +1.4-1.5% on 4x P100 is the figure needing explanation, not ours.

CORRECTION TO FABLE'S REASONING, which was load-bearing for the reopen: it argued "a
correctness fix should move answers TOWARD the reference, not produce a different wrong one."
But 17x23 = 391 and the CONTROL ALREADY ANSWERS 391. When the baseline is exact there is no
"toward" available -- any numerics perturbation can only break it. The probe can DETECT a
change; it cannot GRADE one. That argument could not have carried the weight put on it.

QUALITY (Fable's axis A, first time ever run): 10 scored deterministic probes + degeneracy.
  fp16ctl (SHIPPED CONFIG) 10/10, distinct-token ratio 0.645 over 169 words -- healthy.
  fp16fix                   9/10, fails arith1 (411 for 17x23), AND its 220-token generation
                            returned EMPTY (ratio 0.000, len 0) where the control produced
                            169 clean words. One sample, so held as a flag not a finding --
                            but it is a failure mode the old single-probe gate could never
                            have seen.

VERDICT: carve-out REJECTED on clean evidence -- slower by 11.2% and not quality-neutral.
F145 re-tagged: conclusion CONFIRMED, method VOID, superseded by this entry.
SHIPPED CONFIG NOW QUALITY-VALIDATED beyond the single arithmetic probe for the first time.

## F150 -- SERVING PATH MEASURED: transport is FREE (0.11%); the front-end's cost is
## ARCHITECTURAL, not per-token. Fable's axis B CLOSED. (2026-09-05)

The last unmeasured axis. Everything in this ledger was timed at the llama-server level;
nothing had ever measured what a CLIENT experiences. Fable named it because "end-to-end" was
the framing at the start of the project.

INSTRUMENT: one identical deterministic request (temp 0, cache_prompt false, fixed n_predict)
issued three ways against the SAME live server instance, so compute is bit-identical and any
wall-time difference is transport/framing. Arms INTERLEAVED so thermal drift cancels.
Client wall time is compared against the server's OWN reported prompt_ms + predicted_ms --
a WITHIN-REQUEST difference, which is the cleanest form available here.

LEG 1 -- HTTP + LAN. n_predict 32, 5 iterations (iter1 discarded: 1815 ms prefill, cold).
  on-box  127.0.0.1 : overhead 1.7 / 1.8 / 1.7 / 1.6 ms   -> ~1.7 ms
  desktop over LAN  : overhead 2.8 / 3.5 / 3.3 / 3.3 ms   -> ~3.2 ms
  LAN RTT independently measured at 0.258 ms avg (20 pings).
  TOTAL CLIENT-VISIBLE TRANSPORT COST = 3.2 ms on a 2.85 s request = 0.11%.
  => THE 19.44 t/s FIGURE IS HONEST END-TO-END. There is no hidden per-token tax. This is a
  NULL RESULT and is recorded as one.

NOTE ON METHOD, and an UNRESOLVED CONTRADICTION: the server listens on 0.0.0.0:8081 (ss
confirms) and is healthy from 127.0.0.1, but a connect from the desktop TIMES OUT (curl exit
28, not a refusal -- the signature of a DROP rather than a closed port). ICMP works fine.
HISTORY_PAGE section 0 states "ufw inactive, iptables ACCEPT" as of 2026-08-28, which
CONTRADICTS that, and I could not read the live firewall without a sudo password. So the
CAUSE IS UNVERIFIED and is not asserted here; only the symptom is measured. I did NOT change
any firewall -- an outward-facing change, made while the operator was away, for a measurement
that did not require it. Used an ssh tunnel instead; at 0.258 ms RTT the tunnel is noise.
JUDGMENT CALL, flagged for review. The observable that matters either way: THE DEPLOYED MODEL
IS NOT REACHABLE OVER THE LAN TODAY -- anything wanting it must tunnel.

LEG 2 -- THE LOCAL FRONT-END (frontend.py, upstream repointed at the Z640 via its upstream URL).
Its per-request framing cost is NEGLIGIBLE and was verified from its own log, not inferred:
11 upstream POSTs for 11 client requests -- ONE upstream call each, no retries, no fan-out --
and routing is a cheap regex shortcut check ("Pure chat -> HighIQ direct").
But two ARCHITECTURAL behaviours dominate perceived latency:
  (a) IT DOES NOT STREAM. It issues a BLOCKING upstream call and returns only when the whole
      completion is done, so time-to-first-byte becomes TOTAL GENERATION TIME:
      TTFT direct 1.4-5.3 s vs front-end 16-59 s, interleaved, same request.
  (b) IT OVERRIDES THE CLIENT'S max_tokens. Asked for 32; got 62/88/88/76/79.
So the front-end costs nothing per token and everything in latency. NOT A THROUGHPUT FINDING --
it does not touch the 19.44 figure, which is why the axis closes -- but it is the single
largest gap between "measured t/s" and "what a user feels", and it applies to the WHOLE local
stack on the 3060, not just this box. Logged; out of scope here.

UNEXPLAINED, NOT SPUN: direct TTFT drifted 1.40 -> 5.33 s across the five iterations. With
--parallel 1 a still-running upstream generation would queue the next request, which is the
obvious candidate, but the loop is sequential so that should not happen. Recorded as an open
oddity rather than folded into either conclusion.

CLEANUP: front-end stopped, tunnel closed, server killed, box powered down and confirmed. The
local front-end was NOT running before this measurement and is NOT running after it.

THERMAL: the box reached 80 C during leg 2 -- the Xid 79 threshold, and the front-end legs ran
WITHOUT the thermal guard because they were interactive client-side timings, not guarded arms.
That was an error of the same class as F109/F116: a measurement that felt cheap because the
work was on the client side, while the GPU was doing full generations throughout. Killed on
sight and powered down. THIRD occurrence. Any future client-side timing must still run the
guard, because the card does not care which side of the wire the request came from.

## F151 -- GSQ-RCO IQ3_S MEASURED: loses 20%; and the 19.44 champion figure is RE-VERIFIED (2026-09-09)

Candidate `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` IQ3_S-mtp (12,120,016,960 B, 3.5457 bpw) vs the
champion, model-axis bracket (`scripts/bracket_model.sh`), shipped binary/libs/flags, prose_p1,
guard 75 C, cold start 45 C after a ~17 min powered-off soak with NO fan fitted.

| arm | decode | prefill | T | distinct |
|---|--:|--:|---|--:|
| champ_a | **19.42** | 8.97 | 46->63 | 0.657 |
| cand    | **15.27** | 6.53 | 60->71 | 0.802 |
| champ_b | GUARD TRIP 75 C on model load -- DIED | | | |

Raw -21.4%; mean-temp gap 11 C is worth +1.43% to cand at -0.13%/C, so **-20.2% corrected**.
Prefill -27.2%. Bracket INCOMPLETE (no champ_b), so drift is not bounded -- but the gap is an
order of magnitude larger than any plausible drift and the sign is not in question.

PREDICTED BEFORE THE RUN, FROM THE SHIPPED RCO MANIFESTS ALONE (no GPU time): the file contains
**zero q4_1** and puts **31% of params in iq3_s**, which F118 measured as DOMINATED on sm_60
(5.191 ps/w at n=4 vs q4_1 3.481, while 0.94 bpw larger than iq2_s). RCO's search minimises
quality-loss per byte under a size budget and is blind to per-format kernel cost on Pascal.

**A caution about F136.** The candidate is 0.4 GB SMALLER than the champion and still 20% slower.
"Bytes are time" is necessary, not sufficient -- format mix dominates byte count here. Any future
allocation work must carry BOTH terms.

**The 19.44 t/s record is RE-VERIFIED and the 2026-09-08 UNVERIFIED caveat can be lifted:**
champ_a = 19.42 @ 46->63 C vs 19.44 @ 47 C on record, achieved with NO fan on the card. Degeneracy
healthy (0.657 vs 0.645). This also retires the worry that the fan regression had cost throughput.

**Open, and the only axis where the candidate may still win:** its distinct-token ratio is 0.802
vs the champion's 0.657. ONE sample, ONE prompt -- NOT a quality finding. If pursued, use
`scripts/quality_gate.sh` (F149's 10-probe instrument), not another throughput run. Note the
champion has never been PPL'd and GSQ ships an imatrix, so a quality comparison is the one thing
this candidate is still good for.

VERDICT: do NOT adopt. Champion stands. (Session notes not included.)

## F152 — Xid 79 recurrence, and the instrument gap that hid its cause (2026-09-09/10)

The P100 fell off the bus at uptime 4882 s (23:14:27 UTC), 81 minutes into a fanless boot.
Xid 79 followed by Xid 154 (GPU Reset Required); nvidia-smi returned "Unable to determine the
device handle" while /proc/driver/nvidia/gpus/0000:02:00.0 and all three lspci BARs remained.

**Recovered by power-off + WoL.** Not damage — state. Warm reboot was not attempted; the
documented ritual worked first time. GPU came back at 34 C / 26 W.

AER evidence, which narrows the cause: from boot 21:53:14 to 23:14:27 there were ZERO AER
errors; all 21 arrived in the same second, first a Correctable Physical Layer (Transmitter ID)
sourced from 02:00.0, then CmpltTO/UnsupReq at the root port, then "device recovery failed".
A degrading link throws correctables for minutes to hours beforehand. This link was clean and
then stopped. So: ABRUPT DEVICE STOP, consistent with our thermal history (dies ~84 C) but NOT
separable from a power-delivery sag on these logs alone.

**Instrument gap (now closed).** xid_watch.sh queried nvidia-smi AT fault time — when the card
is already off the bus — so it recorded the error string in the temperature field, six times.
The one reading that would settle thermal-vs-power was the one it could not take. Added
`gpu-trace.service` (5 s sampling to a capped rolling log) and xid_watch now carries the last
pre-fault sample into the event line. Next occurrence will be diagnosable rather than inferred.

**Idle drift confirmed again:** 79-80 C at 0% utilisation and ~41 W with no CUDA context.
Fanless, the card is unsafe at idle, not merely under load. Shroud remains the whole fix.

**Added `gpu-guard.service`** (GUARD=78): kills processes holding a CUDA context when the
limit is reached, filtered so a CPU-only server at -ngl 0 is never killed. Replaces
watch-it-by-hand, which is not a safety mechanism.

Claim extraction was moved to the dev box's RTX 3060 (Ridge 3.7bpw, already resident on
:8080) and completed there: 10 chunks, 46 claims, 1 rejected, 0 parse failures.

## F153 — DRAM ceiling MEASURED; F136's saturation closure is unsupported; hardware counters blocked by R580 (2026-09-11 PDT)

After the user's cold power-cycle and fan installation, the P100 was healthy. No idle-curve
characterization was performed. Local nvcc 12.0.140 compiled `triad.cu` for sm_60 with static
cudart; the binary ran **only on the Z640**. No CUDA toolkit/compiler was installed remotely.
Complete commands, source, raw per-trial temperatures, profiler failures and provenance:
[F153 report](f136_measurements_20260911/REPORT.md).

**MEASURED — direct STREAM-style triad, A=B+1.5*C.** Three device-resident arrays; CUDA events
time the kernel only. Five warmups followed by 30 individual trials per invocation. Byte count
is two reads plus one write, decimal GB/s. ECC enabled, HBM clock 715 MHz, 4 MiB L2; each array
is 64–128 times larger than L2. Initialization and host transfers are outside the timed region.

| sequence | array size / total | trials | temperature on every trial | mean GB/s | min–max GB/s | sample SD |
|---|---|---:|---|---:|---:|---:|
| before champion decode | 512 MiB / 1.5 GiB | 30 | 41→41 C | **544.531** | 540.637–546.246 | 1.385 |
| independent size check, before decode | 256 MiB / 768 MiB | 30 | 48→48 C | **546.535** | 544.491–547.678 | 0.747 |
| after champion decode | 512 MiB / 1.5 GiB | 30 | 62→62 C | **541.084** | 539.379–542.390 | 0.659 |

These are achieved streaming rates, not proof of an absolute maximum across every possible
kernel. **426 GB/s is nevertheless disproven as this card's achieved DRAM ceiling.** Its origin
is F110's f16 MUL_MAT m=4096,k=14336,n=1: 117.4 MB / 275.66 us at approximately 66 C. That is a
specific matvec's effective byte rate, not STREAM or DRAM counters. NVIDIA's PCIe 16 GB spec is
732 GB/s. Gen1 PCIe does not enter the timed device-memory benchmark.

**COUNTERS UNAVAILABLE — not replaced with wall-clock arithmetic.** Installed Ubuntu's
`nvidia-profiler` 12.0.146 and its CUPTI/injection dependencies; all collection attempts ran
under sudo because RmProfilingAdminOnly=1. Metric enumeration recognizes both DRAM metrics,
and activity-only kernel tracing succeeds. Collection fails with exit 12, `CUDA profiling
error`, and no events/metrics. A direct diagnostic resolves both metric IDs, then
`cuptiMetricCreateEventGroupSets` returns **27 / CUPTI_ERROR_NOT_SUPPORTED**. Single-metric
and replay-disabled attempts also fail. Warm-loaded champion profiling reaches its first
warmup kernels and fails at **62→62 C**, before profiled token generation. A prior 20-second
cold-load profiler timeout is recorded separately and is inconclusive.

[NVIDIA's R580 explanation](https://forums.developer.nvidia.com/t/cupti-event-api-fails-with-error-cupti-error-not-supported-on-v100/364454)
corroborates this failure: legacy Event/Metric collection needs 575 or earlier; enumeration
can still work. No driver change was made. **Champion hardware-counter DRAM throughput is
unknown, so the requested decode/STREAM ratio is undefined.**

**F136 timing provenance resolved.** The ledger explicitly labels 51.5 ms as champion time,
consistent with the reciprocal of F132's historical normalized 19.43 t/s (51.4668 ms/token).
It is not a separately logged raw kernel time, and is per emitted token, not per MTP round.
F136's splice ran at 18.14 raw t/s, or 55.1268 ms/token. The 266 GB/s mixes the splice's 13.705 GB
size with champion timing. Replacing the numerator gives approximately **243.0 GB/s**, but still
retains that historically normalized/rounded denominator. Existing raw champion timing is
available in `results_PMIN2.txt`, so no splice timing was reassigned:

| raw champion record | temperature | tokens / decode ms | ms/token | file-byte effective GB/s, NOT DRAM |
|---|---|---|---:|---:|
| F132 opening p_min=0.6 | 44→60 C | 192 / 9849.3 | 51.2984 | 243.954 |
| F132 closing p_min=0.6 | 65→71 C | 192 / 10085.4 | 52.5281 | 238.243 |

**MEASURED AGAIN — unchanged champion model and flags**, `run_ridge_mtp.sh` on port 8083,
LD_LIBRARY_PATH=sub4_2:lucefix, n-max 2 / p-min 0.6, prose_p1, cache_prompt=false. Exact model
file size 12,514,461,248 bytes. Each request has 15 prompt tokens and 192 emitted tokens.

| trial | temperature before→after | decode ms | raw t/s | file-byte effective GB/s, NOT DRAM |
|---|---|---:|---:|---:|
| 0 | 52→60 C | 9819.677 | **19.5526** | 244.690 |
| 1 | 59→63 C | 9900.367 | **19.3932** | 242.696 |
| 2 | 62→66 C | 9982.164 | **19.2343** | 240.707 |

No temperature corrections were applied. File size is not DRAM traffic per emitted token:
it includes gather-only embeddings, while MTP verification/drafting change traffic. These
effective rates resolve the historical arithmetic only and are **not** utilization estimates.

Continuous session trace: 268 samples, observed peak **66 C**. Every triad trial and decode
request has before/after temperature. No current-boot Xid or guard trip was recorded. Final
device healthy at 62 C; existing gpu-trace/gpu-guard remained active. Session-owned processes
were stopped; shipped model, launcher, kernel libraries and driver were unchanged.

**CONCLUSION:** F136 does not establish saturation, and its 426 GB/s ceiling and 266 GB/s
champion figure are invalid for that purpose. Whether the actual decode kernels saturate DRAM
is still **UNRESOLVED**, pending counter collection on a compatible driver. Arithmetic-side
optimization cannot be closed on this argument, but no optimization gain or measured decode
headroom is claimed. All previously measured optimization losses retain their own evidence.

## F154 -- DECODE IS NOT BANDWIDTH-BOUND. Clock sweep settles what counters could not. (2026-09-12)

F153 left saturation unresolved because CUPTI counters are unavailable under R580. This is the
indirect discriminator Codex designed in (not included), executed by Claude while the
Codex quota was expired. Harness: (not included) mode `clock`, unmodified except thermal
constants (see PROTOCOL below). Champion config, prose_p1, 192 emitted tokens, seed 5, temp 0.

**Decode, SM application clock varied, HBM PINNED at 715 MHz throughout:**

| arm | SM MHz | decode t/s | T before->after | output sha256 |
|---|---:|---:|---|---|
| clock_v2_0 | 1189 | **18.55** | 63 -> 68 C | c1453f54 |
| clock_v2_1 |  885 | **14.44** | 66 -> 70 C | c1453f54 |
| clock_v2_2 | 1189 | **18.48** | 66 -> 72 C | c1453f54 |

Identical output hash on all three arms = identical work. The two 1189 points agree to **0.4%**
despite a 3 C difference in start temperature, so thermal drift explains essentially none of the
885 deficit.

**Triad negative control at the same clocks (same session, 256 MiB x3, 30 trials each):**

| SM MHz | triad GB/s | spread | T |
|---:|---:|---|---:|
| 1189 | 544.8 | 543.9-545.9 | 63 C |
|  885 | **547.5** | 545.6-549.1 | 66 C |
| 1189 | 544.8 | 542.4-545.7 | 70 C |

**THE RESULT:** SM clock -25.6% -> decode **-22.0%** (elasticity **0.86**) while achieved DRAM
bandwidth is FLAT (+0.5%, i.e. unchanged) across the same clock change. A bandwidth-saturated
workload is by definition insensitive to core clock; decode tracks core clock nearly 1:1.

**F136's "aggregate bandwidth-bound" conclusion is WRONG, not merely unproven (F153).** Memory is
not the wall at ~243 GB/s of file bytes against the 544 GB/s measured ceiling. The arithmetic /
execution axis was closed on a faulty premise and is REOPENED.

**NOT ESTABLISHED, do not overstate:** clock sensitivity localises the limit to the execution side
but does NOT distinguish ALU/integer work from instruction issue, occupancy, per-kernel launch
overhead, or the serial dependency chain of batch-1 matvecs. Speculative decoding (draft-mtp
n-max 2) puts draft AND verify work on the same clock. These imply different fixes. No optimization
gain is claimed here. Narrowing this is the next experiment.

**POWER / THERMAL, incidental but operationally important:** sustained decode at 1189 MHz draws
**143.9 W**; a 2048-token prefill draws **123.6 W** and heats ~0.5 C/s. The fan fitted 2026-09-11
fixes the IDLE runaway (flat 59-60 C for 80+ min, vs pre-fan monotonic climb to 84 C and Xid 79)
but does NOT create load headroom: after repeated arms the floor drifts to 67-70 C and each arm
trips. Measurement remains gated at roughly one 192-token arm per cooldown cycle, which
independently reproduces the closeout's "two arms per boot" constraint.

**PROTOCOL deltas from Codex's DESIGN.md, both forced and both disclosed:** the thermal kill was
raised 70 -> 74 C because warmups alone reached 68 C and full arms self-invalidated; `cool()` was
raised to <64 C because its original <55 C target is UNREACHABLE -- the fan's idle floor is
59-60 C, and the loop hung at 67 C with 0 MiB resident. System `gpu-guard` still kills CUDA at
78 C and did not fire. Prefill arm not yet complete at time of writing.

## F155 -- PREFILL: -b/-ub 1024 beats 512 by +14.4%. F48's void 1024 point re-measured clean. (2026-09-12)

Codex's `prefill` arm from (not included), executed by Claude. Champion config, full
offload, identical 2048 token IDs, `cache_prompt=false`, unchanged model/libs/speculation.
Every arm pre-cooled to 62 C before the timed forward (see THERMAL below).

| arm | -b/-ub | prompt_n | prompt_ms | prefill t/s | T before->after | output sha |
|---|---:|---:|---:|---:|---|---|
| prefill_0 |  512 | 2048 | 16246.1 | **126.06** | 62 -> 70 C | 7c9752f7 |
| prefill_1 | 1024 | 2048 | 14197.0 | **144.26** | 62 -> 70 C | 7c9752f7 |
| prefill_2 |  512 | 2048 | 16232.9 | **126.16** | 62 -> 70 C | 7c9752f7 |

**+14.4%** for 1024 over the 126.11 two-point 512 mean. The 512 bracket closes to **0.08%**, all
three arms ran the identical thermal bracket (62 -> 70 C), and the output hash is identical across
all three, so the work is identical. This supersedes F48, whose 1024 point was throttled and
explicitly void.

**This is a LAUNCH-FLAG change, not a kernel change** -- nothing to compile, nothing to verify
beyond VRAM headroom at the production context length. NOT YET DONE: confirm VRAM at the shipped
context (these arms ran `-c 8192`) and confirm 1024 does not regress decode, before shipping.

**Consistency with F154:** a larger microbatch increases work per kernel launch and improves
occupancy. That helps an EXECUTION-limited workload and would do nothing for a bandwidth-saturated
one. Two independent measurements, same direction.

**THERMAL:** prefill draws ~123.6 W and heats ~0.5 C/s, so a 2048-token forward (~16 s) needs ~9 C
of headroom and MUST start at or below 62 C; from a warm start it trips. An earlier attempt from
66 C died at 74 C mid-forward. Prefill on this card is POWER-limited, not throughput-limited --
long prefills will trip the guard regardless of kernel speed. The harness pre-cools to 62 C before
each timed forward for this reason.


## F156 — Review of F154/F155: prefill gain retained; decode mechanism and thermal protocol qualified (2026-09-12 PDT)

F154 and F155 were executed by the other agent. I reviewed the surviving raw
results, per-request telemetry, triad CSVs, server commands/logs and the modified
harness before continuing. The completed prefill speedup is valid for the tested
first-long-request procedure. Several mechanistic claims need narrowing.

* **Output equality is not work equality.** The clock controls each report 119
  drafted / 104 accepted tokens, versus 120 / 106 at 885 MHz. Thus the target
  round counts are 88, 86, 88; mean proposed widths are 1.3523, 1.3953, 1.3523.
  Round times are 117.644, 154.587, 118.054 ms, respectively, at the recorded
  63→68, 66→70 and 66→72 C. Matching content hashes did not control internal
  speculative work exactly. The large slowdown remains; exact attribution of
  its magnitude does not.
* High-clock throughput averages 18.5137 t/s; the low arm is 14.4421 t/s:
  −21.992%. The ratio of fractional changes is 0.8602, while logarithmic
  elasticity is 0.8411. Neither is a measured arithmetic fraction or headroom.
* The **surviving** closing triad file is 545.007 GB/s at **67 C**, rather than
  the F154 table's 544.8 at 70 C. Opening is 544.803 GB/s at 63 C; low-clock
  control is 547.467 GB/s at 66–67 C. All three contain 30 trials with the
  requested SM clock and HBM 715 MHz. Reused filenames were overwritten on
  retries, so only the surviving artifacts can support the revised table.
* F154 rejects a **pure streaming-bandwidth explanation across this interval**.
  It does not establish that every decode kernel is non-bandwidth-bound at
  1189 MHz: reducing clocks can cross from bandwidth to execution limitation.
  Cache bandwidth, load issuance and memory latency also depend on SM speed.
  True DRAM traffic and saturation remain unmeasured. The 243/544 file-byte
  ratio must be removed from the bottleneck inference, not rehabilitated.
* A GPU clock change does not directly vary host launch cost. The experiment
  therefore is not evidence that host launch overhead is the cause. A GPU
  activity timeline, with an unprofiled control, can bound launch/host gaps
  without retrying unavailable hardware counters.
* Restarting after the prefill warmup discarded that warmup's process state.
  All completed prefill arms nevertheless used the same first-long-request
  procedure; startup's own model warmup remained enabled. This preserves the
  512/1024 comparison but does not establish repeated-request steady-state
  performance. All evaluated 2048 tokens, returned `391`, and had matching
  prompt text and output hashes. The 1024 arm peaked at 15409 MiB versus 13835
  for 512; some 1024 telemetry samples were below the 1189 MHz application
  setting (1075–1189 MHz). The measured gain survives those actual clocks.
* F155's assertion that larger batches cannot help bandwidth-limited work is
  incorrect: they can amortize repeated weight traffic/dequantization as well
  as launches. Prefill is a separate regime. Its result does not identify the
  decode mechanism. Heat accumulation is also not proof of power capping;
  'power-limited, not throughput-limited' is withdrawn. The observed constraint
  is thermal duty cycle, and no universal long-prefill failure is established.
* The 70→74 C abort and <55→<64 C cooldown changes violated the ORIGINAL
  thermal policy, though disclosed. The user's latest continuation explicitly
  authorizes keeping them; subsequent work uses that revised policy with the
  existing 78 C system guard. No temperature corrections are applied. Earlier
  interrupted requests are excluded; retries are not extra replicates.
* At review, the decode batch-size comparison has only two completed arms,
  18.8156 and 18.7563 t/s, both 62→69 C, with identical 123/107 draft counts and
  output. Its remaining bracket is being completed before shipping. A 0.35%
  control spread is an observed bracket spread, not a universal noise floor.

Next discrimination: a small downward SM step (1101 MHz) and an upward step
(1328 MHz), each bracketed by 1189 MHz, all fresh-process identical requests,
with actual clocks and speculative work verified. This probes for a knee near
normal operation. Separately, an activity-only CUDA timeline (no metrics/events)
will compare kernel-busy union time with decode wall time; profiler overhead
must be checked against surrounding unprofiled controls before attribution.
The IQ2_S candidate remains gated on backend correctness before any timing.


## F157 — Prefill batch 1024 SHIPPED after decode bracket and VRAM check (2026-09-12 PDT)

`$HOME/run_ridge_mtp.sh` now uses `-b 1024 -ub 1024`. Backup:
`$HOME/run_ridge_mtp.sh.bak-F157-20260912`. The exact before/after scripts
are saved in perf_20260911; bash syntax and the installed flag line were checked.
The model, library selection, K/V policy, n-max 2 / p-min .6 and default context
8192 are unchanged. This takes effect on the next launcher invocation.

F155 prefill is 126.0613 / 144.2561 / 126.1636 t/s at batch 512 / 1024 / 512,
all 62→70 C, 2048 tokens actually evaluated: **+14.3869%** over the control mean.
The prompt text SHA256 is identical (c6d12ec4...), as is the `391` output.

The decode gate now has a COMPLETE 512/1024/512 bracket:

| batch | raw decode t/s | temperature | drafted / accepted | output hash |
|---|---:|---|---|---|
| 512 opening | 18.8156 | 62→69 C | 123 / 107 | eb77c530... |
| 1024 | 18.7563 | 62→69 C | 123 / 107 | eb77c530... |
| 512 closing, this continuation | 18.8187 | 62→69 C | 123 / 107 | eb77c530... |

512 controls differ by 0.016%; candidate is about −0.32% versus their mean.
This is a small observed cost, not proof of exact zero effect. It is far smaller
than the prefill gain. All sampled clocks in these decode arms were 1189 MHz SM /
715 MHz HBM. A second 1024 replicate remains queued, as corroboration.

VRAM is checked at the actual production **8192 allocation**, with a real
2048-token prefill and generation, not merely successful model loading. Observed
peak: 15409 MiB at 1024 versus 13835 at 512; decode peaks 15325 versus 13795 MiB.
The 16 GiB card reserves approximately 114–115 MiB, leaving about **860 MiB** at
the larger prefill peak. This establishes the measured 8K-allocation workload,
not maximum VRAM at fully resident 8K or at larger CTX overrides. Those were not
silently promoted to tested configurations. No kernel change is shipped here.


### F157 corroboration — fourth decode arm completed

The additional 1024 arm completed at **18.7046 t/s, 62→69 C**, with the same
123 drafted / 107 accepted and eb77c530... output. Its pre-load cooldown was
relaxed from <62 to the user's retained <64 C after waiting at 62 C; the actual
measurement endpoints still matched. It is corroboration, not a replacement
for either original control. Across the two 1024 observations, mean 18.7304 t/s
versus 18.8171 for the two 512 controls is **−0.4606%**. The two 1024 observations
differ by 0.276%; both were below the controls. The shipment therefore accepts a
small measured short-prompt decode cost for +14.39% prefill; 'free' or a universal
0.4% noise floor would overstate the evidence. No temperature correction.

## F158 — IQ2_S sign-add candidate passes the GPU correctness gate

Command: `LD_LIBRARY_PATH=$HOME/perf_20260911/signadd:$HOME/lucefix
$HOME/perf_20260911/test-backend-ops test -b CUDA0 -o MUL_MAT`.
The full operator selection ran, without a type filter: **1261/1261 tests passed,
0 FAIL**, exit 0. There were 426 explicitly unsupported CUDA cases, not silently
counted as numerical passes. All 11 IQ2_S test entries passed, including n=1,2,3.
The CPU backend was skipped as a selected test target; the test harness still
uses its reference comparisons for the CUDA tests.

UTC 2026-09-13 05:09:37–05:10:47; start **67 C**, finish **66 C**, monitored samples
65–66 C, no thermal trip. Correctness is not a throughput measurement. Logs and
telemetry: `remote/micro_testall_0_signadd.log`, `remote/micro.results.jsonl`,
`remote/micro.telemetry.jsonl`; persistent thermal evidence is also in the
Z640 `/var/log/gputrace.log`. The candidate has not yet been promoted or timed.
Library SHA256: 4a07cf8e140505176661587cc8a01a59a573192a37436c358a509f94d7f298a3.

## F159 — IQ2_S sign-add: small reproducible whole-model decode gain (2026-09-13)

Recovered the completed detached harness after the quota interruption; the six-arm
comparison ran after F158's full CUDA correctness pass. No successful arm was rerun
merely because the chat stopped. **All six explicitly used -b 512 -ub 512**, not
the newly shipped 1024 default. Fresh server per arm; champion model, turbo4 flags,
8192 context allocation, n-max 2 / p-min .6, prose_p1, 15 prompt / 192 output tokens.
Startup warmup enabled; no separate request warmup. These are decode timings,
not the approximately 1.1-second first short-prompt evaluation times.

| sequence | libraries | decode t/s | temperature before→after | drafted / accepted |
|---|---|---:|---|---|
| 0 | shipped | 18.7275 | 65→72 C | 123 / 107 |
| 1 | control | 18.7279 | 65→72 C | 123 / 107 |
| 2 | signadd | 18.8433 | 65→72 C | 123 / 107 |
| 3 | control | 18.7373 | 65→72 C | 123 / 107 |
| 4 | signadd | 18.8692 | 65→72 C | 123 / 107 |
| 5 | shipped | 18.7365 | 65→72 C | 123 / 107 |

Mean shipped **18.7320**, rebuilt control **18.7326**, sign-add
**18.8562 t/s**, all 65→72 C. Candidate versus control **+0.6601%**;
rebuilt versus shipped **+0.0032%**. The two rebuilt controls
span 0.0503%, shipped controls 0.0484%, and candidate replicates 0.1376%.
Both candidate observations beat every control. This is a small observed gain
with two candidate replicates, not a universal effect size or a formal confidence
interval across prompts. All sampled SM clocks were 1189 MHz, HBM 715 MHz.
All output hashes are eb77c530..., and all draft/accept counts match (123/107).
The /proc loaded-maps artifacts confirm the intended shipped/control/signadd
CUDA library paths for each arm; selection is not inferred from environment alone.

F158 remains the correctness gate: 1261/1261 supported CUDA MUL_MAT tests passed,
0 FAIL at 67→66 C, including all 11 IQ2_S entries. Candidate SHA256 is unchanged.
Historical timing arms used the then-documented <65 C pre-load schedule and
74 C abort; none tripped. The latest <=62 C long-forward policy applies to new
measurements, not a retroactive claim that these started at 62 C.

**Decision:** retain the candidate as a measured improvement at explicit batch512.
Before promotion, a new control/signadd/control comparison at the shipped batch1024
is running with <=62 C measured starts. No candidate library is shipped yet.
Raw evidence: remote/kernel_review_*.response.json, *.command.json,
*.loaded_maps.json, results.jsonl and telemetry.jsonl. The persistent thermal
trace is /var/log/gputrace.log on the Z640; it is archived with the final evidence.


## F160 — Nearby-clock knee: normal-clock sensitivity persists; 1328 MHz was not sustained

The detached harness completed all five arms before this continuation. Every arm
used the shipped CUDA libraries, explicit **-b 512 -ub 512**, fresh process,
8192 allocated context, prose_p1, 15 prompt and 192 emitted tokens, n-max2/p-min.6.
All five have identical output eb77c530... AND 123 drafted / 107 accepted tokens.
HBM was 715 MHz in every timed sample and every triad trial. No thermal correction.

| requested SM MHz | decode t/s | request temperature | sampled SM range MHz | triad mean GB/s | triad temperature |
|---:|---:|---|---|---:|---|
| 1189 | 18.7334 | 64→71 C | 1189–1189 | 545.868 | 64–64 C |
| 1101 | 17.6505 | 65→71 C | 1101–1101 | 545.436 | 65–65 C |
| 1189 | 18.7275 | 65→72 C | 1189–1189 | 545.796 | 65–65 C |
| 1328 | 19.0197 | 65→73 C | 1202–1328 | 542.357 | 65–66 C |
| 1189 | 18.7484 | 66→72 C | 1189–1189 | 545.731 | 66–66 C |

The 1101 point is **-5.766%** versus its 1189 bracket mean
18.7305 t/s (64→71 and 65→72 C), for a 7.401% clock reduction. The controls
close within 0.0315%; triad changes only approximately −0.073%.
Thus the earlier 885 result was not solely a regime reached only by that large
clock reduction: decode remains sensitive close below normal speed.

The requested 1328 point is **+1.504%** versus its surrounding mean
18.7380 t/s (65→72 and 66→72 C; closing control sampled peak 73 C), whose
controls differ 0.1114%. But **1328 MHz was delivered only in the first two
samples, during prompt evaluation**. Decode-window samples were 1202–1278 MHz,
mostly 1215/1227/1240; their unweighted mean was 1225.95 MHz. The full-request
mean 1235.67 MHz includes prefill and must not be substituted as a fixed decode
clock. No t/s clock correction is made. The triad itself DID run at 1328, at
542.357 GB/s (65–66 C), about 0.624% below its neighbouring controls.

At review, NVIDIA management telemetry reports an existing **150 W power limit**
(default hardware limit 250 W). The upper-clock request sampled approximately
136–152 W during decode. Power capping is a plausible explanation for the
frequency reduction, but no throttle-reason samples were captured inside that
historical arm, so its precise cause is not retrospectively proven. The current
management status/cumulative reasons are archived as part4_power_state.txt;
no CUPTI metrics or events were requested. The power limit is not being raised.

**Decision:** a modest delivered clock increase above 1189 improves decode while
streaming bandwidth does not improve. A pure aggregate streaming-bandwidth wall
at the normal setting is inconsistent with these controlled workload results.
However, the requested full 1189→1328 comparison is **not a fixed-frequency
experiment** and does not quantify a large pool of headroom, exclude a higher
knee, or identify ALU versus issue/cache/load latency. True DRAM bytes remain
unknown; 243/544 is not a utilization or headroom estimate. Default SM application
clock remains 1189. Raw evidence: remote/clock_local_* and per-request telemetry.


## F161 — Activity-only timeline: kernel execution dominates; no large pool of launch gaps

Recovered the completed unprofiled / nvprof / unprofiled interleave. Explicit
**-b 512 -ub 512**, shipped libraries, fresh server each arm, n-max2/p-min.6,
8192 context allocation, 15 prompt / **64 output tokens**. All three start/end
**65→70 C**, all sampled SM 1189 / HBM 715 MHz, identical 160a2c00... output and
35 drafted / 30 accepted. This shorter prefix has a different speculation mix
from the 192-token candidate test; do not pool their absolute t/s.

| arm | decode ms | decode t/s | temperature |
|---|---:|---:|---|
| activity_0_control | 3725.977 | 17.1767 | 65→70 C |
| activity_1_trace | 3855.502 | 16.5997 | 65→70 C |
| activity_2_control | 3728.261 | 17.1662 | 65→70 C |

Unprofiled decode mean **3727.119 ms at 65→70 C**; profile overhead is
**+3.445% elapsed time**, or −3.330% throughput. Controls differ only
0.0613% in elapsed time. This overhead is disclosed, not subtracted from kernel
durations or treated as zero.

The command used only `nvprof --print-gpu-trace --profile-api-trace none
--unified-memory-profiling off --csv --normalized-time-unit us --timeout 25`.
No metrics, CUPTI events, or counter setup. The timeout let nvprof flush its trace
after the completed request. Its final 'Execution timeout' is expected profiler
shutdown, not a failed generation or a thermal trip. The trace contains 98149
GPU activities across loading, startup and the request.

**Window alignment:** trace timestamps and server-log timestamps have different
origins, and no request markers were injected. Use the reported 3855.502 ms decode
length ending at the last device-to-host transfer, and separately at the final
kernel (2.257 ms later). Also inspect windows trimmed 50 and 100 ms at BOTH ends,
plus a full-length window shifted earlier by 50 ms. This is an explicitly
approximate boundary alignment, not a claim of synchronized clocks. Kernel-busy
fractions stay about 93.4% under these alternatives. The interior windows also
avoid the initial draft-prefill tail. Loading and the long first prompt phase
are excluded. Every duration below comes from the profiled **65→70 C** request.

| approximate decode-window quantity | measured duration | fraction of 3855.502 ms |
|---|---:|---:|
| union of kernel intervals | 3601.788 ms | 93.419% |
| union of all GPU activities, including copies | 3691.045 ms | 95.734% |
| no traced GPU activity | 164.457 ms | 4.266% |
| quantized MMVQ kernels | 2974.511 ms | 77.150% |
| IQ2_S MMVQ subset | 1709.082 ms | 44.328% |

Intervals are clipped and merged across streams; overlapping work is NOT counted
twice as GPU-busy time. The maximum no-activity gap is 11.128 ms, and 75 gaps exceed
100 us. Gaps include host work, synchronization and possible profiler overhead;
this trace does not attribute all of them to kernel launch. Raw data and the
reproducible parser are remote/activity_1_trace.nvprof.csv, analyze_activity.py,
and activity_analysis.json. The 4.3% profiled gap share is not a promised
unprofiled optimization gain, especially with 3.44% profiling overhead.

**Decision:** the dominant delay is on-device execution, particularly quantized
matvec kernels, rather than a large pool of dead time between launches. Combined
with F160, SM-dependent kernel work is an active limit at normal operation. This
still does not distinguish ALU throughput from instruction issue, dependency or
cache/load latency inside those kernels. It supplies the missing time partition;
it does not infer DRAM throughput from a model file size or from GPU utilization.
CUDA graphs and the directly measured closed axes remain closed.


## F162 — Fixed-byte IQ2_S microbenchmarks: n=1 improves; n=3 largely does not

These completed after the full F158 correctness gate and the six whole-model
arms. Each shape ran control / signadd / signadd / control. Same tensor data
format and shape, same 128-thread block geometry, same mathematical operation;
only the compiled sign-application rewrite changes. GGML_PERF_MIN_US=3000000;
the harness reports mean us/run across thousands of operations. These are
isolated MUL_MAT rates, not model tokens/s or measured DRAM bandwidth.

| m,n,k | control us/run (temperature) | signadd us/run (temperature) | signadd us/run (temperature) | control us/run (temperature) |
|---|---|---|---|---|
| 17408,3,5120 | 435.80 (70→72 C) | 436.22 (63→67 C) | 436.27 (67→69 C) | 435.99 (68→70 C) |
| 5120,3,17408 | 364.95 (69→71 C) | 364.43 (71→72 C) | 361.64 (63→68 C) | 363.90 (67→69 C) |
| 17408,1,5120 | 246.86 (69→70 C) | 222.73 (70→71 C) | 223.16 (71→72 C) | 245.72 (64→68 C) |

For m17408/k5120, n=1 candidate mean 222.945 us versus control 246.290 us:
**9.48% less time / 10.47% greater operation throughput**, at the temperatures in
the table. Candidate replicates were hotter than the closing control yet faster;
no thermal correction is used. At n=3 for the same shape, mean 436.245 versus
435.895 us is approximately **0.08% slower**, effectively no useful gain in this
sample. The transposed-dimension n=3 shape is about **0.38% less time**, but the
thermal and clock differences make that small number inconclusive.

Micro telemetry covers process startup/calibration as well as the reported timed
loops, so its SM ranges are not mislabeled as exact timing-window clocks. Most
runs held 1189 MHz throughout; the n=3 down-shape runs include startup or load
samples below 1189, and one n=1 candidate sample is 1177 MHz. HBM stayed 715 MHz.
Interrupted attempts (micro_perf_1_signadd, micro_perf_6_signadd and
micro_perf_11_control) hit the 74 C
session guard, were excluded, cooled, and resumed. The retained final log per tag
is the successful retry; micro.results.jsonl preserves the failed attempts and
telemetry. Successful rows above have no guard trip. No failed timing was averaged.

Driver occupancy calculation on extracted sm_60 cubins (no launched workload,
no counters) gives **the same** resident-block ceilings in both builds: n=1:
5 blocks/SM (31.25% thread capacity); n=2: 4 (25%); n=3: 3 (18.75%). Static SASS
instruction counts are 780→684 at n=1, 900→882 at n=2, 1110→1092 at n=3, with no
spills. The candidate crosses no occupancy-allocation boundary. These are
resource ceilings and static code facts, not measured achieved occupancy.

**Inference supported:** less sign-handling instruction work improves the n=1
kernel while bytes and occupancy ceiling are held constant. This identifies a
real instruction-path lever, without distinguishing ALU throughput, issue or
scheduling/dependency effects. The benefit does not scale simply with static
instruction count at n=3. In the separate 64-token profile, IQ2_S n=1 occupies
about 10% of decode wall time at 65→70 C, whereas all IQ2_S widths occupy about
44%; a large isolated n=1 gain can therefore coexist with the small +0.66%
whole-model gain at 65→72 C. This comparison explains scale; it is not an Amdahl
prediction across different prompt lengths/speculation mixes.

Evidence: remote/micro_perf_*.log, micro.results.jsonl, micro.telemetry.jsonl,
occupancy.py, remote/occupancy.json, compiler_resources.json and the saved SASS.
The rewrite is a measured narrow gain. It does not establish that no other
arithmetic optimization exists or justify reopening directly closed axes.


## F163 — Production-batch confirmation in progress; CUDA module provenance verified

New measurements explicitly use **-b 1024 -ub 1024**, 8192 allocated context,
1189 MHz application clock, the same 192-token prose request and fresh server
per arm. Before every timed request, the harness checks **temperature <=62 C**.
A startup that exceeds that limit is stopped before timing and restarted after
further cooling. The 32-token initial preconditioning request is excluded.

Completed so far, before the closing control:

| arm | decode t/s | temperature | drafted / accepted |
|---|---:|---|---|
| rebuilt control opening | 18.7629 | 61→68 C | 123 / 107 |
| signadd | 18.8559 | 62→69 C | 123 / 107 |

Outputs match eb77c530... . The candidate is about +0.496% against this opening
control, but no final production-batch effect is assigned before the closing
control. These observations are being recorded as they finish, not held until
the entire task ends. No kernel promotion has occurred yet.

**Code provenance check:** cuobjdump extraction restricted to sm_60 found 167
CUDA modules in each library (164 unique hashes). The shipped and rebuilt
control module MULTISETS are byte-identical, including the MMVQ module. Candidate
versus control differs in exactly ONE module, the intended MMVQ cubin; the other
166 module instances are byte-identical. The differing hashes match the separately
extracted mmvq.control.o and mmvq.signadd.o cubins used for the occupancy check.
This strengthens the library-selection control and confirms that the large-batch
GEMM/dequantization device kernels are unchanged. It is code evidence, not a new
prefill throughput measurement. Host linking/build provenance is separately
preserved; equality of GPU modules is not claimed as whole-library byte equality.
See module_identity.json, module_difference.json, build_provenance.json and the
three library artifacts. No extra GPU workload was run for this comparison.


### F156 count-convention correction identified during the part-4 log review

F156 called N_emitted − N_accepted (88/86/88) the target round counts. That
convention includes the initial token obtained from prompt evaluation. The
server's cumulative #calls(b,g,a) differences show **87/85/87 subsequent
generation iterations**, while the new 192-token fresh-process arms have 84
(the 64-token trace has 33). Thus F156's derived mean proposed widths should be
119/87 = 1.3678, 120/85 = 1.4118, and 119/87 = 1.3678 when describing those
iterations. Whole decode elapsed divided by those counts is 118.997,
156.405, 119.411 ms, respectively, at the original **63→68, 66→70,
66→72 C**. These include other decode overhead, not just target-kernel time.
The raw t/s, temperatures, draft/accept counts and clock comparisons in F154,
F159 and F160 are unchanged. The original warning that equal output did not
establish equal speculative work remains valid. This corrects my own off-by-one
terminology and derived per-iteration figures rather than carrying them forward.


### F163 closing control completed — production batch1024 gate passed

The closing control completed at **18.6909 t/s, 62→69 C**, with the same
123/107 draft/accept counts and eb77c530... output. The complete bracket is:

| libraries | decode t/s | temperature |
|---|---:|---|
| control opening | 18.7629 | 61→68 C |
| signadd | 18.8559 | 62→69 C |
| control closing | 18.6909 | 62→69 C |

Control mean **18.7269 t/s**; candidate **+0.6889%**. Controls span 0.3842%,
so this production-batch check is less tightly closed than F159. Candidate still
beats the colder opening control by 0.4958% and the closing control at matching
temperatures by 0.8827%. No temperature normalization is applied. Combined with
F159's tighter replicated six-arm result, this supports a small improvement on
this prose request, not a universal per-prompt gain or a precise global effect.
All timed samples have SM 1189 / HBM 715 MHz. Measured peak VRAM is 15325 MiB for
all three arms. /proc mappings confirm the intended control/signadd libraries.
No timed arm tripped the guard; above-limit startup attempts were stopped before
timing. The initial preconditioning request is excluded.

The production-batch check supports promotion of the same F158-tested artifact.
Raw evidence: remote/prod1024_[012]_*.response.json, *.loaded_maps.json,
*.command.json, part4_prod.log, results.jsonl and telemetry.jsonl.


## F164 — Sign-add promoted after correctness and production-batch confirmation

`$HOME/run_ridge_mtp.sh` now selects
`$HOME/iq2_signadd_F159:$HOME/sub4_2:$HOME/lucefix` as LD_LIBRARY_PATH. The validated
CUDA library is `$HOME/iq2_signadd_F159/libggml-cuda.so.0`, SHA256
4a07cf8e140505176661587cc8a01a59a573192a37436c358a509f94d7f298a3.
The original sub4_2 library remains intact. The launcher still uses **-b 1024
-ub 1024**, n-max2/p-min.6, turbo4 flags and the same default 8192 context.

Backup/rollback source: `$HOME/run_ridge_mtp.sh.bak-F164-20260913`.
Exact scripts are archived as run_ridge_mtp.before_part4.sh and
run_ridge_mtp.after_part4.sh. Installed launcher SHA256:
908f7b7213abdf33d48d78f21c2fe10cce27a2fca1993b5a8c6e1bcdc054defc.
The update checked the old launcher hash to avoid overwriting concurrent edits,
made the backup, preserved executable mode 775, and atomically replaced the
launcher. Bash syntax passes; ldd resolves CUDA to the installed candidate, with
no missing dependencies. The copied library matches the tested artifact exactly.
This takes effect at the **next launcher invocation**; no production server was
left running just to exercise the launcher.

Evidence for adoption: F158 full CUDA MUL_MAT **1261/1261 supported tests passed,
0 FAIL at 67→66 C**; F159 six-arm decode **+0.66%, all 65→72 C**, explicit batch512;
F163 production-batch1024 confirmation **+0.69%**, control 61→68 / candidate 62→69 /
control 62→69 C, all output and draft/accept counts matching. The tighter six-arm
control spread was about 0.05%; the production controls span 0.38%. This is a
small measured improvement on prose_p1, not a universal gain across prompts.

The F157 prefill flag improvement remains in place: historical F155 **+14.39%**
at 2048 tokens, all 62→70 C. Prefill throughput was not retimed with sign-add in
part4. Instead, module comparison confirms all non-MMVQ CUDA device modules,
including the large-batch GEMM/dequantization modules, are byte-identical; only
the intended MMVQ module changed. Do not relabel the historical prefill number
as a new measurement of this deployed library. No larger-context or fully
resident 8K VRAM guarantee is added.

Final checks: GPU healthy and idle at **64 C, 0 MiB**, no session-owned CUDA
workload left. Application clocks restored to 1189 SM / 715 HBM; idle SM was
405 MHz. Existing power cap stays 150 W. gpu-guard is active with GUARD=78;
no gpu-guard entries or kernel Xids were found in the reviewed continuation
window. No reboot or cold cycle was needed in part4. The persistent
**/var/log/gputrace.log** is preserved locally as **gputrace_part4.log**;
per-request 0.5-second telemetry and full response/server logs are also retained.

**Measured conclusion:** ordinary decode is dominated by SM-sensitive quantized
matvec execution, with a demonstrated narrow instruction-path improvement.
The activity trace does not reveal a large pool of launch gaps. The requested
1328 MHz point did not sustain that clock, so it does not establish a full
upper-clock knee or a large headroom percentage. Actual DRAM traffic remains
unmeasured, and the file-byte/triad ratio is not used as utilization. These
measurements do not establish that all further optimization is exhausted.

## F165 — Context measurement BLOCKED by thermal budget; VRAM ceiling ~32K measured (2026-09-13)

Attempt by Claude to measure the decode-vs-resident-context curve the user asked for. **The curve
was NOT obtained.** Four runs aborted; zero depth points completed. Recording the constraints
because they are the useful result, and so the next attempt is designed correctly.

**VRAM ceiling, MEASURED.** Ridge-q41nonffn + turbo4 K/V + draft-mtp, `-b 1024 -ub 1024`:

| allocated context | VRAM idle | VRAM during prefill |
|---|---:|---:|
| `-c 8192`  | 13795 MiB | 13835 MiB |
| `-c 32768` | 15607 MiB | 16045 MiB |

Against ~16276 MiB usable, `-c 32768` leaves ~230 MiB at peak. **So ~32K is the practical
allocation ceiling for this model, and 128K does NOT fit.** Any "128K on the P100" claim refers to
a different model or quant, not this champion. Not yet measured: whether 32K is stable under a
fully resident 32K workload, as opposed to allocation plus a partial fill.

**THERMAL BUDGET IS THE BINDING CONSTRAINT, and it is tighter than the VRAM one.** Measured costs
at `-b 1024 -ub 1024`, 1189 MHz:

  - a 1500-token prefill chunk costs **~6 C**
  - a 128-token decode sample costs **~5 C**
  - cold boot floor **40-48 C**; abort used **74 C**; system gpu-guard **78 C**

So ONE boot buys roughly **five chunks, ~7500 tokens**, including measurement overhead. Worse, once
soaked the card will not recover: observed **75 C at 38.7 W with 0 MiB resident**, and **72 C at
37.7 W idle**, i.e. it does not return to the 59-60 C fanned idle floor within any usable wait.
`cool()` targets of 62 and 66 C both proved unreachable post-work. The fan (fitted 2026-09-11)
fixed the IDLE RUNAWAY, not sustained-load headroom — see F154's power figures (decode 143.9 W,
prefill 123.6 W).

**CONSEQUENCE FOR THE NEXT ATTEMPT:** a process that builds context in one server lifetime cannot
reach useful depth, because the thermal budget forces cold cycles and a server restart discards the
KV cache. Deep-context work therefore REQUIRES `--slot-save-path` persistence across restarts, as
already recorded in the resident-context axis work (65.4 KiB/token, validated 4000 -> 7000 ->
10000). That was skipped here and is why all four runs failed.

**Method errors in this attempt, disclosed:** (1) a probe prompt that told the model to answer
briefly produced a 4-token sample (`predicted_n` 4) read as 26.03 t/s — invalid, fixed with
`ignore_eos`; (2) three harness instances were launched concurrently onto one log and one port,
since the cleanup killed only the first PID — fixed with a lock file; (3) two cooldown targets were
set below the card's achievable floor and hung. None of these produced a recorded measurement.

**The cold-cycle path DID work unattended for the first time:** power down, 8 min soak, magic packet
re-sent every 10 s, box up, **75 C -> 48 C**. `z640_coldcycle.sh` is now reliable.

## F166 — RETRACTION: F165's "128K does not fit" is WRONG, and F157/F164 are REVERTED (2026-09-13)

**F165 is retracted on its VRAM conclusion.** It measured `-c 32768` at 15607 MiB using
`lucefix/llama-server` with NO `LUCE_MTP_WINDOW` and no `-ctkd/-ctvd`. Per the history page
(SESSION 2026-08-30/31), the MTP context otherwise **inherits the target's full `n_ctx`**, so that
run allocated a second complete KV. The figure is an artifact of the wrong binary and flags, not a
property of the card.

**The verified long-context configuration is on record and works:**
`-c 131072 -b 256 -ub 256 --ctx-checkpoints 1 -ctkd turbo4 -ctvd turbo4 LUCE_MTP_WINDOW=1024`
-> **16.28 t/s decode, 100% acceptance, 15619 MiB, peak 65 C** (2026-08-31), using the windowed
build `~/lucewin/` (commit `e56e140b8`). The 128K blocker was the **compute buffer (990.95 MiB)**,
which `-b/-ub 256` shrinks enough; the MTP window saving scales with context and is load-bearing at
128K. F165's thermal-budget figures stand; only its VRAM/context-ceiling conclusion is void.

**CONSEQUENCE — F157 and F164 are REVERTED in the launcher.** F157 raised `-b/-ub` from 512 to
1024 for +14.39% prefill, measured ONLY at `-c 8192`. Larger microbatch inflates the compute buffer
— precisely the allocation that had to be shrunk to 256 to make 128K fit. Shipping 1024 therefore
trades away the verified 120-128K capability for a prefill gain measured where VRAM is not scarce.
That trade was never measured at long context and was not the user's intent.

`$HOME/run_ridge_mtp.sh` is restored byte-identical to `run_ridge_mtp.sh.bak-F157-20260912`
(sha256 26fe20dc...): `LD_LIBRARY_PATH=$HOME/sub4_2:$HOME/lucefix`, `-b 512 -ub 512`. The Codex
variant is preserved as `run_ridge_mtp.sh.codex-F164-variant`. The model file
`Ridge-q41nonffn.gguf` was never modified by any of this work.

**The F159/F164 sign-add kernel result (+0.66% decode, 1261/1261 correctness) is NOT disputed** —
it is reverted only because it rode in on the same launcher edit. It can be re-shipped on its own
by changing LD_LIBRARY_PATH alone, leaving `-b/-ub 512` intact, once someone confirms it at long
context. Same for the prefill flag: it is a legitimate gain AT 8K and should be exposed as an
opt-in (e.g. honouring a `BATCH` env var) rather than baked into the default that must also serve
128K.

**Standing lesson:** any launcher change must be validated at the LONG-CONTEXT configuration, not
only at the `-c 8192` benchmark default. VRAM headroom at 8K says nothing about 128K.

## F167 — Current parameterised launcher VERIFIED at 128K allocation; headroom is smaller than the historical record (2026-09-14 UTC)

First workload of this continuation used the unmodified current launcher: champion
Ridge-q41nonffn, sign-add/sub4_2/lucefix, CTX=131072 BATCH=256 CKPT=1
DRAFT_KV=turbo4 LUCE_MTP_WINDOW=1024, n-max2 / p-min.6, full GPU offload.
The 15-token prose_p1 prompt / 192-token decode completed at **18.5506 t/s,
50→58 C**. Actual timed SM=1189 and HBM=715 MHz. 119 drafted / 104 accepted,
output c1453f54...; resident slot ends at 206 tokens. This is **128K allocated,
not 128K resident**. Short prompt evaluation was 2077.682 ms; it is not a bulk
prefill throughput benchmark. No thermal correction or claim against the 47 C
historical reference is made.

Ready VRAM **15687 MiB at 50 C**, measured request peak **16039 MiB at 50→58 C**,
card reports 16269 MiB total: only **230 MiB** remains at this observed peak.
Startup succeeded and drafting remained active. Thus the CURRENT recipe works
for this short request, but the historical ~657 MiB spare is not its measured
headroom. Long-prefill and batch-variant peaks still need measurement.

The ledger's PROVENANCE section and surviving ~/longctx_server.log establish
that the August 31 16.28 t/s record used Ridge-3.7bpw, n-max1 / p-min0, 407 prompt
and 16 output tokens. F166 correctly preserves a historical allocation capability,
but does not establish identical model/configuration or full-resident context.
The initial cold model read took nearly four minutes; this was disk I/O, not
compute. No driver/counter retries or clock/power-limit increases were made.
Evidence: perf_20260911/remote/part5/verify128_launcher.*, results.jsonl,
telemetry.jsonl; remote persistent /var/log/gputrace.log.

After this successful first verification, UBATCH was added as an independent
launcher override, defaulting to BATCH (which still defaults to 512). No default
batch, context, checkpoint, speculation or library selection changed. Backup:
~/run_ridge_mtp.sh.bak-F167-UBATCH-20260914. Bash syntax checked. The LIBS comment
was corrected: an empty LIBS uses the default with ${LIBS:-...}; selecting stock
requires the explicit sub4_2:lucefix path. This is a flag-interface change, not a
new kernel promotion. F158 still identifies the installed sign-add artifact's
1261/1261 supported MUL_MAT passes, 0 FAIL at 67→66 C.

## F168 — Batch decomposition at 128K allocation (measurements appended as completed)

All arms: current sign-add libraries, n-max2 / p-min.6, target K=q8_0
(auto-asymmetric) / V=turbo4, draft turbo4 flags, MTP window1024, CKPT1.
The common long request uses exactly 2048 token IDs calibrated by /tokenize,
cache_prompt=false, followed by a simple arithmetic answer. Its 5-token output
is a prefill correctness check, **not a meaningful decode speed sample**.
Decode is measured separately with the 192-token prose_p1 request.

First baseline **b256/ub256: 100.8374 prefill t/s, 55→63 C**, 20309.922 ms
for 2048 newly evaluated tokens; output hash 7c9752f7..., same as F155.
Ready VRAM 15687 MiB, request peak **16059 MiB** (55→63 C), leaving 210 MiB
of reported card total. No MTP process failure, guard trip or OOM.
This validates a real prefill at the 128K allocation, not 128K resident context.
Before/after telemetry, exact request, launcher, command, maps and response are
in remote/part5/ctx128_pf_a0.*. A closing baseline and candidate arms follow.

### F168 — ub512 is a measured allocation failure, not a slow timing

The 128K b512/ub512 arm exited during startup with `cudaMalloc failed: out of
memory`, requesting **990.95 MiB** for the prefill compute buffer. MTP window1024
was active with reserve512; it was not accidentally omitted. No request ran;
prefill/decode rates are **unavailable**, not zero and not estimated from 8K.
No fallback/offload or automatic batch reduction was accepted. Full server log
and sampled startup VRAM/temperatures are in ctx128_pf_ub512.* and telemetry.jsonl.
This closes this flag combination under the current precise recipe, not all
possible memory-saving trades.

### F164/F162 mechanism clarification — target n=3 is a maximum, and IQ2_S is not the MTP FFN

The current champion GGUF header has Q6_K blk.64 FFN/nextn projections and Q4_1
attention matrices; no IQ2_S MTP-layer matrix. Draft generation is ncols_dst=1
per token, but its catch-up process can use wider batches. Target verification
appends the sampled token plus ACTUAL proposals, so p_min=.6 produces widths
1,2,3, all directly seen for IQ2_S in F161. Thus a draft-only IQ2_S improvement
is not the explanation for F163. A narrow target-n1 improvement is mechanically
consistent with the observed sub-1% end-to-end effect: F161 puts target IQ2_S n1
at about 10% of profiled wall time (65→70 C), and F162 reduces that one measured
n1 shape's time by 9.48% at its recorded temperatures. This is a scale check,
not a cross-request Amdahl prediction. The +0.689% F163 bracket alone remains
suggestive rather than established against its 0.384% control span; stronger
production-batch replication needs independent paired blocks at matched temps
and speculative work, enough replicates for a paired interval excluding zero.
New evidence: perf_20260911/remote/part5/champion_header.json; no GPU timing was
needed for the tensor-type/source inspection.

### F168 ongoing arm — ctx128_pf_a1

Allocated context 131072; b256/ub256; prefill. Temperature **50→59 C**; sampled peak 59 C; VRAM peak **16059 MiB**. **98.3663 prefill t/s**, 2048 evaluated tokens in 20820.133 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx128_pf_a1.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx128_dec_a0

Allocated context 131072; b256/ub256; decode. Temperature **56→62 C**; sampled peak 63 C; VRAM peak **16039 MiB**. **18.5625 decode t/s**, 192 generated tokens in 10343.458 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx128_dec_a0.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx128_dec_b1024

Allocated context 131072; b1024/ub256; decode. Temperature **60→66 C**; sampled peak 66 C; VRAM peak **16039 MiB**. **18.5080 decode t/s**, 192 generated tokens in 10373.898 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx128_dec_b1024.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx128_dec_a1

Allocated context 131072; b256/ub256; decode. Temperature **62→69 C**; sampled peak 69 C; VRAM peak **16039 MiB**. **18.5619 decode t/s**, 192 generated tokens in 10343.747 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx128_dec_a1.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 — 128K logical-batch decode bracket CLOSED

b256/ub256 **18.5625 t/s (56→62 C, sampled peak63 C)**;
b1024/ub256 **18.5080 (60→66 C)**;
b256/ub256 **18.5619 (62→69 C)**.
All 192 output tokens, 119/104 drafted/accepted, c1453f54..., sampled clocks
1189/715 MHz, peak16039 MiB. The b256 controls close to **0.0028%** despite
bracketing the candidate temperatures. Candidate is **−0.292%** versus their
mean18.5622 t/s. That is a small observed cost with only one candidate sample,
not proof of exact invariance. No temperature correction. The b1024 prefill
replicate and final prefill baseline are still queued, so its net trade is not
accepted or recommended yet.

The first closing prefill baseline after a cold boot was **98.3663 t/s at50→59 C**,
peak16059 MiB, versus the initial100.8374 at55→63 C. Their2.48% spread exceeds
what would justify a precise ~1% logical-batch gain. In particular the colder
arm was slower; this first-long-request procedure includes startup/cache effects
and the difference is not silently assigned to temperature. Replication follows.

### F168 continuation protocol — recovering after quota interruption

The saved b1024/ub256 128K arm DID complete at101.9699 prefill t/s (60→68 C),
peak16059 MiB, but its replication/final baseline and8K decomposition were not
completed before the coordinator ended. No valid completed arm is being rerun.
Current recovery found the GPU idle at65 C,0 MiB; the stated38 C session start
is not substituted for actual measurement temperatures. The coordinator now
runs detached, retaining its per-arm records across an interactive quota exit.

The user's literal8K b256/ub1024 comparison is added. The source clamps effective
n_ubatch to min(n_batch, requested n_ubatch), so that configuration is expected
to execute ub256. It is measured as requested, and the genuine b1024/ub1024 arm
remains in the plan to test a physical1024-token microbatch. No effect from the
capped arm will be attributed to a microbatch that the engine did not execute.
The effective-setting check and timing evidence will be retained separately.

### F168 ongoing arm — ctx128_pf_b1024_r1

Allocated context 131072; b1024/ub256; prefill. Temperature **53→62 C**; sampled peak 62 C; VRAM peak **16059 MiB**. **98.4680 prefill t/s**, 2048 evaluated tokens in 20798.629 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx128_pf_b1024_r1.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx128_pf_a2

Allocated context 131072; b256/ub256; prefill. Temperature **59→67 C**; sampled peak 67 C; VRAM peak **16059 MiB**. **101.8194 prefill t/s**, 2048 evaluated tokens in 20114.054 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx128_pf_a2.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 — 128K prefill replication COMPLETE: logical batch does not recover F155's gain

All five complete 2048-token prefill observations, same token IDs, uncached
prompt, output7c9752f7..., no thermal correction:

| logical b / physical ub | prefill t/s | temperature | request VRAM peak |
|---|---:|---|---:|
|256/256, original opening|100.8374|55→63 C|16059 MiB|
|1024/256, original candidate|101.9699|60→68 C|16059 MiB|
|256/256, first after next cold boot|98.3663|50→59 C|16059 MiB|
|1024/256, first after latest cold boot|98.4680|53→62 C|16059 MiB|
|256/256, final closing baseline|101.8194|59→67 C|16059 MiB|

The first-after-boot pair differs only **+0.103%**; the later-process pair
(original candidate versus final closing control, at similar temperatures)
differs **+0.148%**. These are descriptive comparisons across boots, not a
formal confidence interval. Both settings exhibit the same roughly3.5% shift
between their first-after-boot and later-process observations. Pooling all
observations gives about−0.12% for the candidate, but unequal startup-state
counts make that pooled figure unsuitable as a precise causal effect estimate.

**Measured decision:** increasing the logical batch to1024 while holding ub256
fits128K, but does NOT reproduce the14.39% prefill gain. No useful prefill gain
is established here. It is not correct to promote the original one-off+1.12%
comparison as a win. The completed decode bracket observes−0.292% for this
logical-batch change (temperatures/work in the preceding block); peak VRAM is
unchanged. Keep b256/ub256 for the128K recipe. The physical-ub512 alternative
already OOMed at startup59→60 C, so the original large-microbatch gain and this
128K recipe remain incompatible among these tested settings. This does not
close other memory-saving approaches or claim full-resident128K validation.

The8K decomposition, including the user's literal b256/ub1024 capped case and
the genuine b1024/ub1024 physical-microbatch case, is continuing before the
whole F168 experiment is closed. Evidence remains remote/part5/*.request.json,
*.response.json, *.maps, *.server.log, results.jsonl, telemetry.jsonl, and
persistent Z640 /var/log/gputrace.log.

### F168 recovery note — pre-start temperature boundary, no invalid timing accepted

The first8K baseline was refused three times BEFORE CUDA/server startup: the
coordinator read62 C, but the arm's own pre-start check read63 C. No8K request
ran and no result was averaged. The earlier progress message attributing this
to model loading was incorrect; the raw errors say `COOL_REQUIRED before startup`.
The coordinator now waits for<=61 C before starting a process (the actual timed
long-request gate remains<=62 C) and cold-cycles after any thermal refusal rather
than repeating starts at the same boundary. Earlier failed-attempt files are
preserved before retry. The detached queue resumes from existing completed markers.

### F168 continuation — interrupted CPU prefault recovered, 2026-09-15

The detached queue stopped on an SSH disconnect during CPU-only model prefault, before its first8K CUDA/server arm. Recovery found the GPU healthy at59 C,0 MiB, with no surviving measurement process. This is not classified as a GPU fault or a failed timing. Completed128K observations were preserved; the remaining8K queue resumed. The coordinator now reconciles completed responses and surviving remote processes before retrying a transport interruption. The n=3 candidate remains unpromoted and will be tested only after the batch decomposition closes.

## F169 — llama.cpp BF16→FP32 cuBLAS fallback: DOES NOT APPLY to this champion (2026-09-15)

Upstream merged 2026-09-13: `ggml_cuda_mul_mat_cublas` routes some BF16 work to FP32 on older
GPUs; a P40 showed ~21% (ub 256) / ~28% (ub 512) prefill gain on a **BF16 Llama 1B**. A 2026-09-14
experimental P100 fork patch extends the same fallback to decode, with **no measured speedup**.

**Rejected without testing, on two independent grounds.**

1. **The champion GGUF contains ZERO BF16 tensors**, so the fallback path can never fire. Tensor
   census from (not included), 866 tensors:

   | type | count | share |
   |---|---:|---:|
   | F32 (norms/biases) | 360 | 41.6% |
   | Q4_1 | 213 | 24.6% |
   | IQ2_S | 160 | 18.5% |
   | Q8_0 | 96 | 11.1% |
   | IQ3_S | 32 | 3.7% |
   | Q6_K | 5 | 0.6% |
   | **BF16** | **0** | **0%** |

   (A grep hit for "BF16" in the header is a metadata string, not a tensor type.)

2. **The VRAM cost is disqualifying even if it did apply.** A Qwen user reported ~4 GB extra VRAM
   from FP32 conversion. F167/F168 measured only **210-230 MiB** free at the 128K peak
   (16039-16059 MiB of 16269). Even a tenth of that cost breaks 128K.

Note also that the upstream gain is a PREFILL/large-batch effect on a BF16 model; the decode
extension is unmeasured by its own author. Do not revisit unless a future champion ships BF16
tensors AND long-context VRAM headroom grows substantially.

### F168 ongoing arm — ctx8_prefill_a0

Allocated context 8192; b256/ub256; prefill. Temperature **60→68 C**; sampled peak 68 C; VRAM peak **13049 MiB**. **97.9304 prefill t/s**, 2048 evaluated tokens in 20912.811 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_prefill_a0.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_prefill_ub512

Allocated context 8192; b512/ub512; prefill. Temperature **50→59 C**; sampled peak 59 C; VRAM peak **13835 MiB**. **121.4322 prefill t/s**, 2048 evaluated tokens in 16865.383 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_prefill_ub512.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_prefill_b1024

Allocated context 8192; b1024/ub256; prefill. Temperature **56→64 C**; sampled peak 64 C; VRAM peak **13049 MiB**. **100.5699 prefill t/s**, 2048 evaluated tokens in 20363.952 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_prefill_b1024.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_prefill_b256_ub1024

Allocated context 8192; b256/ub1024; prefill. Temperature **61→69 C**; sampled peak 69 C; VRAM peak **13049 MiB**. **100.5353 prefill t/s**, 2048 evaluated tokens in 20370.952 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_prefill_b256_ub1024.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_prefill_ub1024

Allocated context 8192; b1024/ub1024; prefill. Temperature **48→57 C**; sampled peak 57 C; VRAM peak **15409 MiB**. **138.8118 prefill t/s**, 2048 evaluated tokens in 14753.791 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1139–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_prefill_ub1024.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_prefill_a1

Allocated context 8192; b256/ub256; prefill. Temperature **54→62 C**; sampled peak 62 C; VRAM peak **13049 MiB**. **100.5252 prefill t/s**, 2048 evaluated tokens in 20372.997 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_prefill_a1.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_decode_a0

Allocated context 8192; b256/ub256; decode. Temperature **59→65 C**; sampled peak 65 C; VRAM peak **13029 MiB**. **18.9118 decode t/s**, 192 generated tokens in 10152.397 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_decode_a0.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_decode_ub512

Allocated context 8192; b512/ub512; decode. Temperature **61→68 C**; sampled peak 68 C; VRAM peak **13795 MiB**. **18.9065 decode t/s**, 192 generated tokens in 10155.256 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_decode_ub512.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 — 8K prefill decomposition and closing baseline COMPLETE

The final b256/ub256 baseline is **100.5252 t/s at54→62 C**, peak13049 MiB, versus the opening97.9304 at60→68 C. Their2.615% span prevents a precise small-effect claim from the pooled mean. The logical-only b1024/ub256 arm was100.5699 at56→64 C, only+0.0444% versus the nearby-temperature closing baseline. Literal b256/ub1024 was100.5353 at61→69 C, only+0.0100%, with the same13049 MiB peak. It is capped to effectiveub256 in source; runtime diagnostic follows. No useful logical-batch-only gain is established at8K either.

Physical b512/ub512 achieved121.4322 t/s at50→59 C, peak13835 MiB; b1024/ub1024 achieved138.8118 at48→57 C, peak15409 MiB. Both were the first model process after their respective cold boots. The latter is+14.312% versus the former, consistent with F155's independent bracket, but these two single observations are not a new confidence interval. The genuineub1024 arm had28 request samples at1189 MHz and one at1139 MHz, HBM715 throughout; peak sampled power154.25 W. It is not mislabeled as fixed1189 throughout, and no clock/thermal correction is applied.

The gain follows the effective physical microbatch, while increasing logical b alone atub256 does not reproduce it. Together with the128K ub512 startup OOM, this answers the requested decomposition: the tested logical-batch knob does not preserve the large-microbatch prefill gain under the current128K memory recipe. Decode comparisons are still running before whole-F168 closeout. All exact2048-token requests gave the same7c9752f7... answer; that five-token answer is not a decode-speed measurement.

### F168 ongoing arm — ctx8_decode_b1024

Allocated context 8192; b1024/ub256; decode. Temperature **48→56 C**; sampled peak 56 C; VRAM peak **13029 MiB**. **18.8942 decode t/s**, 192 generated tokens in 10161.842 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_decode_b1024.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_decode_b256_ub1024

Allocated context 8192; b256/ub1024; decode. Temperature **52→59 C**; sampled peak 59 C; VRAM peak **13029 MiB**. **18.9133 decode t/s**, 192 generated tokens in 10151.581 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_decode_b256_ub1024.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_decode_ub1024

Allocated context 8192; b1024/ub1024; decode. Temperature **56→62 C**; sampled peak 63 C; VRAM peak **15325 MiB**. **18.8685 decode t/s**, 192 generated tokens in 10175.703 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_decode_ub1024.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — ctx8_decode_a1

Allocated context 8192; b256/ub256; decode. Temperature **59→65 C**; sampled peak 65 C; VRAM peak **13029 MiB**. **18.9169 decode t/s**, 192 generated tokens in 10149.665 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/ctx8_decode_a1.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

## F168 — Batch decomposition COMPLETE (all planned valid arms recovered)

Rates below are per-arm observations, without temperature correction. 128K means allocated context, not resident context. 8K uses CKPT2/unwindowed draft cache; 128K uses CKPT1, draft turbo4 flags and window1024. All are current sign-add libraries; no kernel/default change was made for this experiment.

| Allocation | b / requested ub | effective ub | regime | rate t/s | temperature | peak MiB | drafted / accepted |
|---:|---|---:|---|---:|---|---:|---|
|131072|256/256|256|prefill|98.3663|50→59 C|16059|2/2|
|131072|256/256|256|decode|18.5625|56→62 C|16039|119/104|
|131072|1024/256|256|decode|18.5080|60→66 C|16039|119/104|
|131072|256/256|256|decode|18.5619|62→69 C|16039|119/104|
|131072|1024/256|256|prefill|98.4680|53→62 C|16059|2/2|
|131072|256/256|256|prefill|101.8194|59→67 C|16059|2/2|
|8192|256/256|256|prefill|97.9304|60→68 C|13049|2/1|
|8192|512/512|512|prefill|121.4322|50→59 C|13835|2/1|
|8192|1024/256|256|prefill|100.5699|56→64 C|13049|2/1|
|8192|256/1024|256|prefill|100.5353|61→69 C|13049|2/1|
|8192|1024/1024|1024|prefill|138.8118|48→57 C|15409|2/1|
|8192|256/256|256|prefill|100.5252|54→62 C|13049|2/1|
|8192|256/256|256|decode|18.9118|59→65 C|13029|123/107|
|8192|512/512|512|decode|18.9065|61→68 C|13795|123/107|
|8192|1024/256|256|decode|18.8942|48→56 C|13029|123/107|
|8192|256/1024|256|decode|18.9133|52→59 C|13029|123/107|
|8192|1024/1024|1024|decode|18.8685|56→62 C|15325|123/107|
|8192|256/256|256|decode|18.9169|59→65 C|13029|123/107|

Effective ub uses the engine clamp verified in src/llama-context.cpp: min(n_batch, requested n_ubatch). The literal b256/ub1024 arm is retained as requested, but does not execute a1024-token microbatch. The separate b1024/ub1024 arm tests that physical size. Prefill rates evaluate2048 exact token IDs; the five-token arithmetic answer is not a decode sample. Decode uses a separate192-token prose request.

8K prefill: b256/ub256 control mean99.2278t/s, control span2.615%, at the temperatures above. Relative descriptive changes: ub512 +22.377%, b1024 +1.353%, b256_ub1024 +1.318%, ub1024 +39.892%. These small-sample differences are not confidence intervals; startup-state and thermal spreads must be considered.

8K decode: b256/ub256 control mean18.9143t/s, control span0.027%, at the temperatures above. Relative descriptive changes: ub512 -0.042%, b1024 -0.106%, b256_ub1024 -0.005%, ub1024 -0.242%. These small-sample differences are not confidence intervals; startup-state and thermal spreads must be considered.

The128K logical-batch result and its paired closing baseline are recorded immediately above: startup-matched prefill differences only+0.10–0.15%, and decode−0.292%. Raising logical b while retaining ub256 does not recover F155’s14.39% prefill gain. Physical ub512 OOMs at128K. Keep b256/ub256 for that recipe. A genuine larger-ub short-context option remains an explicit knob; do not silently apply it to128K. Full-resident128K decode, quality and duty-cycle throughput are not established by these allocation tests.

All raw requests, responses, exact launcher snapshots, loaded maps and0.5-second telemetry are retained in remote/part5; persistent thermal evidence is /var/log/gputrace.log on the Z640. No hardware performance counters were requested. No kernel was promoted.

### F169 n=3 gridshare candidate — gridshare_gate

{'tag': 'gridshare_gate', 'arm': 'gridshare', 'mode': 'test'}; temperature 61→62 C, sampled peak 62 C; exit 0; thermal_trip=False. Full CUDA0 MUL_MAT gate: **1261/1261 tests passed, 0 FAIL**. This is correctness evidence, not throughput. Raw log/result/telemetry: remote/part5/gridshare_gate.*. The candidate rearranges IQ2_S n=3 unfused dot-product work, retaining block geometry and per-output accumulation order; based on sign-add. Candidate SHA256 130c0099d8b7c2344394ef7d9219682876ca5d1909bc87dbc0cb6dced17ac829. No promotion before model validation at BOTH allocations.

### F169 n=3 gridshare candidate — gridshare_perf_0_0_signadd

{'tag': 'gridshare_perf_0_0_signadd', 'arm': 'signadd', 'mode': 'perf', 'gate': 'gridshare_gate', 'pattern': 'type_a=iq2_s,type_b=f32,m=17408,n=3,k=5120'}; temperature 62→65 C, sampled peak 66 C; exit 0; thermal_trip=False. Reported operation timings (not model t/s):   MUL_MAT(type_a=iq2_s,type_b=f32,m=17408,n=3,k=5120,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):                6919 runs -   436.20 us/run - 534.77 MFLOP/run - [1;34m  1.23 TFLOPS[0m. Raw log/result/telemetry: remote/part5/gridshare_perf_0_0_signadd.*. The candidate rearranges IQ2_S n=3 unfused dot-product work, retaining block geometry and per-output accumulation order; based on sign-add. Candidate SHA256 130c0099d8b7c2344394ef7d9219682876ca5d1909bc87dbc0cb6dced17ac829. No promotion before model validation at BOTH allocations.

### F169 n=3 gridshare candidate — gridshare_perf_0_1_gridshare

{'tag': 'gridshare_perf_0_1_gridshare', 'arm': 'gridshare', 'mode': 'perf', 'gate': 'gridshare_gate', 'pattern': 'type_a=iq2_s,type_b=f32,m=17408,n=3,k=5120'}; temperature 63→67 C, sampled peak 68 C; exit 0; thermal_trip=False. Reported operation timings (not model t/s):   MUL_MAT(type_a=iq2_s,type_b=f32,m=17408,n=3,k=5120,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):                6732 runs -   447.61 us/run - 534.77 MFLOP/run - [1;34m  1.19 TFLOPS[0m. Raw log/result/telemetry: remote/part5/gridshare_perf_0_1_gridshare.*. The candidate rearranges IQ2_S n=3 unfused dot-product work, retaining block geometry and per-output accumulation order; based on sign-add. Candidate SHA256 130c0099d8b7c2344394ef7d9219682876ca5d1909bc87dbc0cb6dced17ac829. No promotion before model validation at BOTH allocations.

### F169 n=3 gridshare candidate — gridshare_perf_0_2_gridshare

{'tag': 'gridshare_perf_0_2_gridshare', 'arm': 'gridshare', 'mode': 'perf', 'gate': 'gridshare_gate', 'pattern': 'type_a=iq2_s,type_b=f32,m=17408,n=3,k=5120'}; temperature 65→67 C, sampled peak 68 C; exit 0; thermal_trip=False. Reported operation timings (not model t/s):   MUL_MAT(type_a=iq2_s,type_b=f32,m=17408,n=3,k=5120,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):                6732 runs -   447.50 us/run - 534.77 MFLOP/run - [1;34m  1.20 TFLOPS[0m. Raw log/result/telemetry: remote/part5/gridshare_perf_0_2_gridshare.*. The candidate rearranges IQ2_S n=3 unfused dot-product work, retaining block geometry and per-output accumulation order; based on sign-add. Candidate SHA256 130c0099d8b7c2344394ef7d9219682876ca5d1909bc87dbc0cb6dced17ac829. No promotion before model validation at BOTH allocations.

### F169 n=3 gridshare candidate — gridshare_perf_0_3_signadd

{'tag': 'gridshare_perf_0_3_signadd', 'arm': 'signadd', 'mode': 'perf', 'gate': 'gridshare_gate', 'pattern': 'type_a=iq2_s,type_b=f32,m=17408,n=3,k=5120'}; temperature 66→68 C, sampled peak 69 C; exit 0; thermal_trip=False. Reported operation timings (not model t/s):   MUL_MAT(type_a=iq2_s,type_b=f32,m=17408,n=3,k=5120,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):                6919 runs -   436.24 us/run - 534.77 MFLOP/run - [1;34m  1.23 TFLOPS[0m. Raw log/result/telemetry: remote/part5/gridshare_perf_0_3_signadd.*. The candidate rearranges IQ2_S n=3 unfused dot-product work, retaining block geometry and per-output accumulation order; based on sign-add. Candidate SHA256 130c0099d8b7c2344394ef7d9219682876ca5d1909bc87dbc0cb6dced17ac829. No promotion before model validation at BOTH allocations.

### F169 n=3 gridshare candidate — gridshare_perf_1_0_signadd

{'tag': 'gridshare_perf_1_0_signadd', 'arm': 'signadd', 'mode': 'perf', 'gate': 'gridshare_gate', 'pattern': 'type_a=iq2_s,type_b=f32,m=5120,n=3,k=17408'}; temperature 66→69 C, sampled peak 71 C; exit 0; thermal_trip=False. Reported operation timings (not model t/s):   MUL_MAT(type_a=iq2_s,type_b=f32,m=5120,n=3,k=17408,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):                8415 runs -   362.21 us/run - 534.77 MFLOP/run - [1;34m  1.48 TFLOPS[0m. Raw log/result/telemetry: remote/part5/gridshare_perf_1_0_signadd.*. The candidate rearranges IQ2_S n=3 unfused dot-product work, retaining block geometry and per-output accumulation order; based on sign-add. Candidate SHA256 130c0099d8b7c2344394ef7d9219682876ca5d1909bc87dbc0cb6dced17ac829. No promotion before model validation at BOTH allocations.

### F169 n=3 gridshare candidate — gridshare_perf_1_1_gridshare

{'tag': 'gridshare_perf_1_1_gridshare', 'arm': 'gridshare', 'mode': 'perf', 'gate': 'gridshare_gate', 'pattern': 'type_a=iq2_s,type_b=f32,m=5120,n=3,k=17408'}; temperature 67→70 C, sampled peak 71 C; exit 0; thermal_trip=False. Reported operation timings (not model t/s):   MUL_MAT(type_a=iq2_s,type_b=f32,m=5120,n=3,k=17408,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):                5423 runs -   562.87 us/run - 534.77 MFLOP/run - [1;34m950.08 GFLOPS[0m. Raw log/result/telemetry: remote/part5/gridshare_perf_1_1_gridshare.*. The candidate rearranges IQ2_S n=3 unfused dot-product work, retaining block geometry and per-output accumulation order; based on sign-add. Candidate SHA256 130c0099d8b7c2344394ef7d9219682876ca5d1909bc87dbc0cb6dced17ac829. No promotion before model validation at BOTH allocations.

### F169 n=3 gridshare candidate — gridshare_perf_1_2_gridshare

{'tag': 'gridshare_perf_1_2_gridshare', 'arm': 'gridshare', 'mode': 'perf', 'gate': 'gridshare_gate', 'pattern': 'type_a=iq2_s,type_b=f32,m=5120,n=3,k=17408'}; temperature 67→71 C, sampled peak 71 C; exit 0; thermal_trip=False. Reported operation timings (not model t/s):   MUL_MAT(type_a=iq2_s,type_b=f32,m=5120,n=3,k=17408,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):                5423 runs -   563.05 us/run - 534.77 MFLOP/run - [1;34m949.77 GFLOPS[0m. Raw log/result/telemetry: remote/part5/gridshare_perf_1_2_gridshare.*. The candidate rearranges IQ2_S n=3 unfused dot-product work, retaining block geometry and per-output accumulation order; based on sign-add. Candidate SHA256 130c0099d8b7c2344394ef7d9219682876ca5d1909bc87dbc0cb6dced17ac829. No promotion before model validation at BOTH allocations.

### F169 n=3 gridshare candidate — gridshare_perf_1_3_signadd

{'tag': 'gridshare_perf_1_3_signadd', 'arm': 'signadd', 'mode': 'perf', 'gate': 'gridshare_gate', 'pattern': 'type_a=iq2_s,type_b=f32,m=5120,n=3,k=17408'}; temperature 68→72 C, sampled peak 73 C; exit 0; thermal_trip=False. Reported operation timings (not model t/s):   MUL_MAT(type_a=iq2_s,type_b=f32,m=5120,n=3,k=17408,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):                8415 runs -   363.73 us/run - 534.77 MFLOP/run - [1;34m  1.47 TFLOPS[0m. Raw log/result/telemetry: remote/part5/gridshare_perf_1_3_signadd.*. The candidate rearranges IQ2_S n=3 unfused dot-product work, retaining block geometry and per-output accumulation order; based on sign-add. Candidate SHA256 130c0099d8b7c2344394ef7d9219682876ca5d1909bc87dbc0cb6dced17ac829. No promotion before model validation at BOTH allocations.

## F169 — gridshare n=3 microbenchmark decision

Shape m×k=17408x5120, n=3. gridshare_perf_0_0_signadd: 436.20 us/run, 62→65 C; gridshare_perf_0_1_gridshare: 447.61 us/run, 63→67 C; gridshare_perf_0_2_gridshare: 447.50 us/run, 65→67 C; gridshare_perf_0_3_signadd: 436.24 us/run, 66→68 C. Candidate operation-throughput change -2.533%; control time span 0.009%.

Shape m×k=5120x17408, n=3. gridshare_perf_1_0_signadd: 362.21 us/run, 66→69 C; gridshare_perf_1_1_gridshare: 562.87 us/run, 67→70 C; gridshare_perf_1_2_gridshare: 563.05 us/run, 67→71 C; gridshare_perf_1_3_signadd: 363.73 us/run, 68→72 C. Candidate operation-throughput change -35.525%; control time span 0.419%.

The compiler reports137→126 registers and1092→1134 static instructions for gridshare, without spills or a block-geometry change. This is not an achieved-occupancy measurement. The experiment changes scheduling/register demand together; it cannot uniquely attribute any gain to occupancy. No temperature correction. Microseconds are operation timings, not model token rates.

The strict automatic advancement criterion was not met. Review the measured shape-specific result; no end-to-end speedup or globally exhausted optimization claim follows. No promotion.

### F168 ongoing arm — anchor_stock_default_a

Allocated context 8192; b512/ub512; decode. Temperature **60→67 C**; sampled peak 67 C; VRAM peak **13795 MiB**. **19.2320 decode t/s**, 192 generated tokens in 9983.373 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1227–1328 MHz, HBM 715–715 MHz. Raw evidence remote/part5/anchor_stock_default_a.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — anchor_signadd_default_a

Allocated context 8192; b512/ub512; decode. Temperature **49→57 C**; sampled peak 57 C; VRAM peak **13795 MiB**. **19.7473 decode t/s**, 192 generated tokens in 9722.854 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1240–1328 MHz, HBM 715–715 MHz. Raw evidence remote/part5/anchor_signadd_default_a.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — anchor_stock_pinned_a

Allocated context 8192; b512/ub512; decode. Temperature **53→60 C**; sampled peak 60 C; VRAM peak **13795 MiB**. **18.8121 decode t/s**, 192 generated tokens in 10206.170 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/anchor_stock_pinned_a.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — anchor_signadd_default_b

Allocated context 8192; b512/ub512; decode. Temperature **57→64 C**; sampled peak 64 C; VRAM peak **13795 MiB**. **19.5156 decode t/s**, 192 generated tokens in 9838.266 ms; 123/107 drafted/accepted. Output eb77c530a9fa; cache_n=0; sampled SM 1227–1328 MHz, HBM 715–715 MHz. Raw evidence remote/part5/anchor_signadd_default_b.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

## F170 — The harness PINS the SM clock; every harness decode number understates production by ~2%. No regression exists (2026-09-15)

F168's 8K decode arms all sat at ~18.91 t/s against the champion's verified 19.44
(-2.7%). The thermal hypothesis was tested and FAILED: colder arms at 48->56 C and
52->59 C also returned ~18.9, and the closing control was 18.9169 at 59->65 C.

**Cause: `batch_arm.py` calls `clock(1189)`, pinning the SM clock. Production runs the
default/boost policy and samples 1227-1328 MHz.** A 2x2 (libs x clock policy) at the
shipped 8K config (b512/ub512, 192-token prose_p1, 123/107 drafted/accepted, output
eb77c530a9fa on every arm) separates the two axes:

| libs | clock | decode t/s | temperature | sampled SM |
|---|---|---:|---|---|
| sign-add F159 | pinned | 18.9169 | 59->65 C | 1189 |
| sign-add F159 | **default** | **19.7473** | 49->57 C | 1240-1328 |
| sign-add F159 | **default** | **19.5156** | 57->64 C | 1227-1328 |
| stock | pinned | 18.8121 | 53->60 C | 1189 |
| stock | default | 19.2320 | 60->67 C | 1227-1328 |

**Clock policy at fixed libs:** +2.23% (stock) and +3.81% (sign-add, vs the mean of its
three pinned observations). The stock pair is the CONSERVATIVE reading: its default arm
ran WARMER (60->67 vs 53->60), so temperature opposes the effect rather than inflating it.
The sign-add pair has the opposite confound. Call it **>=2.2%**, not a point estimate.

**Sign-add at fixed (pinned) clock: +0.557%** (18.9169/18.8121), or +0.530% using the mean
of the three pinned sign-add observations. This independently reproduces the separately
bracketed **+0.66%** from a different session at different instrument settings.
The apparent +2.68% for sign-add at default clocks is CONFOUNDED (49->57 vs 60->67 C,
11 C apart with the faster arm colder) and is rejected in favour of the pinned comparison.

**CONCLUSION: nothing regressed.** Production config (sign-add + default clocks) measures
**19.7473 and 19.5156, mean 19.63** - at or ABOVE the verified 19.44. The 1.2% spread
across ~8 C is ordinary thermal sensitivity.

**PROCESS CONSEQUENCE, applies to the whole ledger:** any decode figure produced by
`batch_arm.py` without `clock_mode: default` carries a pinned 1189 MHz clock and is
~2% BELOW how the box actually runs. Such figures are valid for A/B comparison against
each other (the pin is a constant) but MUST NOT be compared against production baselines
such as 19.44. Pinning remains the right default for A/B work - it removes boost variance -
but the resulting numbers are instrument-relative, not absolute.

Evidence: perf_20260911/remote/part5/anchor_stock_default_a.*, anchor_signadd_default_a.*,
anchor_signadd_default_b.*, anchor_stock_pinned_a.*, anchor2_queue.log, telemetry.jsonl.
Arms run by Claude driving the existing batch_queue.py/batch_arm.py harness after Codex
exhausted its weekly quota (resets 2026-09-19 07:39).

### F168 ongoing arm — fr_ub512_ctx32k

Allocated context 32768; b512/ub512; prefill. Temperature **60→68 C**; sampled peak 68 C; VRAM peak **14379 MiB**. **124.3212 prefill t/s**, 2048 evaluated tokens in 16473.462 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/fr_ub512_ctx32k.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — fr_ub512_ctx64k

Allocated context 65536; b512/ub512; prefill. Temperature **49→57 C**; sampled peak 57 C; VRAM peak **15195 MiB**. **120.6606 prefill t/s**, 2048 evaluated tokens in 16973.236 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/fr_ub512_ctx64k.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — fr_ub1024_ctx16k FAILED

Allocated context 16384; b1024/ub1024; error Remote end closed connection without response; terminal GPU sample {'utc': '2026-09-16T02:57:39.953119+00:00', 'epoch': 1789527459.9531426, 'temp': 59.0, 'sm': 1189.0, 'mem': 715.0, 'watts': 39.13, 'mib': 15551.0}. No valid prefill rate. See remote/part5/fr_ub1024_ctx16k.*; failed attempts are not timing replicates.

### F168 ongoing arm — fr_ub512_ctx96k

Allocated context 98304; b512/ub512; prefill. Temperature **58→66 C**; sampled peak 66 C; VRAM peak **16011 MiB**. **125.6702 prefill t/s**, 2048 evaluated tokens in 16296.622 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/fr_ub512_ctx96k.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — fr_ub1024_ctx32k FAILED

Allocated context 32768; b1024/ub1024; error Remote end closed connection without response; terminal GPU sample {'utc': '2026-09-16T02:58:57.940419+00:00', 'epoch': 1789527537.9404423, 'temp': 67.0, 'sm': 1189.0, 'mem': 715.0, 'watts': 42.23, 'mib': 15915.0}. No valid prefill rate. See remote/part5/fr_ub1024_ctx32k.*; failed attempts are not timing replicates.

## F171 — Context frontier at the SHIPPED microbatch: 96K fits at full prefill speed (2026-09-15)

F168 established that the prefill lever is the PHYSICAL microbatch `-ub`, not the
logical `-b`, and that the shipped default is `BATCH=512` (b512/ub512). That reframed
the long-standing "fast prefill OR long context" trade as a measurable question:
**how much context can the shipped ub512 hold?**

All arms: sign-add libs, **pinned 1189 MHz** (comparable to F168's prefill table),
`window=True` FORCED on every arm (`LUCE_MTP_WINDOW=1024`, `-ctkd/-ctvd turbo4`),
ckpt=1 at >=64K else 2, 2048-token uncached prompt, output hash 7c9752f78f8b on every
successful arm. Forcing the window matters: without it the MTP head allocates a SECOND
full KV cache, which is the artifact that produced F165's bogus "32K VRAM ceiling".

| context | prefill t/s | VRAM peak | headroom | temperature |
|---|---:|---:|---:|---|
| 8192 (F168 baseline) | 121.4322 | 13835 MiB | 2549 MiB | 50->59 C |
| 32768 | 124.3212 | 14379 MiB | 2005 MiB | 60->68 C |
| 65536 | 120.6606 | 15195 MiB | 1189 MiB | 49->57 C |
| **98304** | **125.6702** | **16011 MiB** | **373 MiB** | 58->66 C |

**Prefill is FLAT across a 12x context range** (120.66-125.67, no monotone trend; the
spread is run-to-run scatter and the arms are not temperature-matched). 96K retains
MORE headroom (373 MiB) than the shipped 128K/ub256 recipe (210 MiB, F168).

**This is allocation, not resident context.** Every figure is a 2048-token uncached
prefill; decode at these depths is measured separately and was NOT established here.

**PREDICTION QUALITY, recorded because it is the F165 failure mode:** a 2-point
extrapolation (~25.7 KiB/token from the 8K->128K ub256 delta) predicted 14438 / 15241 /
16116 MiB against measured 14379 / 15195 / 16011 - errors of 0.4% / 0.3% / 0.7%, all
CONSERVATIVE. It nonetheless called 96K "too tight" when 96K fits with 373 MiB spare.
The extrapolation was fit for choosing which arms to run; it was NOT fit for deciding
a boundary, and was not used as one.

### F171 — `-ub 1024` requires an MTP draft window LARGER than the microbatch

Both ub1024 arms FAILED identically, and **not on VRAM**:
`decode: failed to find a memory slot for batch of size 1020` ->
`llama_decode(ctx_dft) failed rc=1 (pos=1024)` -> `failed to process speculative batch`.
`ctx_dft` is the DRAFT context. A 1024-token microbatch submits a ~1020-token batch to
an MTP draft context whose KV was capped at `LUCE_MTP_WINDOW=1024`; it does not fit and
the server aborts.

This combination had never been exercised: ub1024 had only ever run at 8K where the
window is OFF (F168's 138.8118 arm is therefore UNAFFECTED and stands), and the window
had only ever been used at 128K where ub is 256. The constraint is
**`LUCE_MTP_WINDOW` must exceed the physical microbatch**, which costs VRAM precisely
where VRAM is scarce. Retest with window_size=2048 is queued.
`batch_arm.py` gained a `window_size` knob (default 1024, behaviour unchanged) and
`ctk`/`ctv` passthrough; `run_ridge_mtp.sh` gains `CTK`/`CTV` (default turbo4,
behaviour unchanged), backed up as `.bak-preCTK-20260915`.

Evidence: perf_20260911/remote/part5/fr_*.{request,response,server.log,maps,command}.json,
frontier_queue.log, telemetry.jsonl. Arms run by Claude driving the existing
batch_queue.py/batch_arm.py harness; Codex is out of quota until 2026-09-19 07:39.

### F168 ongoing arm — fr2_dec_ub512_ctx96k

Allocated context 98304; b512/ub512; decode. Temperature **49→56 C**; sampled peak 56 C; VRAM peak **15971 MiB**. **18.5464 decode t/s**, 192 generated tokens in 10352.398 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/fr2_dec_ub512_ctx96k.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — fr2_dec_ub512_ctx64k

Allocated context 65536; b512/ub512; decode. Temperature **53→59 C**; sampled peak 59 C; VRAM peak **15155 MiB**. **18.5747 decode t/s**, 192 generated tokens in 10336.641 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/fr2_dec_ub512_ctx64k.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — fr2_dec_ub512_ctx8k

Allocated context 8192; b512/ub512; decode. Temperature **56→63 C**; sampled peak 63 C; VRAM peak **13727 MiB**. **18.5912 decode t/s**, 192 generated tokens in 10327.463 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/fr2_dec_ub512_ctx8k.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — fr2_pf_ub512_ctx112k FAILED

Allocated context 114688; b512/ub512; error Remote end closed connection without response; terminal GPU sample {'utc': '2026-09-16T03:16:12.771812+00:00', 'epoch': 1789528572.771837, 'temp': 61.0, 'sm': 1189.0, 'mem': 715.0, 'watts': 39.61, 'mib': 16223.0}. No valid prefill rate. See remote/part5/fr2_pf_ub512_ctx112k.*; failed attempts are not timing replicates.

### F168 ongoing arm — fr2_pf_ub1024_ctx32k

Allocated context 32768; b1024/ub1024; prefill. Temperature **61→67 C**; sampled peak 68 C; VRAM peak **15921 MiB**. **145.5484 prefill t/s**, 2048 evaluated tokens in 14070.920 ms. The 5-token answer is not a decode benchmark. Output 7c9752f78f8b; cache_n=0; sampled SM 1101–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/fr2_pf_ub1024_ctx32k.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — f16k_ctx8k_ctrl_a

Allocated context 8192; b512/ub512; decode. Temperature **62→69 C**; sampled peak 69 C; VRAM peak **13727 MiB**. **18.5746 decode t/s**, 192 generated tokens in 10336.689 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/f16k_ctx8k_ctrl_a.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — d32k_ub1024_w2048_a

Allocated context 32768; b1024/ub1024; decode. Temperature **49→56 C**; sampled peak 56 C; VRAM peak **15837 MiB**. **18.5157 decode t/s**, 192 generated tokens in 10369.564 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/d32k_ub1024_w2048_a.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — d32k_ub512_w1024_a

Allocated context 32768; b512/ub512; decode. Temperature **53→60 C**; sampled peak 60 C; VRAM peak **14339 MiB**. **18.5545 decode t/s**, 192 generated tokens in 10347.874 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/d32k_ub512_w1024_a.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — d32k_ub1024_w2048_b

Allocated context 32768; b1024/ub1024; decode. Temperature **56→63 C**; sampled peak 63 C; VRAM peak **15837 MiB**. **18.5211 decode t/s**, 192 generated tokens in 10366.582 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/d32k_ub1024_w2048_b.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — d32k_ub512_w1024_b

Allocated context 32768; b512/ub512; decode. Temperature **60→66 C**; sampled peak 66 C; VRAM peak **14339 MiB**. **18.5563 decode t/s**, 192 generated tokens in 10346.902 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/d32k_ub512_w1024_b.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

## F172 — Decode at the SHIPPED and FAST microbatches across the context frontier; `BATCH=1024` costs 0.20% decode and 1.5 GB (2026-09-15)

F171 measured PREFILL across the frontier. `BATCH=1024` had NEVER been measured on
decode at any context, so its +17% prefill was one-sided. It is now bracketed.

All arms: sign-add libs, pinned 1189 MHz, `window=True`, 192-token prose_p1,
**119/104 drafted/accepted and output c1453f54b802 on EVERY arm** - identical
speculative work, so these are like-for-like.

**Decode vs context at the shipped ub512 (window1024), same session:**

| context | decode t/s | vs 8K | temperature | VRAM peak |
|---|---:|---:|---|---:|
| 8192 | 18.5912 | - | 56->63 C | 13727 MiB |
| 65536 | 18.5747 | -0.089% | 53->59 C | 15155 MiB |
| 98304 | 18.5464 | **-0.241%** | 49->56 C | 15971 MiB |

**Decode is FLAT to 96K.** Temperature works AGAINST the effect - the 96K arm ran
COLDEST and was slowest - so the true context cost is if anything smaller than -0.24%.

**Microbatch at fixed 32K context (paired, alternating, n=2 each):**

| 32K arm | decode t/s | mean | VRAM peak |
|---|---:|---:|---:|
| b1024/ub1024, window2048 | 18.5157 / 18.5211 | **18.5184** | 15837 MiB |
| b512/ub512, window1024 | 18.5545 / 18.5563 | **18.5554** | 14339 MiB |

**-0.1994%.** Each pair closes to 0.029% and 0.010% across 7 C and 6 C of spread.
Against the SAME-CONTEXT ub512 prefill control (124.3212, F171), ub1024's 145.5484
is **+17.07%** - not the +19.8% obtained by comparing against the 8K baseline.
**The real cost is +1498 MiB (decode) / +1542 MiB (prefill)**, which is why the fast
microbatch caps near 32K while ub512 reaches 96K.

### F172 — the ~1.7% draft-KV cost is the price of the long-context MACHINERY, not of depth

This session's 8K decode reads **18.5912** while F168's 8K arms read **~18.91**. Same
model, same shipped b512/ub512, same pinned clock. The command records differ:
F168's `ctx8_decode_a0` has DRAFT_KV and LUCE_MTP_WINDOW **unset**; this session's
`fr2_dec_ub512_ctx8k` has `DRAFT_KV=turbo4, LUCE_MTP_WINDOW=1024`. Speculative work
drops with it: **123/107 -> 119/104**, i.e. the draft head accepts slightly less with a
turbo4-quantized draft cache. b/ub also differ (256 vs 512), but F168 measured the 8K
microbatch effect on decode at ~0 (18.9118 vs 18.9065), so the ~1.7% is attributed to
the draft-KV/window. **NOT isolated** - it was not varied alone - but well supported.
Consequence: the ~1.7% is paid ONCE for enabling long context, and depth beyond that
is nearly free (-0.24% out to 96K).

### F172 — 112K is a genuine allocation wall
`fr2_pf_ub512_ctx112k` died with `CUDA error: out of memory` (ggml-cuda.cu:107) at a
sampled 16223 MiB. Not a config conflict, unlike the ub1024/window failures of F171.
The ub512 ceiling is therefore between 98304 (16011 MiB, fits, 373 MiB spare) and 114688.

**MODE TABLE (all ALLOCATION, not resident context):**

| mode | ctx | prefill t/s | decode t/s | VRAM peak |
|---|---|---:|---:|---:|
| BATCH=512 (shipped) | 8K | 121.4322 | 18.91 (no draft-KV) | 13835 |
| BATCH=1024 + window2048 | 32K | **145.5484** | 18.5184 | 15921 |
| BATCH=512 + window1024 | **96K** | 125.6702 | 18.5464 | 16011 |
| BATCH=256 + window1024 | 128K | ~100.8 | 18.5625 | 16059 |

Evidence: remote/part5/{fr2_dec_*,d32k_*,fr2_pf_*}.*, dec32k_queue.log,
frontier2_queue.log, telemetry.jsonl. Arms run by Claude on the existing harness;
Codex out of quota until 2026-09-19 07:39.

### F168 ongoing arm — f16k_ctx8k_f16_a

Allocated context 8192; b512/ub512; decode. Temperature **61→68 C**; sampled peak 68 C; VRAM peak **13847 MiB**. **18.5354 decode t/s**, 192 generated tokens in 10358.570 ms; 126/105 drafted/accepted. Output 9d2c737cc6fe; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/f16k_ctx8k_f16_a.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — f16k_ctx32k_ctrl_a

Allocated context 32768; b512/ub512; decode. Temperature **48→56 C**; sampled peak 56 C; VRAM peak **14339 MiB**. **18.5638 decode t/s**, 192 generated tokens in 10342.720 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/f16k_ctx32k_ctrl_a.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — f16k_ctx32k_f16_a

Allocated context 32768; b512/ub512; decode. Temperature **53→60 C**; sampled peak 60 C; VRAM peak **14819 MiB**. **18.5554 decode t/s**, 192 generated tokens in 10347.409 ms; 126/105 drafted/accepted. Output 9d2c737cc6fe; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/f16k_ctx32k_f16_a.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — f16k_ctx8k_f16_b

Allocated context 8192; b512/ub512; decode. Temperature **56→62 C**; sampled peak 63 C; VRAM peak **13847 MiB**. **18.5374 decode t/s**, 192 generated tokens in 10357.429 ms; 126/105 drafted/accepted. Output 9d2c737cc6fe; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/f16k_ctx8k_f16_b.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

### F168 ongoing arm — f16k_ctx8k_ctrl_b

Allocated context 8192; b512/ub512; decode. Temperature **59→66 C**; sampled peak 66 C; VRAM peak **13727 MiB**. **18.5978 decode t/s**, 192 generated tokens in 10323.825 ms; 119/104 drafted/accepted. Output c1453f54b802; cache_n=0; sampled SM 1189–1189 MHz, HBM 715–715 MHz. Raw evidence remote/part5/f16k_ctx8k_ctrl_b.* and telemetry.jsonl. This individual arm is recorded before its bracket is interpreted.

## F173 — REJECTED: f16 K cache gives NOTHING on this configuration, and does not settle the bandwidth question (2026-09-15)

**Lead (external):** qr0n/p100-llama-cpp documents that on GP100 an **f16 K cache beats
a quantized K**, the opposite of standard advice, because a quantized K sends attention's
K.Q dot through `__dp4a` which sm_60 lacks and emulates as four scalar XMADs (3.95 TOP/s),
while f16 uses GP100's packed fp16 unit (15.80 TOP/s). Their figures: +2.4% at depth 0,
+17.5% at 49K, **+24.1% on a real 75K request**. Their stated mechanism ALSO claims it as
a bandwidth-boundedness disproof: f16 K reads **52% more memory and wins anyway**.

**MEASURED HERE — null, and slightly negative.** Paired, alternating, pinned 1189 MHz,
shipped b512/ub512 + sign-add, `CTK=f16` vs the default turbo4/auto-asymmetric K:

| depth | control K | f16 K | delta | VRAM cost |
|---|---:|---:|---:|---:|
| 8192 | 18.5746 / 18.5978 (mean **18.5862**) | 18.5354 / 18.5374 (mean **18.5364**) | **-0.268%** | +120 MiB |
| 32768 | 18.5638 | 18.5554 | **-0.045%** | +480 MiB |

Each 8K pair closes to 0.011-0.012%. Note the arms are NOT hash-identical
(f16: 126/105 drafted/accepted, output 9d2c737c; control: 119/104, c1453f54) - expected,
since f16 K is HIGHER precision than the quantized cache, so numerics and sampling diverge.
Both produced 192 tokens.

**WHY IT DOES NOT REPRODUCE — and why this is NOT a bandwidth result.** The source's own
caveat: the optimization applies only to the vector kernel at `Q->ne[1] <= 2`, and under
speculative decoding (`Q->ne[1] > 2`) a different kernel converts to f16 REGARDLESS,
giving them only +5.1%. **The champion runs draft-mtp at n-max 2 = B=3**, so our K.Q dot
was never taking the emulated-dp4a path the optimization targets. There was nothing for
the format swap to fix, which is exactly what ~0% means.

**Consequence for F136/F153/F154/F156:** the "reads 52% more, wins anyway" disproof
DOES NOT TRANSFER to this configuration, because in our speculative regime the experiment
does not test the byte-vs-arithmetic trade at all. **The bandwidth question remains OPEN.**
It also remains UNMEASURABLE by direct means on this box: CUPTI returns
CUPTI_ERROR_NOT_SUPPORTED under R580, and `nvidia-smi -q -d SUPPORTED_CLOCKS` reports
exactly **ONE** memory clock (715 MHz) against many graphics clocks - so the complementary
"pin SM, vary HBM" sweep that would settle it is PHYSICALLY UNAVAILABLE on this card.
That is why F154 could only run the sweep in the other direction.

**VERDICT: do not adopt f16 K.** Zero gain, and it costs the scarce resource (VRAM is what
caps context: +480 MiB at 32K, and it would be impossible at 96K/128K where only
373/210 MiB remain). Launcher gained `CTK`/`CTV` knobs (default turbo4, behaviour
unchanged, backup `.bak-preCTK-20260915`); they stay unset.

Evidence: remote/part5/f16k_*.{request,response,server.log,command}.json, f16k_queue.log,
telemetry.jsonl. Source: github.com/qr0n/p100-llama-cpp docs/kv-cache-type.md.

---

## F174 — REJECTED for production: the width 2..8 PQ2 geometry lever is real only at width 4, where no shipping profile spends its time (2026-09-22)

**Context.** The user put verification-width geometry FIRST on the agenda ("We've never
touched the width yet? — that's much more relevant"). The r4 four-row win (+39%, F-series
earlier) gates on `c_ncols_dst == 1`; speculative verification runs wider. Codex built a
bitwise-exact candidate (`prismbin_width_exact`, summation order preserved) after 576 kernel
comparisons; the faster changed-warp variants FAILED full-model parity and were discarded.
Kernel-level claim: **+10-14% on the width-4 K=5120 projections**, ~8% on K=17408 at eight rows,
**nothing available at widths 2, 3, 5-8**.

Codex hit a HARD usage limit 13:04 PDT (reset Sep 26 08:42) with `run_profile.py` written but
never executed, so the end-to-end table below was run by me on its harness, unchanged.

**Paired results.** All arms `prismbin_width_exact` vs their recorded baselines, ctx 32768,
same two-request payload (req1 novel prose 192 tok, req2 copy-heavy 150 tok).
**Every content SHA identical to baseline** (`8ff4379d`, `6f5593fd`) in all six arms.

| profile | verify width | prose base -> cand | copy base -> cand | arm |
|---|---|---|---|---|
| no-spec | 1 | 24.63 -> **24.63** | 24.51 -> **24.54** | wc_none_k |
| MTP n-max 1 | 2 | 28.96 -> **29.10** | 31.39 -> **31.61** | wc_mtp_k |
| MTP n-max 2 | 3 | 29.08 -> **29.21** | — -> **34.81** | wc_mtp2_l |
| chain (ngram->MTP) | 2-5 | 28.26 -> **28.48** | 37.27 -> **37.78** | wc_chain_k |
| DFlash2, 2 drafts | 3 | 26.74 -> **26.04** | 33.23 -> **32.39** | wc_dflash2_l |
| DFlash2, 3 drafts | **4** | 22.62 -> **23.69** | 32.59 -> **34.46** | wc_dflash3_l |

Draft/accept counts are byte-identical to baseline in the DFlash arms (110/161, 98/102 at
2 drafts; 121/207, 110/114 at 3), so these are clean paired deltas, not different code paths.

**VERDICT: do not promote.** The lever pays **only at width 4 (+4.7% prose / +5.7% copy)** and
is null (<=+1.4%, inside scatter) at widths 1, 2 and 3 — which is where every shipping profile
actually runs. DFlash2 at 2 drafts REGRESSED 2.5% on both payloads. The one profile that does
use width 4 is still the slowest on prose: 23.69 does not beat no-speculation's 24.63. A
correct optimisation with no traffic on it.

**Scale-down from the kernel number is expected and should be remembered:** 10-14% measured on
isolated K=5120 projections became ~5% end-to-end, because those projections are a slice of the
verification pass, not the pass.

**F174b — D IS NOT ZERO for MTP.** `pair_baseline_mtp_trace_i` (LD_PRELOAD decode tracer,
two llama contexts distinguishable by ctx pointer) measured the draft head at **2.58 ms median
over 371 calls, two calls per main-model call => D ~= 5.2 ms**, ~9.5% of the width-2
verification pass. Prior analysis in this project assumed D ~= 0 for MTP and is wrong wherever
it did. Main-model width-2 median was **55.71 ms**, independently reproducing the standalone
sweep's T2 = 54.53 — the cost model cross-validates against live decode. Corrected tau for
MTP-1 is **1.76**, not the 1.58 that D=0 implies.

**T(B) for Bonsai PQ2_0 on the r4 build**, widths 1..8 (ms): 39.55, 54.53, 70.65, 98.31, 98.95,
113.88, 126.31, 139.84. Linear-fit marginal **14.18 ms ~= 36% of T1** (Ridge's was 44%), with
**width 4 sitting 12.65 ms above the fit** and T4 ~= T5 — the anomaly this work targeted.
DO NOT reuse Ridge's `T(B) = 81.0 + 35.7(B-1)`: per Codex those coefficients are Ridge-3.7bpw,
the ledger has different ones for q41nonffn, and 19.44 t/s used further changes plus confidence
filtering. Three configs, never one.

**F174c — THE ACTUAL FIND, incidental to the above: MTP at n-max 2.** 29.21 prose / 34.81 copy.
Best prose in the whole table, **+10% copy over MTP n-max 1** (31.61) at equal prose, and it
drops the chain's ngram stage entirely. Against the shipping chain (28.48/37.78) it trades 8%
copy for 2.6% prose and a simpler configuration. NOT YET A SHIPPING DECISION — measured on two
requests at ctx 32768 only; needs the broader suite and a populated-context arm before promotion.

**Caveat on the DFlash arms.** `prismbin_dflash_width_candidate` was assembled by me, not built:
`prismbin_dflash2` with its `libggml-cuda.so.0.21.0` replaced by the width_exact one (the
geometry change lives entirely in that lib; this is the same construction Codex used for
`prismbin_width_exact` over `prismbin_verify_r4`). Its other ggml libs are from a 14:38-14:48
build. Identical output SHAs and identical draft counts are the evidence the swap is sound.

Thermals: cycles 16-17, starts 36-44 C, peaks 46-53 C, zero aborts, zero Xid. Blower still dead.
Evidence: (not included), `wc_dflash{2,3}_l.*`,
`pair_baseline_mtp_trace_i.decode.jsonl`, `width_all_exact_j.*` + `exact_logits_j.b1..b8.bin`.

---

## F175 — PROJECT CLOSE-OUT: production default changed chain -> MTP-1; the chain's win was an artefact of a 63-token prompt (2026-09-22)

**THE DECIDING TABLE.** t/s against PROMPT LENGTH, r4 build, ctx 32768, q8_0 KV:

| prompt | no-spec | MTP n-max 1 | chain (was shipped) | MTP n-max 2 |
|---|---|---|---|---|
| 18-63 tok | 24.63 / 24.54 | 29.10 / 31.61 | 28.48 / **37.78** | 29.21 / 34.81 |
| 469 tok | **24.48** | 23.55 | **18.08** | — |
| 3348 tok | **23.34 / 23.13** | 22.83 / 21.55 | 20.81 / 20.91 | **17.24** |

**The chain's headline copy number exists ONLY at an 18-63 token prompt.** At 469 tokens it is
18.08 against no-speculation's 24.48 — a **26% loss to doing nothing**. The crossover sits BELOW
469 tokens, i.e. below any real session: the agent harness preamble alone is ~4200 tok re-sent per step. Every populated-context arm ever measured on this box
is won by no-spec or MTP-1; none by the chain.

**WHY, from the width histogram (F175b).** MTP-1 costs 3-4% against no-spec at populated context
and gains 18-29% at short, and never collapses. **EXPORTED: `run_bonsai_p100.sh` default
`SPEC=ngram-simple,draft-mtp` -> `SPEC=draft-mtp`** (backup `.bak-preSPEC-20260922`), which via the
existing case block also pins NMAX=1/PMIN=0 and selects the MTP gguf. Smoke-tested through the
production launcher, not the harness: 25.64 t/s, 120 drafted / 70 accepted, coherent output,
clean shutdown. Note lines 90-91 still append `--spec-ngram-simple-size-m/n` unconditionally;
inert when ngram-simple is not in SPEC, but it is noise. Use `SPEC=ngram-simple,draft-mtp` ONLY
for short-prompt copy-heavy batch work; `SPEC=none` for exclusively long-context work.

**F175b — CHAIN VERIFICATION-WIDTH HISTOGRAM** (arm `wc_chain_hist_m`, LD_PRELOAD decode tracer,
143 main-model decodes). This was Codex's outstanding correction — that the chain uses variable
widths — and it is now measured:

| width | count | share | median ms | % decode time |
|---|---|---|---|---|
| 1 | 1 | 0.7% | 44.19 | 0.4% |
| 2 | 108 | 75.5% | 55.63 | 54.2% |
| 4 | 4 | 2.8% | 91.38 | 2.9% |
| 5 | 2 | 1.4% | 102.76 | 1.6% |
| 6 | 26 | 18.2% | 114.05 | 23.8% |
| 14, 59 | 2 | — | ~1040, ~1119 | PREFILL, not verification |

Excluding the two prefill rows, **width >=4 is ~34% of real decode time**. The two stages are
cleanly separable: width 2 = the MTP stage at n-max 1; **width 6 = the ngram stage at m=5**.
Chain draft overhead: 247 draft decodes, 2.66 ms median, ~8.6% of decode time (consistent with
F174b's D ~= 5.2 ms/token for MTP).

**THIS EXPLAINS F174's NULL RESULT EXACTLY.** The geometry fix improves width 4 = **2.9% of
decode time**. Width 6, nearly a quarter of decode time, was one of the widths with NO
bitwise-exact improvement available. The lever landed on almost nothing. Had a width-6
improvement existed it would have been the one worth having; it does not.

**F175c — MTP n-max 2 REJECTED, and an earlier call in this session RETRACTED.** On two short
requests it looked like the best profile of the day (29.21/34.81, +10% copy over n-max 1) and was
reported as such. The broader suite killed it: **17.24 t/s at 3348 tok, the slowest arm ever
measured at that context**, AND it produced different output (`13aabbdf`) on a case where FOUR
independent arms agree on `d6a84924` (`suite_verify_a`, `context_stock_a`, `suite_mtp1_only_e`,
`suite_chain_n4_mtp1_e` — including two no-spec arms). Deterministic case, real divergence.
Caveat: n-max 2 at long context was only ever run on `prismbin_width_exact`, so "n-max 2 diverges"
is not fully separated from "n-max 2 + width_exact diverges". Either way it does not ship.
LESSON: do not promote a profile on the paired two-request payload alone; it has an 18-63 token
prompt and is not representative of session use.

**F175d — DFLASH2 WIDTH-EXACT REGRESSION CONFIRMED ON REPEAT.** wc_dflash2_l 26.04/32.39,
wc_dflash2_rep_m 25.71/31.91, baseline pair_dflash2_f 26.74/33.23. Draft counts byte-identical
across all three (110/161, 98/102), SHAs identical. ~-3% is real, not scatter (the repeat peaked
56 C vs 49 C so some of the further drop is thermal). Reinforces F174: do not promote
`prismbin_width_exact`. Production stays on `prismbin_verify_r4`.

**STATE AT CLOSE.** Production: `prismbin_verify_r4`, `SPEC=draft-mtp`, NMAX=1, Z640 launcher
CTX=262144 UB=512 CTK/CTV=q4_0. Codex rate-limited until 2026-09-26 08:42 with
`resume_longcontext.sh` armed for that time and (not included) rewritten to start at
STEP 2; if the project stays closed, that waiter should be killed. Blower still dead; all of
2026-09-22's 12 arms ran 36-56 C with zero aborts and zero Xid.

**STILL OPEN, deliberately not pursued:** the acceptance-vs-context crossover is bracketed
(below 469 tok) but not located; width 6 geometry unexplored; `--kv-mean-center` calibration
never done though production now ships q4_0 KV.

## F176 — Prefix-cache reuse on Bonsai 2 (hybrid GatedDeltaNet): checkpoints sit at PROMPT END − ubatch only (2026-09-23)

Arms `cr_cms8192_a`, `cr_cms2048_a`, `cr_harnessshape_cms8192_a` (burst cycle 19). All used production
settings: `prismbin_verify_r4`, draft-mtp, CTX 262144, q4_0 KV, ub512. Each arm sends four
`/completion` requests, as token-id lists, to one fresh server. Case files are
`cache_reuse_cases{,2}.json`. A = 6000 tokens of the agent harness READMEs, B = different text.

| arm | request | shape | re-read (prompt_n) | reused (cache_n) |
|---|---|---|---|---|
| cms 8192 | 1 | A[:6000] cold | 6000 | 0 |
| | 2 | + 300 appended | 304 | 5996 |
| | 3 | A[:4129] + B (diverge at 4129) | **6000** | 0 |
| | 4 | A[:1000] + B (diverge at 1000) | 6000 | 0 |
| cms 2048 | 1–4 | identical shapes | identical: 6000 / 304 / **6000** / 6000 | |
| the agent harness-shape, cms 8192 | 1 | A[:4129] cold (like a session's first request) | 4129 | 0 |
| | 2 | A[:6000] (append) | 1864 | 4136 |
| | 3 | A[:4129] + B (compaction shape) | **2383** | **3617** |
| | 4 | + 300 appended | 304 | 5996 |

**Mechanism:** the fork checkpoints the recurrent state **once per request, one ubatch before
the prompt ends** (3617 = 4129 − 512). A divergence can only roll back to the end of an EARLIER
request. It cannot roll back to an arbitrary position, and `--checkpoint-min-step` does not change
this (8192 and 2048 behave identically). `--cache-reuse` is a no-op on this architecture
(PrismML PROMPT-CACHE.md).

**For the agent harness:** an appended step always reuses (304 tokens re-read). After a compaction, the summary
replaces everything after the preamble, so the re-read goes back to the session's first request:
preamble + first user message − 512. That means summary + retained tail + ~500 tokens, not the
whole prompt. **Nothing survives a server restart:** checkpoints and cache-ram are both in-process,
so the first request after any guard kill is fully cold. That is the death loop from 2026-09-23.

**Thermal, this card, no blower:** cold prefill of 6000 tokens took 49 s and went **40 → 57 °C (~0.35 °C/s)**
at 118–165 W, faster than the 0.15 °C/s decode figure. From 40 °C a single cold pass has roughly
**8–9k tokens** before the 68 °C guard. The 13.1k the agent harness compaction threshold is therefore safe only
because most passes are warm.

**Harness bug found and fixed:** `context_control.py`'s prefill pause had never fired. This fork's
progress events carry `tokens:[0]`, which the harness read as decode on the first event. It is
fixed with (not included) (backup `.bak-20260923`). With the fix, peak was
**58 °C** across all pauses (344–418 s cooling each), against 66 °C with the bug in `cr_cms8192_a`.
Pausing (SIGSTOP at 58 °C, resume at 41 °C) lets a cold prefill of any length finish without a kill.
