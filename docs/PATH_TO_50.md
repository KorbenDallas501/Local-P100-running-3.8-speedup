# Path to 50 t/s — measured ledger

Target: **50 t/s on BOTH workloads**. This file carries MEASURED numbers and the arithmetic that
follows from them. Projections are labelled. Nothing enters as a champion until reproduced.

Decomposition. Throughput is `t/s = 1000 * tau / T(B)` with `T(B) = a + d*(B-1)`:
  **a** = per-round intercept (fixed cost: launches, syncs, sampling, one weight pass)
  **d** = marginal cost per verification column
  **tau** = 1 + accepted-per-round (speculation quality)
Only three terms exist. Every experiment must say which one it moved.

---

## Current best, MEASURED

| workload | config | t/s (43 C) | tau | a | d | round ms |
|---|---|--:|--:|--:|--:|--:|
| prose | B=4, p_min=0.6, our fork | **18.77** | 2.341 | — | — | 127.4 |
| prose | B=4, p_min=0 (control) | 17.51 | 2.743 | 90.9 | 22.1 | 156.6 |
| echo | B=6, p_min=0 | **28.58** | 5.756 | 90.9 | 22.1 | 201.4 |

Current T(B) fit (champion, MMVQ, two cold bracketed points): **T(B) = 90.9 + 22.1(B-1)**.
NOTE this supersedes the older `81.0 + 35.7(B-1)`, which predates the surgical requant (F102).

**Remaining multiplier to target: prose 50/18.77 = 2.66x. Echo 50/28.58 = 1.75x.**

---

## What each term can plausibly supply

### 1. Marginal cost `d` = 22.1 ms/column — MMQ (batch31, PENDING)
MMVQ re-decodes the quantized weight per output column; MMQ stages a dequantized tile in shared
memory once and reuses it across columns. Attacks `d` directly. Pascal MMQ configs ship in-tree
(`mmq-config-pascal-older.cuh`); the gate is policy (`n_experts > 0`), not capability (F100).
MEASURED elsewhere (L4 sm_89, same model/engine): d 9.30 -> 1.98 ms, a +9.7 ms, +16.2% e2e.
**Must measure our own T(B) curve, not just e2e t/s.** Do not assume the L4 ratio transfers.
Ceiling if d -> 0 entirely, a unchanged, tau unchanged: prose 1000*2.341/90.9 = **25.8 t/s**;
echo 1000*5.756/90.9 = **63.3 t/s**. So **echo can reach 50 on `d` alone; prose cannot.**

### 2. Intercept `a` = 90.9 ms — batch32, PENDING
  (a) FA converts the WHOLE q8_0 K and V cache to f16 every round at B>=3 (TILE kernel), zero at
      B<=2 (VEC). 17 attention layers. Candidate explanation for our measured n-max 1 > n-max 3.
      Sits in BOTH terms: the conversion is per-round (a) but only above B=2.
  (b) CUDA graphs are disabled below Volta by an undocumented arch cutoff. ~10^3 kernel launches
      per round at 3-8 us. Pure `a`.
Roofline says one weight pass is 17.1 ms at 732 GB/s, so **~74 ms of the 90.9 ms intercept is not
weight movement** — it is launches, syncs, sampling, and the FA conversion. That is the headroom.
Ceiling if a -> 40 ms with d unchanged: prose 1000*2.341/(40+66.3) = 22.0; echo 1000*5.756/(40+110.5) = 38.2.

### 3. Speculation quality `tau` — p_min (F104, MEASURED) + drafter
prose tau 2.341 at the p_min=0.6 optimum-so-far; 2.743 at p_min=0. echo tau 5.756.
**Prose is the tau-limited workload.** Even with a=40 and d=0, prose at tau 2.341 gives 58.5 t/s
— reachable in principle, but only if BOTH intercept and marginal collapse. At today's a=90.9
with d=0, prose needs **tau 4.55** to hit 50. Measured prose tau has never exceeded 3.36 (B=8,
master MTP), and DFlash2's dedicated drafter did not beat MTP on prose (F85/F98).

---

## Honest assessment of the 2.66x for prose

No single mechanism supplies it. The plausible stack, all PROJECTED:

    MMQ on d (22.1 -> ~8?)      prose 127.4 -> ~104 ms      18.77 -> ~22.5   (1.20x)
    graphs + FA on a (90.9 -> ~60?)  round -> ~74 ms        -> ~31.6         (1.40x)
    tau via wider B once d is cheap (2.341 -> ~3.3?)        -> ~44           (1.39x)
                                                            ------
                                                            ~2.35x  => ~44 t/s

That lands short of 50 and every factor is a guess. **Stated plainly: we do not currently have a
measured path to 50 t/s on prose.** We have a measured path to roughly the mid-20s (MMQ), a
plausible path to the low 30s (intercept work), and a speculative path to the low 40s that
requires wider speculation to become economical AFTER `d` falls — which is exactly the interaction
batch31's B=6/B=8 arms test.

**Echo is different and much closer.** 1.75x needed; MMQ alone could supply most of it if `d`
falls substantially, because echo's tau 5.756 multiplies against the saved columns five times over.

### What new capability would be required for prose 50
Either (i) `a` below ~55 ms AND `d` below ~10 ms AND tau >= 3.3 simultaneously, or (ii) a
fundamentally better prose drafter (tau >= 4.5), which the external evidence says does not exist
today for this model, or (iii) a structurally cheaper target — a smaller model or a sparser
architecture — which changes the problem rather than solving it.

---

## Log
- 2026-09-03 opened. prose 18.77 (p_min 0.6), echo 28.58. MMQ/FA/graphs pending.
