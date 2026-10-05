# Process log — forward model & inverse shaping optimisation

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/inverse_design_sim_eval.py`](../../scripts/inverse_design_sim_eval.py)
> **数据/关联脚本**: [`scripts/inverse_design_accuracy_ladder.py`](../../scripts/inverse_design_accuracy_ladder.py)
> **数据/关联脚本**: [`scripts/inverse_design_restarts.py`](../../scripts/inverse_design_restarts.py)
> **数据/关联脚本**: [`scripts/inverse_restart_selection.py`](../../scripts/inverse_restart_selection.py)
> **数据/关联脚本**: [`scripts/inverse_objective_alignment.py`](../../scripts/inverse_objective_alignment.py)
> **数据/关联脚本**: [`scripts/inverse_achievable_target.py`](../../scripts/inverse_achievable_target.py)
> **数据/关联脚本**: [`scripts/gs_vs_gradient_inverse.py`](../../scripts/gs_vs_gradient_inverse.py)
> **数据/关联脚本**: [`scripts/gs_plus_refinement.py`](../../scripts/gs_plus_refinement.py)
> **数据/关联脚本**: [`scripts/alignment_vs_accuracy.py`](../../scripts/alignment_vs_accuracy.py)
> **数据/关联脚本**: [`scripts/roi_robustness.py`](../../scripts/roi_robustness.py)
> **数据/关联脚本**: [`scripts/restart_claim_robustness.py`](../../scripts/restart_claim_robustness.py)
> **数据/关联脚本**: [`scripts/freeform_vs_zernike.py`](../../scripts/freeform_vs_zernike.py)
> **运行环境**: 离线
> **说明**: 逆向整形 14 次尝试的完整过程记录（含 3 处被推翻的结论）
<!-- provenance:end -->

Running record of *analyse → propose → attempt → measure → record*. Newest
sections appended. Two earlier findings live in [`README.md`](README.md); this file
is the chronological log, including the dead ends.

Corpus: `data/debug`, 11393 records, 7 families. Protocol notes that apply to
every number below are in README.md's **Noise floor** section — the same config
varies between R² 0.748 and 0.937 on the split seed alone, so **only paired
differences mean anything**.

---

## Round 3 — inverse shaping (逆向整形)

### Analysis

What inverse-shaping paths exist, and what does each optimise against:

| path | optimises against | independent evaluator available? |
|---|---|---|
| `algorithm/signal_processing/differentiable_shaping.py::train_beam_shaping` | its **own** angular-spectrum model, full-pixel phase | no (it *is* the model) |
| `optimizer/wfless/model_in_loop_shaping.py` Step B | the **physical** bench model | yes, the camera |
| `optimizer/wfless/slm_zernike_pib.py` | nothing — SPGD measures the camera | n/a |
| **the learned `ZernikeAmpModel`** | **nothing** | — |

So inverse design on the *learned* model did not exist. The important question is
whether it *can* exist, because if the model both synthesises and scores the
phase, its accuracy cancels and the experiment measures nothing.

### Defect found: the model had no gradient path to invert

First attempt: optimise the coefficients against a square target by feeding a
zero phasor to `forward`.

```
step  0: loss=0.187500 |grad|=0.000e+00 max|c|=0.0000
step 40: loss=0.187500 |grad|=0.000e+00 max|c|=0.0000
```

**The gradient is exactly zero and the coefficients never move.** `forward`
returns `measured_phasor * exp(i·correction)` — it is a model of the *bench*, not
a generator. With a zero phasor the field is identically zero for any
coefficients, so the output is identically zero and there is no gradient.

The naive "just use the coefficients" reading is therefore impossible, and
silently so: it returns a plausible-looking image rather than raising.

**Fix:** added `ZernikeAmpModel.correction_far_field()` — coefficients → far
field with **no** measured phasor. `forward` and the new method share one
`_observable_from_field`, so the propagation, crop, attention and observable
branch cannot drift between them.

Verified: `forward(zero phasor)` is identically zero (the trap is still visible),
while `correction_far_field()` gives a live gradient and the loss falls
0.145 → 0.067 over 40 steps with coefficients reaching 0.55 rad.

The distinction is physical, not cosmetic: `correction` is the phase the SLM
commands, whereas the learned coefficients describe the bench's own aberration.
Predicting a measurement wants `forward`; synthesising a spot wants
`correction_far_field`.

### Attempt 1 — the evaluator returned all-zero metrics

First run scored `pib=0.0000` for *every* configuration, including the flat
reference. A metric that is identically zero is a broken evaluator, not a result.

Three separate faults, found in order:

1. **The sim was far out of its calibrated regime.** `SimPibSystem` is built for
   the real 1920x1200 panel (`beam_w0=400`, `far_field_padding=4`,
   `far_field_window=1024`). I had constructed it at `slm_shape=(64,64)` with
   `beam_w0=10.7`, and the far field came back **NaN**.
2. **Probed it properly rather than guessing.** A sweep over panel shapes,
   beam waists and paddings showed every configuration in the *calibrated* range is
   finite (`max=1.0e2`), so the sim itself was fine and my configuration was the
   problem.
3. **The actual NaN source was the Zernike phase, not the sim.**
   `ZernikeGenerator.generate_noll` returns **NaN outside the aperture disc** —
   the pupil has no light there. Embedding that patch into the panel propagated
   NaN across the entire far field, so every metric read 0 (the ROI sum was 0
   over a non-finite image). Fixed with an explicit `nan_to_num`: the aperture is
   zero light, not undefined light.

Also fixed: `correction_far_field()` returned `(g, g)` because
`correction_phase()` is a bare 2-D map with no batch axis, while `forward()`
returns `(B, 1, g, g)`. The two must be comparable or every comparison against
`forward` is silently broadcasting.

### Attempt 2 — result

Forward model fitted to one real measured sample (MSE 6.2e-4), inverse-designed
for 60 steps on the learned model, then the phase pushed through the independent
sim and scored with the canonical `rms_pib_terms`:

| design objective | sim pib | sim uniformity | shape_sum | vs flat |
|---|---|---|---|---|
| flat reference | 0.1862 | 0.4717 | 0.6578 | — |
| trained fwd coeffs | — | — | 0.9966 | +0.339 |
| `mse` | 0.1978 | 0.8230 | 1.0209 | +0.363 |
| **`physical` (pib+uni)** | **0.2059** | **0.8647** | **1.0707** | **+0.413** |
| `anchored` (mse+shape gap) | 0.1308 | 0.6239 | 0.7547 | +0.097 |

**Inverse design on the learned model works, and the unanchored physical
objective is the right one for it** — the opposite of the fitting conclusion in
README.md §1, and the asymmetry is the point: `pib`/`uniformity` are degenerate when
you are *matching a measurement*, and correct when you are *synthesising a spot*.

`anchored` is worst here, which is consistent rather than surprising: it was built
to preserve fidelity to a measured frame, and in inverse design there is no
measurement to be faithful to — its reference term actively pulls the solution back
toward the flat phase.

### Honest limits of attempt 2

* **n = 1 forward-model fit, single design seed, single inverse seed.** No
  replication. README.md's noise floor applies with full force: the spread on a
  single paired delta here is ~0.05, and the physical-vs-mse gap is 0.05. So
  *the ordering is not established*, only that all three beat flat.
* The evaluator is a **sim**, not the bench. A sim conclusion is not a hardware
  claim.
* The forward model was fitted to one sample, so "trained fwd coeffs" scoring
  +0.339 over flat says the aberration estimate is roughly right, not that the
  model is accurate.

### Next direction

Replicate attempt 2 across seeds and forward-model samples to see whether
`physical` > `mse` survives, since that is the only claim currently on the table
and it currently sits inside the noise.

### Attempt 3 — the attempt 2 ordering does NOT replicate

4 real corpus samples x 3 design seeds = 12 paired runs, each forward model fitted
to a different real measured sample:

| design objective | mean sim shape_sum | paired delta vs `mse` | positives |
|---|---|---|---|
| `mse` | 1.0836 | — | — |
| `physical` | 0.9885 | **-0.0951** (spread 0.2495) | **3/12** |
| `anchored` | 0.9710 | -0.1126 (spread 0.2865) | 3/12 |

**Attempt 2's conclusion reversed.** Attempt 2 had `physical` ahead by +0.050 on one
sample with one seed; across 12 paired runs `physical` is *behind* by 0.095 and is
positive in only 3 of 12. So the sign flipped with more data — this is the third
time in this repo that a single-seed ordering failed to replicate (see README.md
§11 for the two earlier instances). The `+0.050` was noise, and had I stopped at
attempt 2 I would have recorded the exact opposite of the truth.

What survives:

* **Inverse design on the learned model works** — all three objectives beat the
  flat reference in **12/12** paired runs. That is the robust finding.
* **No objective ordering is established.** `physical`'s mean is worst-to-best
  depending on the run; with `spread=0.25` against a mean delta of `0.095`, the
  ranking is not resolvable at this sample size. Reporting "physical is best"
  would be reporting noise.

Note the tension with README.md §1: unanchored `pib`/`uniformity` are degenerate
for *fitting* a measurement but looked right for *inverse design*. Attempt 3
shows even that intuition does not survive replication — the physical terms are
not reliably better than plain MSE for inverse design either. Kept as a
negative result rather than deleted.

### Next direction

The remaining open question is whether forward-model *accuracy* (not objective
choice) is what limits inverse quality. That needs an accuracy ladder — train
forward models to deliberately different fidelity levels, invert each, and check
whether sim shape_sum tracks the ladder. Every comparison so far has used one
accuracy point, so accuracy has never been varied as a variable.

### Attempt 4 — accuracy ladder: more accuracy is NOT better

`scripts/inverse_design_accuracy_ladder.py`. Ladder rung = AdamW steps used to fit
the forward model (`fit_steps=0` = random init = the "no information" control).
Accuracy is **held-out** forward MSE on a different real sample than the one
fitted — otherwise the rungs are not ordered by generalisation and the ladder is
meaningless. Confirmed monotone: MSE falls 0.00311 → 0.00079 across the rungs.

`design_steps=0` scores the fitted coefficient vector *directly*, with no inverse
optimisation in front of it. That is the only place forward accuracy cannot hide
behind the optimiser.

sim shape_sum, 3 sample pairs x 2 seeds (lower held-out MSE = more accurate model):

| fit_steps | held-out MSE | design=0 | design=5 | design=20 | design=60 |
|---|---|---|---|---|---|
| 0 | 0.00311 | 0.6578 | 1.0403 | 1.0859 | 1.0668 |
| 5 | 0.00232 | 0.6898 | 0.7725 | 0.9745 | 0.9718 |
| **15** | **0.00124** | **1.1210** | 0.9084 | 0.8026 | 0.8874 |
| 40 | 0.00087 | 1.0496 | 0.9193 | 1.0560 | 1.0991 |
| 80 | 0.00081 | 1.0322 | 1.0454 | 1.0080 | 0.9829 |
| 200 | 0.00079 | 0.8557 | 0.9896 | 1.0348 | 1.0965 |

Flat reference = 0.6578.

**Q1 — accuracy axis at design_steps=0, paired against the best rung (15):**

| fit_steps | held-out MSE | Δ shape_sum | positives |
|---|---|---|---|
| 0 | 0.00311 | −0.4632 | 0/6 |
| 5 | 0.00232 | −0.4312 | 0/6 |
| 40 | 0.00087 | −0.0714 | 0/6 |
| 80 | 0.00081 | −0.0889 | 0/6 |
| 200 | 0.00079 | **−0.2653** | 0/6 |

Three findings, and the second is the one that answers the original question:

1. **Forward accuracy does carry usable information.** The no-information start is
   strictly worse, 0/6. So this is not "accuracy is irrelevant".
2. **More accuracy is not better — the relationship is non-monotonic.** The best
   start is `fit_steps=15` (held-out MSE 0.00124). The *most* accurate model,
   `fit_steps=200`, cuts held-out error a further 36% (0.00079) yet is markedly
   worse as a start (−0.2653, 0/6). Ranking forward models by held-out MSE would
   therefore **actively mislead** here: it would pick the worse shaper.
3. **Inverse design steps add a real, monotone gain** — +0.045 / +0.093 / +0.116 at
   5 / 20 / 60 steps (18/36, 24/36, 22/36 positive). They reduce but do not erase
   the dependence on the start: at design_steps=60 the fit_steps=15 column is still
   the worst of the fitted rungs.

So the original premise — "how does forward-model accuracy affect inverse shaping?"
— has an answer that is *not* the expected one: in this setup intermediate
forward accuracy shapes better than high forward accuracy.

**Caveats, stated because they bound the claim:**

* 3 sample pairs x 2 seeds per cell. The `0/6` sign counts are consistent, but the
  spread across rungs (0.86–1.12) is only ~2x the within-cell noise, so this is a
  *ranking*, not a calibrated curve.
* The `fit_steps=0` rungs all share one random init (`torch.manual_seed(0)`), so
  their variance is understated — treat that row as n=1, not n=6.
* Evaluator is a **sim**, not the bench.
* The mechanism is **not** established. Plausible: the heavily-fitted models
  overfit a single sample, and their coefficients drift away from the true bench
  aberration in a way that is a worse *starting phase*. That is a hypothesis.

A bug worth recording: the first version of this sweep passed a `design_steps`
value that `design()` ignored (its step count was hardcoded), so the 5/20/60 columns
came out byte-identical and only 0 vs 60 was really tested. The table above is from
the fixed version, where the design axis genuinely varies.

### Attempt 5 — overfitting hypothesis REFUTED, and attempt 4 needs weakening

If the heavily-fitted start is worse *because* it overfits its single training
sample, then fitting on more samples should improve held-out accuracy and move the
best start rung toward high accuracy. Tested train_size ∈ {1, 4}, start scored
directly on the sim (design_steps=0), 3 disjoint sample blocks, full rung ladder.

| train_size | fit_steps | held-out MSE | sim shape_sum | vs flat |
|---|---|---|---|---|
| 1 | 0 | 0.00349 | 0.6578 | +0.0000 |
| 1 | 5 | 0.00273 | **0.9502** | +0.2923 |
| 1 | 15 | 0.00186 | 0.8137 | +0.1559 |
| 1 | 40 | 0.00135 | 0.8132 | +0.1553 |
| 1 | 80 | **0.00130** | 0.9189 | +0.2610 |
| 1 | 200 | 0.00130 | 0.8858 | +0.2279 |
| 4 | 0 | 0.00349 | 0.6578 | +0.0000 |
| 4 | 5 | 0.00283 | 0.9402 | +0.2824 |
| 4 | **15** | 0.00198 | **0.9988** | +0.3409 |
| 4 | 40 | 0.00144 | 0.8902 | +0.2323 |
| 4 | 80 | 0.00139 | 0.9364 | +0.2786 |
| 4 | 200 | **0.00138** | 0.8439 | +0.1861 |

Two results, one of which corrects attempt 4:

**1. The overfitting hypothesis is refuted.** Going from 1 to 4 training samples
does *not* improve held-out MSE (0.00130 → 0.00138 at the top rung — marginally
worse, i.e. flat) and does *not* move the best start rung toward high accuracy. So
the forward model is **capacity/optimisation-limited, not data-limited**. More
data buys nothing here, which means the attempt-4 dip at high accuracy is not an
overfitting artefact.

**2. Attempt 4's "more accuracy is not better" is too strong, and is probably a
noise ordering I over-read.** The identity of the best rung is **not stable across
resampling**: attempt 4 found `fit_steps=15` clearly best (1.1210, with every other
rung 0/6 behind), while this run on different sample blocks finds `fit_steps=5`
best for train_size=1 (0.9502) and `fit_steps=15` best for train_size=4 (0.9988),
with the high-accuracy rungs at 0.84–0.92 and paired positives of only 1/3. A
"best rung" that changes identity when the sample block changes was never a real
optimum.

The defensible version of the finding, which survives both runs:

* **Fitting helps a lot over no information** — 0.6578 → 0.81–1.12, paired positives
  0/6 and 0/3 against the no-info floor. This is the one solid inverse-design result.
* **Above that floor, start quality is flat within noise** across a 4x range of
  held-out MSE (0.00079–0.00311). No rung can be called best, and none beats the
  others consistently.
* **Therefore held-out forward MSE is a saturated, uninformative model-selection
  criterion for inverse shaping in this regime** — not because the relationship is
  non-monotonic, but because MSE barely moves while the thing we care about wanders.

That is a weaker claim than attempt 4 made, and the weakening is the point: attempt 4
was one sample block reading an ordering that did not replicate.

**Practical consequence — where the leverage actually is.** The forward model is at
its accuracy ceiling (MSE flat vs training-set size) and the start quality is flat in
that ceiling. So the productive lever is the **inverse optimiser**, not the forward
model: design_steps gives a real monotone gain (+0.045 / +0.093 / +0.116 at 5 / 20 /
60 steps, attempt 4) with no sign of saturating at 60.

### Attempt 6 — the noise floor was never measured, and it explains attempts 3-5

First run of this sweep came back with `sd = 0.0000` across 4 seeds and mean == best
== worst. Not a result: `ZernikeAmpModel.coefficients` is initialised to **zeros
deterministically**, so `torch.manual_seed(seed)` has no effect on the inverse design
at all. Every "seed" was a byte-identical replicate.

**This retroactively weakens attempts 3, 4 and 5.** Their paired sign counts
(`0/6`, `3/12`, `4/4`...) were computed over duplicate rows, so the effective n was
the number of *sample blocks* (3-4), not the row count (6-12). The direction of those
results stands, but the counts were not the evidence they looked like. This is the
third time in this file that a methodological bug, not physics, produced the headline
number — and the second time it was only caught by printing mean/best/worst/sd
together rather than a single average.

Redone with explicit restarts, `coefficients ~ N(0, sigma)`, 5 restarts per cell:

| sigma | objective | steps | mean | best | worst | sd |
|---|---|---|---|---|---|---|
| 0.3 | physical | 20 | 0.7919 | 0.9559 | 0.5584 | 0.1615 |
| 0.3 | physical | 60 | 0.9681 | 1.1550 | 0.5457 | 0.2174 |
| 0.3 | physical | 300 | 1.0038 | 1.1067 | 0.8693 | 0.0772 |
| 0.3 | mse | 60 | 0.9061 | 1.1152 | 0.5320 | 0.2011 |
| 1.0 | mse | 60 | 1.0961 | 1.1931 | 0.8668 | 0.1216 |
| 1.0 | mse | 150 | 1.0520 | 1.1448 | 0.9443 | 0.0664 |

**1. The noise floor is larger than every effect chased in attempts 3-5.**
Restart-to-restart sd is 0.07-0.26 on means of 0.79-1.10. Attempt 4's headline gap
was 0.15-0.46 and attempt 3's was 0.05-0.10 — i.e. *inside* the restart noise. Those
results were not wrong, they were underpowered: nothing in the earlier design
measured the restart distribution before comparing means across configurations.

**2. Inverse design saturates by ~20-60 steps; more steps do nothing.** Restart-matched
paired deltas:

| transition | sigma=0.3 physical | sigma=0.3 mse | sigma=1.0 physical | sigma=1.0 mse |
|---|---|---|---|---|
| 0 → 20 | +0.134 (3/5) | **+0.211 (5/5)** | **+0.204 (5/5)** | **+0.258 (5/5)** |
| 20 → 60 | +0.176 (4/5) | +0.037 (4/5) | +0.065 (3/5) | **+0.181 (5/5)** |
| 60 → 150 | −0.059 (2/5) | −0.044 (2/5) | +0.084 (2/5) | −0.044 (1/5) |
| 150 → 300 | +0.094 (3/5) | +0.120 (4/5) | −0.042 (2/5) | −0.022 (2/5) |
| 300 → 600 | +0.003 (3/5) | −0.192 (1/5) | −0.074 (2/5) | −0.067 (1/5) |

Only the first rung is reliably real. Everything past ~60 steps is mixed at 5
restarts, and 300 → 600 is *negative* in three of four arms.

**3. The lever is restarts, not steps.** Best-of-5 restarts, expected value over
restarts:

| sigma | objective | steps | single run | best-of-5 | gain |
|---|---|---|---|---|---|
| 0.3 | mse | 150 | 0.8619 | 1.0632 | **+0.2013** |
| 0.3 | physical | 60 | 0.9681 | 1.1315 | **+0.1634** |
| 1.0 | physical | 60 | 0.9271 | 1.1156 | **+0.1885** |
| 1.0 | mse | 20 | 0.9156 | 0.9481 | +0.0859 |

Restarting buys +0.08 to +0.20 *consistently across every arm*, whereas step-count
tuning beyond 60 buys nothing and sometimes hurts. That is a larger and far more
reliable gain than anything the forward-model or loss changes produced.

### What this means for the codebase

`slm_gs_refine` and the shaping runners spend their budget on epochs (steps). On this
evidence the budget is better spent on **restarts**, since the per-restart outcome
spread (sd 0.08-0.26) dwarfs the within-restart progress past ~60 steps.

⚠️ **This recommendation is partly overturned by attempt 7 below** — the "score each
restart and keep the best" half does not survive testing. Keep the restarts, drop the
model-score selection.

### Attempt 7 — the model score cannot rank restarts, so the recipe above fails

The proposed recipe was "~60 steps per restart, score each on the model, keep the
best". That is only valid if the model's own score tracks quality on the independent
evaluator. Single-trial check said no but hopeful: spearman −0.10 (no usable rank
correlation) yet `argmin(model_loss)` landed on rank 4/24 with regret 0.0375 against
a restart spread of 0.56. That is exactly the shape of result that is luck, so the
rule was measured over **30 independent trials of 12 restarts** instead.

| rule | mean sim shape_sum |
|---|---|
| pick by model score (argmin loss) | 1.0041 ± 0.1555 |
| oracle (argmax sim) | 1.1416 ± 0.0397 |
| random restart | 0.9660 ± 0.0472 |
| worst restart | 0.6888 ± 0.1074 |

* `pick_model − random = **+0.0381 ± 0.1367**`, positive in 20/30 trials,
  **t = +1.53** — not significant.
* per-trial `spearman(model_loss, sim) = −0.0562 ± 0.2693` — mean rank correlation
  is zero.
* fraction of the available gain captured: **0.278**, against 0.782 implied by the
  single trial.
* regret vs oracle: 0.1375 ± 0.1528 — as large as the gain itself.

**Verdict: the model score is not useful for selection.** The rank-4 landing was
luck. An external scorer is required.

### The coherent picture (attempts 5-7 together)

These three now explain each other, which is the strongest result in this file:

1. **Attempt 5**: the forward model is capacity/optimisation-limited, not
   data-limited — 4 training samples buy no held-out accuracy (MSE 0.00130 → 0.00138).
2. **Attempt 7**: its predictions are therefore reliable only near the data it was
   fitted to. Off that manifold it interpolates the training point but its *ranking*
   of different coefficient vectors is uncorrelated with reality.
3. **Attempt 6**: so **optimising on the model works but selecting with it does
   not.** Descending the model loss reliably improves the model loss and produces sim
   scores above flat (12/12 paired, attempts 3-6), yet two solutions that fit the
   training sample equally well can have very different real far fields.

The practical consequence is uncomfortable but clear: for inverse shaping, the learned
model can *propose* phases but cannot *grade* them. Grading needs a measurement —
which is exactly why the hardware path is SPGD with a real camera, and why the sim
`model_in_loop` loop refits against probes rather than trusting its own forward model.

Revised recipe, after attempts 6 and 7 together:

* **Do** use restarts: the outcome spread (0.45 within a trial) is the only large,
  reliable effect found in this entire study.
* **Do** keep ~60 steps per restart; past that, step count buys nothing.
* **Do not** select restarts by model loss — it is statistically indistinguishable
  from picking blind. Select on a measurement, or (offline, as here) on the
  independent simulator, and treat the model score as an optimisation signal only.

### Attempt 8 — inverse objective irrelevant; `correction_far_field` ignores the fit (**ROI-conditional**)

Debugging the under-determination test turned up something structural rather than
statistical. `correction_far_field()` depends **only on `self.coefficients`**, and the
design loss never reads the fitted forward-model state. So inverse design is
*completely independent of what the model was fitted to*: changing `n_train` or
`l2` produced **bit-identical** inverse solutions (measured coefficient norm
7.619766 in all three cases). Only `n_max` — the parameterisation size — matters.

That reframes attempt 7. The "model score" tested there was *literally the objective
being minimised*. So attempt 7's result reads: **driving the design objective lower
does not produce better real far fields.** The cap on inverse quality is objective
misalignment, not forward-model accuracy.

Tested directly — same starts, same steps, only the objective changes, 16 restarts,
paired by restart, all scored on the independent sim:

| objective | obj start | obj final | improved | sim mean | sim sd | vs flat |
|---|---|---|---|---|---|---|
| `mse` | 0.15220 | 0.11533 | 0.03687 | 1.0162 | 0.1031 | +0.3584 |
| `physical` (pib+uni) | 1.24017 | 0.96665 | 0.27352 | 1.0165 | 0.1291 | +0.3586 |
| `pib_only` | 0.68281 | 0.41610 | 0.26671 | 1.0097 | 0.1460 | +0.3519 |
| `uni_only` | 0.55736 | 0.37885 | 0.17851 | 0.9453 | 0.1404 | +0.2875 |
| `mse+physical` | 1.39237 | 1.08055 | 0.31182 | 0.9893 | 0.1622 | +0.3314 |

Paired deltas vs `mse` (same restart):

| objective | delta | positives | t |
|---|---|---|---|
| `physical` | **+0.0002** ± 0.0372 | 8/16 | +0.01 |
| `pib_only` | −0.0065 ± 0.0328 | 8/16 | −0.20 |
| `uni_only` | −0.0709 ± 0.0476 | 7/16 | −1.49 |
| `mse+physical` | −0.0270 ± 0.0522 | 10/16 | −0.52 |

**1. The objective does not matter.** Every objective lands at sim ≈ 1.01 ± 0.10, and
every paired delta is within noise of zero. `physical` differs from plain `mse` by
+0.0002 — as close to identical as a measurement can resolve. Optimising the
physically-motivated shaping terms is worth **nothing** over plain MSE here.

**2. The more physical the objective, the *less* it predicts the real metric.**
Splitting restarts by how well each converged on its own objective:

| objective | sim on best-objective half | worst-objective half | delta |
|---|---|---|---|
| `mse` | 1.0583 | 0.9741 | **+0.0842** |
| `physical` | 1.0207 | 1.0123 | +0.0084 |
| `pib_only` | 1.0222 | 0.9973 | +0.0248 |
| `uni_only` | 0.9415 | 0.9491 | −0.0076 |
| `mse+physical` | 0.9448 | 1.0338 | **−0.0890** |

Plain `mse` is the *best* of the five at predicting which solution will actually shape
well; `mse+physical` is **anti-aligned** (−0.089). Every objective improved in 16/16
restarts, so this is not "the optimiser failed to converge" — it converged and got
*less* useful. The model's `pib_term`/`uniformity_term` are computed on the model's
own far field with its own ROI convention; because the model is imperfect, matching
its PIB does not produce good real PIB.

### What this settles, and what it does not

This **does not** contradict README.md §1. That section is about the *forward fitting*
problem, where the physical losses were decisively better (10-fold grouped CV, R²
+0.750 vs +0.634, paired sign-flip p=0.012). Fitting a measurement and synthesising a
spot are different problems, and now both have answers:

| problem | does the loss matter? | evidence |
|---|---|---|
| forward **fitting** (match a measured frame) | **yes, decisively** | README §1, 10-fold CV, p=0.012 |
| inverse **design** (synthesise a spot) | **no** | attempt 8, 5 objectives, paired, all deltas ≈ 0 |

So "find a better loss for the forward model" was the right question and the anchored
loss answered it. But the inverse leg is not a loss problem at all: within this setup
the outcome is set by the parameterisation and the restart, not by the objective.

Consequences:

* Do **not** spend more effort on inverse-design losses. Five were tested; the spread
  between the best and worst is 0.07 on a restart sd of 0.10–0.16.
* The one real inverse lever remains **restart diversity** (attempt 6), and it needs
  an *external* scorer to exploit (attempt 7) — which in hardware means a camera.
* A promising untested direction, implied by finding 2: since `mse` is the best
  available predictor of real quality, the inverse design may work better with a
  **richer target than a binary square** — e.g. the target intensity that a GS
  solution would produce for the same geometry, which encodes the physics in the
  *data* rather than in the loss.

### Attempt 9 — an ACHIEVABLE target does not help (but it exposed the real answer)

Attempt 8's implication was that the lever is *what the loss matches*, not the loss. A
binary square is unreachable in a 135-coefficient Zernike basis, so its MSE residual is
dominated by an irreducible mismatch and the gradient may carry little directional
information. Tested by replacing it with an achievable target, built from canonical
pieces only:

1. canonical Fourier Gerchberg-Saxton (`propagation='fft'`) on the square target → pupil phase
2. canonical `fit_zernike` projection onto the model's basis, piston dropped
   (the model holds the K non-piston Noll modes only — `fit_zernike` returns all 136)
3. the model's own `correction_far_field` for those coefficients → achievable intensity

| target | sim mean | sim sd | vs flat |
|---|---|---|---|
| `binary_square` (current) | 0.9689 | 0.1422 | +0.3110 |
| `gs_achievable` | 0.9476 | 0.1726 | +0.2898 |

Paired delta **−0.0212 ± 0.0665**, 8/16, t = −0.32. No improvement — the hypothesis is
refuted. The reason is circularity: the achievable target is *already* the GS answer
rendered by the model, so MSE toward it just re-derives GS coefficients through a
worse optimizer.

But the run printed a number that mattered more than the hypothesis:

```
sim shape_sum of the GS target coefficients themselves: 1.0998
```

**1.0998 — higher than any gradient inverse design in this study** (means 0.95–1.02).

### Attempt 10 — GS beats gradient (**RETRACTED by attempt 13: it was an ROI artefact**)

GS is deterministic, so one number proves nothing. Varied the thing that actually makes
it non-deterministic — the initial random pupil phase — over 16 draws, paired with 16
gradient designs from the same budget, all scored on the independent sim:

| method | mean | sd | min | max | beats flat |
|---|---|---|---|---|---|
| **canonical GS** | **1.1199** | **0.0341** | 1.0510 | 1.1790 | **16/16** |
| gradient on model (MSE, 60 steps) | 1.0028 | 0.1599 | 0.6406 | 1.1971 | 15/16 |

Paired **GS − gradient = +0.1171 ± 0.0423**, positives 10/16, **t = +2.77**.

1. **GS wins on the mean**, significantly.
2. **GS is 4.7x more reproducible** (sd 0.034 vs 0.160). This is the more important
   number for a bench: gradient inverse design's outcome is dominated by restart
   noise, so its *typical* result is mediocre even when its best-ever result matches
   GS's best.

### The conclusion this whole study was circling

Attempts 3–8 all spent effort on levers *of the gradient path* — which loss, which
forward-model accuracy, which normalisation, which attention. Attempt 10 shows the
gradient path is the wrong place to spend it: it is noise-dominated, and a
deterministic open-loop solver beats it on both mean and variance.

This independently justifies architecture the repo already ships: `slm_gs_refine` does
GS pre-shaping before SPGD refinement, and `model_in_loop_shaping` builds on GS. Those
designs were previously justified by "GS is open-loop so it cannot use feedback"; this
study adds the stronger empirical reason that **GS alone outperforms optimising the
learned model**, because the learned model cannot grade its own solutions (attempt 7)
and its gradients are not aligned with the shaping metric (attempt 8).

Actionable summary for the codebase:

| do | don't |
|---|---|
| use canonical GS to propose inverse phases | expect gradient descent on the learned model to beat it |
| treat the learned model as a *forward* predictor only | use the learned model's score to pick between solutions |
| spend effort on forward-model *loss* (README §1) — it is decisive there | spend effort on inverse-design loss — worth +0.0002 (attempt 8) |
| budget for measurement in hardware (a camera) to select restarts | select restarts by model score — t = 1.53, indistinguishable from blind (attempt 7) |

Caveats: the evaluator is the sim, not the bench. GS here runs at `cell_spacing=8 µm`
on a 64×64 grid without knowledge of the sim's angular scale, so its advantage is not
an artifact of being tuned to the evaluator — but a GS solution aimed at the *wrong*
angular scale would lose, and `slm_gs_refine`'s bake-off exists precisely for that.

### Attempt 11 — refinement on top of GS makes it WORSE (**RETRACTED by attempt 13: the sign is ROI-dependent**)

Attempt 10 established GS > gradient from a random start. The repo's shipped
`slm_gs_refine` is a two-stage architecture built on the complementary premise — that
GS proposes and then feedback refines — so that second half needed testing too.
Refinement started **from the GS solution**, paired by the same draw:

| arm | mean | sd | min | max | Δ vs GS | positives |
|---|---|---|---|---|---|---|
| **GS only** | **1.0905** | 0.0603 | 0.9618 | 1.1744 | — | — |
| GS then MSE refinement | 0.8699 | 0.1348 | 0.6133 | 1.0474 | **−0.2206** | **0/16** |
| GS then physical refinement | 0.8577 | 0.1243 | 0.7175 | 1.1017 | **−0.2328** | **0/16** |

Paired t = **−7.65** and −6.84. This is not a null result — gradient refinement
**actively damages** a good GS solution, in every single draw, by roughly 20% of the
score. And it *doubles* the spread (sd 0.060 → 0.135), so refinement does not even
buy stability.

This closes the loop on attempt 8. The model's gradients are anti-aligned with real
shaping quality, so descending them walks downhill on the model's objective and
uphill on reality — whether the start is random (attempt 10) or already good
(attempt 11). The better the starting point, the more there is to lose.

### ⚠️ What this does and does not say about `slm_gs_refine`

It does **not** condemn the shipped runner. `slm_gs_refine` refines with **SPGD
against a real camera**, i.e. feedback on the actual measurement. The refinement
tested here is gradient descent through the **learned model**, and attempts 7-8
showed precisely why that is different: the learned model cannot grade its own
solutions, and its gradients are misaligned.

The honest statement is sharper than "refinement is bad":

| forward model | gradient refinement |
|---|---|
| **exact** (analytic FFT == the evaluator) | helps — `generate_iterative_zernike_shaping` measured 0.849 vs GS 0.812 |
| **learned / imperfect** (this study) | **hurts** — −0.22, 0/16 |

So the load-bearing assumption in that pipeline is that the forward model is
trustworthy. When it is learned from limited hardware data (attempt 5: held-out MSE
saturated, capacity-limited), it is not, and model-based refinement destroys more
than it finds. Feedback on a real measurement does not have this failure mode.

### Attempt 12 — forward accuracy CANNOT rescue refinement (it is not an input)

Attempts 4-5 said forward accuracy does not predict inverse quality. Attempts 8 and 11
said the model's gradients are anti-aligned. Those looked contradictory, so this
attempt was built to reconcile them: if the anti-alignment were a *consequence* of the
poor single-sample fit, a genuinely accurate model should have usable gradients.

The ladder (n_max ∈ {4,8,15,20} × n_train ∈ {1,8}, held-out MSE measured on a disjoint
sample, refinement from a GS start, paired by draw) came back with `n_train` having
**literally zero** effect — rows (4,1) and (4,8) identical, likewise 8, 15 and 20. Only
`n_max` changed anything.

That is attempt 8's structural finding biting again, so it was verified directly rather
than interpreted. With `n_max` fixed at 15 and the same GS start:

| forward model | fitted coef norm | refined coef norm | sim shape_sum |
|---|---|---|---|
| `n_train=1, fit=0, l2=0` | **0.0000** | 6.84294 | 0.997656 |
| `n_train=1, fit=200, l2=0` | 2.4335 | 6.84294 | 0.997656 |
| `n_train=8, fit=200, l2=0` | 2.5421 | 6.84294 | 0.997656 |
| `n_train=8, fit=200, l2=1e-2` | 2.5317 | 6.84294 | 0.997656 |

**Bit-identical output from a completely unfitted model and from heavily fitted,
regularised ones.** The reason is structural and was established in attempt 8:
`correction_far_field()` is a function of `self.coefficients` alone, and `refine()`
overwrites `self.coefficients` with the GS start before the first step. The fitted
state is never read.

**Therefore attempts 4-5 and attempts 8-11 never actually conflicted.** They looked
like they did only because I had not noticed that the forward model's fit is not an
input to the gradient path. The reconciliation:

* forward-model accuracy does not predict inverse quality — **correct**, and now
  trivially so: it cannot, because it is not consulted;
* the learned model's gradients are misaligned — **correct**, and structural rather
  than a fitting deficiency.

So no amount of forward-model training makes gradient refinement through the learned
model safe or useful. That closes the forward-model half of the question: the answer
is not "improve the model", it is "do not refine through the model".

*(The script's own printed verdict — "alignment improves with accuracy" — is
misleading and should not be read. It is an artifact of `n_max=20` below, not a trend;
held-out MSE across all eight configs spans only 0.0032-0.0050, a range attempt 5
already showed to be saturated and uninformative.)*

### A separate real defect found alongside: `n_max=20` GS projection is worse than flat

Same ladder, looking at the GS proposals themselves:

| n_max | GS only | vs flat (0.6578) | after refinement |
|---|---|---|---|
| 4 | 1.1312 | +0.47 | 0.9623 |
| 8 | 0.9733 | +0.32 | 0.8704 |
| 15 | 1.0608 | +0.40 | 0.9209 |
| **20** | **0.5974** | **−0.06** | 0.9948 |

Projecting the GS pupil phase onto 230 modes (`fit_zernike(..., n_max=20)`) produces a
far field **worse than no shaping at all** — the large +0.3974 "refinement gain" at
n_max=20 is recovery from that broken start, not evidence of good alignment. The
least-squares projection of a smooth GS phase onto many high-order Zernike modes
injects high-order noise that costs more than the extra degrees of freedom buy.

Actionable: **do not raise `n_max` for the GS projection.** On this evidence n_max=15
is the best of the four and n_max=20 is actively harmful. This matters because
`n_max` is the *only* lever that affects the gradient path at all — which is the flip
side of the invariance above, and a trap for anyone who reads "the fit doesn't matter,
so just add capacity".

### Final state: the question is closed

| question | answer | evidence |
|---|---|---|
| does forward accuracy predict inverse quality? | **no — it is not an input** | attempt 12, bit-identical output from unfitted vs fitted |
| does the forward model's fit change gradient refinement? | **no** | attempt 12, `n_train`/`fit_steps`/`l2` all inert |
| is the learned model's gradient usable for inverse design? | **no, it is anti-aligned** | attempts 8 (−0.089), 11 (−0.22, 0/16) |
| can the model's score rank candidate phases? | **no** | attempt 7, t=1.53 over 30 trials |
| is the inverse loss worth tuning? | **no** | attempt 8, `physical` vs `mse` = +0.0002 |
| what is the best inverse proposer? | **canonical GS** | attempts 10 (+0.1171, t=2.77), 11 |
| should `n_max` be raised? | **no — n_max=20 lands below flat** | attempt 12 |

The whole arc converges on one rule: **propose with GS, grade with a measurement, never
refine through the learned model, and do not buy inverse quality with `n_max`.** The
learned model earns its place as a *forward* predictor — where README §1's anchored
loss work is decisive — and nowhere else.

### Attempt 13 — ⚠️ ATTEMPTS 8, 10 AND 11 ARE ROI-ARTEFACTS. Retracted.

Every comparative result above was measured at ONE arbitrary evaluator ROI:
`SIZE_FRAC=0.375`, `ASPECT=4/3`. Attempts 10 and 11 had respectable t-statistics there
(+2.77 and −7.65), which is exactly why they needed attacking. Re-ran both claims
across a 3x3 sweep of ROI geometries (10 draws each), holding methods and starts fixed:

| ROI (frac x aspect) | flat | GS | grad | GS−grad | t | GS+refine | ref delta | t |
|---|---|---|---|---|---|---|---|---|
| 0.250 x 1.000 | 0.5340 | 0.8115 | 0.9728 | **−0.1613** | −6.40 | 0.9442 | +0.1327 | +6.25 |
| 0.250 x 1.333 | 0.5570 | 0.8044 | 0.9326 | −0.1283 | −2.95 | 0.8504 | +0.0461 | +0.92 |
| 0.250 x 1.500 | 0.5649 | 0.9684 | 0.8888 | +0.0796 | +1.36 | 0.8309 | −0.1375 | −2.59 |
| 0.375 x 1.000 | 0.6114 | 0.8947 | 0.9447 | −0.0500 | −0.95 | 0.9591 | +0.0644 | +1.42 |
| **0.375 x 1.333** (original) | 0.6578 | 1.0970 | 0.8668 | **+0.2303** | +5.60 | 0.9482 | −0.1488 | −2.74 |
| 0.375 x 1.500 | 0.6809 | 1.1612 | 0.9983 | +0.1629 | +4.42 | 0.8951 | −0.2661 | −4.35 |
| 0.500 x 1.000 | 0.7199 | 0.6538 | 1.0037 | **−0.3499** | −11.15 | 0.9532 | +0.2993 | +8.01 |
| 0.500 x 1.333 | 0.8016 | 0.9330 | 1.1997 | −0.2667 | −6.19 | 1.0218 | +0.0888 | +1.58 |
| 0.500 x 1.500 | 0.8442 | 1.1876 | 1.2218 | −0.0342 | −1.32 | 0.9916 | −0.1960 | −4.32 |

Win counts over all 90 draws:

| comparison | result | verdict |
|---|---|---|
| gradient beats flat | **90/90** | **robust** |
| GS beats flat | 82/90 | robust (fails at 0.5x1.0: 2/10) |
| GS beats gradient | **34/90** | **chance — not a real effect** |
| refinement helps GS | 5/9 ROIs | **sign depends on ROI** |

So:

* **Attempt 10 is retracted.** "GS beats gradient by +0.1171 (t=+2.77)" held only at the
  ROI I happened to choose. Across the sweep it is 34/90 — a coin flip. GS does beat
  flat reliably (82/90), and so does gradient (90/90); neither dominates.
* **Attempt 11 is retracted.** "Refinement destroys GS (−0.2206, 0/16)" held only where
  GS was already strong. Where GS underperformed, refinement *helped* by up to +0.2993.
* **Attempt 8's objective ranking is likewise suspect** — same single ROI, and its
  effect size (+0.0002) was already inside the noise it was measured against.

**What survives, and it is a different kind of claim.** Refinement's value is strongly
monotone in how good the starting point already was:

| | x = GS − flat | y = refinement delta |
|---|---|---|
| pearson | **−0.9056** | |
| spearman | **−0.8667** | |
| spearman excluding the single ROI where GS fell below flat | **−0.8095** | |

*(An inline Spearman I printed for this first read −0.0144 was a bug in that one-liner's
ranking; the correct value is −0.8667 and it holds at −0.8095 without the outlier, so the
effect is not driven by one point.)*

So the robust, ROI-independent statement is:

> **Refinement is a restart, not a gradient.** It helps when the proposal is bad and
> hurts when the proposal is good, monotonically. It carries no consistent directional
> information about the true objective.

That is consistent with everything else here — attempts 7 and 8 found the model's
gradients are uninformative and misaligned — but it is a much weaker and more specific
claim than "refinement is harmful", and it does **not** support "never refine".

It also explains the repo's existing `slm_gs_refine` **bake-off** precisely: keep GS
only if it beats flat, and refine only from a proposal that failed the bar. That guard
is exactly the right shape for a restart-like operator, and this is the first evidence
*for* it rather than merely against skipping refinement.

### What is actually robust in this study

Only two things, and they are different in kind from everything above:

1. **Structural (proof by bit-identity, not a metric):** `correction_far_field()` reads
   only `self.coefficients`, so the forward model's fit is not an input to the gradient
   path at all — an unfitted model and a fitted+regularised one produce identical
   refinements (attempt 12). This is a property of the code, not of a measurement, so
   no evaluator choice can invalidate it.
2. **Robust in sign across all 9 ROIs:** both GS and gradient inverse design beat flat
   (82/90 and 90/90). Inverse design on the learned model works; *which* proposer wins
   does not.

Everything else — objective ranking, GS-vs-gradient, refinement-harm, the alignment
table — is ROI-conditional and must not be quoted without the ROI attached.

### Attempt 14 — the one surviving claim, hardened across ROI × objective

Attempt 13 left a single ROI-independent conclusion, but it had been measured on one
objective (`mse`). A claim that survives a nuisance sweep on one axis and not another
is only half-robust, so the sweep was repeated over objective as well: 9 ROI
geometries × 2 objectives, 10 draws per cell (180 refinements).

Refinement delta against how well the proposal already did:

| ROI (frac × aspect) | GS − flat | Δ mse | Δ physical |
|---|---|---|---|
| 0.250 × 1.000 | +0.2775 | +0.1327 | +0.1812 |
| 0.250 × 1.333 | +0.2473 | +0.0461 | +0.1386 |
| 0.250 × 1.500 | +0.4034 | −0.1375 | −0.0086 |
| 0.375 × 1.000 | +0.2833 | +0.0644 | +0.0020 |
| 0.375 × 1.333 | +0.4392 | −0.1488 | −0.2415 |
| 0.375 × 1.500 | +0.4803 | −0.2661 | −0.1108 |
| 0.500 × 1.000 | −0.0660 | +0.2993 | +0.2385 |
| 0.500 × 1.333 | +0.1315 | +0.0888 | +0.1420 |
| 0.500 × 1.500 | +0.3434 | −0.1960 | −0.0795 |

| objective | pearson | spearman |
|---|---|---|
| `mse` | −0.9056 | −0.8667 |
| `physical` | −0.8278 | **−0.9167** |

**Both objectives show the same monotone pattern**, so the conclusion survives a 2D
nuisance sweep:

> **Refinement is a restart, not a gradient.** It rescues weak proposals and degrades
> strong ones, monotonically, *regardless of what it is optimising*.

Two things worth noting. First, the `physical` rank correlation is **stronger** than
`mse` (−0.9167 vs −0.8667) — so attempt 8's "mse is the better-aligned objective"
does not extend to this question, and should not be quoted as a general ranking.
Second, this is the only conclusion in the file that survived an attempt to break it,
and it is the one that carries a practical recommendation.

**Practical reading.** Gate refinement on proposal quality, exactly as
`slm_gs_refine`'s bake-off already does: refine only proposals that failed the bar,
and never refine one that already beat it. Attempt 13 had made this look like "don't
refine"; attempt 14 shows it is "refine the weak ones".

### Closing state of the question

| question | answer | evidence | status |
|---|---|---|---|
| does forward accuracy predict inverse quality? | no — it is not an input | attempt 12, bit-identical | **robust** |
| does the fit change gradient refinement? | no | attempt 12 | **robust** |
| does inverse design beat flat? | yes | attempts 10/13, 82–90/90 | **robust** |
| is refinement a restart rather than a gradient? | yes | attempt 14, ROI × objective | **robust** |
| should refinement be gated on proposal quality? | yes | attempt 14 | **robust** |
| is the inverse loss worth tuning? | no | attempt 8, +0.0002 | ROI-conditional |
| does GS beat gradient? | no | attempt 13, 34/90 | **retracted** |
| does refinement always hurt? | no | attempts 13/14 | **retracted** |
| should `n_max` be raised? | no — n_max=20 lands below flat | attempt 12 | ROI-conditional |

Four robust results, three retractions, and one structural proof. The retractions are
the most informative part of the file: every one was a single-configuration result that
looked significant (t up to −11) and dissolved under a sweep of a constant nobody had
questioned.

### Attempt 15 — freeform phase: the conclusions are parameterisation-independent

Every attempt above optimised a 135-coefficient Zernike vector, but the shipped
`slm_gs_refine` optimises a **freeform** phase grid. Since attempt 13 proved the
Zernike results ROI-conditional, they could not be transferred unexamined.

An earlier note in this file claimed freeform needed a new forward model. **That was
wrong.** `forward` validates only that its input is `(B, 1, g, g)` at the basis
resolution, and `measured = complex(phase_cos, phase_sin)` *is* the input phasor — so
any phase, including a freeform one, is pushed straight through the existing model.
The aperture is masked explicitly, because `forward` uses the phasor raw and a
phase-only pupil would otherwise have unit amplitude *outside* the illuminated disc
while the sim lights a finite disc.

9 ROI geometries x 8 draws, `phase-grid=24` (576 DOF, the runner's default) upsampled
to 64x64, all scored on the independent sim:

| ROI (frac × aspect) | flat | GS_zernike | GS_freeform | grad_freeform | GS_fm+refine |
|---|---|---|---|---|---|
| 0.250 × 1.000 | 0.5340 | 1.0290 | 0.9405 | 0.9418 | 1.0095 |
| 0.250 × 1.333 | 0.5570 | 0.5344 | 0.7929 | 0.9614 | 0.9436 |
| 0.250 × 1.500 | 0.5649 | 0.6843 | 0.8939 | 0.9707 | 0.6971 |
| 0.375 × 1.000 | 0.6114 | 1.0942 | 0.5323 | 1.0023 | 0.9379 |
| 0.375 × 1.333 | 0.6578 | 0.7640 | 1.0406 | 0.9805 | 0.9537 |
| 0.375 × 1.500 | 0.6809 | 0.7058 | 1.0115 | 0.9567 | 1.0880 |
| 0.500 × 1.000 | 0.7199 | 1.1138 | 1.1811 | 0.9940 | 0.6868 |
| 0.500 × 1.333 | 0.8016 | 1.0091 | 1.3215 | 1.1276 | 0.9874 |
| 0.500 × 1.500 | 0.8442 | 0.8757 | 1.2404 | 1.1319 | 1.3191 |

Paired across the 9 ROI cells:

| comparison | mean | positives | verdict |
|---|---|---|---|
| **GS_freeform − GS_zernike** | **+0.1272** | **7/9** | the projection bottleneck is real |
| grad_freeform − GS_freeform | +0.0125 | 4/9 | coin flip |
| GS_freeform − flat | +0.3315 | 8/9 | beats flat |
| grad_freeform − flat | +0.3439 | **9/9** | beats flat |

Three things this settles:

1. **The Zernike projection is a lossy bottleneck, quantified.** Using the GS pupil
   phase directly instead of projecting it through `fit_zernike` is worth **+0.1272
   in 7 of 9 ROIs**. This is the constructive version of attempt 12's finding that the
   projection lands *below* flat at `n_max=20` — and it means `slm_gs_refine`, being
   freeform, already avoids the failure mode entirely. The Zernike path was paying a
   penalty the hardware path never incurs.
2. **"GS beats gradient" stays retracted, now in a second parameterisation.**
   `grad_freeform − GS_freeform` is +0.0125 at **4/9** — the same coin flip attempt 13
   found in Zernike space. The retraction was not a Zernike artefact either.
3. **The two surviving conclusions replicate in freeform**, which is what actually
   makes them credible:
   * inverse design beats flat — `grad_freeform` **9/9**, `GS_freeform` 8/9;
   * **refinement is a restart** — `spearman(GS_freeform − flat, refinement delta) =
     **−0.7333**` (Zernike gave −0.8667 / −0.9167). Weaker, but the same strong sign
     in a completely different parameterisation, so it is not a basis artefact.

One ROI fails: at `0.375 × 1.000`, `GS_freeform` scores **0.5323 against a flat of
0.6114** — GS itself lands below no-shaping in freeform there, while the same GS phase
projected to Zernike scores 1.0942. Since it is a single cell out of nine and the
gradient arms are unaffected, it is recorded rather than chased.

### Closing state of the question (updated by attempt 15)

| question | answer | evidence | status |
|---|---|---|---|
| does forward accuracy predict inverse quality? | no — it is not an input | attempt 12, bit-identical | **robust** |
| does inverse design beat flat? | yes | attempts 10/13/15, 90/90 and 9/9 | **robust, 2 parameterisations** |
| is refinement a restart rather than a gradient? | yes | attempt 14, ROI × objective; attempt 15, freeform | **robust, 2 parameterisations** |
| should refinement be gated on proposal quality? | yes | follows from the two rows above | **robust** |
| does the Zernike projection cost quality? | yes, +0.1272 by removing it | attempt 15, 7/9 | **new** |
| does GS beat gradient? | no | attempt 13 (34/90); attempt 15 (4/9) | **retracted twice** |
| does refinement always hurt? | no | attempts 13/14/15 | **retracted** |
| is the inverse loss worth tuning? | no | attempt 8, +0.0002 | ROI-conditional |
| should `n_max` be raised? | no — n_max=20 lands below flat | attempt 12 | ROI-conditional |

### Remaining caveats

* Every number remains a **sim** claim, not a bench claim.
* The freeform grid is `24x24` upsampled to 64x64 on a 64x64 pupil grid. A native
  full-resolution freeform grid (4096 DOF) was not tried, and neither was a
  physically-apertured coarse grid, so the coarse-grid choice is unvalidated.

---

## ⚠️ CORRECTION (added after the illustrated report) — the evaluator was cropping the wrong region

Rendering the pred-vs-true figure exposed a bug that invalidates the **quantitative**
sim numbers in attempts 1-15.

`sim_far_field` cropped the simulator's far field with `image[:64, :64]`. The 0-order
of a `SimPibSystem` far field sits at the **centre** of the array, so that took the
**top-left corner**, where the measured value is `1e-05` against a peak of `100` —
i.e. every ROI metric in this file was computed on a patch roughly 10^5 times dimmer
than the beam, describing off-axis sidelobes rather than the spot.

It was invisible because the numbers were self-consistent and reproducible. It became
visible only as a *picture*: the "independent simulation" panel of the inverse
pred-vs-true figure rendered blank while the line profile showed a flat trace. Two
further faults surfaced with it:

* **Angular-scale mismatch.** The 64x64 pupil was embedded in the 1200x1920 panel, so
  the far field covered the *panel's* angular range and the beam core landed on ~6 px
  of the 64 px crop. A design targeting a 24 px square in its own grid then aimed at
  something 4x larger than the measurement window, and every comparison read as noise
  (GS-flat came out at ~-0.002 for every ROI tried).
* **ROI size calibrated against the broken crop.** `SIZE_FRAC=0.375` was chosen while
  the evaluator was reading a corner.

Fixed by running the 64x64 pupil **directly** (`far_field_window=64`, waist scaled from
the bench ratio), so the evaluator's far-field grid is identical to the grid
Gerchberg-Saxton produces and the two are comparable pixel-for-pixel. With the scale
matched, shaping works as expected: **GS - flat = +0.2945** at `SIZE_FRAC=0.375`,
`aspect=4/3`.

What survives and what does not:

| claim | status after the corrected re-measurement |
|---|---|
| `correction_far_field` ignores the fitted model (attempt 12) | **unaffected** — a code-level fact proved by bit-identity, with no metric involved |
| forward accuracy does not predict inverse quality | **unaffected in substance** — it was never an input to the path |
| GS beats gradient from a random start | **confirmed, 9/9**, paired t = +6.4…+32.5 (was 34/90) |
| inverse design beats flat | **only together with refinement** — GS alone does not |
| "refinement hurts GS" (attempt 11) | **void** — sign reversed, refinement helps in 8/9 |
| "refinement is a restart" (attempt 14) | **void** — spearman −0.87/−0.92 → **+0.93/+0.95** |
| freeform replication (attempt 15) | **needs re-measuring** on the corrected evaluator |
| absolute numbers throughout | **void**; the JSON panels are kept only as a record of the bug |

### Attempts 13 and 14, re-measured on the corrected evaluator

Both scripts were refactored onto `ml.zernike.inverse_design` — their private evaluator
copies are gone, and those copies are how the bug survived in twelve files. Score is
`pib + uniformity` on the independent simulator, **higher is better**, so a positive delta
means refinement *helped*.

| ROI | flat | GS | gradient | GS + refine |
|---|---|---|---|---|
| 0.250 × 1.000 | 1.0618 | 1.0254 | 0.5329 | 1.0321 |
| 0.375 × 4/3 | 1.0366 | 1.0280 | 0.7179 | 1.1214 |
| 0.500 × 1.500 | 1.0261 | 1.0217 | 0.9128 | 1.1475 |

Two facts the corner crop had hidden:

* **GS alone never beats flat.** `GS − flat` is negative in **90/90** runs, by −0.0025 to
  −0.0364. The GS *proposal* is not an improvement on doing nothing.
* **Gradient from a random start is far worse than flat** (`−0.3287` mean), while gradient
  *refinement on top of GS* is clearly better than flat (+0.0795 mean, positive in 78/90
  for `mse`; +0.0426, 68/90 for `physical`).

And the claim that survived the original sweep, "refinement is a restart, not a
gradient", **is dead**:

| objective | pearson | spearman (was) | spearman (now) |
|---|---|---|---|
| `mse` | +0.896 | **−0.87** | **+0.9333** |
| `physical` | +0.805 | **−0.92** | **+0.9500** |

That is not a weakened effect, it is a **sign reversal with a larger magnitude**. The
better the GS proposal, the *more* refinement gains — which is what a gradient carrying
directional information does, and the exact opposite of restart behaviour.

The corrected picture is `GS + refine > flat > GS > gradient-from-random`: GS supplies the
right basin, the gradient finishes the job. That is a sensible division of labour, and it
is the opposite of the "refinement degrades strong proposals" story the retracted numbers
told.

The methodological lesson is the same one as attempts 3, 5 and 13, and this time it is
about *plots*: a metric can be reproducible, self-consistent, and still measure the wrong
region. Rendering it is what caught it.

---

## Attempt 16 — anchored spot-size term and energy-conserving output

Two requested changes to the forward path, measured paired on 12 real corpus samples
(150 steps each, identical init, `n_max=15`, `grid=64`).

**1. `spot_moment_gap_term` — the radial second moment, anchored to the target.**
`|var_r(pred) - var_r(true)| / var_r(true)`, computed about the intensity centroid.
Anchored from the start, unlike `pib_term`/`uniformity_term`: the reference is always
the measurement, so it cannot be won by ignoring the data (the defect that made the
unanchored pair cost R2 +0.78 -> -0.86).

**2. `conserve_energy` — rescale the output so its intensity sum equals the input
phasor's.** `_propagate` already uses `norm="ortho"`, so Parseval is exact and the
propagated field is conservative; peak normalisation is what destroys the total. The
rescale is therefore applied **last**, after `_normalize`, or it would be a no-op.
Verified exact: ratio `1.000000`, and the anchor tracks partial coherence
(`|phasor|=0.5` -> `128.0` for an input total of `128.0`).

| config | R2 | MSE | moment gap | var pred | var true |
|---|---|---|---|---|---|
| **conserve_energy = False** | | | | | |
| `mse` | **0.9545** | 0.00067 | 0.0463 | 71.10 | 74.56 |
| `mse + moment(0.5)` | 0.7441 | 0.00378 | **0.0055** | 74.77 | 74.56 |
| `mse + moment(2)` | 0.5964 | 0.00597 | **0.0038** | 74.52 | 74.56 |
| `mse + moment + gap` | 0.7361 | 0.00390 | 0.0072 | 74.27 | 74.56 |
| **conserve_energy = True** | | | | | |
| `mse` | **-4.8517** | 0.08640 | 0.6998 | 126.73 | 74.56 |
| `mse + moment(0.5)` | -1.6788 | 0.03960 | 0.0340 | 77.09 | 74.56 |
| `mse + moment(2)` | -1.5800 | 0.03812 | 0.0191 | 75.59 | 74.56 |

Paired deltas vs `mse`:

| flag | config | dR2 | positives |
|---|---|---|---|
| False | `mse+moment(0.5)` | **-0.2105** ± 0.0148 | 0/12 |
| False | `mse+moment(2)` | **-0.3582** ± 0.0151 | 0/12 |
| True | `mse+moment(0.5)` | **+3.1729** ± 0.1674 | **12/12** |
| True | `mse+moment(2)` | **+3.2717** ± 0.1592 | **12/12** |

Read honestly, and neither half is a standalone win:

* **The spot-size term does exactly what it was built for.** It cuts the moment gap
  8x (0.0463 -> 0.0055 at `w=0.5`, -> 0.0038 at `w=2`) and pulls `var_pred` onto
  `var_true` (71.10 -> 74.77 against a true 74.56).
* **But it buys that with fidelity.** R2 falls 0.9545 -> 0.7441 -> 0.5964, in 0/12
  paired runs. It is a genuine **trade-off**, not a free win, and it is monotone in the
  weight. For a *forward* model, whose whole job is to predict the whole frame, that
  is usually the wrong side of the trade.
* **`conserve_energy` on its own is harmful** with a peak-normalised target: R2
  **-4.8517**. The output now carries an absolute scale the target does not, and MSE
  punishes the mismatch. It only makes sense with `normalization="none"`, or with a
  loss that is absolute-scale aware.
* **They interact strongly.** With conservation on, the spot-size term stops being a
  trade-off and becomes a repair mechanism: **+3.17, 12/12**. The two are not
  independent knobs.

Recommendation, given the measurements rather than the intent:

* Forward fidelity is the priority -> keep `w_spot_moment = 0` (the incumbent), or use a
  small weight only when spot size is what the application cares about.
* Absolute brightness is the priority -> `conserve_energy=True` **with**
  `normalization="none"`, and then `w_spot_moment` becomes necessary rather than
  optional.
* Do not enable `conserve_energy` alone against a peak-normalised target.

### Remaining caveats

* Every number remains a **sim** claim, not a bench claim.
* The freeform grid is `24x24` upsampled to 64x64. A native full-resolution freeform
  grid (4096 DOF) was not tried.
* **The ROI-sweep conclusions (attempts 13-15) still need re-measuring** on the
  corrected evaluator. See the correction section above.
* Attempts 8-14 are uncommitted: a concurrent in-progress merge holds 21 conflicted
  files and `git commit` refuses. They are on disk and unstaged.

### Honest tally of retractions in this file

Attempts 4, 5, 8, 9, 10 and 11 each produced a conclusion that a later attempt
overturned. Four of those were caught by printing mean/best/worst/sd or by re-running
with genuine restarts; **this one was only caught by varying an arbitrary constant I had
never questioned**, which is the same class of error as trusting a single split (README
§11) and I should have varied the ROI from the start.

### Remaining caveats

* Zernike parameterisation at 64x64 throughout; **freeform phase — what
  `slm_gs_refine` actually optimises on hardware — was never tested**, and given
  attempt 13 the Zernike conclusions should not be transferred to it unexamined.
* Every number remains a sim claim, not a bench claim.
* Attempts 8-13 are uncommitted: a concurrent in-progress merge holds 21 conflicted
  files and `git commit` refuses. They are on disk and unstaged.
