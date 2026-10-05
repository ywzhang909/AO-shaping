# Process log — forward model & inverse shaping optimisation

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
evidence the budget is better spent on **restarts with pick-the-best**, since the
per-restart outcome spread (sd 0.08-0.26) dwarfs the within-restart progress past
~60 steps. Concretely: ~60 steps per restart, N restarts, score each on the model,
keep the best. That is a cheap change and it is the one this study actually
supports.

Caveats: 5 restarts per cell is thin for an sd estimate; the evaluator is a sim, not
the bench; and "score the restarts on the model" assumes the model's score correlates
with the independent one, which is the next thing worth checking.

### Next direction

Check the assumption the recipe depends on: does picking the best restart *by the
learned model's own score* also pick a good one *on the independent sim*? If model
score and sim score are uncorrelated across restarts, then best-of-N needs an
external scorer and the recipe above is wrong.