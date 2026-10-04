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