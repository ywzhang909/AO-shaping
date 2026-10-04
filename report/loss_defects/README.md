# Forward-model / loss defect hunt

Conclusions from metric- and gradient-probing `ZernikeAmpModel` and
`ml/zernike/losses.py` on the real corpus (`slm_zernike_shaping`, 1010 records,
10 pickles, `grid=64`, `n_max=15`, `far_field_padding=10`, sim/offline only).

Reproduce with `python -m ml.zernike.train_amp` plus the panels described below.
Absolute R² is **not** comparable across runs — see "Noise floor" at the end.

---

## 1. FIXED — the physical loss terms were unanchored (category error)

`pib_term` / `uniformity_term` read only the **prediction**; the target was never
consulted. Correct for *shaping* (you do want the brightest, flattest box you can
synthesise), wrong for *fitting* a forward model, where the optimum is "ignore
the data and emit an ideal spot".

Real corpus, 8 epochs, paired seed:

| objective | val R² | pib | uniformity | shape_sum |
|---|---|---|---|---|
| `mse` | +0.7771 | 0.6736 | 0.5114 | 1.1850 |
| `physical` (unanchored) | **−0.8621** | 0.7769 | 0.7192 | 1.4961 |
| blend `mse`+unanchored | −0.2818 | 0.7934 | 0.6997 | 1.4931 |
| *measured frame* | — | — | — | *1.1127* |

`shape_sum = 1.4961` is **34% above the physics being predicted**, R² fell below
a constant predictor, and the best epoch was **0** — it never beat its own
initialisation. `w_mse=1.0` could not stop it: the physical terms are O(1) and a
peak-normalised MSE is O(0.003), a ~450× gap, so the fidelity anchor carried
~0.2% of the gradient.

**Fix:** `shape_gap_term` = `|shape_sum(pred) − shape_sum(reference)|`, divided by
the reference's own scale (`shape_gap_relative`, default on). `loss="physical"`
now defaults to `w_mse=1, w_shape_gap=1`.

| objective | val R² | pib | uniformity | shape_sum | vs measured |
|---|---|---|---|---|---|
| anchored | **+0.6764** | 0.7479 | 0.4681 | 1.2159 | +9.3% |
| anchored, gap ×10 | +0.5109 | 0.8324 | 0.3881 | 1.2206 | +9.7% |

R² −0.86 → +0.68; overshoot 34.5% → 9.3%; best epoch 0 → 7. The ×10 row is a
monotone fidelity/shape trade-off, not a cliff.

## 2. FIXED — the panel could not see what the loss optimises

`evaluate()` reported correlation / efficiency / spot diameter but nothing about
*where the light landed in the box*, i.e. exactly what `pib` and `uniformity`
optimise. A loss change was literally unmeasurable. Added
`ml.zernike.metrics.roi_shape_terms` via the canonical `rms_pib_terms`, wired
through `evaluate(roi_size_frac=...)` using the same fraction the loss is
configured with, returning the measured frame's own terms for relative reading.

## 3. FIXED — CCD intensity normalisation (`image_mode="robust"`)

`abs255` keeps absolute intensity but leaves per-frame gain in (measured frame
maxima span 255 / 239.6 / 100.9 across dtypes); `peak` removes gain but divides by
a *single* pixel, and this bench has hot pixels that outrank the real 0-order
(peak 22–46 against a frame mean of 0.26).

`robust` median-subtracts then divides by a robust beam level. The first attempt
used the 99.5th percentile and was **wrong**: taken over the lit pixels — only
~65 elements for a small beam — a single hot pixel *was* the quantile, so the
scale became 60000 instead of 255. Now the median of the lit pixels.

**Not fixed here:** a hot pixel that wins the `argmax` in `_anchored_window` and
moves the crop. That is a locator hazard, not a scaling one, and is still open.

## 4. FIXED — silent gradient death in the trainable self-attention

The additive-residual + `clamp(min=0)` form was a dead end: Adam's first
sign-based step drove `out_proj` negative enough to zero the whole far field, so
the output and its gradient were both identically 0 and **every** tensor — the
coefficients included — stopped training, with nothing raised. Measured: 3 tensors
with gradient at step 0, then **0** from step 1 onward.

Now a bounded multiplicative gate `intensity * (1 + tanh(·))`, identity at init.
All 9 tensors receive gradient from step 1.

---

## 5. REFUTED — one global coefficient vector is **not** a capacity bottleneck

`slm_zernike_shaping` looks like one family but is **four** optimisation
objectives (rms_pib 404 / rmse_out 303 / shape 202 / roi_pib 101 records over 10
pickles). Each objective's run leaves the SLM at a different aberration, so the
family holds four bench states — while the model has exactly **one** global
coefficient vector. Hypothesis: a per-objective vector must fit better.

Three seeds, paired (identical held-out records, stratified file split):

| per-group − plain | mean | spread | sign |
|---|---|---|---|
| rms_pib | −0.0149 | 0.0633 | 1/3 |
| rmse_out | −0.0228 | 0.0519 | 0/3 |
| shape | −0.0311 | 0.0188 | 0/3 |

**Refuted, and in the opposite direction: 8/9 paired deltas are negative.** A
per-objective vector is *consistently worse*, because it trains on 43–86% less
data. The four runs evidently share nearly the same underlying aberration, so one
vector suffices. **No architecture change is warranted** — do not "fix" this.

Note `roi_pib` is excluded: a single pickle cannot be split by file, and a
record-level split would leak consecutive epochs of one run across the split.

## 6. NOT DEMONSTRATED — the self-attention buys nothing measurable

Same 3 seeds, paired:

| attention − plain | mean | spread | sign |
|---|---|---|---|
| rms_pib | +0.0018 | 0.0177 | 2/3 |
| rmse_out | +0.0012 | 0.0033 | 2/3 |
| shape | +0.0045 | 0.0105 | 3/3 |

6/9 positive, mean +0.001 to +0.005, spread up to 0.018 — **inside the noise**.
On a *single* seed this looked like a consistent +0.008/+0.002/+0.010 gain with a
large physical improvement (`shape_sum` gap cut 24–78%); repeating it collapsed
that to nothing. The single-seed reading was noise.

The attention is implemented, identity-initialised, gradient-alive and correct —
but at 4321 parameters (309× the 14-parameter physics) for a mean R² delta of
~+0.002 it is **not justified on this corpus**. It stays `attention=False` by
default, and enabling it should be treated as unproven until a protocol with the
repo's usual statistical power (grouped CV, ≥10 folds) says otherwise.

---

## Noise floor — read this before trusting any number above

The **same configuration** gave val R² between **0.748 and 0.937** depending only
on the split seed. Absolute R² here is meaningless; only *paired* differences are
informative, and the paired spread on a delta is ~0.003–0.06 depending on the
objective. Any comparison at a single seed — including several conclusions that
were reversed during this hunt — is inside that band.