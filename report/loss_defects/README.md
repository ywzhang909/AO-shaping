# Forward-model / loss defect hunt

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/sweep_far_field_padding.py`](../../scripts/sweep_far_field_padding.py)
> **运行环境**: 离线
> **说明**: 前向模型/loss 缺陷排查结论（人工撰写）；同目录 *.json 为各探针面板
<!-- provenance:end -->

Conclusions from metric- and gradient-probing `ZernikeAmpModel` and
`ml/zernike/losses.py` on the real corpus (`slm_zernike_shaping`, 1010 records,
10 pickles, `grid=64`, `n_max=15`, `far_field_padding=10`, sim/offline only).

Reproduce with `python -m ml.zernike.train_amp` plus the panels described below.
Absolute R² is **not** comparable across runs — see "Noise floor" at the end.

> **Sections 1–7 are the forward/loss hunt. The inverse (shaping) investigation is a
> separate, longer record in [`PROCESS.md`](PROCESS.md)** — 16 numbered attempts, of
> which **four headline conclusions were retracted**: three when the evaluator's ROI
> geometry was swept, and one more when the evaluator itself was found to have been
> cropping the wrong region of the far field. **That last one invalidated every
> quantitative inverse result in this file**, so §8 below has been re-measured rather
> than inherited.

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

## 5. FIXED — `far_field_padding=10` is wrong for most families

`ZernikeAmpConfig.far_field_padding` defaults to 10 and its own docstring warns
"the right value depends on the family's `fov_px`, so re-run the sweep for a new
family rather than copying the default". That warning was never acted on.

Swept all 6 usable families (`scripts/sweep_far_field_padding.py`, coefficients
at Z = 0 so this is *purely* geometric agreement — no training, no fitting;
48 samples/family, `grid=64`, `n_max=15`, `normalization="peak"`, judged on R²):

| family | fov_px | best pad | R²(best) | R²(10) | gain vs default |
|---|---|---|---|---|---|
| `model_in_loop_hw_collect` | 64 | **8** | +0.4839 | +0.3063 | **+0.178** |
| `model_in_loop_hw_sweep` | 64/1944 | 10 | +0.4967 | +0.4967 | 0.000 |
| `slm_pib` (**7866 rec**) | 320 | **16** | +0.7350 | +0.5367 | **+0.198** |
| `slm_pib_online` | 248 | **14** | +0.6819 | +0.5167 | **+0.165** |
| `slm_zernike_shaping` | 248 | 12 | +0.5530 | +0.4945 | +0.058 |
| `slm_gsnet_square` | 1944 | 4 | **−1.106** | −1.373 | — |

**The default is optimal for exactly one family.** It is badly wrong for
`model_in_loop_hw_collect`: R² falls from +0.484 at pad 8 to **−4.46** at pad 20,
so a large padding is not a "conservative" choice on a narrow window. The largest
family in the corpus (`slm_pib`, 7866 records) peaks at 16, not 10.

The real signal is the **trend** — the optimum increases with `fov_px`
(64→8, 248→14, 320→16), which is what a zero-padding/angular-extent argument
predicts. Adjacent optima differ by ~0.05 R² in places, near this repo's noise
floor, so treat the argmax as ±1 step and the trend as the finding.

`fov_px = 1944` is **negative at every padding** and is deliberately absent from
the lookup: `slm_gsnet_square` stores freeform phase cells rather than Zernike
coefficients, so a Zernike-parameterised model has nothing to fit. Do not read its
−1.11 as a padding problem.

Added `PADDING_BY_FOV_PX` and `recommended_padding(fov_px)` (unmeasured values fall
back to the default rather than being guessed at) so this is queryable instead of
folklore.

---

## 6. REFUTED — one global coefficient vector is **not** a capacity bottleneck

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

## 7. NOT DEMONSTRATED — the self-attention buys nothing measurable

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

---

## 8. INVERSE DESIGN — re-measured on the corrected evaluator

Summarised from [`PROCESS.md`](PROCESS.md) for the reader who does not need the full
16-attempt record. **Inverse design here means: synthesise a Zernike phase that
shapes the far field, then score the result on an independent simulator
(`SimPibSystem`, separate numpy FFT and illumination model) so the model never
grades its own work.** Score is `pib + uniformity`, **higher is better**.

⚠️ **Every number below was re-measured.** The first evaluator cropped the simulator's
far field with `[:64, :64]` — the top-left corner, while the 0-order sits at the
array centre (value `1e-5` against a peak of `100`). All of it therefore described
off-axis sidelobes. It is now `ml.zernike.inverse_design`, which runs the 64×64 pupil
directly so its far-field grid matches the grid GS designs on. Caught by *drawing* the
result: the simulator panel rendered blank while the numbers stayed self-consistent.

### The results that hold

| finding | evidence |
|---|---|
| **Inverse design beats flat — but only with refinement.** Gradient refinement on top of GS gains **+0.0795** mean (`mse`, positive in 78/90) and **+0.0426** (`physical`, 68/90). | 9 ROI geometries × 10 paired draws |
| **GS beats gradient from a random start.** | **9/9 ROIs**, paired t = **+6.4 … +32.5** (was 34/90 before the fix) |
| **Gradient refinement carries directional information — it is not a restart.** The better the GS proposal, the *more* refinement gains. | pearson +0.90 / +0.80, spearman **+0.93 / +0.95** over 9 ROI × 2 objectives |
| **Forward-model accuracy is not an input to the gradient path.** `correction_far_field()` reads only `self.coefficients`, so an unfitted and a fitted+regularised model produce *bit-identical* refinements. | coef norm 0.0000 vs 2.5421 → refined 6.84294, sim 0.997656 in both (proof, not a measurement) |

### The corrected division of labour

`GS + refine (1.1214) > flat (1.0366) > GS alone (1.0280) > gradient-from-random (0.7179)`
at the canonical ROI. Two facts the broken crop had concealed:

* **GS alone never beats flat** — `GS − flat` is negative in **90/90** runs (−0.0025 to
  −0.0364). The GS *proposal* is not an improvement on doing nothing.
* **Gradient from a random start is far worse than flat** (−0.3287 mean).

So GS supplies the basin and the gradient does the work. That is the opposite of the
"refinement degrades strong proposals" story the retracted numbers told.

### What was retracted, and why it matters

| retracted claim | original result | what actually happened |
|---|---|---|
| "GS beats gradient inverse design" | +0.1171, t=+2.77, **34/90** across 9 ROIs — a coin flip | **9/9**, t = +6.4…+32.5. The coin flip was the broken evaluator |
| "Refinement destroys a GS solution" | −0.2206, 0/16, t=−7.65 | **sign reversed** — refinement helps in 8/9, mean +0.0795 |
| "Refinement is a restart, not a gradient" | spearman **−0.87 / −0.92** | **spearman +0.93 / +0.95** — sign reversal, larger magnitude |
| "Inverse loss choice is worth +0.0002" | 5 objectives, paired | ROI-conditional |
| "The Zernike projection costs +0.1272" | 9 ROI × 8 draws | **not yet re-measured** — void until it is |

The first three all had good t-statistics and all were reproducible. Two constants broke
them, in order: the ROI box (`SIZE_FRAC=0.375`, `ASPECT=4/3`, chosen once and never
questioned) and then the evaluator's crop. Both are the same failure mode as trusting a
single split — a nuisance parameter fixed once and never swept. Recorded so the next
person sweeps the nuisance parameter instead of trusting the conclusion.

### Freeform does *not* replicate Zernike — and that is the interesting part

`freeform_vs_zernike.py` re-measured on the same corrected evaluator, same 9 ROIs.
Score is `pib + uniformity`, higher is better.

| quantity | Zernike space | freeform space |
|---|---|---|
| `GS − flat` | −0.0147, negative in **90/90** | **+0.2843, 9/9** |
| `GS − GS(projected)` | — | **+0.2128, 9/9** |
| refinement delta | **+0.0795**, 78/90 (helps) | **−0.2779, 0/9** (hurts every ROI) |
| spearman(proposal quality, delta) | +0.93 / +0.95 | +0.45 (positive, weak) |

Two conclusions, opposite in sign:

* **The Zernike projection bottleneck is real** and larger than the retracted run
  reported: using the GS pupil phase directly is worth **+0.2128 in 9/9** (+0.1272 in
  7/9 before). `slm_gs_refine` is freeform, so it never pays this.
* **Freeform refinement destroys the GS solution**, where the same gradient helps in
  Zernike space. A freeform refinement moves **576 DOF** while the forward model it is
  scored through carries **135 Zernike modes** — it descends a space the objective
  cannot see. This is attempt 12's invariance result from the other side: the refinement
  must stay inside what the model can represent.

⚠️ That is a statement about **model-based** refinement. It is **not** a claim about
`slm_gs_refine` on hardware, where the forward model is the measurement and therefore
does span all 576 DOF. It does say a model-based freeform loop is unsound unless the
model covers the refinement space.

### Scope limits that are not optional

* Every number is a **sim** claim, not a bench claim. A native full-resolution freeform
  grid (4096 DOF) and a physically-apertured coarse grid were never tried, so the choice
  of `phase-grid=24` remains unvalidated.
* The two parameterisations **disagree** on whether refinement helps. Treat "refine
  after GS" as a claim that must be re-established per parameterisation, never carried
  over.
