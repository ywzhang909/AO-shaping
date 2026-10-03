# Zernike far-field model — development report

Task: train the physics forward model on the real hardware corpus, record the
diagnostics that matter, optimise it, and compare it against a U-Net baseline.

Everything below was measured on `data/debug/slm_zernike_shaping` (1010 records,
10 pickles, `fov_px=248`) through `src/ml/hwdataset`. Numbers are reproducible
with the commands in [Reproduction](#reproduction).

---

## 1. What the model is

`src/ml/zernike/models.py` implements a differentiable pupil → far-field model
whose only free parameters are one **global** Zernike vector `Z`:

```
U    = (phase_cos + i·phase_sin) · exp(i · Σ_k Z_k B_k)     # complex domain
focal = fftshift(fft2(ifftshift(center_pad(U)), norm="ortho"))
loss  = MSE(observable(focal), ccd)
```

- `Z` is **trained**; `n_max` (the coefficient count `K`) is a **hyperparameter**.
- The basis `B` is built once by the canonical `ZernikeGenerator` — no Zernike
  maths is re-derived.
- The forward pass never calls `atan2`; it stays in the complex domain because
  the corpus phase is wrapped to `[0, 2π)` and an angle would carry a branch cut.
- **Piston is excluded** (`K = (n_max+1)(n_max+2)/2 − 1`): `|FFT(e^{iφ₀}U)| ≡ |FFT(U)|`
  identically, so a piston coefficient has exactly zero gradient forever.

## 2. Data pipeline

`src/ml/hwdataset` serves `(phase_cos, phase_sin, image, exposure_log10, …)`.
Three decisions that are load-bearing:

| decision | why |
|---|---|
| phase as a coherent `(cos, sin)` pair | corpus wraps to `[0, 2π)`; `atan2` would put a branch cut on the input |
| target **not** peak-normalised (`abs255`) | exposure is a model input, so absolute brightness *is* the signal encoding it |
| FOV surfaced as `fov_px`, never resampled | families genuinely differ (64/248/320/1944 px windows); resampling would fabricate pixels |

The on-disk cache is bit-exact against the direct path (1010/1010 records) and
**317× faster** (127.0 s → 0.4 s, 10 pickles opened → 0).

## 3. Three traps found by measurement

**3.1 Scale mismatch — the big one.** `_anchored_window` takes a centre-cropped
`grid×grid` window out of the 248-px CCD frame; it does **not** resize. A far
field computed at `far_field_padding=1` spans the pupil's *full* diffraction
field, ~4× the target's angular scale, so the two are not pixel-comparable:

| `far_field_padding` (centre-cropped) | val R² at `Z=0` |
|---|---|
| 1 | −0.20 |
| 4 | −0.12 |
| 8 | +0.33 |
| **10** (default) | **+0.46** |
| 12 | +0.53 |
| 16 | +0.49 |
| 20 | +0.18 |

A clear **interior optimum** — this is a real calibration, not "sharper is
better". After the fix, `max|Z|` fell from a runaway **1.77 rad → 0.165 rad**,
because a small correction then genuinely explains the data instead of the
optimiser reaching for huge phase to fake the loss.

**3.2 `normalization="sum"` is a metric trap.** Dividing both prediction and
target by their own total energy makes them agree almost trivially:

| `normalization` | MSE | R² | PSNR | SSIM |
|---|---|---|---|---|
| `peak` | 0.00429 | **+0.706** | 24.8 dB | 0.594 |
| `sum` | **0.00000** | +0.632 | **72.1 dB** | **0.9996** |
| `none` | 0.47534 | **−158.98** | 3.2 dB | 0.037 |

`sum` reads as *perfect* on MSE/PSNR/SSIM while R² is **worse**. Any
sum-normalised metric measures the normalisation, not the fit. `none` is unusable
because the raw FFT amplitude carries an arbitrary scale.

**3.3 `observable`: intensity, not amplitude.** The original spec said `amp`;
the measurements say otherwise, consistently at every capacity:

| `observable` | n_max=4 | n_max=11 | n_max=15 |
|---|---|---|---|
| `amplitude` | +0.634 | +0.659 | +0.675 |
| `intensity` | **+0.750** | **+0.792** | **+0.797** |

A CCD integrates intensity; it does not report field amplitude. The default is now
`intensity`; `amplitude` remains available.

## 4. Metrics

`src/ml/zernike/metrics.py` reports the img2img-standard set (MSE, RMSE, MAE,
NRMSE, PSNR, SSIM) plus the beam set (correlation, efficiency, centroid offset,
90 % encircled spot diameter, peak ratio), reusing the canonical `beam_metrics`
helpers rather than reimplementing them.

**LPIPS/FID are deliberately absent** — they need a pretrained backbone and this
environment has no `torchvision`, so
`torchmetrics.image.LearnedPerceptualImagePatchSimilarity` is genuinely not
importable. `available_perceptual_metrics()` reports that honestly rather than
returning a constant.

`perplexity` is reported as `exp(MSE / Var(target))`. Perplexity is
`exp(cross-entropy)` and has no exact MSE analogue; this is a monotone rescaling
of the normalised error, valid *across epochs of one run* only. **Trust R² and
SSIM.**

## 5. Optimisation, one lever at a time

Single-variable sweeps from a common baseline, `slm_zernike_shaping`, fixed split:

| lever | outcome |
|---|---|
| **`n_max`** | **The only large effect.** R² 0.48 → 0.80 monotonically over 1→20; still **0 dead modes** at K=230 |
| `far_field_padding` | interior optimum ≈10–12 |
| `observable` | intensity wins by ~0.12 R² |
| `normalization` | `peak` only valid choice |
| `lr` | 0.1 best alone, but see the non-additivity warning |
| `l2_penalty` | 1e-4 neutral; 1e-2 over-regularises (R² 0.469, `max|c|`=0.039) |
| `grad_clip` | no effect — gradients (~1e-3) never reach the threshold |
| `optimizer` | SGD much worse (R² 0.466); adam ≡ adamw is *correct* at `weight_decay=0` |
| `max_train` | more data helps monotonically |

**Two things the sweeps taught that a single run would have hidden:**

*Non-additivity.* `far_field_padding=12` and `lr=0.1` each beat the defaults
alone, but combined with `n_max=11` both turn **worse** (R² 0.794 → 0.753).
Greedy coordinate descent fails here; every winner must be re-tested jointly.

*A dead lever.* `optimizer` initially returned byte-identical scores for
adam/adamw/sgd — `train()` hard-coded `torch.optim.Adam` and ignored
`cfg.optimizer`. It had measured nothing until it was fixed.

## 6. The noise floor — and why it was never the fold size

Same config, same seed, repeated → **bit-identical** (3/3 runs, 5 decimals), so the
pipeline is deterministic. Sweeping only the split seed gave:

| seed | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| val R² | +0.780 | +0.797 | +0.920 | +0.923 |

**σ ≈ 0.07, range 0.14.** For a long time I attributed this to "only 128 validation
records". **That diagnosis was wrong**, and finding the real cause is what made the
comparison resolvable.

The corpus is not i.i.d. The 1010 usable records sit in 10 pickles of exactly 101
records each, but those pickles are 2–4 timestamps of only **four** optimisation
objectives:

| objective | files | records | mean intensity | total variance |
|---|---|---|---|---|
| `rms_pib` | 4 | 404 | 0.0335 | 2.057 |
| `rmse_out` | 3 | 303 | 0.0325 | 2.247 |
| `shape` | 2 | 202 | 0.0303 | 2.093 |
| `roi_pib` | 1 | 101 | 0.0271 | 1.274 |

and those objectives have **different image distributions**. Scoring one
objective's mean image against another's targets:

| | rms_pib | rmse_out | roi_pib | shape |
|---|---|---|---|---|
| **rms_pib** | 1.000 | 0.977 | **−0.670** | 0.710 |
| **rmse_out** | 0.979 | 1.000 | **−0.223** | 0.852 |
| **roi_pib** | **−1.696** | **−1.157** | 1.000 | 0.040 |
| **shape** | 0.715 | 0.842 | 0.416 | 1.000 |

`R² = −0.670` means the `rms_pib` mean image predicts `roi_pib` targets *worse
than a constant*. So a random validation fold receives a **random objective
mixture**, and since `roi_pib` is 10 % of the corpus and nearly orthogonal to the
rest, the pooled R² depends on the mixture. **The split-seed sweep was measuring
the mixture lottery, not model quality.**

Two further corrections to earlier statements in this report:

* the split is **75/25 by file** (`val_fraction = 0.25`, `_select_records`
  `train_amp.py:302`), then validation is head-truncated to `max_val = 128`
  records (`:315`) — not 80/20;
* `HWRecordRef.source` is the *phase representation*
  (`panel_gray`/`panel_rad`/`zernike`/`freeform`), **not** the source file. The
  file identity is `HWRecordRef.path`.

## 7. Grouped cross-validation

`scripts/compare_models_cv.py` replaces the single split. The repo has **no
sklearn dependency at all** (verified: zero references to `sklearn` /
`model_selection` anywhere), so folds are built by hand; `_select_records`
already used the right key, `str(record.path)`, and this keeps that convention.

| protocol | folds | train / val | what it answers |
|---|---|---|---|
| `objective` | 4 | 606 / 404 | generalisation to an **objective never seen** |
| `file` | 10 | 909 / 101 | every record validated exactly once |

Statistics: exact two-sided **sign-flip permutation** test (2¹⁰ = 1024
enumerations, min p = 0.00195, no normality assumption), Cohen's `d_z`, and
Holm–Bonferroni across the model×metric family.

### Result (identical inputs, target, loss, optimiser, schedule, 50 epochs, lr 0.01)

| model | params | val R² | val SSIM | val PSNR |
|---|---|---|---|---|
| hybrid (w=32) | 10,408 | +0.8797 ± 0.0359 | 0.7676 ± 0.0480 | 29.57 ± 1.24 |
| physics (n_max=15) | **135** | +0.8766 ± 0.0434 | 0.7437 ± 0.0323 | 29.58 ± 1.39 |
| unet [16…256] | 7,778,465 | **+0.9004 ± 0.0310** | **+0.8516 ± 0.0363** | **+31.34 ± 1.82** |

*(objective protocol, 4 folds)*

| model | params | val R² | val SSIM |
|---|---|---|---|
| hybrid (w=32) | 10,408 | +0.8721 ± 0.0799 | 0.7404 ± 0.0817 |
| physics (n_max=15) | **135** | +0.8727 ± 0.0867 | 0.7320 ± 0.0704 |
| unet [16…256] | 7,778,465 | **+0.9004 ± 0.0610** | **+0.8419 ± 0.0618** |

*(file protocol, 10 folds)*

Paired differences, negative = physics worse:

| comparison | metric | diff | d_z | p (exact) | verdict |
|---|---|---|---|---|---|
| physics − unet | SSIM | −0.1099 | **−2.49** | **0.0020** | unet better |
| physics − unet | PSNR | −1.84 dB | −1.76 | 0.0039 | unet better |
| physics − unet | R² | −0.0277 | −0.89 | 0.0117 | unet better |
| physics − unet | NRMSE | +0.0045 | +1.01 | 0.0117 | unet better |
| physics − hybrid | all 5 | — | ≤0.27 | 0.61 – 0.98 | **no difference** |

### Two things this settles, and one it cannot

**1. The U-Net is genuinely better.** SSIM by a wide margin (`d_z` ≈ −2.5), and R²
by a small but significant margin. So my §6 retraction below was *also* wrong:
"no metric separates the three" was an artefact of the mixture lottery, not a
real null result. Two conclusions in this report have now been overturned in
opposite directions by better statistics — see §11.

**2. The hybrid adds nothing.** Every metric, both protocols, `p ≥ 0.61`. It is
statistically indistinguishable from the physics model it wraps, at 77× the
parameters. It is therefore **not** promoted to the training entry point; it stays
in the comparison harness as a documented negative result.

**3. Four folds cannot reach significance, by construction.** With n = 4 the
exact test's smallest attainable two-sided p is 2/2⁴ = **0.125**. The
`objective` protocol is scientifically the *right* question (unseen objective)
but is underpowered — it can only ever *suggest*. Only the 10-fold protocol can
confirm. More folds cannot manufacture more independent units than there are
groups, so this is a ceiling, not a tuning problem.

## 8. Optimisation, one lever at a time

Single-variable sweeps, `slm_zernike_shaping`, fixed split:

| lever | outcome |
|---|---|
| **`n_max`** | **The only large effect.** R² 0.48 → 0.80 over 1→11, then saturates (below) |
| `far_field_padding` | interior optimum ≈10–12 |
| `observable` | intensity wins by ~0.12 R² |
| `normalization` | `peak` is the only valid choice |
| `lr` | 0.1 best alone, but see non-additivity |
| `l2_penalty` | 1e-4 neutral; 1e-2 over-regularises |
| `grad_clip` | no effect — gradients (~1e-3) never reach the threshold |
| `optimizer` | SGD much worse; `adam ≡ adamw` is *correct* at `weight_decay=0` |
| `max_train` | more data helps monotonically |

**`n_max` is saturated.** Re-measured under the powered 10-fold protocol:

| n_max | K | val R² |
|---|---|---|
| 11 | 77 | +0.8695 ± 0.0837 |
| 15 | 135 | +0.8727 ± 0.0823 |
| 20 | 230 | +0.8775 ± 0.0803 |
| 25 | 350 | +0.8760 ± 0.0812 |
| 30 | 495 | +0.8797 ± 0.0777 |

6.4× the coefficients buys **+0.010 R²** against a fold-σ of ~0.08.

### The physics optimum is a *joint* setting, and it moved

`lr` and `n_max` are **non-additive**, so sweeping them one at a time gets the
wrong answer. Joint sweep, same 10 folds:

| n_max | lr | val R² | val SSIM |
|---|---|---|---|
| 15 | 0.01 | +0.8727 ± 0.0823 | 0.7320 |
| 20 | 0.01 | +0.8775 ± 0.0803 | 0.7548 |
| **20** | **0.02** | **+0.8803 ± 0.0781** | **0.7646** |
| 15 | 0.02 | +0.8719 ± 0.0842 | 0.7370 |
| 20 | 0.005 | +0.8730 ± 0.0811 | 0.7361 |

`lr = 0.02` **alone** at `n_max=15` buys nothing (+0.8719 vs +0.8727), yet the same
`lr` at `n_max=20` is the best cell in the table. That is the non-additivity of §8
reproduced under the powered protocol, and it is why greedy coordinate descent
fails here.

Paired over the same 10 folds, `(15, 0.01) → (20, 0.02)`:

| metric | before → after | diff | d_z | p (exact) |
|---|---|---|---|---|
| R² | +0.8727 → +0.8803 | +0.0076 ± 0.0096 | +0.79 | 0.0234 |
| SSIM | 0.7320 → 0.7646 | +0.0326 ± 0.0164 | +1.98 | 0.0039 |

**This is where paired testing earns its keep.** The between-fold spread of R² is
0.078, so a +0.008 change is invisible to any unpaired comparison; the paired σ
is **0.0096** because fold difficulty cancels. The improvement is consistent in
8 of 10 folds (the two negatives are −0.004 each).

⚠️ **But treat it as suggestive, not confirmatory.** `(20, 0.02)` was chosen *as
the best of five on these same folds*, so these p-values carry a winner's-curse
bias — the classic selection effect that nested CV exists to remove. Confirming
it properly needs an outer loop (select on inner folds, score on the held-out
one), which was not affordable here. `n_max=20, lr=0.01` (+0.8775 / 0.7548) is the
defensible fallback if one wants a cell that was not selected on this data.

### The U-Net configuration was *not* over-fitted to the noisy split

Its hyperparameters were originally chosen on the underpowered single split, so
they were re-verified on the powered protocol:

| config | val R² | val SSIM |
|---|---|---|
| **[16…256], 50 ep, lr 0.01** (previous choice) | **+0.8993 ± 0.0603** | +0.8390 ± 0.0593 |
| [16…256], 100 ep, lr 0.01 | +0.8931 ± 0.0510 | +0.8407 ± 0.0653 |
| [16…256], 100 ep, lr 0.005 | +0.8982 ± 0.0481 | +0.8382 ± 0.0578 |
| [16…256], 50 ep, lr 0.005 | +0.8889 ± 0.0476 | +0.8167 ± 0.0364 |
| [24…384], 100 ep, lr 0.01 | +0.8950 ± 0.0572 | +0.8476 ± 0.0680 |

The original choice is already the best on R² and tied on SSIM; all five configs
span 0.010 R² and 0.031 SSIM, far inside the ±0.06 fold-σ. **The U-Net is already
converged and insensitive to these knobs**, so the earlier pick did not over-fit
the noisy split. Kept unchanged.

**Non-additivity, first sighting.** `far_field_padding=12` and `lr=0.1` each beat
the defaults alone, but combined with `n_max=11` both turn **worse** (R² 0.794 →
0.753) on the original single-split protocol.

**A dead lever, twice.** `optimizer` initially returned byte-identical scores for
adam/adamw/sgd because `train()` hard-coded `torch.optim.Adam` — it had measured
nothing. And exposure-rescale augmentation, which is *physically exact* here
(exposure is a model input, the CCD is linear, and the corpus never clips)
changed R² by **exactly zero** to 4 decimals. The reason is measurable: exposure
is **constant** across this family (`log10 = −1.0`, so the dataset's exposure
standardisation falls back to mean 0 / std 1) *and* no model consumes it —
`forward(phase_cos, phase_sin)` only. A valid augmentation on an input the model
never sees, over a dimension that never varies, is a no-op. It would matter on a
multi-exposure corpus.


## 9. Final configuration and verdict

```
physics : ZernikeAmpConfig(n_max=20, grid=64, observable="intensity",
                           normalization="peak", far_field_padding=10, center_crop=True)
          50 epochs, Adam lr=0.02, cosine, batch 64, seed pinned
          (n_max=20 / lr=0.01 if you want a cell not selected on the CV folds)
unet    : [16…256], 50 epochs, Adam lr=0.01, cosine, batch 64
```

| metric | physics (230) | hybrid (10,408) | unet (7,778,465) |
|---|---|---|---|
| val R² (10-fold) | +0.8803 ± 0.0781 | +0.8721 ± 0.0799 | **+0.9004 ± 0.0610** |
| val SSIM (10-fold) | 0.7646 | 0.7404 ± 0.0817 | **+0.8419 ± 0.0618** |
| val PSNR (10-fold) | — | 29.07 ± 2.51 | **+31.05 ± 2.44** |
| params | 230 | 10,408 | 7,778,465 |
| wall time / fit | ~9 s | ~13 s | ~15 s |

**Verdict.** The U-Net wins every metric significantly (R² p = 0.012, SSIM
p = 0.002, PSNR p = 0.004) against the physics model at its *pre-tuning*
configuration, and it remains ahead after the physics model is tuned
(+0.9004 vs +0.8803 R²). So my earlier "they tie" claim was wrong; what survives
is that the physics model is a far cheaper way to get most of the way there.

What the physics model retains is a real operational advantage: it emits a
**realisable SLM phase** in closed form, `Σ Z_k B_k`, as 230 interpretable
radians, using 1/34,000th of the U-Net's parameters and 40 % less wall time. The
U-Net emits an image, and recovering a commandable phase from it is a separate
inversion problem. So the two answer different questions, and for "what phase do I
command" the physics model is the only one of the three that answers it directly.

The hybrid is a clean negative result: indistinguishable from physics on all five
metrics across both protocols (p ≥ 0.61) at 45× the parameters.

## 10. Honest limitations

- **One family, one `fov_px`.** All conclusions are for `slm_zernike_shaping`
  (`fov_px=248`, `fov`-consistent). `far_field_padding` is a per-family constant
  and **must be re-swept** for another family.
- **The sampling unit is the pickle (n = 10).** That is the hard ceiling on
  statistical power here; more folds cannot create more independent groups. At
  n = 10, d_z ≈ 0.9 is detectable and d_z ≈ 0.25 is not.
- **k-fold differences are not independent** — fold *i*'s training set overlaps
  fold *j*'s by 8/9. Paired-t and permutation p-values are therefore mildly
  anti-conservative, which is why the effect sizes and CIs, not the p-values, are
  the honest headline.
- **Only 4 objectives exist**, so leave-one-objective-out gives 4 folds whose
  minimum attainable p is 0.125. That protocol answers the most relevant
  question and cannot resolve it. More objectives — not more folds — is what
  would fix this.
- **A single global `Z`** can only represent bench-wide systematic phase, not
  per-sample aberrations. That is the structural reason it plateaus near R² 0.87
  while the U-Net, which can vary its prediction per sample, reaches 0.90.
- **SSIM on speckle is a known-weak metric.** Its structure term is a point-wise
  normalised cross-correlation, and Larson & Chandler show SSIM returns near-
  identical scores (~0.64) for Gaussian noise, speckle noise, salt-and-pepper,
  JPEG and blur — it largely cannot distinguish distortion *type*. The
  domain-standard alternatives are Strehl ratio, encircled energy and FWHM. SSIM
  is reported here as a secondary descriptor; R² and the beam metrics carry the
  weight, and `perplexity` is reported only as a within-run monotone rescaling.
- **`grad norm` is tiny** (~1e-3) because there is one parameter vector and a
  scale-normalised loss. This is why `lr` mattered so much and why early sweeps
  either did nothing (1e-4) or ran away (1e-1).

## 11. Conclusions overturned in this report

Kept as a record, because getting it wrong twice is the point:

| version | claim | why it was wrong |
|---|---|---|
| v1 | "U-Net wins R² by +0.022; SSIM is a real non-overlapping gap" | right direction, wrong evidence: one seed, and the U-Net given 25 epochs while physics converges by 13 |
| v2 | "no metric separates the three" | **over-correction.** The ±0.07 split noise swamped a real effect; the mixture lottery, not model quality, was being measured |
| v3 | "U-Net significantly better; hybrid indistinguishable from physics" | grouped CV + paired exact tests; stands on 4 and 10 folds |

The lesson is not "the v1 estimate was noisy". It is that **the noise floor was
itself misdiagnosed**, so a correct effect was first hidden and then, after an
over-correction, briefly denied. Both errors came from trusting a variance
estimate whose *source* had not been identified.

## 12. Reproduction

```bash
# train + log + compare images (wandb offline; sync later if a key is available)
python -m ml.zernike.train_amp --n-max 15 --epochs 50 --lr 0.01 \
    --max-train 1010 --beam-samples 96 --out-dir logs/zernike_amp_final

# the authoritative comparison: grouped CV, paired exact tests
python scripts/compare_models_cv.py --protocol both --epochs 50 --lr 0.01 \
    --out logs/models_cv.json

# recompute the statistics from saved folds without retraining
python scripts/compare_models_cv.py --analyse logs/models_cv_objective.json

# the older single-split baseline (kept for continuity; known underpowered)
python scripts/compare_unet_baseline.py --seeds 3 --models physics hybrid unet

# tests
python -m pytest tests/ao_shaping/ml/zernike -q      # 67 tests
python -m pytest tests/ao_shaping/ml -q             # 418 tests
```