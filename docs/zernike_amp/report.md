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

## 6. The noise floor (and a retraction)

Same config, same seed, repeated → **bit-identical** (3/3 runs, 5 decimals). The
pipeline is deterministic; all variance is the split seed:

| seed | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| val R² | +0.780 | +0.797 | +0.920 | +0.923 |

**σ ≈ 0.07, range 0.14.** With val = 2 files / 128 records, a single run's R²
carries ±0.13 of noise.

> **Retraction.** An earlier version of this comparison reported that the U-Net
> "wins on R² by +0.022" and that "SSIM is a real, non-overlapping gap". Both
> claims were wrong. The first came from **one seed at an unequal budget** (the
> U-Net was given 25 epochs while the physics model converges by epoch 13). The
> second came from the same single seed. Re-run at a matched budget with 3 seeds,
> neither claim survives — see below.

## 7. Model comparison at a matched budget

Identical split, inputs, target, loss, optimiser, schedule and budget (50 epochs,
lr=0.01), **3 seeds each**:

| model | params | val R² (mean ± std) | val SSIM | PSNR | corr | centroid off | spot d90 ratio | time |
|---|---|---|---|---|---|---|---|---|
| `hybrid` (physics + CNN residual, w=32) | 10,408 | +0.8756 ± 0.0655 | 0.7712 ± 0.0362 | 29.34 dB | 0.949 | 1.11 px | 1.098 | 11.4 s |
| **`physics` (n_max=15)** | **135** | **+0.8790 ± 0.0593** | 0.7552 ± 0.0419 | 29.33 dB | **0.951** | **1.05 px** | **1.052** | **8.1 s** |
| `unet` [16…256] | 7,778,465 | +0.8739 ± 0.0574 | 0.8115 ± 0.0878 | **29.67 dB** | 0.949 | 1.22 px | 1.084 | 13.2 s |

**No metric separates the three.** The three R² means span 0.005 against a
per-seed σ of ~0.06. The SSIM means differ by 0.056, but the U-Net's SSIM σ is
0.088 — the largest in the table — and the ranges **overlap** (physics max 0.790
vs U-Net min 0.704). The earlier "non-overlapping SSIM gap" was a single-seed
artefact.

**What this means.** A 135-parameter physical model matches a 7.78M-parameter
U-Net on this task, using **1/58,000th** the parameters and 40 % less wall time.
The physics model also wins the two beam metrics that matter operationally
(centroid offset 1.05 px, spot-diameter ratio 1.052) and ties on correlation.

Capacity is non-monotonic in the U-Net, confirming the physics model sits
between the underfit and overfit regimes rather than at an optimum:

| U-Net | params | val R² (3 seeds) |
|---|---|---|
| [8…64] | 486,481 | +0.8352 ± 0.0972 |
| **[16…256]** | **7,778,465** | **+0.9012 ± 0.0711** |
| [32…512] | 31,100,225 | +0.8603 ± 0.0780 |

(Those three rows are at the earlier 25-epoch budget; only the 3-seed means are
comparable to each other, not to §7.)

**Not interchangeable.** The U-Net emits an *image*; recovering a realisable SLM
phase from it is a separate inversion problem. For predicting the correction to
command on the SLM, the physics model is the only one of these that can emit a
realisable phase, in closed form `Σ Z_k B_k`, as 135 interpretable radians.

## 8. The hybrid experiment (a negative result)

Hypothesis: the physics model's only visible deficit was local speckle texture,
which a smooth Zernike basis cannot synthesise, so add
`pred = physics(phase) + residual_cnn(phase)` with the residual **zero-initialised**
so the hybrid starts bit-identical to the physics model.

**It did not close the gap.**

| config | val R² | val SSIM |
|---|---|---|
| physics | +0.8790 | 0.7552 |
| hybrid w=16, 25 ep | +0.8738 | 0.7661 |
| hybrid w=32, 25 ep | +0.8697 | 0.7499 |
| hybrid w=64, 25 ep | +0.8332 | 0.6595 |
| hybrid w=32, 50 ep | +0.8756 | 0.7712 |

The first attempt at 25 epochs was simply **under-trained** — with 60 epochs the
residual reaches SSIM 0.726 (vs physics 0.698), a consistent but small +0.028.
At 3 seeds that gain is inside the physics model's own SSIM σ of 0.042, so it is
**not a demonstrated improvement**, and larger residuals degrade monotonically
(overfitting on 808 samples).

The informative part is *why* it fails. Adding a small CNN to the physics output
does not recover the U-Net's texture advantage, which means that advantage comes
from the U-Net's deep multi-scale encoder–decoder with skip connections — genuine
architectural capacity — and not merely from "having a CNN in the loop".

**Default unchanged.** The hybrid stays available
(`ZernikeAmpHybrid`) but physics remains the default: equal within noise on R²,
simpler, and faster.

## 9. Final configuration

```
ZernikeAmpConfig(n_max=15, grid=64, observable="intensity",
                 normalization="peak", far_field_padding=10, center_crop=True)
train: 50 epochs, Adam lr=0.01, cosine, batch 64, all 1010 records, seed pinned
```

| metric | value |
|---|---|
| val MSE | 0.00173 |
| val R² | +0.8790 |
| val PSNR | 29.33 dB |
| val SSIM | 0.7552 |
| val correlation | 0.951 |
| centroid offset | 1.05 px |
| spot-d90 ratio | 1.052 |
| grad norm | 2.3e-03 → 1.0e-03 |
| dead modes | 0 / 135 |
| max&#124;Z&#124; | 0.591 rad |
| wall time | 8.1 s |

## 10. Honest limitations

- **One family, one `fov_px`.** All conclusions are for `slm_zernike_shaping`
  (`fov_px=248`). The `far_field_padding` calibration is a per-family constant and
  **must be re-swept** for another family.
- **A single global `Z`** can only represent bench-wide systematic phase, not
  per-sample aberrations. That is why it reaches R² ≈ 0.88 and not 0.95+; the
  U-Net's per-sample capacity is the honest reason it can go higher on some seeds.
- **3 seeds is the minimum** the noise floor permits. Nothing with a margin below
  ~0.1 R² should be concluded from this report — including several of the
  intermediate comparisons above, which is why §6 exists.
- **`grad norm` is tiny** (~1e-3) because there is only one parameter vector and
  the loss is scale-normalised. This is why `lr` mattered so much and why early
  lr sweeps either did nothing (1e-4) or ran away (1e-1).
- The 135 modes are fitted on 808 training records with 0 dead modes, which is
  reassuring but not proof of identifiability on a different corpus.

## Reproduction

```bash
# train + log + compare images (wandb offline; sync later if a key is available)
python -m ml.zernike.train_amp --n-max 15 --epochs 50 --lr 0.01 \
    --max-train 1010 --beam-samples 96 --out-dir logs/zernike_amp_final

# baseline comparison (identical split / inputs / loss / budget)
python scripts/compare_unet_baseline.py --seeds 3 --models physics hybrid unet \
    --residual-width 32 --epochs 50 --lr 0.01

# tests
python -m pytest tests/ao_shaping/ml/zernike -q      # 67 tests
python -m pytest tests/ao_shaping/ml -q             # 412 tests
```