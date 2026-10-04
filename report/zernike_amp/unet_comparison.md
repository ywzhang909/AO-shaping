# physics vs U-Net on `slm_zernike_shaping`

<!-- provenance:start -->
> **生成脚本**: [`scripts/compare_models_cv.py`](../../scripts/compare_models_cv.py)
> **复现命令**: `python scripts/compare_models_cv.py --analyse`
> **数据/关联脚本**: [`scripts/compare_unet_baseline.py`](../../scripts/compare_unet_baseline.py)
> **运行环境**: 离线
> **说明**: grouped CV 对比 (physics / hybrid / unet) 的原始对照表
<!-- provenance:end -->

> **Superseded.** [`report.md`](report.md) §7 holds the authoritative result:
> **grouped cross-validation** with exact sign-flip permutation tests, which
> resolves the comparison that this file could not. The short version: the U-Net
> *is* significantly better (R² p = 0.012, SSIM p = 0.002), and the hybrid is
> indistinguishable from physics (p ≥ 0.61).
>
> This file is kept as the record of the earlier **single-split** runs. Its
> numbers remain reproducible but are **underpowered**: the split is 75/25 *by
> file* then truncated to `max_val=128` records, so the validation fold gets a
> random mixture of the four objectives — and `roi_pib` (10 % of the corpus) has
> a nearly orthogonal image distribution, which is what produced the ±0.07
> seed-to-seed swing. Read it as history, not as the verdict.

Head-to-head produced by `python scripts/compare_unet_baseline.py --seeds 3`.

Identical split (same `_select_records`, same files), identical inputs
`(phase_cos, phase_sin)`, identical target (peak-normalised `image`), identical
loss (MSE), optimiser (Adam), schedule (cosine) and budget (**25 epochs**).
`grid=64`, `far_field_padding=10`, `observable="intensity"`. **3 seeds each.**

| model | params | val R² (mean ± std) | val SSIM | PSNR | corr | centroid off | spot d90 ratio | train s |
|---|---|---|---|---|---|---|---|---|
| `physics(n_max=15)` | **135** | +0.8796 ± 0.0586 | 0.7552 | 29.36 dB | 0.951 | 1.10 px | 1.047 | 4.2 |
| `unet` [8…64] | 486,481 | +0.8352 ± 0.0972 | 0.8352 | 27.93 dB | 0.937 | 1.01 px | 1.110 | 4.8 |
| `unet` [16…256] | 7,778,465 | **+0.9012 ± 0.0711** | **0.8683** | **30.81 dB** | **0.957** | **0.96 px** | 1.053 | 6.7 |
| `unet` [32…512] | 31,100,225 | +0.8603 ± 0.0780 | 0.8519 | 28.88 dB | 0.943 | 0.97 px | 1.035 | 13.6 |

Per-seed R²:

- `physics(n_max=15)`: [0.797, 0.921, 0.921]
- `unet` [8…64]: [0.710, 0.946, 0.850]
- `unet` [16…256]: [0.801, 0.949, 0.954]
- `unet` [32…512]: [0.761, 0.868, 0.952]

## Reading

**The R² differences are all inside the noise.** Seed-to-seed std is 0.06–0.10 on
this corpus. `physics − unet[16…256] = +0.022`, `physics − unet[32…512] = +0.019`.
No capacity is significantly better than the 135-parameter model on R², and the
physics model is *better* than the 486 K and 31 M variants.

**Capacity is non-monotonic.** 486 K underfits (R² 0.835), 7.8 M is best (0.901),
31 M overfits (0.860). The physics model sits between those two failure modes using
**1/58,000th** the parameters of the best U-Net and 3× less wall time.

**SSIM is the one real gap.** Only the 7.8 M U-Net separates from physics on SSIM
(0.817 minimum vs physics' 0.785 maximum — the ranges do not overlap); the 486 K
and 31 M variants both overlap physics. This is physically interpretable: SSIM
rewards local high-frequency texture, and a 135-mode *smooth* Zernike correction
cannot synthesise speckle. It fits the envelope; the U-Net also fits the grain.

**Centroid offset is where physics is worst** (1.10 px vs 0.96–1.01 px). The global
Zernike vector places the spot marginally worse than a per-sample network does,
which is consistent with its low capacity for sample-specific structure.

## Not interchangeable

These models are not drop-in replacements for each other. The U-Net emits an
**image**; recovering a physical SLM phase from one is a separate inversion
problem. For the actual use case — *predict the correction to command on the SLM* —
the physics model is the only one of these that can emit a realisable phase, and it
does so in closed form (`Σ ZₖBₖ`) with 135 interpretable numbers in radians.

## Caveats

- These are **different model classes**, not two points on one curve: the physics
  model has `n_max` trainable numbers shared by the whole corpus; the U-Net is a
  per-sample function approximator. A U-Net beating it is expected and does not
  invalidate the physics prior — it says the mapping is not globally low-rank.
- 3 seeds is the minimum the ±0.13 noise floor permits. Any conclusion with a
  margin below ~0.1 R² should not be drawn from this table.
- Single-family, single-`fov_px` corpus. `centroid_offset_px` in particular depends
  on the target window, so it is not an absolute accuracy figure.
