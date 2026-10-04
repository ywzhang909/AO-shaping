# Model-in-the-loop bench calibration (SLM-200 + Daheng)

<!-- provenance:start -->
> **生成脚本**: [`scripts/model_in_loop_hw_runbook.py`](../../scripts/model_in_loop_hw_runbook.py)
> **复现命令**: `python scripts/model_in_loop_hw_runbook.py --stage all --target-cam-px 40`
> **运行环境**: 硬件
> **说明**: 正向模型台架几何标定；runbook 采集 + 本文档结论
<!-- provenance:end -->

Measured constants and known failure modes for running
`ZernikeCoefficientOptimizer` (Step A) and the A→B loop against the **real**
2f-Fourier bench, as opposed to `SimFourierGSNetEnv`.

Everything here was measured on 2026-09-29 with the devices online; the numbers
are what the model needs, not what the twin assumes.

> ⚠️ **2026-10-01 复核，3 处已订正**（其余内容仍有效）：
> 1. §L459 焦面标定常数 `132940/P` → **`7600/P`**（本文档内部原本自相矛盾）。
> 2. §L276 相机序列号 `FJB24112232` 与 §L550 的 `FJB24112222/32` 冲突 → 以前者为准
>    （`report/slm/bench_calibration_20261001.md:8,35` 只承认 `FJB24112232`）。
> 3. §L434 「`--exposure-ms 1.1` 是 runbook 默认」与 §L366 的「3 ms（peak≈97）」不一致，
>    且已被 2026-10-01 的曝光扫描取代：**t ≥ 0.4 ms 单调近线性，建议工作点 1.0–1.5 ms**。
>
> ⚠️ **仍未收口的常数冲突（见 `TODO.md` H-11 / H-16）**：焦面标定在三处不一致 ——
> `AGENTS.md:697` 写 `5021/Λ`（对应 3.31 µm 像元）、本文档与 `README.md:517` 写 7400–7600
> （对应 2.2 µm 像元）、`report/slm_pib_heuristic_hw/report.md:159` 主张改 **10954**。
> 2.2 µm 像元推得的是 ~7557 而非 10954，故该主张本身也待复核。**引用本文件前请先确认用哪一套。**

## Bench capability limit: the panel cannot resolve pixel-scale phase

**This retires the speckle-correlation route on this bench, and no amount of
re-scaling fixes it.** Measured 2026-09-30 by driving the pupil and reading the
far field (`core40` = fraction of frame energy within 40 px of the 0-order,
exposure 3.0 ms, beam radius 450 px at panel centre):

| pupil phase | peak | `core40` |
|---|---|---|
| flat | 86 | 0.0261 |
| per-pixel random, 1 SLM px | 92 | 0.0247 — **essentially unchanged** |
| random Zernike n≤4 | 35 | 0.0195 |
| random Zernike n≤8 | 24 | 0.0250 |
| random Zernike n≤14 | 12 | 0.0127 |

Per-pixel random phase does **nothing**; smooth Zernike phase degrades the focus
monotonically with order. So the panel's effective phase resolution is far
coarser than one pixel — high spatial frequencies are strongly low-pass filtered
by the LCOS and the optics.

The consequence is visible in `data/model_in_loop_hw/model_vs_measured.png`:
under a full random pupil phase the **measured** far field is a single tight
spot on a dim halo, whereas the forward model predicts a speckle field filling
the frame. The model's premise — random pupil phase ⇒ far-field speckle — is not
realisable here, so the speckle correlation cannot rise above ~0.07 no matter
how the geometry is fitted, and **any Step A / Step B coefficient derived from it
would be meaningless**.

Do not spend more time tuning `disc_candidates`, `far_field_size` or the waist
search against this bench. The options are:

1. **Band-limit the probes** and accept a smooth pupil model — then the far field
   is a perturbed focus, not a speckle field, and the calibration metric has to
   change from speckle correlation to something like spot width / centroid
   response versus a modelled defocus sweep.
2. **Calibrate the panel's transfer function** (the `slm-lut` tool already does
   this for grayscale→phase) and pre-compensate, so the requested phase is the
   realised phase. This is the prerequisite for any pixel-level work.
3. **Use a different bench or a smaller beam** where the beam only illuminates a
   region small enough that the panel's resolution is not the limit.

## Sweep calibration (the route that works on this bench)

Speckle correlation is out (see above). The replacement drives **smooth Zernike
modes** and reads the **spot centroid and width**, which the panel can realise.

### Why tilt is the right measurement

A unit-RMS Zernike tilt of coefficient `t` over an aperture of radius `a` imposes
a phase gradient `2t/a`, deflects the beam by `2t/(k·a)` and shifts the focal
spot by `f·t·λ/(π·a)`. The model needs
`camera_px_per_model_px = λf / (P · d_model · camera_pixel)`. Dividing the two
cancels λ, f, the model pitch and the camera pixel entirely:

```
camera_px_per_model_px = k_tilt · π · a / P
```

where `k_tilt` is the **measured** centroid shift in camera pixels per radian of
tilt coefficient. Nothing about the bench has to be assumed — which is the whole
point, since every assumed constant was wrong before.

### Measured (2026-09-30, 3.0 ms, aperture r=450 px at panel centre)

| quantity | value |
|---|---|
| tilt slope, panel-x → camera-y | **−3.467** cam px/rad |
| tilt slope, panel-y → camera-x | **−3.355** cam px/rad |
| `k_tilt` (mean) | 3.411 cam px/rad |
| `camera_px_per_model_px` @ P=4096, a=128 model px | **0.335** |
| reproducibility over 3 sweeps | 0.325 / 0.331 / 0.335 (~3 %) |
| flat spot FWHM | 11.9 cam px |
| bench defocus offset | **−0.171 rad** |

The two tilt slopes agreeing to 3 % is also a direct confirmation of the **90°
axis swap** — a run whose slopes disagree means the panel was indexed
inconsistently.

### What the sweep does NOT give: the beam waist

The defocus response is usable but not fittable to a pure-defocus model:

| c (rad) | −4.0 | −2.5 | −1.5 | −0.75 | +0.75 | +1.5 | +2.5 | +4.0 |
|---|---|---|---|---|---|---|---|---|
| FWHM (cam px) | 12.5 | 30.9 | 17.8 | 13.4 | 12.8 | 28.0 | 33.4 | 42.2 |
| peak | 62 | 19 | 58 | 113 | 90 | 31 | 16 | 9 |

- The **−4.0 rad point reads narrower than −2.5 rad** (12.5 vs 30.9), which
  breaks the monotonicity a single lobe must satisfy. Dropping non-monotonic
  points is what turns the fitted curvature from −18.9 (unphysical) into +94.6.
  Width thresholds and centroid hollowness both fail to catch it — the outlier's
  hollowness (0.80) sits inside the single-lobe range.
- After that, the response is still **asymmetric by +16 %** at matched |c|, and a
  pure-defocus pupil is symmetric about its own offset (−0.171 rad here). The
  residual is non-defocus aberration (astigmatism/coma) that a defocus sweep does
  not contain, so the illumination profile is **not identifiable**.

The code reflects this honestly: when the defocus fit residual exceeds 10 %,
`beam_waist_panel_px` is emitted as the flat-top default and
`calibration_notes` says `waist NOT identifiable, flat-top default`. The
tilt-derived scale is unaffected — it never used the width data.

**Next step to close the gap:** add an astigmatism sweep and fit defocus + astig
together, which separates the two contributions and makes the waist meaningful.

### Commands

```bash
# smooth Zernike tilt + defocus sweep, with phase/CCD images
python scripts/model_in_loop_hw_runbook.py --stage sweep \
    --exposure-ms 3.0 --pupil-center 960,600 --zernike-radius 450 --collect-disc 450

# fit (offline, re-runnable against the saved npz)
python scripts/model_in_loop_hw_runbook.py --stage calibrate --method sweep \
    --region 256 --far-field-size 4096 --collect-disc 450
```

## Bench probes (`ao_shaping.tools.slm`)

These five probes produced every measured number in this file. They are not
registered as CLI commands; run them with `python -m ao_shaping.tools.slm.<name>`.

| Tool | What it establishes |
|---|---|
| `slm_tilt_probe` | **Is the panel actually modulating?** A 2*pi ramp over `P` px moves the spot by `7600/P` camera px, independent of diffraction efficiency. This is the probe that overturned the false "frozen panel" verdict — gratings could not, because oblique incidence moves where the first order lands. Also reports the displayed memory slot, so a firmware no-op is visible. |
| `slm_panel_locate` | **Where is the beam on the panel?** Writes a random-phase disc at candidate positions and keeps the one that scatters the 0-order most. Replaces the old habit of converting the camera 0-order into a panel offset, which produced `(+426, -290)` px and rolled the pupil phase clean off the beam. |
| `slm_beam_extent` | **How big is it?** Half-plane random-phase boundary sweep; the knee is the beam edge. Replaces a stale "~192 SLM px" note that had been used to modulate under half the pupil. |
| `slm_phase_resolution` | **Can the panel resolve pixel-scale phase?** Per-pixel random phase vs random Zernike n<=4/8/14. |
| `slm_exposure_check` | **Is the camera drifting?** Auto-exposure state plus peak/sum drift at a fixed setting, which separates "camera drift" from "the SLM still holds the previous run's pattern". |

`slm_bench_probe.py` is their shared measurement core: pure functions, devices
passed in, no CLI — so it is unit-tested offline in
`tests/ao_shaping/tools/slm/test_slm_bench_probe.py`.

### The three rules they exist to enforce

1. **Never locate a spot with a raw `argmax` on a dim frame.** At peak 22-46
   against a frame mean of 0.26 a single hot pixel wins. A box blur alone is not
   enough either — a 5000-count defect spread over 5x5 still reads 200, beating a
   spot at 20 — so `measure_spot` **despikes with a median first**, then blurs. The
   reference centroid used to wander 60 px between repeats, which was enough to
   call a healthy panel "unmoved".
2. **Display flat before reading a flat reference.** The panel retains the last
   pattern, so a reference read before any write is the previous run's speckle.
   That is why the same 3 ms setting measured 100 counts in one run and 23 in the
   next.
3. **Never pass a fixed `memory_number=`.** The firmware treats `display_memory` on
   the already-displayed slot as a no-op, so every frame after the first is stale.
   `display_data()` rotates the slot itself and estimates the LCOS flip time.

## Measured bench constants

### Beam on the panel (measured, not assumed)

Measured by sweeping the boundary of a half-plane of random phase and watching
the 0-order concentration fall (`core40`); the knee is the beam edge. The
**flat reference must be written first** — the panel retains whatever pattern
was last displayed, so a "flat" read before any write is the previous run's
speckle (this is what made the beam look 4x dimmer than it is between runs).

- centre: panel **(960, 600)** — the panel centre (beam was re-centred 2026-09-30)
- radius: **450 panel px** (confirmed by the operator and by the x-scan, which
  saturates at T≈1440, i.e. 960 ± 480)
- x extent 510–1410, y extent 150–1050 at the time of the scan

The x-scan matched a centred 450 px radius cleanly; the first few y points were
non-monotonic because each is only 2 random draws, so the y extent is the
least certain number here.

### Camera

- Daheng MER2-507, 2592×1944, pixel 2.2 µm.
- Auto exposure is **off** and stays off: `get_auto_exposure_state()` reports
  `enabled: False`, `ExposureTime.get()` reads back exactly. 12 successive grabs
  at a fixed setting drift **4.4 % in peak / 0.7 % in total sum**.
- Brightness is **non-linear in exposure**, so calibrate by interpolation, not by
  a linear fit. At ~1.1 ms the 0-order peak lands on 60; the same setting has
  measured 55 and 63 on different days (laser drift ~15 %), so re-check the peak
  before trusting a long run.

### SLM ↔ camera geometry (measured, do not re-derive)

- **The panel and camera axes are swapped 90°**: a panel-x phase ramp moves the
  spot in camera-y (confirmed independently by `slm_tilt_probe`).
- **The focal scale must be measured, not derived.** A 2π ramp over `P` panel px
  moves the spot by `S/P` camera px, and `S` is now measured *inside the sweep*
  (`--sweep-ramps`), so the calibration no longer depends on a Zernike fit:

  ```
  S = 7615 cam px per 1/period        (TILT_SHIFT_SCALE = 7400, probe 7418)
  ```

  Two independent routes now agree to 0.5 %: the ramp gives
  `S = 7615`, and a Zernike tilt of coefficient `c` deflects `5.362c` cam px,
  which is the same statement since `S = k·π·R = 5.362 × π × 450 = 7579`. The
  physical cross-check (`k_tilt` against `f·λ/(π·R·camera_pixel)`) reads
  **1.00×**.

- **⚠️ The LCOS needs a *stability* criterion, not a longer fixed wait.** The
  driver's automatic flip-time estimate is driven by how much the gray map
  changed, and two different phases with similar gray statistics make it report
  **0.0 ms**. Measured: the same ramp displayed twice read fwhm 43.2 px on the
  first grab and 12.8 px three seconds later, with the centroid moving 62 px. A
  single-shot acquisition therefore records an *unsettled* frame that still looks
  plausible — a ramp sweep came out non-monotone (P=120 reading "unmoved" while
  P=240 moved 44 px) purely for this reason, and it made the Zernike tilt slope
  read **1.63 cam px/rad instead of 5.36**, a 3.3× error that survived three
  runs. Repeats had been masking it. `display_and_average` now discards frames
  until two consecutive (peak, centroid) readings agree (`--stable-tol`,
  `--max-wait-s`).

- **Keep the Zernike tilt coefficient small anyway.** Above ~1.5 rad the spot
  visibly deforms (hollowness falls to 0.65) and the centroid stops tracking the
  peak; the ramp has no such failure mode, which is why it is now authoritative.

- **Oblique incidence is a known, unquantified ~1.6×.** The independent
  field-of-view route (which assumes normal incidence) predicts
  `camera_px_per_model_px = 0.829` where the measured ramp route gives `0.529` —
  a ratio of 0.64, i.e. `cos θ` for an incidence of roughly 50°. The bench *is*
  known to be oblique, so the FOV route is the one carrying the assumption and
  the measured ramp value is the one to use. Both numbers, and the percentage
  disagreement, are recorded in `calibration_notes` on every run.

- **Waist still not identifiable** (joint width fit 14.5 % residual, defocus
  asymmetry +70 %). The bench carries aberration the swept modes do not cover;
  separating it needs coma/spherical sweeps. `beam_waist_panel_px` is emitted as
  the flat-top default and labelled `NOT identifiable` rather than pretending.

### Waist identification — what was tried and why it fails

`--stage sweep` now sweeps **six** modes (`defocus`, `astig_x/y`, `coma_x/y`,
`spherical`; `--sweep-coma` / `--sweep-spherical`) and the fitter fits them
jointly. The cubic/quartic pair was added because the quadratic pair is
degenerate with the bench's own residual defocus. They *do* respond strongly and
distinctly — fitted curvatures `astig 116/140`, `coma 100/92`, `defocus 219`,
`spherical 148` — so adding them moved the joint residual 24.1 % → 21.5 %.

**The dead guard that was actually worth more.** `_DEFOCUS_MIN_HOLLOWNESS` and
`_DEFOCUS_SINGLE_LOBE_FACTOR` were defined but never referenced: the single-lobe
check did not exist. Implementing it (hollowness alone — see below) and applying
it to *every* fitted mode took the joint residual **21.5 % → 14.5 %** and made
the curvatures self-consistent (`astig_x` 68 → 116 against `astig_y` 81 → 140).
It drops 6 points across 4 modes, the largest being `defocus +1.5 rad` at 24.4 px
— a ring being fitted as a Gaussian lobe.

*Rejected alternative:* "drop points wider than `1.8 × min(w)` in their own
sweep". It sounds like the same test in measurement units, but it deletes exactly
the large-`|c|` points that carry the curvature signal — it removed `astig_y`
from a synthetic sweep whose widths were correct. Hollowness measures
single-lobedness *directly*; the width ratio does not. The constant is retained
with that note so the alternative is not re-tried.

**Remaining 14.5 %, and why more modes will not fix it.** The residual is
dominated by the **defocus asymmetry, +70 %**: the width response is not
symmetric about its own minimum, i.e. the bench carries an aberration the swept
basis does not span. A wider basis cannot remove a component that is absent from
the basis. Identifying the waist on this bench needs either a WFS (a direct
wavefront measurement instead of an inferred one) or a bench whose residual
aberration is genuinely negligible. Until then the calibrator is right to refuse
and label the number.


| Quantity | Value | How it was obtained |
|---|---|---|
| SLM | Santec #1, serial `22030108`, 1920×1200, 1064 nm, 2π = 993 gray | driver `open()` log |
| SLM mode | **memory mode only** (`video_mode=0`) | DVI mode can wedge the controller |
| Camera | Daheng `FJB24112232`, 2592×1944 | driver log |
| Camera pixel | ≈ 2.2 µm (nominal `d_ccd` in the twin) | nominal |
| 0-order position | **(674, 1028)** on a 2592×1944 frame | `argmax` on the flat frame |
| Geometric frame centre | (1296, 972) | — |
| 0-order offset | **622 px in x** from the geometric centre | — |
| Peak repeatability | 0.50 % std/mean (5 frames, 3 ms) | flat phase |
| Flat-phase spot σ | ≈ 12.5 camera px = **27.5 µm** | intensity-weighted 2nd moment |
| Beam waist `w0` | `λf/(π σ_f)` = **1.54 mm ≈ 192 SLM px** | derived from the above |
| Illuminated pupil | disc of **radius ≈ 190–200 px** on the panel | derived from `w0` |
| Working exposure | 3 ms → peak ≈ 97 (8-bit, non-saturating) | measured |

**The 0-order is 622 px off the geometric centre.** Always locate it by
`zero_order_center(..., refine=False)`, never by assuming the frame centre. This
is the AGENTS.md rule, now with a measured magnitude.

## Model ↔ camera geometry

The far-field pixel pitch of the model is

```
p_fft = λ f / (P · d_slm)
```

With the pupil transformed unpadded (`P = region = 256`):

| Quantity | Value |
|---|---|
| `p_fft` | 64.94 µm |
| camera pixel | 2.2 µm |
| **ratio** | **29.5×** (the model samples the far field ~29× too coarsely) |

Cross-check: the real 27.5 µm spot is 0.42 model-px, and the model reported
σ = 0.44 px — agreement to 4 %. The model is *correct*, merely under-sampled.

Fix with the `far_field_size` padding option (added for this purpose; it is a
pure sampling change — the field of view stays `λf/d_slm` = 16625 µm, verified
by an unchanged spot σ and peak/mean across P = 1024/2048/4096):

| `far_field_size` | `p_fft` | real spot σ in model px |
|---|---|---|
| 256 (default) | 64.94 µm | 0.42 — unusable |
| 2048 | 8.12 µm | 3.4 |
| 4096 | 4.06 µm | 6.8 |
| 8192 | 2.03 µm | 13.6 — 1:1 with the camera |

8192 costs ~1.1 s per forward, so a 600-step Step B is ~11 min; 4096 is the
practical compromise.

**The model `region` must map to the illuminated pupil (~192 SLM px), not to the
whole 256-px grid.** A 256-px model's Airy width is ~65 µm against a real 27.5 µm
spot, i.e. the modelled aperture is ~2.4× too large.

## Known failure modes (each one cost a measurement)

| Symptom | Cause | Fix |
|---|---|---|
| Step A wanders to ‖c‖ ≈ 10 rad and the fit loss *rises* | The CCD re-normalises every frame to `peak_photons` (60000) while the model peaks in the thousands — a **47× unit mismatch**. Step A divides the model by the *measured* peak, which silently turns "match the pattern" into "maximise brightness under the bright core". | `_to_model_units()` rescales by the model's own peak before fitting |
| 56 % (later 28 %) of the fitted ‖c‖ lands in Noll 1–3 | Piston is a global phase (invisible to \|E\|²); tilt only displaces the spot. Unidentifiable directions absorb the residual when there is no real signal. | `frozen_modes=(1, 2, 3)`, matching the repo's `slm_square_shaping --zernike-mask` convention |
| Every frame clips at 255, σ inflates 12.5 → 29.8 px | `enable_auto_exposure(True)` on this Daheng body leaves it in a boosted state even though `get_auto_exposure_state()` then reports `enabled: False` | Do **not** call it; sweep exposure manually |
| `ExposureTime.set: is not writeable`, everything saturated | `reset_exposure_time` takes **milliseconds**; passing µs silently clamps to the 1000 ms maximum | Pass ms |
| `create_phase_from_array` warns "将从中心裁切" | It expects `(height, width)` = (1200, 1920); `ZernikeGenerator` returns `(width, height)` | Build the phase into an explicit `(1200, 1920)` buffer and assert the shape |
| Model "agrees" at MSE 0.0068 but is wrong | **MSE is not discriminative** when both images are peak-dominated: the number mostly reflects the spot's tail energy | Compare σ and the Pearson correlation, not MSE alone |

## Result: Step A has nothing to fit on this bench

Fitting a shared aberration to real captures from
`data/debug/slm_pib_shape_20260926_175334` (733 records, commanded \|`_c`\| up to
5 rad, per-mode spread 4.07 rad — plenty of pupil diversity):

| Configuration | MSE at `c=0` | MSE after fit | fit loss | ‖c‖ | piston+tilt share |
|---|---|---|---|---|---|
| all 66 modes | 0.006824 | 0.006809 (+0.21 %) | 1.04e-3 → 9.32e-3 | 9.26 rad | 28.5 % |
| Noll 1–3 frozen | 0.006824 | 0.006802 (+0.32 %) | 1.04e-3 → 9.31e-3 | 8.81 rad | 0.0 % |

The model already explains the measurement to MSE 0.0068 with **zero**
aberration, and every fitted coefficient vector makes the agreement slightly
worse while the loss rises. The residual is speckle-level model imperfection
(SLM LUT, wavefront-correction residual, departure of the real beam from a
Gaussian, camera PSF, quantisation) — **not** low-order aberration.

So on this bench Step A correctly reports "no aberration", and the piston/tilt
freeze is what makes it say so cleanly instead of inventing a 9 rad answer.
For Step A to have work to do, either inject a known aberration or raise the
model's fidelity; the A→B loop is still meaningful, with Step B carrying it.

## Hardware runbook (when the devices are back)

1. Locate the 0-order by `argmax`; expect ≈ (674, 1028), not the centre.
2. Exposure 3 ms (peak ≈ 97). Do not use `enable_auto_exposure`.
3. Display phases through `slm.create_phase_from_array` with an explicit
   `(1200, 1920)` array; flat phase is zeros (radians), not a gray level.
4. Model: `region` mapped to the ~192 SLM px illuminated pupil,
   `far_field_size=4096`, `w0` matched to the real beam, `frozen_modes=(1,2,3)`.
5. Size the target square from the *measured* scale (1 model px at
   `far_field_size=4096` is 4.06 µm ≈ 1.85 camera px), not from the model grid.

## Step B at the hardware-valid sampling (verified on the twin)

Every loop test in the suite runs at `region=64 / far_field_size=256`, the
regime measured above to be ~29.5× under-sampled for this bench. Re-running Step
B on the twin configured with the *measured* geometry
(`region=256`, `far_field_size=4096`, `w0=192` panel px → `p_fft=4.06 µm`,
target 40 camera px = 22 pupil px = 352 far-field px) gives:

| | EE | uniformity CV | flatness |
|---|---|---|---|
| baseline (flat phase) | 1.0000 | 13.83 | 0.0026 |
| after Step B, 150 it, lr 0.05 | **1.0000** | **0.651** | **0.187** |

So the model and Step B do reach a flat top at the real scale — CV improves
21× with no energy lost. Two things had to be right for this, both easy to get
wrong:

- **A target authored on the pupil grid must be rescaled onto the far-field
  grid** by `far_field_size / region` (16× here). `shape_phase_with_frozen_aberration`
  does this, and `ZernikeCoefficientOptimizer` exposes `far_field_size` for it.
- **Step B's own FFT must honour the same padding.** It used to transform at
  `region` regardless, so on a padded run it optimised against a target on a
  different grid than the model far field. It now zero-pads exactly as
  `forward_intensity` does, which keeps Step A and Step B on one grid.

Measuring the result needs the same factor: scoring the shaped spot in a
22-px box *on the 4096-px far field* reports a spurious `EE -> 0.005` collapse,
because the square that was actually produced is 352 px wide. The apparent
"optimizer empties the box" failure was this measurement bug, reproducible at
every learning rate from 0.01 to 0.05.

Cost: a 4096² forward with autograd is ~0.77 s, so a 150-iteration Step B is
~2 min and the runbook's 600 iterations ~8 min per round. Budget accordingly.

## Known aberration injection (Step A ground truth)

On an aberration-free bench Step A has nothing to fit: against real captures the
fitted coefficients only changed the agreement by +0.3 % (noise) and the fit loss
*rose*. To make the Step A result meaningful, the runbook injects a known
aberration that the model never sees:

- `--defocus-rad` (default **1.0**) builds a Noll-4 `(2,0)` defocus over a
  `--zernike-radius` (default 200) px disc and adds it to the **displayed** phase
  only. The stored probe phase, which is what Step A sees, is the bare random
  phase — so the aberration is genuine ground truth.
- `--stage fit` then runs `calibrate_shared_aberration` over the real captures
  and prints injected vs recovered Noll-4 with the error and a 0.3 rad verdict.

This is the check that cannot be done on a flat bench, and it is the one that
makes the A→B iteration measurable rather than self-fulfilling.

## CCD exposure calibration

Peak brightness is **not** linear in exposure (3.0 ms → 154, 2.0 → 111,
1.0 → 55, 0.8 → 49), so extrapolating a linear fit mispredicts badly (it
suggested 1.07 ms for a target of 60; the actual peak there was 47). Calibrate
by interpolation in the measured curve instead:

| exposure | 1.20 ms | **1.10 ms** | 1.00 ms | 0.95 ms | 0.90 ms | 0.85 ms |
|---|---|---|---|---|---|---|
| peak | 63 | **60** | 55 | 52 | 48 | 46 |

**`--exposure-ms 1.1` is the runbook default** and lands the 0-order peak on 60.

Caveat: the laser output drifts. The same 1.0 ms setting measured 63 counts in
one run and 55 in the next (~15 %), so re-check the peak before trusting a long
run, and prefer recording the measured peak with the results.

## Measurement pitfalls that produced a false fault

The panel was never frozen. Three independent mistakes stacked into a
convincing false verdict:

1. **Normalised full-frame L2 is the wrong metric for "did the pattern
   change".** A ~60 px spot moving 63 px changes ~1e-4 of a 5.0-Mpx frame, so
   the metric reads ~5e-5 no matter what the SLM does. It cannot resolve a
   moving spot. Use the 0-order position, or a difference map.
2. **argmax/centroid without smoothing is noise-limited here.** At peak 22–46
   against a mean of 0.26, a single hot pixel wins `argmax`; the *reference*
   centroid wandered 60 px between repeats, so "no motion" was not evidence of
   anything. Box-smooth (5×5) before locating the spot.
3. **A fixed `memory_number=` is a firmware no-op.** `display_data` rotates the
   slot itself and estimates the LCOS flip time when `wait_time_s=None`; passing
   a fixed slot makes the panel hold the old image (the driver warns loudly —
   this is what the vendor tool avoids by using `10 + len(frames)`).

A tilt probe settles it in one shot: a 2π phase ramp over `P` panel px moves
the focal spot by `7600/P` camera px, independent of diffraction efficiency.

> ⚠️ **2026-10-01 修正**：此处原写 `132940/P`，与本文档自己的另外三处
> （`7600/P`、`S = 7615`、L471-473 的 `shift_px ≈ 7600 / P`）**相差 17.5×**。
> `report/slm/bench_calibration_20261001.md:42-43` 已正式裁定 **7400–7600 为实测支持值、
> `132940` 是错的**，本文档即引用来源之一。按 `132940` 读会得到下表两行荒谬的
> "expected shift"（16 px 实测 vs 277 px 预期）。**已按正确值统一。**

| panel ramp period P | 480 | 120 |
|---|---|---|
| expected shift (naive f=125 mm) | 277 px | 1108 px |
| **measured** | **16 px** | **63 px** |

Two real geometric facts fall out, both of which the old code got wrong:

- **The panel and camera axes are swapped 90°.** A tilt along panel-x moves the
  spot in camera-**y**, and vice versa. (The `spgd-square --rotation-search`
  option exists for the same reason.)
- **The focal scale is ~17.5× smaller than the naive `λf/d_slm` estimate**, so
  the empirical relation is `shift_px ≈ 7600 / P`. Do not derive panel↔camera
  geometry from first principles on this bench; measure it.

`slm-diagnose` passes all three steps once it is pointed at the right camera
(`--camera-type daheng`; it defaults to `miicam` and fails to open). Its
`modulate` step is the strong one: `set_grayscale` 0→1023 gives bucket
6362→12408, `rel_spread=0.540`, versus 0.9 % when I measured it badly.

### The camera is NOT the noise source

`get_auto_exposure_state()` reports `enabled: False`, `ExposureTime.get()`
reads back exactly 3000 µs, and 12 successive grabs at a fixed setting drift
only **4.4 % in peak / 0.7 % in total sum**. So exposure is manual and stable.

What *does* move the numbers between scripts: **the SLM retains whatever pattern
was last displayed.** A "flat" reference captured before any write is really the
previous run's speckle — that is why the same 3 ms setting measured peak 100 in
one run and 23 in the next. Always write flat first, and treat any cross-run
brightness comparison as meaningless without it.

## Hardware fault log

> **⚠️ RETRACTED — the "frozen panel" verdict below was a MEASUREMENT artifact,
> not a hardware fault.** The panel was modulating correctly the whole time.
> See "Measurement pitfalls that produced a false fault" for the three
> independent mistakes. Kept only because the failure modes are worth knowing.

### 2026-09-30 — after SLM restart: digital side healthy, optics dead

Post-restart re-test localises the fault precisely.

| Check | Result | Meaning |
|---|---|---|
| Displayed memory slot across 4 writes | advances 3 → 4 → 5 → 6 | driver writes **and** displays correctly |
| Coarse blazed grating, 160 px period (first order ~47 px away) | peak/mean 101.8 → 103.7; **max outside ±60 px stays 2**; in-box energy 0.0092 → 0.0091 | no first-order beam produced at all |
| Half-field π step | 0-order peak 52 → 37, position unchanged | only the documented amplitude coupling, no diffraction |
| Full-panel random phase | frame rel-diff 1e-5 … 1e-4 | indistinguishable from flat |

A *full-panel* grating producing literally zero first-order light leaves only
two possibilities, and they are indistinguishable from software:

- **(a) The LCOS analog drive (VOA bias) is absent or faulty.** Phase
  modulation comes from the analog voltage on the VOA electrodes; the digital
  memory write and slot display can both be perfectly healthy while no phase is
  applied optically.
- **(b) The light reaching the camera never traverses the addressed panel
  area.** A pattern is visible on the panel, but the beam forming the measured
  0-order does not pass through it.

Separating them needs a look at the beam path and the panel's drive/interlock
state, not another script.

Note for anyone repeating this: the 0-order peak *does* change with the average
gray level (52 → 35 for a random phase). That is the amplitude coupling
AGENTS.md documents for 1064 nm, **not** evidence of modulation — the
normalised *pattern* and the frame sum are what stay put, and those are the
numbers to judge.

### 2026-09-30 — SLM powered but LCOS not modulating (re-confirmed, brighter laser)

Re-checked after the bench was powered up again, with the laser output raised
(0-order peak 142 counts at 3 ms, previously 85–100). The fault is unchanged:
flat / full-area blazed grating / full-panel random phase still give frames that
agree to **1e-5 – 2e-5**, and the 0-order never moves. So the laser and camera
are fine; the LCOS is still not being driven.

### 2026-09-30 — SLM powered but LCOS not modulating

The bench powered up and the devices enumerate, but **no phase pattern reaches
the panel**, so no shaping run can produce a result — this affects the repo's
existing `slm-gsnet` / `spgd-square` / `slm-pib` runners too, not just the
model-in-the-loop test.

What works:

- `Santec(slm_number=1, wavelength=1064, video_mode=0)` opens; serial `22030108`;
  status OK; `Panel_Res` 1920×1200; 2π = 993 gray; `_max_gray` = 993 (mapping
  correct, so this is not a gray-scale configuration problem).
- Camera `FJB24112222`/`FJB24112232` delivers frames with the beam present:
  peak 84–100 counts at 3 ms, total frame sum ≈ 3.43e6, 0-order stable at
  **(669, 1027)**.

What fails — the two documented checks from the AGENTS.md chain:

| Check | Expectation | Measured |
|---|---|---|
| ① Freeze: flat / full-area blazed grating / top-half / bottom-half / full-panel random phase | frames must differ; a 64 µm-period grating must split the 0-order ~945 px away | pairwise relative difference **2e-5 … 1.3e-4**; 0-order never moves |
| ② Modulation: grayscale sweep 0…993 | 0-order bucket should show a ~993-gray period, **>15%** relative spread | **0.86%** relative spread |

Conclusion: the LCOS is not being driven (panel power / driver board / cable /
LUT state), not a phase-mapping or software fault. Recovery order:

1. `python src/ao_shaping/main.py slm-diagnose` — the canonical tool, which
   automates ① / ② / ③.
2. If ① passes but ② fails, check the wavelength/LUT configuration; if ① also
   fails, power-cycle the panel and re-seat the drive cable.
3. Do **not** interpret a flat response as "the model cannot shape": with a
   frozen panel every phase looks identical, so any optimisation silently
   converges to nothing while reporting plausible-looking metrics.

Once ① and ② pass, re-run the geometry calibration — the records captured
during the fault (`data/model_in_loop_hw/calibration_records.npz`, saved
2026-09-30) are **not** usable: the probe phases never reached the panel, so
they carry no pupil-phase information.

## Offline geometry calibrator

`calibrate_bench_geometry(records, region=..., far_field_size=...)` derives the
disc radius, beam waist and far-field scale from saved captures, so a hardware
session does not have to repeat the manual defocus sweep. It searches
(disc radius × beam waist) jointly and scores by **Pearson correlation** of the
speckle pattern, then returns a `BenchGeometry` with
`model_side_for_camera(camera_px)` for sizing a target square.

```python
geo = calibrate_bench_geometry(records, region=256, far_field_size=4096)
side_model_px = geo.model_side_for_camera(40)   # 40 camera px -> model px
```

Two things it deliberately does **not** do:

- **It does not match spot size.** The FWHM of a speckle cut through the global
  max is a noisy, non-monotonic function of beam waist; a bisection on it
  converged to ~60 % error. Correlation peaks only when the pupil mapping is
  right, so it is the sole score. FWHM is reported as a diagnostic.
- **It does not return a plausible answer when the data cannot discriminate.**
  Below a correlation of 0.75 — or when the winner sits on the edge of the
  search grid — it logs a "NOT trustworthy" warning naming the cause.

### Which captures can calibrate the geometry

Not all of them. On
`data/debug/slm_pib_shape_20260926_175334` (733 records, commanded `|_c|` up to
5 rad) the calibration returns correlation **0.55–0.57** and saturates at the
edge of the search grid, and correctly refuses to call that a result. The
reason is physical: those runs build their Zernike over **radius 600 px** on the
panel, while the beam only illuminates ~192 px, so the pupil phase inside the
beam is a smooth, nearly flat quadratic. With no speckle to correlate, the score
is dominated by the spot envelope and simply rewards the smoothest
(broadest) pupil.

Use captures whose phase **varies across the illuminated area** — free-form /
random pupil phases, or Zernike sets built with a radius matched to the beam.
The `slm_gsnet` free-form runs are the right shape of data but store only the
576-element free-form vector `_c`, not the panel phase, so the phase has to be
reconstructed before they can be used.

## Environment note

OOPAO is installed **only in `.venv`** (editable, `libs/OOPAO`), and the import
name is uppercase `OOPAO`; the scoop Python that some tooling uses does not have
it. This OOPAO version has **no `OOPAO.fft`** module and warns that
`Telescope` is no longer the master class — the repo's
`ao_shaping.drivers.sim.oopao_backend` handles that through the
`_oopao_compat` shim, and `propagate()` works with `AO_OOPAO_BACKEND=1`. The
shim covers ASM/turbulence, not focal-plane FFT, so it does not replace the
model's far-field propagation.

## 与 slm-model-in-loop 的关系

slm-model-in-loop 复用了本文件所述的台架几何自校准流程：通过 calibrate_bench_geometry 拟合几何参数，并在运行时检查实测远场与模型预测的相关性（--min-geometry-correlation）。当相关性低于阈值时，命令会拒绝整形并提示改用 slm-gs-refine。此外，它在 Step A 中交替拟合单个共享 Zernike 像差以校正正向模型，再在 Step B 中冻结该像差并合成目标光斑相位，从而避免两步互相抵消。
