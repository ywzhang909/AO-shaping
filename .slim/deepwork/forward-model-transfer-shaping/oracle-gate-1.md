Illumination amplitude is not the cause — the camera field of view is; the freeform gate is unmeetsable by construction, so the fix is the metric's support, not the threshold.

# Gate 1 — oracle review (session `ses_eacefec36ffe5uky72h`, attempt 1 of 3)

Asked to review the plan for "use transfer learning to fix the forward model, then
differentiably shape arbitrary shapes on the bench". Verdict: the first phase
cannot answer the question it is designed to answer, and the oracle had already
executed it.

## What the oracle found

Fitting a **free 128x128 amplitude** (16384 params) to the 16 training frames of
run `20261019_111941` and scoring the holdouts:

| holdout | control (calibrated symmetric Gaussian) | fitted 128x128 amplitude |
|---|---:|---:|
| TRAIN (16) | 0.785 | 0.926 |
| smooth00 | 0.821 | 0.902 |
| smooth01 | 0.810 | 0.894 |
| zern02 | 0.870 | 0.957 |
| zern03 | 0.757 | 0.896 |
| **free04** | **0.118** | **0.129** |
| **free05** | **0.108** | **0.105** |

Textbook overfitting: train +0.14, smooth +0.09, zernike +0.09, **freeform
+0.01 / -0.00**. Fitting a free amplitude to *one* freeform frame reaches train
corr 0.920 and transfers to the *other* freeform frame at 0.053.

Identifiability answer (Q1): worse than underdetermined. 16 smooth-bandwidth
frames cannot pin 16384 amplitude parameters; and the training family contains
**no freeform-like probe** (`_probe_phase(..., smooth=True)` on hardware,
`slm_model_in_loop.py:2606-2611`), so the training distribution itself is the
problem. Smallest defensible parameterisation if an amplitude model is ever
wanted: 4-6 parameters (two waists, de-centre, rotation), fitted on
*mixed-bandwidth* frames including freeform, holdout = a different seed at the
same bandwidth, fitted on log-amplitude bounded to +-1 e-fold of the calibrated
Gaussian, and rejected unless it beats the control on **all six** holdouts.

## The actual dominant cause: camera field of view

The 300x300 camera frame covers r <= 150 camera px, but the model grid extends to
r = 536 camera px. For a freeform probe only **8.5%** of the modelled energy lands
inside the camera crop (smooth 99.6%, zernike 93.1%). `_to_model_grid`
(`slm_model_in_loop.py:1775-1793`) zero-pads outside, so the holdout correlation
is computed against 91.5% structural zeros.

Achievable ceiling, per family:

```
holdout_04: corr(perfect_model, camera_masked_perfect_model) = 0.4876
holdout_05: corr(perfect_model, camera_masked_perfect_model) = 0.4353
achieved:                                             0.241 / 0.277
```

**A perfectly correct forward model scores 0.44-0.49 on the freeform family and
still fails the 0.5 gate.** The freeform block is a mis-specified metric, not a
mis-specified model. No amount of Step B iterations, exposure or sampling changes
this - consistent with the measured 0.11 -> 0.28 saturation under raising signal.

Two secondary mechanisms:

1. **Speckle registration chaos.** A white-noise pupil phase decorrelates to 0.81
   at a 0.7-SLM-pixel pupil shift and 0.60 at 1.4 px; the same shift leaves a
   smoothed probe at 1.000. The freeform holdout is therefore a sub-pixel
   registration stress test the bench cannot pass, not a model-accuracy test.
2. **Higher-order diffraction loss.** The freeform phase has 0.804 rad/SLM-pixel
   gradient, so zeroth-order efficiency `exp(-sigma^2)` ~ 0.52: ~48% of the light
   leaves the 0th order entirely. Measured in-window flux is 0.077x flat while
   the single-order FFT model predicts 0.036x. No amplitude parameter can express
   this.

Null test: `corr(measured_freeform_04, predicted_freeform_05) = 0.087`. The achieved
0.24 is barely distinguishable from comparing two different speckle realisations.

## Answers to the five questions

1. **P1 well-posedness:** no, and it fails in the worst direction - it flatters.
2. **Degeneracy with Zernike:** separate by *bandwidth*, not statistically.
   Defocus/tilt live below ~0.05 cyc/model-px; freeform probes put 80% of power
   above 0.25 cyc/model-px and the n_orders=10 basis (66 modes) cannot represent
   them at all. Fit illumination on low-bandwidth probes, aberration on
   high-bandwidth ones, never jointly. Moot given the FOV finding.
3. **Is illumination amplitude dominant?** No - the pattern falsifies it. Smooth
   and Zernike probes are insensitive to illumination shape (0.82-0.87) and
   freeform is not, but freeform is equally insensitive to anything else because
   it is simply not observable through a 300-pixel crop.
4. **Sequencing:** P3 before P1 is wasteful; P1 as written is worse than wasteful.
   Reorder: redefine the gate first (offline, from recorded frames), then
   re-measure once on hardware. Either score the correlation inside the camera
   crop only / normalise both arrays to the camera support, or replace `freeform`
   in the gate set with a **band-limited freeform** probe (sigma >= 2 model px,
   verified immune to sub-pixel shift). One hardware run, not two.
5. **Repo rules:** analysis/experiment generators belong in `scripts/`, not
   `src/ao_shaping/tools/` (hardware-interaction tools only). Append-only report
   edits are fine but `scripts/_common/provenance.py::REPORTS` is the single
   mapping source - add the entry and run `scripts/sync_report_provenance.py`;
   hand-editing the provenance block is a hard test failure. Chinese prose must
   follow `docs/terminology_glossary.md`. **Do not touch the 0.5 constant** - the
   finding is that the gate is applied to a family whose ceiling is below it;
   fix the metric's support/normalisation, not the threshold, and say so with the
   ceiling number so nobody reads it as gate-lowering.

## Recommended replacement

- **P1' (offline, ~30 min, no hardware):** from the recorded frames only, compute
  the per-family achievable ceiling and report it beside the achieved value.
  This is the single experiment that answers the question.
- **P1'' (offline):** recompute the gate with the correlation restricted to the
  camera support and show whether the freeform family then passes *for a real
  reason*.
- **P3' (hardware, one run):** re-measure with the corrected gate. Do not add
  amplitude parameters until P1'' implicates amplitude.
- **P2 is dropped** - there is no illumination model to implement.

Open caveat the oracle could not close offline: whether the freeform family was
*intended* as a chaos/registration stress test rather than a model-accuracy gate.
If it was, the 0.5 threshold on it was always unmeetsable by design and the
runbook should say so. The constant stays where it is either way.