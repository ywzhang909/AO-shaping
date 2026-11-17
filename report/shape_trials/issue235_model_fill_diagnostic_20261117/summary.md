# Issue #235 model-fill diagnostic (2026-11-17)

- Session: `data\debug\square1000_model_diag_20261117_140741`
- Script: `scripts/slm/issue235_model_fill_diagnostic_20261117.py`
- Model floor at 0.1 ms: 5.713 counts (background 0.2049 + bias/exposure 5.508)
- Predicted/measured planes: 1944x2592 native pixels

## Q1 - does the frozen model predict the measured concentration?

Displayed-phase gate (rebuilt panel gray vs archived `panel_gray`): mean |diff| 0 gray levels, max 0; equivalent 0 rad (one LSB = 0.00614 rad).

| frame | role | power within +/-50 px (meas / pred) | pixels above half max (meas / pred) | FWHM x,y meas (px) | FWHM x,y pred (px) | Pearson | cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | A | 0.2571 / 0.007099 | 194 / 990239 | 18.46, 14.59 | 991.3, 998.6 | -0.0101 | 0.0161 |
| 2 | N4+ | 0.2591 / 0.007105 | 193 / 990614 | 18.43, 14.5 | 991.4, 998.8 | -0.0101 | 0.0158 |
| 3 | N4- | 0.2483 / 0.007092 | 202 / 989842 | 19.67, 13.9 | 991.2, 998.3 | -0.0104 | 0.0168 |

Denominators: ``power_*`` is a fraction of each plane's own floor-subtracted signal (measured floor 15 counts, model floor 5.7132 counts).  The measured spot rides on a broad halo that extends past +/-50 px, so its fraction measures relative concentration rather than a compact energy budget, whereas the model's signal is the square itself: 0.9898 of it falls inside the 1015 x 1003 px predicted footprint.

Verdict: **the frozen model does NOT predict the measured point-like concentration: it puts 0.007099 of its signal within +/-50 px and 990239 pixels above half max, versus 0.2571 and 194 measured; whole-plane Pearson is -0.0101.**

## Q2 - which single parameter could reproduce the measurement?

Target (measured anchor 0): power within +/-50 px 0.2571, pixels above half max 194.

| parameter | frozen | required value | factor | pred +/-50 px | pred half-max px | ratio (power, half-max) | reproduces? | plausibility |
|---|---:|---:|---:|---:|---:|---:|---|---|
| phase_scale | 1 x frozen command phase | 0.9 | 0.9 | 0.008479 | 798830 | 0.03298, 4118 | no | refit fitted 0.9992 (fwdmodel_fit_20261117.md, marginal dPearson 0.0019). The required value would sit outside the fitted [0.90, 1.10]. |
| amplitude_width_x_px | 352.2 panel px | 2818 | 8 | 0.185 | 161 | 0.7197, 0.8299 | no | not fitted by the refit (fixed at the checkpoint's (352.25, 132.72)). Values above 960 px no longer fit the panel's half span. |
| sigma_ccd_px | 0 CCD px | 16 | n/a | 0.0104 | 990602 | 0.04044, 5106 | no | refit blur_ccd_px 2.15. Blur only spreads power, so it cannot turn a 1000 px square into a 16 px spot; a negative value is undefined. |
| defocus_diopter | -0.02439 diopter | -1 | n/a | 0.01794 | 28278 | 0.06981, 145.8 | no | checkpoint value -0.0244 diopter (frozen_parameters.json). Extra defocus adds curvature/blur and cannot concentrate the square. |
| far_field_pixel_scale | 1 x frozen CCD px per far-field sample | 0.0142 | 0.0142 | 0.9186 | 230 | 3.513, 1.041 | no | refit scale_px 2.009 CCD px/sample at 2x padding (geometric 1.9698); the frozen model samples 3.94 CCD px/sample at full-panel width. A required factor far below 1 means sub-CCD-pixel sampling. |

Verdict: **no single parameter reproduces the measured power and half-max pixel count inside the scanned ranges.**

## Q3 - does the measured spot imply a full-panel pupil?

Measured spot FWHM: 18.46 px (40.6 um).

One full-panel DFT sample = lambda*f/(N*p) = 8.66e-06 m = 3.94 CCD px; K = lambda*f/(p*p_cam) = 7557 vs calibration 7564 (relative error 0.000937).

| candidate pupil | predicted FWHM (px) |
|---|---:|
| full panel, Gaussian width 960 px | 2.95 |
| full panel, square side 1920 px | 3.49 |
| amplitude_width x = 352 px | 8.04 |
| amplitude_width y = 133 px | 21.3 |

Inverted: Gaussian width 153.5 px, square side 362.7 px.

Measured x-FWHM 18.46 px = 40.6 um.  A full 1920 px square pupil focuses to 3.49 px and a 960 px Gaussian (1/e^2 amplitude radius) to 2.95 px, i.e. 5.3x narrower than the measurement.  Inverting the same relation gives an effective square side of 363 px (Gaussian width 153 px), 19% of the panel -- not a full panel.  The frozen ``amplitude_width`` (352.2, 132.7) px predicts 8.04 px (x) and 21.34 px (y); the measured 18.46 px sits 2.30x above the x prediction and 1.16x below the y one, so the spot scale is that of a few-hundred-px pupil.  The analytic scale 7556.8 agrees with the stored calibration 7563.9 to 0.09%.

Verdict: **the measured spot does NOT imply a full panel: a 1920 px square pupil would focus to 3.49 px FWHM, 5.3x narrower than the measured 18.46 px; the measured spot inverts to an effective square side of 362.7 px (19% of the panel) or a Gaussian width of 153.5 px, the scale of the frozen amplitude_width (352.2, 132.7) px rather than panel-sized illumination.**

## Provenance

- Command: `python scripts/slm/issue235_model_fill_diagnostic_20261117.py --session data/debug/square1000_model_diag_20261117_140741 --out data/debug/issue235_model_fill_20261117`
- Generated (UTC): 2026-11-17T11:39:07.231109+00:00
- All numbers above are keys in the sibling JSON.
