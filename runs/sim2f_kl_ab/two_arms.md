# Two-arm comparison — `iid` vs `kolmogorov` aberration ensemble

Same corpus, same checkpoint, same 24 samples, same seed, same rounds, detector off — **only** the high-order coefficient ensemble differs.

- corpus: `runs/sim2f_fair/corpus`
- checkpoint: `runs/sim2f_fair/train/model_best.pt`
- loop: samples=24, rounds=2, seed=7, use_detector=False, probe_fit=True, refit=True
- baseline reproduced: `runs/sim2f_fair/loop/a_no_noise/eval_loop.json`
- mode: ran ml.sim2f.loop.run_loop for every arm
- git HEAD: `41cc7b7902649a3cc96bff9b764bd359dec9294f`

## 0. Pre-registration (text written before the numbers)

> The ordering does NOT change.  ``flat`` still beats every net arm on the calibrated bench, under both ensembles.

Reasoning: The calibrated 2f twin's dominant optical error is the static defocus (10.0847 mm), not turbulence.  The drawn aberration state is a *second* error on top of it: 0.05-0.45 rad RMS over Noll 4..22 plus 0-6 px of tip/tilt.  Changing the *shape* of that ensemble's high-order part cannot repair the defocus-dominated flat spot, which is what the fair run measured (``net_open`` 1.4588 vs ``flat`` 1.1160); the plausible outcome is therefore that the numbers move and the ordering does not.

Significant outcome: ``flat`` losing the aggregate under the kolmogorov ensemble, or the net-vs-flat family count rising above 0/7 far enough to change the published verdict.  Either would be reported as a significant result and not folded into 'the numbers moved'.

Not evidence: A change smaller than the state-draw noise floor measured by the seed-swap replicates.  The detector is OFF in this arm (``a_no_noise``), so the *only* stochasticity left is which 24 aberration states were drawn -- a different 24 states is the right noise floor, not a rerun of the same 24.

## 1. Conceptual verification — does the corpus carry an aberration state?

**No.** `pairs.npz` holds exactly the 8 fields declared by `ml.sim2f.schema.CACHE_FIELDS`: `kind, phase, seed, split, spot, support, target, teacher_metrics`.

- fields naming an aberration/zernike/state/coefficient: none
- `build_pairs` passes an `aberration=` kwarg: **False** (mentions `draw_aberration`: False, `AberrationConfig`: False); its docstring says the teacher bench is ABERRATION-FREE: True
- the teacher documents an aberration-free bench: **True**; passes an `aberration=` kwarg: **False**
- corpus `meta.json` has an aberration block: None

The state is drawn **inside** the loop, per sample: `loop.run_loop` L442 `states = AberrationConfig() if aberration_cfg is None else aberration_cfg`, L478 `rng = np.random.default_rng([int(cfg.seed), int(index)])`, L479 `state = draw_aberration(bench, states, rng)`. So `aberration_model` cannot reach a label or a weight: **the two arms share corpus and checkpoint**, and the comparison is an evaluation-ensemble one.

## 2. Commands and wall clock

One command; it issues four `loop.run_loop` calls internally.

| # | what | command | wall clock |
|---|---|---|---|
| 1 | whole driver (4 loop runs) | `python runs/sim2f_kl_ab/run_two_arms.py --samples 24 --rounds 2 --seed 7` | **103.3 s** |
| — | internal `run_loop` → `iid` | `runs/sim2f_kl_ab/loop/iid/` | 29.9 s |
| — | internal `run_loop` → `kolmogorov` | `runs/sim2f_kl_ab/loop/kolmogorov/` | 25.1 s |
| — | internal `run_loop` → `iid_seed8` | `runs/sim2f_kl_ab/loop/iid_seed8/` | 23.2 s |
| — | internal `run_loop` → `kolmogorov_seed8` | `runs/sim2f_kl_ab/loop/kolmogorov_seed8/` | 23.6 s |

Per-run clock is measured around `run_loop` alone, so it excludes interpreter start, `ml.sim2f` import and CUDA init — which is why the first run is the slowest and why these do not add up to the driver total. The published `runs/sim2f_fair` figure for one loop run is **83.6 s** as a *whole process*; see `COMMANDS.md` for the like-for-like process measurement.

## 3. Baseline reproduction (iid arm vs the published `a_no_noise`)

max |Δ| over 35 columns = **0.000e+00** (35/35 columns exactly equal); bit-exact: **True**.

| arm | column | published | iid arm | abs diff |
|---|---|---|---|---|
| `flat` | objective | 1.115984360 | 1.115984360 | 0.000e+00 |
| `flat` | l1 | 1.014160986 | 1.014160986 | 0.000e+00 |
| `flat` | ncc | 0.634377393 | 0.634377393 | 0.000e+00 |
| `flat` | ee | 0.592706500 | 0.592706500 | 0.000e+00 |
| `flat` | cv_in_support | 0.878612143 | 0.878612143 | 0.000e+00 |
| `flat` | iou50 | 0.135533004 | 0.135533004 | 0.000e+00 |
| `flat` | peak_ratio | 2618.462856293 | 2618.462856293 | 0.000e+00 |
| `net_open` | objective | 1.458799481 | 1.458799481 | 0.000e+00 |
| `net_open` | l1 | 1.300929000 | 1.300929000 | 0.000e+00 |
| `net_open` | ncc | 0.543675479 | 0.543675479 | 0.000e+00 |
| `net_open` | ee | 0.368518078 | 0.368518078 | 0.000e+00 |
| `net_open` | cv_in_support | 0.913205829 | 0.913205829 | 0.000e+00 |
| `net_open` | iou50 | 0.090308186 | 0.090308186 | 0.000e+00 |
| `net_open` | peak_ratio | 1433.749629339 | 1433.749629339 | 0.000e+00 |
| `net_probe` | objective | 1.415102440 | 1.415102440 | 0.000e+00 |
| `net_probe` | l1 | 1.263964387 | 1.263964387 | 0.000e+00 |
| `net_probe` | ncc | 0.595424300 | 0.595424300 | 0.000e+00 |
| `net_probe` | ee | 0.395447837 | 0.395447837 | 0.000e+00 |
| `net_probe` | cv_in_support | 0.726892818 | 0.726892818 | 0.000e+00 |
| `net_probe` | iou50 | 0.106727855 | 0.106727855 | 0.000e+00 |
| `net_probe` | peak_ratio | 1418.221382141 | 1418.221382141 | 0.000e+00 |
| `net_closed` | objective | 1.411641240 | 1.411641240 | 0.000e+00 |
| `net_closed` | l1 | 1.260528229 | 1.260528229 | 0.000e+00 |
| `net_closed` | ncc | 0.604072606 | 0.604072606 | 0.000e+00 |
| `net_closed` | ee | 0.395547922 | 0.395547922 | 0.000e+00 |
| `net_closed` | cv_in_support | 0.696537893 | 0.696537893 | 0.000e+00 |
| `net_closed` | iou50 | 0.106786458 | 0.106786458 | 0.000e+00 |
| `net_closed` | peak_ratio | 1407.053639730 | 1407.053639730 | 0.000e+00 |
| `oracle` | objective | 1.415341655 | 1.415341655 | 0.000e+00 |
| `oracle` | l1 | 1.263268861 | 1.263268861 | 0.000e+00 |
| `oracle` | ncc | 0.602574863 | 0.602574863 | 0.000e+00 |
| `oracle` | ee | 0.391708816 | 0.391708816 | 0.000e+00 |
| `oracle` | cv_in_support | 0.682112221 | 0.682112221 | 0.000e+00 |
| `oracle` | iou50 | 0.123121340 | 0.123121340 | 0.000e+00 |
| `oracle` | peak_ratio | 1365.873495738 | 1365.873495738 | 0.000e+00 |

config match: loop={'samples': True, 'rounds': True, 'probe_fit': True, 'refit': True, 'use_detector': True, 'seed': True}, aberration block shared keys=True (extra keys in this run: ['aberration_model', 'covariance_path']), metadata={'n_samples': True, 'columns': True, 'arms': True}
(paths differ by design: ['cache', 'checkpoint'])

## 4. Aggregate, both arms

### iid

| arm | objective | ncc | l1 | ee | cv_in_support | iou50 |
|---|---|---|---|---|---|---|
| `flat` | 1.115984 | 0.634377 | 1.014161 | 0.592706 | 0.8786 | 0.1355 |
| `net_open` | 1.458799 | 0.543675 | 1.300929 | 0.368518 | 0.9132 | 0.0903 |
| `net_probe` | 1.415102 | 0.595424 | 1.263964 | 0.395448 | 0.7269 | 0.1067 |
| `net_closed` | 1.411641 | 0.604073 | 1.260528 | 0.395548 | 0.6965 | 0.1068 |
| `oracle` | 1.415342 | 0.602575 | 1.263269 | 0.391709 | 0.6821 | 0.1231 |

rank by objective (lower is better): `flat < net_closed < net_probe < oracle < net_open` — `flat` best: **True**

### kolmogorov

| arm | objective | ncc | l1 | ee | cv_in_support | iou50 |
|---|---|---|---|---|---|---|
| `flat` | 1.008276 | 0.680853 | 0.922273 | 0.655988 | 0.8190 | 0.1791 |
| `net_open` | 1.401812 | 0.575040 | 1.249656 | 0.391374 | 0.7663 | 0.1195 |
| `net_probe` | 1.392542 | 0.612578 | 1.242596 | 0.400216 | 0.6849 | 0.1208 |
| `net_closed` | 1.411238 | 0.605402 | 1.259525 | 0.393146 | 0.6908 | 0.1077 |
| `oracle` | 1.415342 | 0.602575 | 1.263269 | 0.391709 | 0.6821 | 0.1231 |

rank by objective (lower is better): `flat < net_probe < net_open < net_closed < oracle` — `flat` best: **True**

## 5. Per family — does `net` beat `flat`?

### iid

| family | n | flat | net_open | net_probe | net_closed | oracle | net_open < flat |
|---|---|---|---|---|---|---|---|
| gaussian | 3 | 0.7446 | 1.0372 | 0.8943 | 0.8765 | 0.9185 | no |
| square | 6 | 0.8509 | 1.4006 | 1.3951 | 1.4099 | 1.3947 | no |
| disc | 4 | 0.9734 | 1.3756 | 1.3335 | 1.3393 | 1.3309 | no |
| rect | 2 | 1.1292 | 1.2923 | 1.3315 | 1.3481 | 1.3654 | no |
| ring | 2 | 1.1597 | 1.6764 | 1.7736 | 1.7762 | 1.7879 | no |
| line | 4 | 1.3612 | 1.6106 | 1.5646 | 1.5429 | 1.5453 | no |
| two_spot | 3 | 1.8428 | 1.8712 | 1.7022 | 1.6711 | 1.6779 | no |

- `net_open` beats `flat` in **0/7** families: none
- `net_probe` beats `flat` in **1/7** families: two_spot
- `net_closed` beats `flat` in **1/7** families: two_spot
- `oracle` beats `flat` in **1/7** families: two_spot

### kolmogorov

| family | n | flat | net_open | net_probe | net_closed | oracle | net_open < flat |
|---|---|---|---|---|---|---|---|
| gaussian | 3 | 0.5841 | 0.9726 | 0.8454 | 0.8854 | 0.9185 | no |
| disc | 4 | 0.8458 | 1.4133 | 1.3139 | 1.3297 | 1.3309 | no |
| square | 6 | 0.8461 | 1.3677 | 1.3879 | 1.4050 | 1.3947 | no |
| rect | 2 | 1.0484 | 1.3333 | 1.3342 | 1.3485 | 1.3654 | no |
| line | 4 | 1.2306 | 1.5227 | 1.5506 | 1.5439 | 1.5453 | no |
| ring | 2 | 1.3315 | 1.8032 | 1.7863 | 1.7825 | 1.7879 | no |
| two_spot | 3 | 1.4349 | 1.5008 | 1.6194 | 1.6757 | 1.6779 | no |

- `net_open` beats `flat` in **0/7** families: none
- `net_probe` beats `flat` in **0/7** families: none
- `net_closed` beats `flat` in **0/7** families: none
- `oracle` beats `flat` in **0/7** families: none

## 6. What the kolmogorov ensemble actually changed

### Pairing (the comparison is paired, not independent)

- sample order identical across arms: **True** (24 samples, 7 families: square, disc, line, two_spot, rect, gaussian, ring)
- `tilt_px` identical: **True** (max |Δ| 0.000e+00)
- `illum_center_px` identical: **True** (max |Δ| 0.000e+00)
- `illum_width_scale` identical: **True** (max |Δ| 0.000e+00)
- total high-order RMS identical per sample (to float64 round-off): **True** (mean iid 0.2123 rad vs kolmogorov 0.2123 rad, max |Δ| 1.110e-16)
- only `coeffs[2:]` differs: max |Δ| over all 21 coefficients = **1.594e+00**, over the 19 high-order ones = 1.594e+00
- **measured pairing proof**: the `oracle` arm cancels the true coefficients exactly, so its objective can only differ by round-off. It differs by **1.19e-07** between the two arms, against 1.08e+00 for the estimated arms — i.e. the runs are paired to float32 precision and only the ensemble separated them.

### Per-mode RMS ladder — the 24 states the loop actually drew

| Noll | iid RMS (rad) | kolmogorov RMS (rad) | ratio ko/iid |
|---|---|---|---|
| 4 | 0.21391 | 0.40675 | 1.901 |
| 5 | 0.22997 | 0.62042 | 2.698 |
| 6 | 0.19310 | 0.41176 | 2.132 |
| 7 | 0.20985 | 0.18925 | 0.902 |
| 8 | 0.25639 | 0.22861 | 0.892 |
| 9 | 0.28135 | 0.23948 | 0.851 |
| 10 | 0.17976 | 0.21239 | 1.182 |
| 11 | 0.34951 | 0.14847 | 0.425 |
| 12 | 0.21417 | 0.21706 | 1.014 |
| 13 | 0.21197 | 0.14632 | 0.690 |
| 14 | 0.20597 | 0.13072 | 0.635 |
| 15 | 0.23122 | 0.14840 | 0.642 |
| 16 | 0.15005 | 0.08618 | 0.574 |
| 17 | 0.27674 | 0.10877 | 0.393 |
| 18 | 0.15972 | 0.10168 | 0.637 |
| 19 | 0.25286 | 0.10878 | 0.430 |
| 20 | 0.35932 | 0.08618 | 0.240 |
| 21 | 0.23563 | 0.11952 | 0.507 |
| 22 | 0.26191 | 0.06714 | 0.256 |

- `iid`: strongest off-diagonal **r = -0.681** (Noll 9–19), median |r| 0.186, mean |r| 0.216, n = 24
  - top 3 pairs: 9–19 -0.68, 8–20 +0.63, 16–21 -0.62
- `kolmogorov`: strongest off-diagonal **r = +0.706** (Noll 8–21), median |r| 0.198, mean |r| 0.230, n = 24
  - top 3 pairs: 8–21 +0.71, 9–19 -0.70, 5–13 -0.61

  ⚠️ **At n = 24 this correlation block is noise, not structure.** With 171 pairs the sampling sd of a single r is ≈ 0.21, so a max |r| near 0.7 (and a median |r| near 0.19) is what an *uncorrelated* ensemble produces — which is exactly why the iid reference row looks as structured as the kolmogorov one here. Read the 4096-draw block below for the real correlation structure.

### Per-mode RMS ladder — a large draw (4096) for the stable estimate

| Noll | iid RMS (rad) | kolmogorov RMS (rad) | ratio ko/iid |
|---|---|---|---|
| 4 | 0.27098 | 0.51669 | 1.907 |
| 5 | 0.27975 | 0.51183 | 1.830 |
| 6 | 0.27459 | 0.51595 | 1.879 |
| 7 | 0.27793 | 0.29562 | 1.064 |
| 8 | 0.27945 | 0.30135 | 1.078 |
| 9 | 0.27518 | 0.29495 | 1.072 |
| 10 | 0.27785 | 0.29567 | 1.064 |
| 11 | 0.27755 | 0.19069 | 0.687 |
| 12 | 0.27375 | 0.18959 | 0.693 |
| 13 | 0.27753 | 0.18597 | 0.670 |
| 14 | 0.27917 | 0.19665 | 0.704 |
| 15 | 0.27617 | 0.19626 | 0.711 |
| 16 | 0.27248 | 0.14024 | 0.515 |
| 17 | 0.26973 | 0.13955 | 0.517 |
| 18 | 0.27661 | 0.13761 | 0.497 |
| 19 | 0.28291 | 0.13252 | 0.468 |
| 20 | 0.27864 | 0.13921 | 0.500 |
| 21 | 0.27986 | 0.13808 | 0.493 |
| 22 | 0.27995 | 0.10248 | 0.366 |

- `iid`: strongest off-diagonal **r = +0.056** (Noll 5–15), median |r| 0.013, mean |r| 0.015, n = 4096
  - top 3 pairs: 5–15 +0.06, 11–19 -0.05, 6–16 -0.04
- `kolmogorov`: strongest off-diagonal **r = -0.612** (Noll 11–22), median |r| 0.013, mean |r| 0.038, n = 4096
  - top 3 pairs: 11–22 -0.61, 7–17 -0.55, 8–16 -0.54

### Bingham shrinkage — does the documented cost show up?

Fixed total RMS puts every draw on a sphere, and an anisotropic shape on a sphere is squeezed toward isotropy. The documented cost (`test_end_to_end_shape_survives_the_rms_rescale`) is low order **0.85×** and the highest mode **1.27×** of its *asset* share, with the defocus-to-Noll-22 span falling 35× → 24×.

| quantity | documented | measured here |
|---|---|---|
| low-order mean share ratio (Noll 4–6) | 0.85 | **0.900** |
| highest mode (Noll 22) share ratio | 1.27 | **1.256** |
| defocus-to-Noll-22 span, asset | 35× | 35.2× |
| defocus-to-Noll-22 span, realised | 24× | **25.4×** |
| extreme share ratio | — | min 0.890 (Noll 5), max 1.297 (Noll 16) |

Measured on an independent draw of n = 4096 from the driver's own `np.random.default_rng([seed, index])` scheme, so this is not the test's numbers being echoed back.

## 7. Noise floor — a different 24 states from the same ensemble

Same ensemble, `LoopConfig.seed=8` instead of 7 — the detector is off, so this is pure state-draw variation.

| ensemble | arm | \|Δobjective\| | \|Δncc\| | \|Δ net_open−flat gap\| |
|---|---|---|---|---|
| iid | `flat` | 0.1334 | 0.0562 | - |
| iid | `net_open` | 0.0232 | 0.0017 | 0.1103 |
| iid | `net_probe` | 0.0852 | 0.0188 | 0.0482 |
| iid | `net_closed` | 0.1088 | 0.0406 | 0.0247 |
| iid | `oracle` | 0.0990 | 0.0377 | 0.0344 |
| kolmogorov | `flat` | 0.2163 | 0.0738 | - |
| kolmogorov | `net_open` | 0.1463 | 0.0836 | 0.0700 |
| kolmogorov | `net_probe` | 0.1122 | 0.0409 | 0.1041 |
| kolmogorov | `net_closed` | 0.1138 | 0.0435 | 0.1025 |
| kolmogorov | `oracle` | 0.0990 | 0.0377 | 0.1173 |

max |Δobjective| over arms: iid 0.1334, kolmogorov 0.2163; max |Δ gap to flat|: iid 0.1103, kolmogorov 0.1173.

## 8. Verdict

### The pre-registered question: does `flat` still beat the net arms?

**No — the net-vs-flat ordering did not change, and the pre-registered expectation held.** `flat` is the best arm on the aggregate under both ensembles (iid 1.1160; kolmogorov 1.0083), `net_open` beats it in **0/7** families under iid and **0/7** under kolmogorov, and the `net_open` − `flat` gap did not close: +0.3428 → +0.3935.

### Inside the net group: a reshuffle that is not resolvable

The net arms' internal order is `net_closed < net_probe < oracle < net_open` under iid and `net_probe < net_open < net_closed < oracle` under kolmogorov — so it **did** reorder. But the whole net group spans only 0.0472 (iid) / 0.0228 (kolmogorov), while a second draw of 24 states moves a single arm's objective by up to 0.2163. Resolvable: **False**.

The fair run already had this shape (`net_closed` − `net_probe` = 0.0035), so the reshuffle is a property of a 24-state draw, not of the ensemble.

### Did the ensemble move the numbers? (paired, the sensitive test)

Per-sample paired differences, `kolmogorov − iid` (negative = kolmogorov scored a *lower*, better objective). Negative mean with a small p means the ensemble genuinely eases that arm; the pairing is exact.

| arm | mean Δobjective | sd | Cohen's d_z | p (sign flip) | p (sign test) | lower/higher/tied |
|---|---|---|---|---|---|---|
| `flat` | -0.1077 | 0.2217 | -0.49 | 0.0174 | 0.0639 | 17/7/0 |
| `net_open` | -0.0570 | 0.2788 | -0.20 | 0.372 | 0.152 | 16/8/0 |
| `net_probe` | -0.0226 | 0.0622 | -0.36 | 0.0727 | 0.307 | 15/9/0 |
| `net_closed` | -0.0004 | 0.0130 | -0.03 | 0.887 | 0.541 | 14/10/0 |
| `oracle` | -1.49e-08 | 5.63e-08 | — (pairing check) | — | — | 0/0/24 |

Method: exact sign-flip enumeration over 2²⁴ = 16.7 M assignments is impractical in pure Python, so p is Monte-Carlo over 200 000 sign flips (seed 0) using the same statistic as `ml.zernike.eval_stats.sign_flip_pvalue`; the minimum attainable p over 24 pairs is 1.19e-07. The exact binomial sign test is shown alongside; it drops |Δ| ≤ 1e-6 as ties. `oracle` is a pairing check, not a measurement: its differences are float32 round-off.

The pre-registered quantity — the per-sample `net_open − flat` gap — moved by +0.0507 (sd 0.2042, d_z +0.25, sign-flip p = 0.248, sign-test p = 0.541); it widened in 14/24 samples.

Pairing check: `oracle` differs by at most 1.19e-07 between the arms (float32 round-off, since it cancels the true aberration exactly), against 1.08e+00 for the estimated arms.

### Reading

The ensemble changes what the loop must survive, not who wins. Two mechanisms are visible in the measurement and both are consistent with the calibrated twin's dominant error being the static defocus (10.0847 mm): (1) the flat spot itself improves under kolmogorov, because a total-RMS-matched von Kármán draw puts far less energy in high order (Noll 22 sits at 0.37× its iid RMS — see the ladder), and high-order aberration destroys a spot faster per radian² than low order, so matched RMS is not matched difficulty; (2) the amortised phase does not convert that into a win, because it is still commanded onto a bench whose remaining error is the defocus the network was never asked to model.

Scope: this is **this** calibrated bench, **this** corpus and **this** checkpoint. It is not a statement about turbulence in general, and it is not evidence that the network is insensitive to aberration *shape* — it is evidence that on a defocus-dominated bench the shape of the high-order ensemble does not decide the ordering.

