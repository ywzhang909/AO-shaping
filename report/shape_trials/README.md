# Shape objective trials (2026-10-09)

All completed optimization runs below used `--debug` and have their own
PNG/PKL/JSON record set under the named directory. Epoch 0 is the initial
phase; later rows are measurements from the search. These short runs are
diagnostic and do not establish a uniform square beam.

## Confirmed outcome and open issues

The Santec SLM #1 and Daheng CCD were online for these trials, and a settled
flat / 64-px blaze / flat replay showed a reversible spot movement. The 100-px
uniform square **was not achieved**. The best interleaved replay in this set
improved the in-box CV to about 2.46 (from about 3.0 for the lower-order
comparison), but only 164 of 10,000 target pixels exceeded half peak and the
spot FWHM was about 20 x 11 px. It remained a compact peak. The devices are
offline now; no further live test is claimed here.

Confirmed measurement defects and fixes:

- Daheng's `OldestFirst` stream buffer returned stale frames after an SLM phase
  change. A phase-step replay showed the first four frames could still represent
  the previous phase. The driver now sets and verifies `NewestOnly` before
  streaming, with an offline SDK-mock test.
- The initial SLM-PIB image used one frame while SPGD evaluations used
  `--n-eval-frames`. It now uses the same requested frame count, covered by a
  production-engine simulation test.
- Multi-frame camera means were quantized back to integer counts. This made
  low-count background pixels disappear and changed in-box energy from roughly
  0.665 (single frame) to 0.805 (three-frame truncated mean) on a fixed flat
  phase. Daheng, MiiCam and FFmpeg now retain fractional means; fixed-frame
  tests compare batch means with the same frames read individually. These tests
  establish arithmetic equivalence, not live optical stability.
- The auto-exposure probe could observe a phase left by the previous run. The
  runner now displays a flat SLM phase before probing; a mock test checks the
  device and probe order. Debug records now include the initial full camera
  frame and the shape weights needed to interpret each trial.

Shape-method results from controlled replays:

- Adding Pearson weight 1 improved CV slightly but lost about one percentage
  point of in-box energy; its score under the common default metric was worse
  than the unweighted baseline. It remains opt-in (`--w-pearson`).
- The staged shape schedule gave no clear advantage in the paired 24-epoch
  replay. Stage weights now stay fixed within a sign pair and best-phase
  selection uses a fixed score scale across stages.
- `--log-uniformity` improved CV and Pearson slightly in the paired short run,
  with a small energy cost. Extending that configuration to 100 epochs improved
  CV further (about 2.69) while retaining approximately 0.808 in-box energy.
- Raising `--n-max` from 8 to 15 gave the clearest visible shape change in a
  short run, at an energy cost of about 0.017. Extending the n_max=15 run to
  100 epochs did not improve the paired replay over its 24-epoch result.

The remaining constraint is the control basis and optical setup: low-order
Zernike modes mainly move and broaden a compact spot and have not filled the
100-px square. The next hardware session should compare a freeform/GS square
phase against the Zernike baseline at the same fixed exposure and frame count,
using interleaved flat/candidate replay and the existing debug PNG/PKL/JSON
record format. Recheck `N`-frame versus N single-frame measurements live after
the fractional-mean fix before comparing new metrics with this table.

The hardware trials in this table predate the 2026-10-09 camera averaging fix.
Previously, `get_numpy_image(n_sample=N)` truncated each pixel mean back to an
integer, while N separate single-frame reads retained each raw count. This
especially suppressed low-count background pixels. Metrics from different
`n_eval_frames` settings in these old records are therefore not directly
comparable. New multi-frame reads retain fractional counts as float32.

| Trial | Changed factor | Observation |
| --- | --- | --- |
| `baseline_rmse/` | `rmse`, simulator, 3 epochs | Initial CLI baseline. |
| `hardware_shape_baseline/` | `shape`, Daheng, 8 epochs | Peak 255/255 in every saved frame; in-box CV about 3.16; Pearson about 0.265. No clear improvement. |
| `hardware_device_verification/` | Flat SLM before exposure probe, Daheng, 3 epochs | SLM and CCD opened and completed the run, but frame peak was 1/255 throughout. This verifies device communication, not optical shaping. |
| `device_modulate_diagnosis/` | Live full-frame camera and SLM pattern/gray checks, 10 ms | On 2026-10-09, SLM #1 (`22030108`) and Daheng CCD (`FJB24112232`) both opened. A 2592x1944 full frame had a peak of 93/255 at (674,1027), so the earlier dark cropped frames do not establish loss of light. Flat/grating/flat had peaks 93/94/94; flat-to-grating mean absolute difference (0.168) was no larger than flat-to-flat (0.170). The `slm-diagnose --step modulate` gray sweep read back 0..1023 correctly, but bucket spread was 9.4%, below its 15% response threshold. `diagnose_modulate.npz` records its metrics. The SLM was returned to flat after diagnosis. Optical modulation is not established, so no valid hardware shape comparison follows from these frames. |
| `device_freeze_diagnosis/` | Existing grating self-check at 10 ms | Reported PASS, but the flat frame was saturated at 255/255. Its different-frame predicate alone cannot distinguish pattern response from changing illumination. |
| `device_abba_diagnosis/` | 3 ms flat/grating alternation, raw full frames | Peaks moved among y=793, 909, and 1029; even successive flat frames differed. Full-frame PNG/NPZ/JSON are saved. The switching transient made this comparison inconclusive. |
| `device_flat_stability/` | One flat SLM phase, 20 full frames at 1 ms | Peak remained 153-157/255 at x=672-673, y=1027-1029. Same-phase short-term camera capture is stable. Raw NPZ and JSON are saved. |
| `device_settled_modulation/` | Flat / 64-px blaze / flat at 1 ms, 2 s settling and 8 discarded frames | Peak moved from (672,1028) to (672,909) under the grating, then returned to (672,1027) under flat. No saturation. Full-frame PNG/NPZ/JSON establish reversible SLM influence on the CCD spot; the earlier 10 ms null result was insufficient. |
| `device_phase_step_newest/` | Daheng stream mode changed from `OldestFirst` to `NewestOnly`, followed by a phase step at 1 ms | Before the driver fix, the first four frames after a phase switch still showed the old phase; with `NewestOnly`, the first frame responds. Raw phase-step PNGs and JSON are saved. This removes a direct source of incorrect SPGD gradients. |
| `hardware_shape_newest_baseline/` | `main.py slm-pib spgd`, `shape`, fixed 1 ms, 3-frame evaluations, 24 epochs after `NewestOnly` | Complete `--debug` PNG/PKL/JSON. The recorded initial frame had CV 3.144, Pearson 0.253, and in-box energy 0.661. The best recorded epoch 24 had CV 2.939, Pearson 0.283, and energy 0.794. The unusually low epoch-0 energy did not reproduce in the controlled replay below and must not be counted as optimization gain. |
| `device_phase_replay_newest/` | Recorded phases 0/1/0/24/0 alternated at fixed 1 ms and 0.5 s phase settling | Standard debug PNG/PKL/JSON include raw full frames. Flat-phase energy was 0.803/0.805/0.807, CV 3.247/3.258/3.257, Pearson 0.259/0.258/0.259. Epoch 1 was worse than adjacent flats by the shape score (-1.2004 vs about -1.1985). Epoch 24 reproducibly improved CV to 3.041 and Pearson to 0.275, with energy 0.796 and no saturated pixels. This is a modest causal improvement, not a uniform square spot. |
| `device_flat_fast_replay/` | Six flat-phase frames at 1 ms with one frame per read and no extra settling | Standard debug PNG/PKL/JSON. After the first capture, in-box energy stayed at 0.665–0.666 and crop median at 1, matching the old epoch-0 records. With 3-frame averaging, the same flat phase had energy about 0.805 and median 0. The old initial acquisition used `CAM_SAMPLE_ITER=1` while each SPGD evaluation used `--n-eval-frames 3`; integer averaging changes the low-count background and metric denominator. The first fast capture (energy 0.582) is a separate startup transient. |
| `hardware_shape_initial_sampling_fixed/` | `main.py slm-pib spgd`, 1 epoch, fixed 1 ms, 3-frame evaluations after initial-capture fix | Complete `--debug` PNG/PKL/JSON. Epoch 0 now has energy 0.800, median 0, sum 58,873; epoch 1 has energy 0.785, median 0, sum 56,928. The spurious 0.66-to-0.79 jump is gone; the candidate did not beat flat and the SLM was left flat. |
| `hardware_shape_pearson1_newest/` | Same 24-epoch settings as `hardware_shape_newest_baseline/`, only `--w-pearson 1` added | Complete `--debug` PNG/PKL/JSON. This run preceded the initial-capture fix, so its recorded epoch 0 has the same biased 1-frame energy (0.660). The final in-run shape score is on a different scale because of the added Pearson term; compare replay metrics instead. |
| `device_phase_replay_pearson1/` | Recorded Pearson-weighted phases 0/1/0/24/0 replayed with 3-frame averages and 0.5 s settling | Standard debug PNG/PKL/JSON. Interleaved flats had CV 3.223–3.246, Pearson 0.259–0.261 and energy 0.804–0.806. Epoch 24 had CV 2.937, Pearson 0.284 and energy 0.797. This is slightly better than the separate zero-Pearson best-phase replay, but a paired comparison after the initial-capture fix is still needed to attribute the difference to `w-pearson`. |
| `hardware_shape_sampling_fixed_baseline/` | Corrected 3-frame baseline, fixed 1 ms, 24 epochs | Complete `--debug` PNG/PKL/JSON. Epoch-0 energy 0.809, median background 0; best canonical `shape` was -1.156 at epoch 19. |
| `hardware_shape_sampling_fixed_pearson1/` | Matched run with only `--w-pearson 1` changed | Complete `--debug` PNG/PKL/JSON. Epoch-0 energy 0.803, median background 0; best own score at epoch 14. Compare physical metrics in the paired replay, since the objective scores use different weights. |
| `device_phase_replay_paired/` | Interleaved flat / baseline epoch 19 / flat / Pearson epoch 14 / flat, repeated in reverse order | Standard debug PNG/PKL/JSON. Baseline best CV 3.011/3.016, Pearson 0.2766/0.2764, energy 0.789/0.791. Pearson-weighted best CV 2.984/2.984, Pearson 0.2780/0.2782, energy 0.779/0.781. Pearson weighting slightly flattens the spot but loses about 1 percentage point of in-box energy; its score under the common default metric is lower than the unweighted baseline. |
| `hardware_shape_sampling_fixed_schedule/` | Matched run with only `--shape-schedule` changed | Complete `--debug` PNG/PKL/JSON. Best canonical `shape` was -1.175 at epoch 8. The stage-dependent `J` is not comparable across epochs or with the fixed-weight trials. |
| `device_phase_replay_schedule/` | Interleaved flat / baseline epoch 19 / flat / schedule epoch 8 / flat, repeated in reverse order | Standard debug PNG/PKL/JSON. Both candidates have CV about 3.00–3.02 and Pearson about 0.278; the scheduled candidate has about 0.802 in-box energy versus 0.811 for the fixed-weight baseline. No clear schedule benefit at 24 epochs. |
| `device_shape_noise_floor/` | Ten fixed-flat reads, 3 frames each, 1 ms exposure | Standard debug PNG/PKL/JSON. Sample standard deviations: CV 0.0089, Pearson 0.00072, in-box energy 0.00144, default `shape` score 0.00230. These are within-session noise floors, not a bound on between-run optical drift. |
| `hardware_shape_sampling_fixed_log_uniformity/` | Matched 24-epoch run with only `--log-uniformity` changed | Complete `--debug` PNG/PKL/JSON. Best log-weighted score at epoch 18. The score scale differs from the bounded-CV baseline; use the replay comparison below. |
| `device_phase_replay_log_uniformity/` | Interleaved bounded-CV baseline epoch 19 and log-uniformity epoch 18 | Standard debug PNG/PKL/JSON. Log-uniformity CV 2.955–2.966 and Pearson 0.281–0.282 versus bounded-CV baseline CV 3.031–3.038 and Pearson 0.276. In-box energy falls about 0.005; common default shape score improves about 0.005. |
| `hardware_shape_log_uniformity_100ep/` | Same log-uniformity settings extended from 24 to 100 epochs | Complete `--debug` PNG/PKL/JSON. Best at epoch 68; in-run energy 0.799 and peak 115. |
| `device_phase_replay_log_100ep/` | Interleaved 100-epoch log best (epoch 68) with 24-epoch bounded-CV baseline best (epoch 19) | Standard debug PNG/PKL/JSON. Long log run CV 2.689–2.690, Pearson 0.3083, energy 0.8083–0.8087; short baseline CV 3.043–3.050, Pearson 0.275, energy 0.8070–0.8086. The longer run improves shape while retaining essentially the same box energy. The spot remains centrally peaked in the saved PNG. |
| `hardware_shape_log_nmax15_24ep/` | Same 24-epoch log-uniformity settings, `--n-max` raised from 8 to 15 | Complete `--debug` PNG/PKL/JSON. Best at epoch 22; in-run energy 0.792, peak 100. This changes both basis coverage and total random perturbation norm. |
| `device_phase_replay_nmax15/` | Interleaved n_max=8 log best (epoch 18) and n_max=15 log best (epoch 22) | Standard debug PNG/PKL/JSON. n_max=15 CV 2.471–2.472, Pearson 0.331 and energy 0.792–0.794; n_max=8 CV 2.981–2.983, Pearson 0.281 and energy 0.809–0.811. Higher order helps visibly but costs about 0.017 box energy in this short run. |
| `hardware_shape_log_nmax15_100ep/` | Same n_max=15 log-uniformity settings extended to 100 epochs | Complete `--debug` PNG/PKL/JSON. Best at epoch 59, with in-run energy 0.776 and peak 104. The in-run best score is below the 24-epoch high-order run; duration alone did not give continued improvement. |
| `device_phase_replay_nmax15_100ep/` | Interleaved n_max=15 short best (epoch 22) and long best (epoch 59) | Standard debug PNG/PKL/JSON. Short best CV 2.461–2.467, Pearson 0.331–0.332; long best CV 2.483–2.494, Pearson 0.328–0.330. Both keep about 0.79–0.80 in-box energy. In the 100x100 target box, the short best lights only 164/10,000 pixels above half peak; its FWHM is about 20x11 px. It is still a compact spot, not a square beam. |
| `hardware_rmse_1ms/` | `main.py slm-pib spgd`, rmse, fixed 1 ms, 3-frame reads, 8 epochs | Complete `--debug` PNG/PKL/JSON. ROI CV began at 3.10 and ended at 3.26; no best objective improvement over the initial phase. |
| `hardware_shape_1ms/` | Same run settings, objective changed to shape | Complete `--debug` PNG/PKL/JSON, including the initial full frame. In-run CV fell from 3.11 to 1.71 at epoch 8, but peak fell from 156 to 49 while coefficient norm was only 0.0058 rad. This is not yet evidence of a causal shaping improvement. |
| `hardware_shape_best_replay/` | Recorded best epoch-4 phase alternated with flat, fixed 1 ms | Five full-frame PNGs/NPZ/JSON. Flat CV itself drifted 3.29 -> 2.98 -> 2.84; best-phase CV was 3.06 and 3.11 between those flats. The in-run improvement did not survive this causal replay. A constant background subtraction did not remove the flat-phase trend. |
| `sim_shape_before_pearson/` | `shape`, simulator seed 42, 8 epochs | At epoch 8: CV 14.00, Pearson 0.0623. Simulator's signed read noise can make the reported energy ratio exceed 1. |
| `sim_shape_clipped/` | Clip negative pixels in the metric only | Rejected: positive read noise then raised the denominator; reported energy fell to about 0.3. Code change was removed. Raw record retained. |
| `sim_shape_pearson_1/` | Same simulator seed and parameters as the preceding unmodified `shape` run, plus `--w-pearson 1` | At epoch 8: CV 13.87, Pearson 0.0626. Direction is favorable but the change is too small to establish useful shaping. The new weight defaults to 0. |
| `sim_shape_size7_fixed/` | 7-px target, fixed weights, seed 42, 8 epochs | Best canonical shape score stayed at the flat-phase value, -0.2746. |
| `sim_shape_size7_scheduled/` | Same 7-px trial plus `--shape-schedule` | Best canonical shape score also stayed at -0.2746. `J` changed with the stage weights, while the `shape` column remained comparable across epochs. No shaping benefit established. |

The two Pearson simulator runs used `--cam-type sim --seed 42 -c max
--objective shape --target-shape square --target-size 100 --delta 0.05
--lr 1 --n-max 8 --epochs 8`. The shared simulator was reset with seed 42
before each invocation. The early hardware baseline used `--auto-exposure` and
the same shaping/search settings. Current live checks confirm optical response,
and the later hardware runs use non-saturated fixed 1 ms exposure. The initial
energy discrepancy came from using 1 frame for the baseline but 3 frames for
SPGD evaluations; the optimizer now uses `config.n_eval_frames` in both places.
The paired comparisons above cover the tested shape weights and schedule. A
uniform 100-px square still needs a broader control basis and a new hardware
comparison once both devices are online.

## Document ideas and next comparisons

`D:\Projects\TIFO\AO-shaping\docs\SLM远程光斑整形.docx` proposes separate
Pearson/CV/out-of-box-energy terms, stage weights, repeated-frame noise
calibration, corner-background subtraction before normalization, geometric
registration, gradient-scale balancing, adaptive perturbation, a trust region,
and backtracking acceptance. The trials above cover the first two scoring
changes and the noise calibration. The 100-px target is still far from uniform:
the best replayed CV is about 2.46. Background/registration and adaptive
search ideas remain to be isolated and tested with the same debug and replay
protocol. The document's example is a 44x33 rectangle with eight modes, so its
numerical settings do not transfer directly to this 100-px square / n_max=8
bench.
