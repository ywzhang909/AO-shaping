# Scripts Directory

This directory contains various utility scripts for the AO-Shaping project, including build scripts, simulation scripts, training scripts, and analysis tools.

## Build Scripts

### build.ps1
PowerShell script for building Cython extensions and creating a standalone executable.

**Usage:**
```powershell
.\scripts\build.ps1
```

**What it does:**
1. Changes to the calculators directory
2. Builds Cython extensions using `uv run setup.py build_ext --build-lib ../ao_shaping/algorithm`
3. Creates a standalone executable using Nuitka with specific options for the DM_cam.py script

### build_cython.bat
Batch file for building Cython extensions.

**Usage:**
```cmd
.\scripts\build_cython.bat
```

**What it does:**
1. Changes to the calculators directory
2. Builds Cython extensions using `python setup.py build_ext --build-lib ../ao_shaping/algorithm`
3. Displays confirmation message and pauses

### setup_pytorch.py
Installs a CUDA-version-matched PyTorch into the current environment.

**Usage:**
```bash
python scripts/setup_pytorch.py
```

**What it does:**
- Detects the GPU compute capability via `torch.cuda.get_device_capability(0)`
- Installs torch/torchvision from the matching PyTorch wheel index:
  - sm >= 120 (RTX 50xx) -> `cu130`
  - sm >= 90 (RTX 40xx) -> `cu124`
  - no CUDA -> `cpu`
- Runs `pip install --index https://download.pytorch.org/whl/<tag> torch torchvision`

### setup_pytorch.sh
Bash counterpart of `setup_pytorch.py` for CUDA-version-matched PyTorch installs.

**Usage:**
```bash
bash scripts/setup_pytorch.sh
```

**What it does:**
- Reads the GPU name from `nvidia-smi`
- RTX 50xx -> `cu130`; GTX 10/16 series and RTX 20/30/40 series -> `cu124`; unknown -> `cu124`
- Installs torch/torchvision from the matching PyTorch wheel index
- Verifies the install with a final `import torch` check

## Simulation Scripts

### simulate_atmospheric_comparison.py
Generates comparison images of different atmospheric turbulence conditions.

**Usage:**
```bash
python scripts/simulate_atmospheric_comparison.py
```

**What it does:**
- Simulates weak, moderate, and strong turbulence conditions
- Generates phase screen and spot intensity comparisons
- Saves output to `report/simulation/atmospheric_spot_phase_comparison.png`

### sim_turbulence_analysis.py
Generates analysis figures for turbulence validation.

**Usage:**
```bash
python scripts/sim_turbulence_analysis.py
```

**What it does:**
- Creates visualization of AO system performance under different turbulence levels (none, weak, medium, strong)
- Shows phase screens, focal intensities, and metrics (Cn2, phase std, Strehl, peak ratio)
- Saves output to `artifacts/sim_turbulence_analysis.png`

### generate_sim_visual_report.py
Generates comprehensive visual reports for simulation studies.

**Usage:**
```bash
python scripts/generate_sim_visual_report.py
```

**What it does:**
- Runs turbulence scan across different Cn2 values
- Compares various optimizers (SPGD, Zernike-SPGD, PSO, GA, SA)
- Performs RL rollout in simulated turbulence environment
- Generates multiple plots and saves them in a timestamped directory under `logs/`
- Creates summary JSON and CSV files
- Output includes:
  - Turbulence scan grid (phase screen, focal image, slope magnitude)
  - Turbulence metrics trends
  - Optimizer histories and summary
  - RL rollout metrics and frames

### optimize_pib_reference.py
Reference implementation of PIB (power-in-bucket) optimization on a simulated
turbulence + DM bench.

**Usage:**
```bash
python scripts/optimize_pib_reference.py
```

**What it does:**
- Simulates a 128x128 pupil with a 5 mm aperture, 1550 nm wavelength, 0.5 m
  focal length and a 1000 m propagation distance under Cn2 = 1e-9 turbulence
- Optimizes an 8-actuator DM with SPGD (200 epochs) and heuristic search
  (100 epochs) toward a 0.80 target PIB ratio (seed 42)
- Saves `pib_convergence.png` (dpi 150) and `pib_spots.png` to the current
  working directory

### run_device_less_full.py
Runs the full device-less benchmark suite and GIF generation in one shot.

**Usage:**
```bash
python scripts/run_device_less_full.py
```

**What it does:**
- Runs the canonical `run_benchmark_suite` (9-row grid) covering GS and SPGD
  simulation across square, circle and gaussian targets
- Generates 6 animated GIFs (gs/spgd-sim x square/circle/gaussian) with
  iterations=300, seed=42, max_frames=30
- Saves everything (including `suite_stdout.txt`) under
  `report/benchmarks/device_less_full/`

### fouriergsnet_sim_train.py
Offline scenario-matrix runner for the FourierGSNet pipeline: drives the real
`fouriergsnet_optimize.py` `ShapingSystem`/`FourierGSNetLite`/`closed_loop`
through `SimFourierGSNetEnv` (no hardware) for every TARGET SHAPE × ABERRATION
SET × TURBULENCE LEVEL cell, saving per-step dynamic frames.

**Usage:**
```bash
.venv/bin/python scripts/fouriergsnet_sim_train.py
.venv/bin/python scripts/fouriergsnet_sim_train.py --shapes square,circle,gaussian \
    --aberrations none,defocus,mixed --turbulence off,slow,fast \
    --steps 30 --k-px 512 --no-replay --out data/fouriergsnet_sim
```

**What it does:**
- Builds the full 3×3×3 scenario matrix (defaults: shapes
  `square,circle,gaussian`, aberrations `none,defocus,mixed`, turbulence
  `off,slow,fast`); multi-options accept comma-separated values
- Per cell: seeded `SimFourierGSNetEnv` (peak photons 5e4, read noise 2e,
  calib noise) → ideal calibration + LUT → `ShapingSystem` with the shape's
  target → `FourierGSNetLite` → `adaptive_gs_init(5)` → `closed_loop(steps)`
  with per-step frame capture (far-field + phase)
- Saves per cell: `config.json`, `metrics.csv` (step/uniformity/encircled/
  inference_ms/mse/correlation/efficiency), `final.json`, and
  `frames/<scenario>/far_%04d.npy` (K×K float32) + `phase_%04d.npy` (N×N
  float32); top level: `config.json`, `summary.json` (per-cell ok/final
  metrics/wall time), `README.md`
- Per-cell try/except so one failure doesn't stop the matrix; output goes to
  a timestamped subdir under `--out` (never overwrites existing runs)
- **Known workaround**: `closed_loop` is wrapped in `torch.no_grad()` when
  `--no-replay` (default) — the net's forward output `phi` requires grad
  (`c_hat` from trainable conv layers) and `closed_loop` calls
  `phi.cpu().numpy()` (`fouriergsnet_optimize.py:938`), which raises
  `RuntimeError` in training mode. The replay finetune path needs autograd,
  so the wrapper is conditional on `replay=False`.

| Option | Default | Description |
|--------|---------|-------------|
| `--shapes` | `square,circle,gaussian` | Target shapes (comma separated) |
| `--aberrations` | `none,defocus,mixed` | Aberration presets (comma separated) |
| `--turbulence` | `off,slow,fast` | Turbulence levels (comma separated) |
| `--steps` | `30` | Closed-loop steps per cell |
| `--k-px` | `512` | Far-field grid K (crop side) |
| `--replay/--no-replay` | `--no-replay` | Online replay finetune (needs autograd) |
| `--seed` | `42` | Base seed (per-cell seed derived per scenario) |
| `--device` | `cpu` | `cpu`/`cuda` (auto-fallback to cpu) |
| `--save-frames/--no-save-frames` | on | Save per-step far/phase frames |
| `-o, --out` | `data/fouriergsnet_sim` | Output root (timestamped subdir) |

## Training Scripts

### run_curriculum_mamba_turbulence.py
Runs curriculum learning for Mamba-based turbulence training.

**Usage:**
```bash
python scripts/run_curriculum_mamba_turbulence.py
```

**What it does:**
- Implements a 3-stage curriculum (easy → medium → target) for SAC training in turbulent environments
- Each stage increases difficulty by adjusting turbulence strength (Cn2) and other parameters
- Saves models, logs, and generates summary reports with plots
- Output saved to timestamped directory under `logs/mamba_curriculum_*`

### run_long_sac_experiments.py
Runs long SAC experiments for both static and turbulent environments.

**Usage:**
```bash
python scripts/run_long_sac_experiments.py
```

**What it does:**
- Trains SAC agents on static (Zernike) and turbulent environments with extended timesteps
- Implements curriculum learning for turbulent environments
- Evaluates convergence and generates comprehensive reports
- Output saved to timestamped directory under `logs/sac_converged_report_*`

### sweep_mamba_turbulence_report.py
Performs parameter sweep for Mamba turbulence training and generates reports.

**Usage:**
```bash
python scripts/sweep_mamba_turbulence_report.py
```

**What it does:**
- Sweeps hyperparameters for SAC training in turbulent environments
- Evaluates different configurations (gentle, balanced, tracking, low gain)
- Selects best configuration and compares with static baseline
- Generates detailed reports with plots and recommendations
- Output saved to timestamped directories under `logs/`

### sweep_sac_rl.py
Performs hyperparameter sweep for SAC RL training.

**Usage:**
```bash
python scripts/sweep_sac_rl.py
```

**What it does:**
- Sweeps hyperparameters for both static and turbulent SAC training
- Tests different learning rates, buffer sizes, action scales, etc.
- Recommends best parameters for static and turbulent environments
- Output saved to timestamped directory under `logs/sac_sweep_*`

### sweep_stage3_target.py
Sweeps parameters for stage 3 target in curriculum learning.

**Usage:**
```bash
python scripts/sweep_stage3_target.py
```

**What it does:
- Tests different configurations for the final stage of curriculum learning
- Varies timesteps, learning rates, action scales, and other parameters
- Ranks candidates based on convergence metrics and performance
- Output saved to timestamped directory under `logs/stage3_target_sweep_*`

### train_ml.ps1
PowerShell launcher for ML training that forwards arguments to the canonical
trainer.

**Usage:**
```powershell
.\scripts\train_ml.ps1 [args...]
```

**What it does:**
- Sets `PYTHONPATH` to `src;libs` and switches to the script directory
- Forwards all arguments to `uv run python src/ao_shaping/ml/train.py $args`

## Analysis and Tuning Scripts

### compare_loss_algorithms.py

Runs a **loss × algorithm** comparison matrix on the offline sim bench, with both
tracks in one `ProcessPoolExecutor` because they share no state. Fully offline —
no hardware.

- **Track A (ML / autograd)** — `train_amp` × `AmpTrainConfig.loss`
  (`mse` | `physical`) × torch optimizer. Reports forward-model **val R²**,
  PSNR, best epoch, max |coefficient|.
- **Track B (AO / measurement)** — `optimize_slm_zernike_pib` × objective ×
  search algorithm (8, incl. `spgd`) × SPGD `optimizer_type` (6), on the
  2f-Fourier sim bench. Reports the **measured** spot: `m_pib`, `m_shape`,
  `m_ee`, `m_rmse`, final `J`, coefficient norm.

**Why two tracks and not one:** the new losses in `ml/zernike/losses.py` are
torch terms differentiated through `ZernikeAmpModel`; the AO optimizer is a
measurement-driven SPGD/heuristic search that never sees a torch tensor. They
cannot be substituted into each other, so both are measured on the same outcome
axis (the repo's canonical ROI terms) rather than pretending to be one knob.

**Usage:**
```bash
python scripts/compare_loss_algorithms.py --quick          # smoke: 2 epochs, 4 cells
python scripts/compare_loss_algorithms.py --workers 8 --epochs 20 --seeds 3
python scripts/compare_loss_algorithms.py --ao-algorithms spgd --ao-spgd-optimizers adam,adamod,muno
```

Writes `results.csv`, `report.md` and `summary.png` under `--out`
(default `report/loss_algorithms`). `.csv` is **not committed** (global
`*.csv` ignore rule), so the markdown is the tracked artefact.

**Read the results with this repo's noise rules, or you will invent a winner:**
- Rank on **R²** and the physical ROI terms — **never** on MSE/PSNR/SSIM.
  `normalization="sum"` once reported PSNR 72 dB / SSIM 0.9996 while its R² was
  *worse* than a constant predictor.
- Seeds pin the *search*, not the measurement. The report aggregates
  mean ± spread over the **same** seed list for every cell (paired), because
  single-seed deltas sit inside the noise.
- The sim bench starts **near-optimal** (flat already scores pib ≈ 0.97), so a
  gain over the baseline is not guaranteed and is not asserted.

| Option | Default | Description |
|---|---|---|
| `--out` | `report/loss_algorithms` | Output dir for csv/md/png |
| `--quick` | off | Smoke mode: 2 epochs |
| `--workers` | `min(4, cpu)` | Parallel cells (Windows `spawn`) |
| `--epochs` / `--seeds` | `6` / `2` | Epochs per cell / paired seeds |
| `--ml-loss` | `all` | `all` \| `mse` \| `physical` |
| `--ml-optimizers` | `adam,adamw,sgd` | Track A torch optimizers |
| `--ao-objectives` | `pib,shape,rms_pib` | Track B objectives |
| `--ao-algorithms` | `all` | `all` or comma-separated subset |
| `--ao-spgd-optimizers` | `adam,adamod,muno` | SPGD inner optimizers |
| `--grid` / `--n-max` | `32` / `4` | Model grid / Zernike order |
| `--ao-pop` / `--cam-size` | `6` / `128` | Heuristic pop / sim window |
| `--device` | `cpu` | Track A device |

### eval_pib_hybrid_sim.py
Evaluates PIB hybrid simulation and saves results.

**Usage:**
```bash
python scripts/eval_pib_hybrid_sim.py
```

**What it does:**
- Runs PIB simulation evaluation suite
- Prints summary dataframe to console
- Saves artifacts to `logs/pib_sim_eval/`

### tune_sim_spgd_zernike.py
Tunes parameters for SPGD Zernike optimization.

**Usage:**
```bash
python scripts/tune_sim_spgd_zernike.py
```

**What it does:**
- Performs grid search over SPGD and AdaMOD optimizer parameters
- Evaluates combinations of gamma, delta, beta1, beta2, beta3
- Uses custom scoring function balancing PIB ratio, Strehl, and elapsed time
- Saves results to `report/simulation/sim_spgd_zernike_tuning.json`
- Prints best SPGD and AdaMOD configurations

### visualize_sac_runs.py
Visualizes SAC training runs with plots and metrics.

**Usage:**
```bash
python scripts/visualize_sac_runs.py
```

**What it does:**
- Finds SAC training logs (static, turbulent, focus experiments, sweep results)
- Loads training curves, evaluation data, and performs rollouts
- Generates comprehensive visualizations:
  - Training curves (reward, Strehl, PIB, losses)
  - Evaluation curves (reward, episode length)
  - Rollout metrics and frames
- Saves output to timestamped directory under `logs/sac_visual_report_*`

### process_1300_data.py
Enriches the 1300-channel Micro-DM wiring data with grid and connector
information.

**Usage:**
```bash
python scripts/process_1300_data.py
```

**What it does:**
- Reads `1300-5.xlsx` and `1300路机柜输出线序表(1).xlsx` from
  `docs/micro deformable mirror/docs`
- Maps each channel to its 36x36 grid position, IP group, sequence number,
  connector group and pin number
- Writes `data/1300-5-enriched.xlsx` and `data/1300-5-enriched.csv`
- Prints warnings for rows with missing grid positions

### objective_rep_logging.py

Pure-NumPy drift-rejection helpers for the shaping-objective repeat study. **A
library module — no CLI, no `__main__`.** Imported as a sibling module by
`repeat_shape_objectives.py` through a deferred loader; pure Python + NumPy, so
it needs no hardware.

It exists because of one measurement: at a fixed 2 ms exposure the 0-order
`frame_peak` was read as 180 and then 121 within a minute, with the SLM phase
unchanged. Illumination can fall by ~1/3 on its own, so **the drift — not the
objective — is the dominant residual between repeats.** Without rejecting those
rounds a variant can "win" purely by having been measured while the laser was
brighter.

| API | Purpose |
|---|---|
| `flag_drift_reps(rows, key="frame_peak", max_rel_dev=0.25)` | Splits rows into `(kept, dropped)` around the median `frame_peak`, dropping those deviating more than `max_rel_dev`. Rounds with no usable peak (missing / zero / non-finite) are **kept but tagged** `drift_unknown` rather than silently dropped |
| `median_over(rows, key)` | NaN-safe column median |
| `summarise_logged(rows, ...)` | Reduces the columns over the **kept** rows only |

`repeat_shape_objectives.py` calls it with `DRIFT_MAX_REL_DEV = 0.25`.

### pyarrow_probe.py

Small `pyarrow` diagnostics used by the tests and the `ZernikeControl` debug
panel. **A library module — no CLI, no `__main__`.**

| API | Purpose |
|---|---|
| `pyarrow_version() -> str \| None` | Installed version, `None` if absent |
| `pyarrow_pandas_compat_ok() -> bool` | Whether the `pyarrow`/`pandas` pair is importable together |
| `probe_pyarrow() -> tuple[bool, str]` | One-shot `(ok, message)` pair |
| `pyarrow_diagnostics() -> list[tuple[str, str]]` | The multi-row panel view |

It is deliberately **not** imported at `pattern_controls` module level: the
control page calls `probe_pyarrow()` inline inside `ZernikeControl.render()`, so
a broken pyarrow can never block the module import or the rest of the page from
rendering. The tests monkeypatch the probe to simulate a broken pyarrow without
breaking the real one.

## Micro-DM Diff Analysis Pipeline

Analysis pipeline for per-channel Micro-DM (R50Power) response images. For each
controller IP, one camera image is acquired per DM channel while that channel is
driven at a fixed voltage (the remaining channels at 0 V). Each image is reduced
to a localized diff signal against the channel-`000` (0 V) reference of the same
IP, then visualized either as a merged overlay or as per-IP animated GIFs.

### Quick Start (Recommended)

Use `md_img_pipeline.py` for the complete workflow — it handles per-IP references
automatically and generates all outputs in one command:

```bash
# Complete pipeline for a voltage group
python scripts/md_img_pipeline.py --input data/md_test/md_img-80v
python scripts/md_img_pipeline.py --input data/md_test/md_img-100v

# Skip FFT notch filter (faster, ~30s per IP vs ~2min)
python scripts/md_img_pipeline.py --input data/md_test/md_img-80v --no-notch

# Custom parameters
python scripts/md_img_pipeline.py --input data/md_test/md_img-100v \
    --threshold 12 --scale 0.5 --fps 10

# Re-run GIF/overlay only (skip diff computation)
python scripts/md_img_pipeline.py --input data/md_test/md_img-80v --skip-diff
```

**Output structure:**
```
<output>/
├── diff/                  # Per-IP diff images (1:1 filenames with source)
│   ├── 192.168.0.101/
│   │   ├── 192.168.0.101-001.png     # same name as original input image
│   │   ├── 192.168.0.101-002.png
│   │   ├── centroids.csv             # filename, channel, cx, cy table
│   │   └── ...
│   └── ...
├── overlay/               # Per-IP max aggregation overlays
│   ├── 192.168.0.101_overlay.png
│   └── ...
├── gif/                   # Per-IP animated GIFs
│   ├── 192.168.0.101.gif
│   └── ...
├── global_overlay.png     # Max aggregation across all IPs
└── combined.gif           # All IPs merged with labels
```

**1:1 filename mapping** — diff output images keep the exact original filename
(`192.168.0.101-001.png` → `.../diff/192.168.0.101/192.168.0.101-001.png`), so IP,
channel number and source image correspond one-to-one. Centroid coordinates are
no longer embedded in the filename; they are stored in `centroids.csv` per IP
(columns: `filename, channel, cx, cy`) and looked up for GIF labels.

### Manual Steps (Advanced)

For more control, run each step individually:

```bash
# 1. Diff computation, denoising & centroid (per channel)
python scripts/md_img_diff_centroid.py --ref <ref.png> --input data/md_test/md_img-100v \
    --output data/md_test/md_img-100v_diff --threshold 15

# 2a. Merge analysis: pixel-wise maximum over all diff images
python scripts/md_img_diff_overlay.py --input data/md_test/md_img-100v_diff \
    --output data/md_test/md_img-100v_overlay.png

# 2b. Per-IP animation: 50 per-channel diff PNGs -> one animated GIF per IP
python scripts/md_img_diff_to_gif.py --input data/md_test/md_img-100v_diff_jet \
    --output data/md_test/md_img-100v_gif --scale 0.25 --fps 8
```

### md_img_diff_centroid.py — diff computation, denoising & centroid

The core analysis script. For every image under `--input` (recursively scanned),
it computes the signed difference against a single reference image, denoises by
thresholding, saves the rendered diff, and appends the intensity-weighted
centroid of the dominant dark blob to the output filename.

**Algorithm (per image):**

1. **FFT notch filter** (`--notch`, default ON) — the raw camera frames carry
   fixed-pattern interference fringes with frequency peaks at
   `±(19,4), ±(19,7), ±(6,-2), ±(5,2), ±(2,7)` in fftshifted space (fringe
   period ≈ 141 px along x). A Gaussian notch mask
   (`1 - exp(-d²/2w²)`, with the exact center of each peak zeroed) is applied to
   the spectrum of the **raw reference and image before diffing**, then the
   inverse FFT restores real space. With `width=1.0` the fringe energy is fully
   removed (validated: fringe energy 15.6M → 0) while the localized DM response
   is preserved (validated: the centroid of channel 119-030 stays at
   (1790, 1241)). The mask depends only on the image shape, so it is built once
   and cached per shape (rebuilding costs ~2 s, ≈ 3× the FFT itself). Disable
   with `--no-notch`.

2. **Signed difference** — `diff = reference - image` (float64). A positive diff
   means the image is darker than the reference, i.e. the DM has pushed the spot
   away from its rest position — this is the per-channel response signal.

3. **Threshold denoising** — keep only pixels where `|diff| >= threshold`,
   zeroing everything else. See *Threshold calculation method* below.

4. **Dominant dark-blob centroid** (`dominant_blob_centroid`) — build the mask
   `diff > threshold` (pixels significantly darker than the reference), label
   8-connected components with `scipy.ndimage.label`, keep the **largest**
   component, and compute its intensity-weighted centroid:
   `cx = Σ(x·w)/Σw`, `cy = Σ(y·w)/Σw` where `w = diff` at those pixels. Returns
   `(cx, cy)` in (x, y) order. A whole-image centroid is deliberately NOT used —
   the diff is dominated by noise, so the largest connected blob is the real
   localized per-channel signal.

5. **Colormap rendering** — `--cmap gray` outputs plain grayscale `|diff|`
   (0–255); `--cmap jet` (default) maps values in `[threshold, vmax]` through the
   jet colormap with the background left black, so differences pop as
   blue→cyan→green→yellow→red as `|diff|` grows. `vmax` defaults to the per-image
   max `|diff|` (floored at `threshold + 1`) to use the full jet range; pass
   `--vmax` for a fixed scale comparable across images.

6. **Filename convention** — the centroid is written into the output name:
   `<stem>_cx<X>_cy<Y>.png` (1-decimal floats); images with no pixel above
   threshold are saved as `<stem>_cxNone_cyNone.png` so they stay visible in the
   pipeline instead of failing it.

After processing, the script prints a sanity overview: total count, centroid
x/y range and spread, and the number of distinct centroids per controller
subfolder.

**Threshold calculation method:**

The default threshold is `--threshold 15` (empirically derived, not statistical).
Rationale from the code comments:

- The per-channel DM response is a **localized** signal — at the moved spot the
  `|diff|` values sit at 15–25 gray levels.
- A statistical threshold such as `3σ` of the whole-image diff is useless here:
  it still retains 88–97% of all pixels because the difference image is
  dominated by noise, and the resulting whole-image centroid is meaningless.
- Hence the threshold is chosen empirically just below the observed signal
  floor (15) so that only genuine response pixels survive, and the centroid is
  computed on the **largest connected component** rather than the whole image.

**Usage:**
```bash
python scripts/md_img_diff_centroid.py --ref <reference.png> \
    --input data/md_test/md_img-100v \
    --output data/md_test/md_img-100v_diff \
    --threshold 15 --cmap jet
```

| Option | Default | Description |
|--------|---------|-------------|
| `--ref` | (required) | Reference image path (per-IP channel-000 frame) |
| `--input` | `data/md_test/md_img-100v` | Input directory (recursively scanned) |
| `--output` | `data/md_test/md_img-100v_diff` | Output directory (mirrors input structure) |
| `--threshold` | `15.0` | Dark-blob threshold on diff; empirically the signal sits at \|diff\| 15–25 |
| `--cmap` | `jet` | Output colormap: `jet` (black bg + blue→red enhancement) or `gray` |
| `--vmax` | per-image max | Fixed jet color scale maximum, for comparable scales across images |
| `--notch/--no-notch` | on | FFT-notch raw frames to remove fixed-pattern fringes before diffing |

### md_img_diff_overlay.py — merged analysis (pixel-wise maximum)

Loads **all** diff images under `--input` (recursively), computes the pixel-wise
**maximum** across the stack (`np.maximum`), and saves one merged overlay image.
This answers "which pixels are ever perturbed by any channel" — the union of all
channel responses. Shape-mismatched images are skipped with a warning. The
script prints the found image count, image shape (RGB vs grayscale), the
**coverage** (percentage of pixels with signal, i.e. `> 0` after the max merge)
and the max intensity.

**Usage:**
```bash
python scripts/md_img_diff_overlay.py \
    --input data/md_test/md_img-100v_diff \
    --output data/md_test/md_img-100v_overlay.png
```

### md_img_diff_to_gif.py — per-IP animated GIFs

For every subfolder under `--input` (one per controller IP), combines the
per-channel diff PNGs into an animated GIF, one per IP. Key mechanics:

- **Channel sorting** — the channel number is parsed from the filename with
  `-(\d{3})_cx` (the 3-digit number right after the last dash of the base name);
  frames are sorted by channel number, not by lexicographic order.
- **Centroid labels** — the centroid is parsed from `_cx<X>_cy<Y>` in the
  filename and drawn as `ch NN (cx, cy)` in the top-left corner of each frame
  (Windows system fonts with graceful fallback).
- **Downscaling** — frames are resized by `--scale` (default 0.25, LANCZOS) to
  keep the GIF size reasonable; the label font scales with it.
- **Palette** — each frame is quantized to an adaptive 256-color palette, and the
  GIF is saved with `save_all`, `duration = 1000/fps`, infinite loop and
  `optimize=True`.
- **Fault tolerance** — a failing IP is reported and skipped without aborting the
  rest; the script prints the byte size of each GIF and the total written.

**Usage:**
```bash
python scripts/md_img_diff_to_gif.py \
    --input data/md_test/md_img-100v_diff_jet \
    --output data/md_test/md_img-100v_gif \
    --scale 0.25 --fps 8
```

### md_img_pipeline.py — complete pipeline (recommended)

End-to-end pipeline that orchestrates all steps with proper per-IP reference
handling. For each controller IP, uses the channel-000 image as reference (instead
of a single global reference), then generates diff images, per-IP overlays, per-IP
GIFs, global overlay, and a combined master GIF with IP labels and metadata.

**Key features:**
- **Shared reference** — by default ALL IPs are diffed against the single common
  baseline `192.168.0.101-000.png` (channel-000 of the first controller); override
  with `--ref <path>`. Channel 000 of every IP is also diffed (not skipped), so the
  output keeps a full 000..N correspondence with the source.
- **1:1 filename mapping** — diff images keep the exact source filename; centroids
  stored in `centroids.csv` per IP
- **Complete pipeline** — diff computation → per-IP overlay → per-IP GIF → combined GIF
- **Global overlay** — pixel-wise maximum across all IPs for coverage analysis
- **Combined GIF** — all IPs merged with red IP-index labels (bottom-left) and
  metadata labels (top-left)
- **Fault tolerance** — failing IPs are reported and skipped
- **Skip diff mode** — `--skip-diff` reuses existing diff images for faster GIF/overlay regeneration

**Usage:**
```bash
# Full pipeline
python scripts/md_img_pipeline.py --input data/md_test/md_img-80v

# Custom parameters
python scripts/md_img_pipeline.py --input data/md_test/md_img-100v \
    --threshold 12 --scale 0.5 --fps 10

# Skip FFT notch filter (faster)
python scripts/md_img_pipeline.py --input data/md_test/md_img-80v --no-notch

# Re-run GIF/overlay only
python scripts/md_img_pipeline.py --input data/md_test/md_img-80v --skip-diff
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Input directory with per-IP subfolders |
| `--output` | `<input>_processed` | Output root directory |
| `--threshold` | `15.0` | Dark-blob threshold for diff computation |
| `--cmap` | `jet` | Colormap for diff images (`jet` or `gray`) |
| `--vmax` | per-image max | Fixed jet color scale maximum |
| `--notch/--no-notch` | on | FFT-notch to remove fringes |
| `--scale` | `0.25` | Downscale factor for GIFs |
| `--fps` | `8` | GIF frame rate |
| `--skip-diff` | off | Skip diff computation, use existing diff images |
| `--ref` | first IP's `-000.png` | Shared reference image for ALL IPs |

## Diff-Beam Shaping Pipeline

Capture and analysis scripts for the diff-beam (differentiable beam shaping)
hardware runs. `diff_beam_capture_flat.py` acquires flat-phase reference frames
from the SLM + camera bench; `diff_beam_frame_analysis.py` renders the per-frame
records saved by the `diff-beam` runner.

### diff_beam_capture_flat.py

Captures flat-phase reference frames for the diff-beam bench (needs hardware:
Santec SLM + MiiCam).

**Usage:**
```bash
uv run --group ml python scripts/diff_beam_capture_flat.py --exposure-us 1200 --output data/diff_beam/flat_tiff
```

**What it does:**
- Opens the SLM in memory mode and writes a flat phase to a randomly picked
  memory slot (excluding the currently displayed one)
- Captures `--n-sample` averaged frames at the given exposure with a
  `--capture-timeout` watchdog
- Saves `flat_<exposure_us>us.tiff` and `flat_<exposure_us>us.npy` per sample
- Computes the 0-order spot centroid with four algorithms (argmax center,
  intensity centroid, debg centroid, binary centroid) and prints them

| Option | Default | Description |
|--------|---------|-------------|
| `--slm-number` | `1` | SLM device number |
| `--slm-wavelength` | `1064` | SLM wavelength (nm) |
| `--cam-id` | `0` | MiiCam camera ID |
| `--exposure-us` | `1200` | Camera exposure (us) |
| `--n-sample` | `3` | Averaged frames per capture |
| `--settle-time` | `0.3` | Settle time after SLM write (s) |
| `--capture-timeout` | `30.0` | Watchdog timeout for SLM open + capture (s) |
| `-o, --output` | `data/diff_beam/flat_tiff` | Output directory |

### diff_beam_frame_analysis.py

Offline analysis of a `diff-beam` hardware run directory. **Fully offline** —
reads saved artefacts, no hardware.

**Usage:**
```bash
python scripts/diff_beam_frame_analysis.py --run-dir data/diff_beam/run_<ts>
python scripts/diff_beam_frame_analysis.py --run-dir data/diff_beam/run_<ts> --plot
```

**What it does:**
- Reads `config.json`, `frames/*.npy` and `frame_meta.jsonl` from the run dir
- Renders `frames_overview.png` (frame grid) and `spot_vs_target.png`
  (measured spot vs target square)
- Exits with an error if `config.json` is missing

| Option | Default | Description |
|--------|---------|-------------|
| `--run-dir` | (required) | Run directory produced by the `diff-beam` runner |
| `--plot` | off | Also render per-frame plots |

## Report Generation Scripts

> **Repo rule**: all markdown/illustrated-report **generation** lives in `scripts/`
> (naming `generate_*_report.py`), never in `src/ao_shaping/tools/` (reserved for
> hardware-interaction tools). See `AGENTS.md` anti-patterns.

> **Output location**: reports go to `report/<topic>/` at the **repo root**, not to
> `docs/`. `docs/` is device documentation only (specs, SDK usage, assembly
> manuals, image galleries); a measurement result and the manual for the device it
> came from are different documents with different lifetimes, and interleaving them
> by topic made "is this a manual or a result?" unanswerable from the path. Moved
> on 2026-10-05; every generator's output path moved with it.

> **Provenance**: every report states the script that produced it. Declare the
> mapping once in `scripts/_common/provenance.py::REPORTS`, then run
> `python scripts/sync_report_provenance.py` (idempotent; `--check` only
> verifies). **Do not hand-write the header or `report/README.md`** — they are
> generated, and `tests/ao_shaping/scripts/test_report_provenance.py` fails if
> they drift from the registry. A generator that **overwrites** its own report must
> stamp the header at write time (`insert_header`), or the next run deletes it —
> three writers do this (`TestReport`, the `test_spots_calc` benchmark, and
> `compare_loss_algorithms.py`).

> **Shared analysis helpers**: `scripts/` report generators in the SLM/Zernike
> family delegate measurement/analysis logic to
> `src/ao_shaping/tools/slm/slm_scan_analysis.py` (`outlier_mask`, `clamp_shift`,
> `parabolic_min`, `latest_match`, `group_raw_scan`, `analyze_linearity`,
> `LINEARITY_AMPS`) — scripts keep only figure/markdown rendering.

### generate_zernike_amp_report.py

Generates the **illustrated Chinese report** for the learned Zernike far-field
model: `report/zernike_amp/report.md` + `report/zernike_amp/figures/*.png`.
**Fully offline** — reads only saved artefacts, never opens a camera or SLM.

**Usage:**
```bash
python scripts/generate_zernike_amp_report.py
python scripts/generate_zernike_amp_report.py --no-figures
```

**Inputs** (both produced by other scripts, so the report is regenerable):
- `logs/zernike_amp_sweep.json` ← `scripts/sweep_zernike_models.py` (grouped-CV grids,
  the tuned final comparison, paired per-fold differences, per-objective breakdown)
- `logs/zernike_amp_final/summary.json` ← `ml.zernike.train_amp` per-epoch history
  (the same series the run logged to wandb)
- `logs/zernike_amp_final/compare_epoch*.png` — true-vs-prediction frames, copied in
- `data/hw_index_cache.json` — only to recompute the objective-distribution matrix

**Figures** (9): training curves · physics `(n_max, lr)` joint grid · `n_max`
saturation · U-Net candidate re-verification · per-fold scores · paired differences
· per-objective breakdown · objective-distribution matrix · true-vs-pred.

**Three things this report encodes that are easy to get wrong**, and which the
figures exist to make checkable:

1. **Every interval is a paired per-fold difference**, tested with an exact
   sign-flip permutation test. The between-fold spread of R² is ~0.08 while the
   paired spread is ~0.01, so an unpaired comparison cannot resolve the effects
   being claimed.
2. **The objective-distribution matrix uses the target group's own `SS_tot`** as
   the denominator, peak-normalised to match what the model sees. An earlier
   hand-computed version divided a pixel-only quantity by a sample+pixel variance
   and reported a spurious **negative** R²; the figure caught it.
3. **A figure that fails to render degrades to no image tag**, never to a broken
   markdown link (`test_report_embeds_only_figures_that_were_produced`).

Regeneration is idempotent; section 10 of the report is a table of every
conclusion this project has overturned, with the reason each time.

### sweep_zernike_models.py

Runs and **persists** the model sweeps the Zernike report is drawn from, under the
protocol that has statistical power: leave-one-pickle-out (10 grouped folds).

**Usage:**
```bash
python scripts/sweep_zernike_models.py                 # full, ~30 min on one GPU
python scripts/sweep_zernike_models.py --quick         # 2 folds, smoke only
python scripts/sweep_zernike_models.py --final-only    # reuse grids, redo the tuned block
```

**Why it exists**: the report's numbers must come from a saved artefact rather than
from prose, and the *fair* comparison is easy to get wrong — `physics` and `hybrid`
are swept at the **same** tuned configuration, because the hybrid wraps the physics
model and holding it at the untuned setting would flatter the physics model.

| sweep | axis | note |
|---|---|---|
| `physics_grid` | `(n_max, lr)` jointly | they are **non-additive**: `lr=0.02` alone does nothing at `n_max=15`, yet is best at `n_max=20`. A coordinate sweep lands on the wrong cell. |
| `unet_grid` | `(features, epochs, lr)` | re-verifies a config originally chosen on the underpowered single split |
| `final` | tuned physics + hybrid + unet | the headline table, with per-objective means retained |

Writes `logs/zernike_amp_sweep.json`. The repo has **no sklearn dependency**, so the
folds are built by hand on `str(record.path)`, matching the group key
`_select_records` already uses.

**Related**: `scripts/compare_models_cv.py` is the general-purpose CV harness
(arbitrary models, two protocols, `--analyse` to recompute statistics from saved
folds without retraining); this script is the batch sweep that feeds the report.

### generate_zernike_wfs_report.py

Generates the illustrated **Zernike phase → WFS readout distribution** report
(needs hardware: Santec SLM-200 + Thorlabs WFS).

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_zernike_wfs_report.py
python scripts/generate_zernike_wfs_report.py -o report/slm/zernike_wfs_report
```

**What it does:**
- Loads a series of Zernike patterns (modes / radii / amplitudes) at the
  calibrated SLM shift; a **flat-phase user reference** is created first so every
  readout is the increment relative to flat
- Per case writes two figures: **SLM phase** (radian source | actual displayed
  grayscale pattern with mod 2π + shift) and **WFS readout** (spots / read phase
  map / Zernike distribution / metrics)
- Writes `report.md` + `phase/` + `wfs/` + `data.json` to the output dir

| Option | Default | Description |
|--------|---------|-------------|
| `--slm-number` / `--slm-wavelength` | `1` / `532` | SLM device / wavelength |
| `--wfs-exposure-ms` | `4.0` | WFS exposure (capped at 7 ms) |
| `--shift-x` / `--shift-y` | device config | SLM shift override |
| `--settle-extra-s` | `0.1` | Extra settle beyond the pixel-flip estimate |
| `-o, --output-dir` | `report/slm/zernike_wfs_report` | Output directory |

### generate_zernike_response_matrix_report.py

Generates the illustrated **Zernike response matrix** report
(creation / analysis / detection). **Fully offline** — reads saved artefacts, no
hardware.

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_zernike_response_matrix_report.py
python scripts/generate_zernike_response_matrix_report.py --h5 <path> -o report/slm/<dir>
```

**What it does** (writes `report.md` + `figures/`):
- **Creation**: acquisition metadata (device, shift, Zernike radius, amplitude,
  push-pull cycles/averages, DLL index ordering) + matrix shape
- **Analysis**: response-matrix heatmap with diagonal markers, diagonal
  dominance, singular-value spectrum / condition number, repeat-variance map,
  and per-(mode, radius) linearity (**CV + direction cosine**, the correct
  criterion — a normalised response is *constant* when linear, so slope/R² is
  meaningless)
- **Detection**: outlier diagnosis (`|resp|` vs amplitude per radius — shows the
  WFS Zernike-fit collapse when R ≈ beam radius), offline inverse demo
  (`c = pinv(M) @ w`, residual reduction), and the measured closed-loop
  before/after
- **Centering** (§3.4, only for same-run reports): shift-scan V-curve, raw WFS
  wavefront maps before/after (µm, from `debug_<ts>/closed_loop/iter*`), Zernike
  coefficient bars (before vs after vs applied c), and RMS/PV iteration history
  with best-frame markers — shows **离心 (off-axis) correction** quantification

**Same-run gating**: the scan/closed-loop/centering sections are only rendered
when the report json and the matrix h5 are from the same run — detected via
`pass_count_by_radius` in the h5 `device_config`, or (fallback) present in the
report json **plus** matching SLM serial number (h5 `slm_serial` vs report
`device.slm.serial_number`). This prevents mixing a matrix from a different
calibration with unrelated scan data.

Sources default to the latest `data/zernike_response_matrix/zm_*.h5`,
`data/zernike_correction/report_*.json` and `raw_scan_*.json`; override with
`--h5`, `--scan-report`, `--raw-scan`.

### generate_zernike_linearity_report.py

Generates the **Zernike response linearity** report — when the Zernike
coefficient loaded on the SLM grows, does the WFS-read coefficient grow
proportionally? **Fully offline** — reads saved scan artefacts, no hardware.

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_zernike_linearity_report.py
python scripts/generate_zernike_linearity_report.py -o report/slm/zernike_linearity
```

**What it does** (writes `linearity.md` + `figures/`):
- For every (mode, radius) in the raw scan, takes the WFS coefficient at the
  **same DLL index** `diag = (z₊[m] − z₋[m])/2` and checks it is proportional to
  the SLM amplitude A: `diag/A` constant (CV < 15%), through-origin linear fit
  R² > 0.98, and `diag(A=10)/diag(A=2)` ≈ 5.0 (display only — noisy at A=2)
- The residual baseline `|z₊ + z₋|/2` is the instability proxy: a combination is
  judged on linearity only when its response rises above that floor (SNR < 1.5 →
  `噪声受限`)
- Verdicts: `成比例` / `成比例 (弱耦合)` / `噪声受限` / `不成比例`
- Renders `01_response_vs_amplitude.png` (response vs amplitude with linear fit)
  and `02_ratio_r2.png` (A10/A2 ratio + R² bars, color-coded by verdict)
- `--append-to <md>` appends the section to an existing report (idempotent —
  replaces the old section; figure paths recomputed relative to the target)

Sources default to the latest `data/zernike_correction/raw_scan_*.json` and
`data/zernike_correction/report_*.json`; override with `--raw-scan`, `--report`.

### generate_zernike_farfield_sim_report.py

Generates the illustrated **Zernike far-field spot-morphology** report — a 2f-Fourier numerical simulation of Noll 4–15 (n≤4, 12 modes) at amplitudes 0–2 λ (8 steps), **fully offline** (pure numpy, no hardware). The optical model is identical to `SimPibSystem.far_field()` in `src/ao_shaping/drivers/sim/slm_pib_sim.py`: far field = `I = |FFT(pupil · e^{iφ})|²`, pupil 512×512 (R=256 px), far field 8192×8192 zero-padded, A=0 shared Airy baseline, 0-order located by argmax.

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_zernike_farfield_sim_report.py
```

**What it does** (writes `report.md` + `figures/` + `metrics.csv` to `report/zernike_farfield_sim/`):
- §4 Far-field morphology: 12×8 log-intensity grid (160×160 px crop around the 0-order, Gaussian σ=1.5 px) + per-mode morphology description (defocus→ring, astigmatism→ellipse, coma→tail + peak offset, spherical aberration→three rings, trefoil→three lobes, tetrafoil→four lobes); phase grid verifies the raw-radian linear scaling of `φ = A·2π·Z_j`
- §4.1 Numerical-artifact diagnostic: m=4 azimuthal-harmonic comparison between the old grid (128/64) and new grid (512/256) at r=25/50/75/100 px — quantifies the 4× pupil oversampling suppressing the 4-fold staircasing square stripes (the theoretical 1/64≈18 dB applies to the staircasing aliasing energy; the measured m=4 depends on the radius, with the low-intensity ring-region / far-field grid sampling floor dominating at some radii)
- §5 Metrics: Strehl (peak/peak_Airy) vs amplitude + 0.8 criterion amplitude table, EE50/EE90, FWHM, peak offset (non-zero only for the coma family Noll 7/8)
- `metrics.csv` 96 rows: mode_noll, n, m, name, amp_waves, strehl, fwhm_px, ee50_r_px, ee90_r_px, peak_dx, peak_dy, peak_r_px
- Run time ≈17 min (84 far-field FFTs)

### generate_heuristic_pib_report.py

Benchmarks all 7 heuristic optimizers in `ao_shaping.algorithm` (GA, PSO, SA,
Hill Climbing, Random Search, Cross-Entropy, Differential Evolution) on the PIB
(power-in-bucket) optimization problem. **Fully offline** — pure numpy, no
hardware, using the synthetic landscape from
`ao_shaping.optimizer.wfless.pib_sim_eval.SimLandscape` (dim=4, bounds ±12,
seed 42).

**Usage:**
```powershell
$env:PYTHONPATH = "src;libs"
python scripts/generate_heuristic_pib_report.py
```

**What it does** (writes to `report/heuristic_pib/`):
- Runs each optimizer via the `HeuristicOptimizer.create()` factory with its
  spec config (GA/DE/CEM pop_size=30, PSO n_particles=30, per-algorithm
  iteration budgets) and records the PIB convergence history
- `pib_curves.png` — overlaid PIB iteration curves (log x-axis), with 0.5/0.9
  threshold lines; a dot + `iter N` label marks each curve's first crossing
  of 0.9 PIB, and each legend label shows `max@N` (first iteration reaching
  final max PIB)
- `convergence_speed.png` — grouped bar chart (log y-axis) showing the first
  iteration each algorithm reaches PIB 0.5 / 0.9 / its final maximum, with
  exact iteration values annotated above each bar
- `spot_before_after.png` — 2×4 grid of initial vs best spot renders
  (`landscape.render`) with shared brightness normalization
- `summary_bars.png` — horizontal bar chart of final PIB, sorted descending
- `summary.csv` — `algorithm, final_pib, init_pib, best_x_0..3, n_loads`
- `report.md` — results table (final PIB / improvement / **iters to max,
  ≥ 0.9, ≥ 0.5** / n_loads) plus a per-algorithm basin interpretation (global
  center `[-4.8,-4.2,-4.5,-4.0]` vs local center `[2.5,3.2,2.2,2.8]`)
- **设备加载语义**: 设备一次只能加载一个相位, 1 次设备加载 = 1 次相位加载 = 1 次目标函数 (PIB) 评估 = 1 次迭代 (evals_per_iter=1); 表格与 CSV 中的 n_loads 即设备相位加载次数/迭代数。

### generate_slm_pib_heuristic_hw_report.py

**Real-hardware** counterpart: runs the camera test and then every heuristic on the
physical bench (Santec SLM-200 + Daheng MER2-507 NIR) through
`optimize_slm_zernike_pib`, writing `report/slm_pib_heuristic_hw/` (`report.md`,
`camera_frame.png`, `pib_curves.png`, `convergence_speed.png`, `spot_before_after.png`,
`summary_bars.png`, `summary.csv`).

```powershell
$env:PYTHONPATH = "src;libs"   # libs/ REQUIRED for gxipy (Daheng); without it Daheng is unavailable
python scripts/generate_slm_pib_heuristic_hw_report.py --exposure-ms 3.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--cam-type` / `--cam-id` | `daheng` / `0` | camera backend / id |
| `--exposure-ms` | `3.0` | fixed exposure (this bench: ≤3 ms is safe) |
| `--slm-number` / `--wavelength` | `1` / `1064` | SLM device / wavelength |
| `--n-max` / `--cam-size` | `4` / `250` | Zernike order / ROI window |
| `--seed` | `42` | random seed |
| `--skip-camera-test` | off | skip the camera section |

> ⚠️ **Budget caveat**: each algorithm runs only ~30–50 device loads (one SLM phase
> load ≈ 0.3 s settle), and run-to-run variance (light drift / centre detection) has
> been observed to exceed the algorithm-to-algorithm differences — the "best
> algorithm" label is indicative only. Increase `ALGORITHMS` budgets and repeat runs
> before drawing conclusions.

### generate_pib_bench_report.py

Generates the **offline bench acceptance report** for the SLM PIB pipeline from
saved debug artefacts. **Fully offline** — reads `data/debug` pickles + JSON
sidecars, never opens a camera or SLM, so it can be re-run any time (including
while the instruments are powered down) to refresh the conclusions.

**Usage:**
```bash
python scripts/generate_pib_bench_report.py
python scripts/generate_pib_bench_report.py --root data/debug -o report/slm_pib_bench
python scripts/generate_pib_bench_report.py --no-figures
```

**What it does** (writes `report/slm_pib_bench/report.md` + `figures/`):
1. **Noise floor + SNR per amplitude** from every `summary_snr.json` (recursive
   glob — the artefacts sit one level deeper than the search runs). Reports
   single- **and** multi-mode SNR, and states that the floor is *not* a bench
   constant (three sweeps on one day spanned **21×**).
2. **Gate breakdown** from the `summary_smoke_*.json` robust-SPGD runs:
   applied / noise-gated / fold-gated, i.e. how often the search actually moved.
3. **best vs sustained** improvement per recorded search run. Rows whose `J`
   carries the `1e3` guard sentinel are excluded from the statistics and counted
   in a `guard` column (including them produced `+1.3e9 %` nonsense). The
   best-vs-sustained gap is what separates real optimisation from drift.
4. **Provenance** per row, read from the JSON sidecar, so a run is never
   silently attributed to the wrong objective or camera.

**Measured conclusions (Daheng MER2-507-23GM NIR + Santec SLM-200, 2026-09-29):**
- Guard-firing runs have a **median sustained improvement of −11.3 %** vs **+0.4 %**
  for guard-free runs → *check the guard count before touching `delta`.*
- `delta<0.001` at `n_max=9` (54 DOF) is unusable: multi-mode SNR ≈ 1.3, so
  ~95 % of epochs are gated. `delta≈0.1` is the working range; `0.2` trips the
  brightness-fold guard (45/60 rejected).

### measure_shape_sensitivity.py

> **Now a thin CLI over `ao_shaping.tools.slm.slm_snr_probe`** (see the
> hardware-tools section). It keeps its own device construction, target-size
> derivation and markdown output, but the noise-floor / ΔJ / SNR measurement is
> delegated so this script, the report generator and the hardware-gated test
> cannot drift apart.


Measures the **sensitivity (noise floor)** of the `slm-pib` shaping objective on the
real bench — the check that says whether SPGD can see its own gradient at all
(needs hardware: Santec SLM-200 + camera).

**Usage:**
```powershell
$env:PYTHONPATH = "src;libs"
python scripts/measure_shape_sensitivity.py
python scripts/measure_shape_sensitivity.py --deltas 0.02,0.05,0.1,0.2,0.3
```

**What it does** (writes `sensitivity.json` + `sensitivity.md` (+ `sensitivity.png`)
into `-o/--output`, default `report/slm_pib_heuristic_hw/`):
- **Noise floor**: evaluates the shaping score on `--n-frames` frames of the *same*
  fixed phase and reports its std (`ΔJ_noise`).
- **Signal**: for each `Δa` in `--deltas`, writes `c ± Δa` (first Zernike mode =
  Noll 4 defocus, no tilt), averages `--n-repeat` pairs, and reports
  `ΔJ_signal = |mean(J+) − mean(J−)|` — exactly what SPGD turns into a gradient.
- **Verdict**: `SNR = ΔJ_signal / ΔJ_noise` — `>= 3` strong, `>= 2` usable, `< 2`
  unusable (raise `Δa` toward 0.2 rad, or average more frames per perturbation).

> ⚠️ **Measurement core now shared.** The noise-floor / ΔJ / SNR logic lives in
> `ao_shaping.tools.slm.slm_snr_probe` and this script delegates to it, so it
> also gains the **multi-mode (SPGD-style)** SNR column. Judge `--delta` by
> `multi_snrs` / `usable_deltas()` — the single-mode column over-reports what
> SPGD can resolve (bench, same `delta=0.0005`: 2.25 single vs 1.34 at 54 DOF).

| Option | Default | Description |
|--------|---------|-------------|
| `--cam-type` / `--cam-id` | `daheng` / `0` | camera backend / id |
| `--cam-size` | `320` | camera window (px), must exceed the target long side |
| `--slm-number` / `--wavelength` | `1` / `1064` | SLM device / wavelength |
| `--n-max` | `4` | Zernike order |
| `--exposure-ms` | `0.0` | fixed exposure (`0` = auto-expose to `--target-brightness`) |
| `--n-frames` | `10` | frames for the noise floor |
| `--n-repeat` | `3` | `±Δa` pairs averaged per amplitude |
| `--deltas` | `0.05,0.1,0.2` | perturbation amplitudes (rad, comma separated) |
| `-o, --output` | `report/slm_pib_heuristic_hw` | output directory |

> 🔑 **Measured result (2026-09-20, 3.0 ms / 320×320)**: `ΔJ_noise = 4.0e-4`;
> `Δa = 0.05 rad → SNR 0.27`, **`0.1 rad → SNR 0.69` (unusable)**,
> `0.2 rad → SNR 2.77`. Hence the `slm-pib` **`--delta` default is 0.2 rad** — at
> 0.1 rad the SPGD gradient is dominated by measurement noise, which is why the
> gradient baselines underperform the heuristics on the shaping objective.

### compare_shape_objectives.py

Compares shaping **objective functions** with the algorithm pinned to a single fast
search, on the real bench (needs hardware: Santec SLM-200 + camera).

**Usage:**
```powershell
$env:PYTHONPATH = "src;libs"
python scripts/compare_shape_objectives.py
python scripts/compare_shape_objectives.py --epochs 80 --algorithm sa
```

**What it does** — the ONLY variable is the objective; everything else is fixed:
- algorithm pinned (`--algorithm`, default `sa`, 82 device loads ≈ 70 s per variant);
- one **fixed target ROI** for every run: rectangle anchored at the fixed spot
  centre, short side = `TARGET_BOX_WAIST_FACTOR` × the flat-field spot **waist**
  (2 × w0, measured by `spot_waist_sigma` — NOT the 99 %-encircled radius, which
  the stray halo inflates ~2×);
- each variant is judged by ONE common yardstick inside that box, independent of
  what it optimised: `energy = ΣI[box]/ΣI` (full-frame denominator, so the
  absolute value is small by construction), `CV = std/mean` (LOWER = more
  uniform), `peak = max/mean` (lower = fewer hot spots). The box is placed on the
  spot **located by argmax inside each frame** (the camera can clamp the requested
  raw window, so the spot is not at the geometric centre).

Variants: `shape` default, `shape` uniformity-heavy (`w_uniformity=5, w_peak=0`),
`shape` energy-only (`w_uniformity=0, w_peak=0`), `shape` log-uniformity, `roi_pib`.

**Outputs** (into `-o/--output`, default `report/slm_pib_heuristic_hw/`):
- `objectives/<n>_<slug>_spot.png` — best un-windowed frame per variant with the target box drawn and the yardstick annotated;
- `objectives_summary.png` — CV / energy / peak bars per variant;
- `objectives.csv` — the raw numbers;
- an `<!-- OBJECTIVES_START --> … <!-- OBJECTIVES_END -->` section **appended idempotently** to `--append-to` (default `report/slm_pib_heuristic_hw/report.md`), so re-runs replace rather than duplicate it.

| Option | Default | Description |
|--------|---------|-------------|
| `--algorithm` / `--epochs` | `sa` / `80` | pinned algorithm / iterations |
| `--cam-type` / `--cam-id` / `--cam-size` | `daheng` / `0` / `320` | camera backend / id / window |
| `--exposure-ms` / `--target-brightness` | `0.0` / `180` | `0` = auto-expose |
| `--zoom` | `300` | display zoom box (px) around the spot |
| `--append-to` | `report/slm_pib_heuristic_hw/report.md` | report to append the section to |
| `-o, --output` | `report/slm_pib_heuristic_hw` | output directory |

> ⚠️ **Measured (2026-09-20, fixed SA, fixed ROI = 21 px = 2×waist)**: CV 0.259 (energy-only)
> / 0.263 (`roi_pib`) / 0.273 (`e-5u`) / 0.278 (`log-u`) / 0.302 (`e-2u-0.5pk`), while repeating
> the SAME variant gave CV 0.354 then 0.302 — the between-variant spread is **inside the
> single-run variance**, so no objective can be declared the most uniform from one run.

### explore_delta.py

Sweeps the SPGD perturbation amplitude (`--deltas`) on the real bench by driving
the genuine `slm-pib` runner once per candidate, then recommends the one that
**converges** — not the one with the biggest first-vs-last jump. Needs hardware
(Santec SLM + camera), unless `--analyze-only`.

Two robustness criteria are computed per candidate from the saved recorder history:
- **dec** (`frac_decreasing`) — fraction of epochs in which `J` actually decreased. The main judge.
- **late** (`late_gain`) — improvement from the first 1/3 mean `J` to the last 1/3, in %.

If no candidate clears the thresholds it recommends `None` ("拒绝封王") rather than
crowning the least-bad delta: a sweep that returns "nothing converged yet" is the
honest answer, and it stops you tuning against a noise floor.

`--analyze-only` re-judges the newest existing debug run per delta without touching
hardware — free, and the intended way to re-evaluate after editing the judging
logic. Debug artefacts are always on.

> ⚠️ **Measured (2026-09-30, bench scan)**: `frac_decreasing` stayed ≈0.5 across a
> 1000× range of delta. **delta is not the limiting factor** — the next step is
> ABBA palindromic sampling in the main loop (`slm-pib --abba-sampling`).

| Option | Default | Description |
|---|---|---|
| `--deltas` | `0.02,0.05,0.1` | Comma-separated perturbation amplitudes (rad) |
| `--epochs` | `200` | Epochs per candidate |
| `--objective` | `pearson` | Shaping objective (`pearson` / `shape` / `roi_pib` / ...) |
| `--n-max` | `9` | Max Zernike radial order |
| `--lr` | `0.5` | SPGD learning rate |
| `--cam-type` / `--cam-id` | `daheng` / `0` | Camera backend (`daheng` / `miicam`) / device id |
| `--cam-size` | `320` | ROI window (px) |
| `--exposure-ms` | `1.2` | Camera exposure (ms) |
| `--zernike-radius` | `480.0` | Aperture (px) |
| `--target-size` | `50.0` | Target size (px) |
| `--target-shape` | `square` | Target shape |
| `--out` | `report/slm_pib_bench/delta_scan.md` | Markdown summary path |
| `--analyze-only` | off | No hardware; re-judge the newest existing run per delta |

### repeat_shape_objectives.py

Repeats the objective comparison N times and ranks the variants by **median**
uniformity, because a single run cannot separate them: repeating the *same*
variant gave CV 0.354 then 0.302 — the whole between-variant spread (0.259–0.302)
sits **inside** the single-run variance.

Every objective variant is run `--repeats` times with the algorithm pinned to one
fast search, and each repeat is scored with the same common yardstick inside a
**fixed** target ROI (2× the spot waist, fixed centre, spot located by `argmax`
in each frame):
- `energy` = ΣI[box] / ΣI (full-frame denominator)
- `CV` = std/mean (lower = more uniform)
- `peak` = max/mean (lower = fewer hot spots)

Ranking uses the **median**; the per-repeat min/max is reported as the spread so
overlap between variants stays visible.

Drift handling is delegated to `objective_rep_logging.py` (`flag_drift_reps`),
because illumination drift — not the objective — dominates the residual between
repeats.

Outputs into `report/slm_pib_heuristic_hw/`:
- `objectives_repeats.csv` — one row per (variant, repeat), plus the medians
- `objectives_repeats.png` — median CV per variant with repeat min/max error bars
  (+ energy / peak panels)
- an `<!-- OBJECTIVES_REPEATS_START -->` … `<!-- OBJECTIVES_REPEATS_END -->`
  section **appended idempotently** to `--append-to`, carrying the median table
  and naming the most uniform objective — but only when the winner's spread does
  not overlap the runner-up's

Needs hardware (default camera backend `daheng`).

| Option | Default | Description |
|---|---|---|
| `--repeats` | `3` | Repeats per variant |
| `--algorithm` | `sa` | Pinned algorithm |
| `--epochs` | `80` | Epochs per run |
| `--cam-type` / `--cam-id` / `--cam-size` | `daheng` / `0` / `320` | Camera backend / id / window |
| `--exposure-ms` | `0.0` | `0` = auto-expose |
| `--target-brightness` | `180.0` | Auto-exposure target brightness |
| `--slm-number` / `--wavelength` | `1` / `1064` | SLM device / wavelength |
| `--n-max` | `4` | Zernike order |
| `--seed` | `42` | Random seed |
| `-o, --output` | `report/slm_pib_heuristic_hw` | Output directory |
| `--append-to` | `report/slm_pib_heuristic_hw/report.md` | Report to append the section to |

### generate_slm_pib_online_report.py

Generates the offline acceptance report for the **`AO_RUN_HARDWARE=1` online
regression suite**, reading `data/debug/slm_pib_online/<stamp>/` as written by the
suite. **Fully offline** — the instruments may be powered down.

It answers three questions:
1. **Is the delta above the noise floor?** SNR per perturbation delta
   (`ΔJ_signal / ΔJ_noise`), with bands at `SNR_STRONG = 2.5` and `SNR_USABLE = 2.0`.
2. **Did the gates actually let the search move?** A per-epoch timeline of
   `applied` / `fold` / `noise` gating, i.e. how many epochs were really updated
   versus rejected.
3. **Is the improvement the optimiser's or the room's?** A `J` / `max_brt`
   trajectory per epoch, annotated with the environment-drift correlation —
   separating shaping gain from illumination drift.

Outputs to `report/slm_pib_online/`:
- `report.md` — SNR verdict table, gate-observability table, J-trajectory
  env-drift diagnosis, per-run sections
- `figures/snr_by_delta.png` — SNR per delta with unusable / usable / strong bands
- `figures/gate_timeline_<tag>.png` — per-epoch `applied` / `fold` / `noise` timeline
- `figures/j_trajectory_<tag>.png` — `J` and `max_brt` vs epoch, drift correlation annotated

**Recorder row contract** (why the gate timeline is trustworthy): row 0 is the
init baseline and carries no `_gate`; for the rest `applied + stalled + 1 == rows`;
and `_gate` holds `applied` / `fold` / `noise`. Older records fall back to
inferring `_c` stagnation, which **over-counts `applied`** — the report labels
those runs accordingly.

| Option | Default | Description |
|---|---|---|
| `--root` | `data/debug/slm_pib_online` | Root of the online-suite artefacts |
| `-o, --out` | `report/slm_pib_online` | Output directory |

### generate_slm_pib_rms_pib_report.py

Generates the offline report for the **`slm-pib rms_pib` hardware matrix** from
the debug artefacts written by `slm-pib --debug` (the per-run HDF5 `/scalars`
columns plus the runner's summary PNG), for every
`data/debug/slm_pib_rms_pib_*` run. **Fully offline** — reads saved artefacts,
never opens a camera or SLM.

The key rendering detail: the per-algorithm curves plot **tracked historical
best** (`best_rms_pib`), not the raw per-epoch objective. `rms_pib` is an
*energy-guard* objective — when an evaluation perturbs the far field so hard that
the ROI loses more than `--max-roi-energy-loss` of its reference energy, the
optimiser abandons that sample and records a sentinel `J = rms_pib = -999.x`.
Plotting raw `J` would put a −999 spike in every curve; tracking the best keeps
the history finite and monotone, and the guard-rejected row count is reported
alongside so the guard activity stays visible.

Outputs to `report/slm_pib_rms_pib_hw/`:
- `figures/matrix_best_curves.png` — per-algorithm `best_rms_pib` evolution
- `figures/summary_bars.png` — final `best_rms_pib` per algorithm, sorted descending
- `figures/run_<algo>_<stamp>.png` — copies of the runner's own summary sketch
- `report.md` — comparison table (best rms_pib / guard-rejected rows / dynamic
  weights / exposure) + per-run sections

| Option | Default | Description |
|---|---|---|
| `--debug-root` | `data/debug` | Root containing the `slm_pib_rms_pib_*` artefact dirs |
| `--max-runs` | `5` | How many runs to render (newest first) |
| `-o, --output` | `report/slm_pib_rms_pib_hw` | Output directory |

### generate_beam_shaping_benchmark_report.py

Regenerates the **authoritative 9-cell beam-shaping benchmark** (3 algorithms x 3
shapes) into `report/benchmarks/device_less_full/`: `beam_shaping_benchmark_metrics.md`
+ `.csv`, `suite_stdout.txt`, and 6 evolution GIFs under `gif/`. **Fully offline**
-- pure numpy/PIL, ~100 s.

```bash
python scripts/generate_beam_shaping_benchmark_report.py
```

This is the report writer for
`algorithm/signal_processing/beam_shaping_benchmark.py` and it lives here rather
than there because of the AGENTS.md anti-pattern: *report generation MUST live in
`scripts/`*. The benchmark module now only computes -- `run_benchmark` and
`run_benchmark_suite` no longer take an `output_dir` at all -- while the three
helpers that are not I/O (`build_gif_frames`, `to_dataframe`, `HPRINT_KEYS`) stay
with the producer, since frame construction and the returned DataFrame's column
contract are computation, not serialisation.

**Read `area_met` before comparing algorithms.** Measured: gs passes 3/3,
backprop 1/3, spgd-sim **0/3** -- `spgd-sim`'s measured area collapses to ~1 px
with CV 16-34 device-less, so its uniformity numbers are meaningless. The report
header repeats this so the table cannot be misread on its own.

The `.csv` siblings are written but **not committed**: the repo has a global
`*.csv` ignore rule and no CSV under `report/` is tracked. The markdown is the
tracked artefact.

`run_device_less_full.py` is kept as a thin forwarder to this script, because
docs referenced it.

### generate_beam_shaping_papers_report.py

Runs a closed-loop SLM far-field beam-shaping **simulation bench** and compares
the literature methods on one identical optical model + target, so the comparison
is like-for-like instead of across papers. **Fully offline** — needs Python 3.13
+ torch, no hardware. No CLI arguments.

Outputs to `report/beam_shaping/papers/`:
- `figures/<method>_<stamp>.png` — far-field intensity per method
- `figures/target_<stamp>.png` — the target pattern
- `beam_shaping_papers.md` — the report (metric table + figure links)
- `results_<stamp>.json` — all metrics as JSON

### generate_cython_optimizer_report.py

Runs the Cython optimizer benchmark (`src.calculators.benchmark`, reading
`src/calculators/benchmark_results.json`) and emits the markdown performance
comparison with tables + analysis. **Fully offline** — no hardware, no CLI
arguments.

**Output**: `report/benchmarks/cython_optimizer_performance.md`

```powershell
$env:PYTHONPATH = "src;libs"
python scripts/generate_cython_optimizer_report.py
```

### generate_strehl_benchmark_report.py

Benchmarks the 7 heuristic optimizers in `ao_shaping.algorithm` (GA, PSO, SA,
Hill Climbing, Random Search, Cross-Entropy, Differential Evolution) **and SPGD**
on a common offline **Strehl** objective: correcting atmospheric turbulence in
the physical simulation `TraditionalAOSystem` (DM influence functions + Fourier
focal plane). **Fully offline** — pure numpy, no hardware, landscape from
`ao_shaping.optimizer.wfless.strehl_sim_eval.StrehlLandscape` (dim=64, bounds
±1, seed 42, turbulence screen fixed). SPGD participates here as the
gradient-based reference that the PIB benchmark leaves out (it is not a
`HeuristicOptimizer` algorithm).

**Usage:**
```powershell
$env:PYTHONPATH = "src;libs"
python scripts/generate_strehl_benchmark_report.py              # n_grid=256 (~20 min)
python scripts/generate_strehl_benchmark_report.py --n-grid 128 # fast smoke (~5 min)
```

**What it does** (writes to `report/strehl_benchmark/`):
- Runs the 7 heuristic optimizers via the `HeuristicOptimizer.create()` factory
  with the same spec configs and load budgets as the PIB benchmark (GA/DE/CEM
  pop_size=30, PSO n_particles=30, per-algorithm iteration budgets), plus SPGD
  (fixed gain gamma=2.0, delta=0.1, momentum beta1=0.9, 4000 steps x 2 loads =
  8000 loads) driven by the canonical `spgd_gradient()` helper; objective =
  Strehl (simulator definition `clip(peak/_ideal_peak, 0, 1)`, so the plateau
  at 1.0 is the metric's saturation, not a bug)
- `strehl_curves.png` — overlaid Strehl curves vs device loads (log x-axis),
  0.5/0.9 threshold lines; a dot + `load N` label marks each curve's first
  crossing of 0.9 Strehl, and each legend label shows `max@N`
- `convergence_speed.png` — grouped bar chart (log y-axis) of the first device
  loads each algorithm takes to reach Strehl 0.5 / 0.9 / its final maximum,
  with exact values annotated above each bar
- `spot_before_after.png` — 2x4 grid (8 algorithms incl. SPGD) of initial vs
  best spot renders (`landscape.render`) with shared brightness normalization
- `summary_bars.png` — horizontal bar chart of final Strehl, sorted descending
- `summary.csv` — `algorithm, final_strehl, init_strehl, n_loads, elapsed_s`
- `report.md` — results table (final Strehl / improvement / **loads to max,
  >= 0.9, >= 0.5** / n_loads) plus per-algorithm principles and a
  cross-benchmark comparison table against `report/heuristic_pib/summary.csv`
  (rendered when that file is present)
- **设备加载语义**: 设备一次只能加载一个相位, 1 次设备加载 = 1 次相位加载 = 1 次目标函数 (Strehl) 评估 = 1 次迭代 (SPGD 每步 2 次加载: v+δ 与 v−δ); 表格与 CSV 中的 n_loads 即设备相位加载次数/迭代数。本基准的 Strehl 为该仿真器定义 (高斯光瞳远场峰值为理想参考并裁剪至 [0,1]) — 原始比值在 n_grid=256 下可达 ~3.6, 因此裁剪后是否达到 1.0 的收敛速度 (而不是最终值) 才是可比指标。


### generate_dm_response_matrix_report.py

Generates the illustrated **DM response matrix** report
(creation / analysis / detection). **Fully offline** — reads saved artefacts, no
hardware.

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_dm_response_matrix_report.py
python scripts/generate_dm_response_matrix_report.py --h5 <path> -o report/dm_response_matrix
```

**What it does** (writes `report.md` + `figures/`):
- **Creation**: acquisition metadata (calibration mode `sequential`/`hadamard`,
  hadamard order, n_actuators, valid actuator indices, disturb voltage,
  averages/cycles, wait time, timestamp, mean/max variance, condition number)
  + `device_config` dict rendered as a table (incl. `dm_type`/`dm_num` when
  present)
- **Analysis**: response-matrix heatmap, repeat-variance heatmap (log10),
  per-actuator response magnitude (column Frobenius norm) with median + 5%
  dead-threshold markers, per-subaperture slope-sensitivity spatial map
  (reshaped from paired dx/dy slopes when the subaperture grid is derivable
  from the mask; otherwise per-channel magnitude), and singular-value spectrum
  / condition number
- **Detection**: weak/dead actuator candidates (column norm < 5% of median)
  and high-variance actuators (>10× median column variance), each with a
  per-actuator table and interpretation notes

Legacy `.h5` files without the `calibration_mode`/`hadamard_order` attrs are
handled via `.get` defaults (`"sequential"` / `None`). The loader prefers
`ao_shaping.optimizer.wf.dm_response_matrix.load_dm_response_matrix` with a
graceful h5py fallback if the package import fails.

Sources default to the latest `data/dm_response_matrix*.h5`; override with
`--h5` and `-o/--output`.

### generate_centroid_test_visualization.py

Generates the centroid algorithm test visualization report. **Fully offline** —
pure numpy, no hardware.

**Usage:**
```bash
python scripts/generate_centroid_test_visualization.py
```

**What it does:**
- Runs 9 Gaussian spot cases through the centroid algorithms
- Writes `centroid_test_report.md` and figures to
  `report/centroid_test_visualization/`

### generate_diff_shaping_report.py

Generates the illustrated differential beam shaping report (GS vs
differentiable shaping). GPU recommended, CPU fallback available.

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_diff_shaping_report.py
```

**What it does** (writes to `report/slm_differential_shaping/`):
- Compares Gerchberg-Saxton and differentiable (PyTorch) beam shaping on
  square / circle / gaussian targets
- Writes `README.md` plus `figures/`, `gifs/`, `charts/` and `data/`
  subdirectories

### generate_fouriergsnet_sim_report.py

Generates the illustrated **FourierGSNet turbulence-sim matrix** report from a
`scripts/fouriergsnet_sim_train.py` output directory. **Fully offline** — reads
saved artefacts only (config/summary/metrics/frames), no hardware, no pipeline
code.

**Usage:**
```bash
python scripts/generate_fouriergsnet_sim_report.py
python scripts/generate_fouriergsnet_sim_report.py --matrix-dir /tmp/fgn_probe512b
python scripts/generate_fouriergsnet_sim_report.py --matrix-dir data/fouriergsnet_sim/<ts> -o report/fouriergsnet_sim
```

**What it does** (writes `report/fouriergsnet_sim/report.md` + `figures/` + `gifs/`):
- **Header**: matrix config (k_px / steps / seed / env noise params), generation
  timestamp, `**Fully offline**` marker
- **Summary table**: all scenarios × shape/aberration/turbulence/
  final+best uniformity/encircled/wall_time_s/ok, sorted by turbulence →
  aberration → shape
- **Per-scenario section** (核心交付): two animated GIFs per cell —
  `gifs/<scenario>_phase.gif` (SLM 整形相位演化, mod 2π 显示, twilight) and
  `gifs/<scenario>_far.gif` (目标光斑/远场演化, inferno), via the repo
  `_frames_to_gif` convention (LANCZOS 128px + adaptive 256 palette, 15 fps);
  `figures/<scenario>_metrics.png` (2×2 指标曲线, 标注 best uniformity) and
  `figures/<scenario>_frames.png` (初值/中段/末态相位+远场蒙太奇); plus an
  auto-generated interpretation (初值→最终均匀度, 湍流跟踪退化判断, EE 趋势)
- **Turbulence impact**: off vs slow vs fast final uniformity per
  (shape, aberration) — table + `figures/turbulence_impact.png` grouped bars
- **算法完整流程 (§5)**: the pipeline walkthrough (环境构造 → 标定/LUT → 网络前向
  → GS 初始化 → 闭环迭代 → 指标定义 → 动画来源) is rendered from the
  module-level `_PIPELINE_SECTION` `string.Template` with the **current** matrix
  config substituted in (command block, matrix dir, `--native`/`--no-native`
  clause, `k_px`, `steps`, GS 初始化迭代数, 噪声模型 Δk / 中心偏移 / 旋转).
  Uses `Template` (not `.format()`) because the prose contains literal `{}`;
  a missing/empty `config.json` degrades to `未记录` placeholders instead of
  raising.
- Robust: per-scenario try/except; missing frames degrade to static-only with a
  warning; scenario dirs are scanned directly so the report can be regenerated
  mid-run or after the matrix completes (summary.json optional)

| Option | Default | Description |
|--------|---------|-------------|
| `--matrix-dir` | latest `data/fouriergsnet_sim/<ts>` | Matrix output dir |
| `-o, --output` | `report/fouriergsnet_sim` | Report output dir |

### generate_oopao_vs_numpy_report.py

Generates the **OOPAO backend vs legacy numpy/FFT backend** comparison report —
a like-for-like aberration × turbulence matrix. **Fully offline** (pure numpy /
OOPAO simulation, no hardware). Unlike the other entries in this section, it is
*not* a re-generator of saved artefacts: it **runs** the two backends in-process
to produce the comparison, driving `beam_backend` directly.

**Usage:**
```bash
python scripts/generate_oopao_vs_numpy_report.py
python scripts/generate_oopao_vs_numpy_report.py --quick
python scripts/generate_oopao_vs_numpy_report.py --n-grid 128 --seed 7
python scripts/generate_oopao_vs_numpy_report.py --aberrations none,defocus --turbulence none,weak
```

**What it does** (writes `--out-dir` / `report.md` + `summary.csv` + `figures/`):
- **Matrix**: 3 aberrations (`none` / `defocus` (Noll 4) / `astig+coma` (Noll 5-8))
  × 4 turbulence levels (`none` Cn2=0 / `weak` 1e-16 / `moderate` 5e-15 /
  `strong` 5e-14) = **12 scenarios × 2 arms = 24 CSV rows**. A `spherical`
  (Noll 11) case exists but is *not* in the default set.
- **Per-arm metrics** (6): `phase_std_rad` (湍流相位 std) + `phase_rms_rad`
  (总相位 RMS) for the screen; `strehl`, `fwhm_px`, `ee_r4` (EE at 4·FWHM) for
  the focal plane; and `energy_frac` (ASM energy-conservation ratio, the one
  metric that *does* pass through the two different propagation kernels).
- **Figures**: one 5×2 comparison figure per scenario
  (`figures/<scenario>_<stamp>.png`) plus `summary_overview.png`; image links
  are validated (every link resolves, no orphans) before the report is written.
- **cn2=0 cross-arm control**: the `none` turbulence arm must be **bit-identical**
  across backends (`turbulence_phase` short-circuits to an all-zero screen at
  `cn2 <= 0`, *before* the backend switch). The script collects the scenarios
  where all `ARM_INVARIANT_METRICS` match exactly and logs them; if **none**
  match it emits a `warning` (an arm-independent phase bias). `energy_frac` is
  deliberately excluded from that check — it goes through `propagate()`, the
  only place the two kernels legitimately differ.
- **Determinism**: fixed `seed` drives an explicit `default_rng`; two full runs
  produce byte-identical `summary.csv`.
- **Backend hygiene**: forces `AO_OOPAO_BACKEND` off/on per arm, calls
  `oopao_backend._get_backend.cache_clear()` between configurations (the backend
  is `@lru_cache`), and **fails fast** if OOPAO is not importable rather than
  silently producing a two-arm-numpy report.
- Reports the `phase_std_rad` ratio per scenario; on the default config the two
  backends are **not** equivalent (≈8.7×, see `report/oopao_vs_numpy/report.md`),
  so the report states that absolute Strehl/FWHM must not be compared across arms.

**Zernike coefficients are radians**: aberration cases use Noll indices fed to
`zernike_utils.generate_zernike_phase()` (canonical entry), not to any local
Zernike table.

| Option | Default | Description |
|--------|---------|-------------|
| `--n-grid` | `64` | Simulation grid side length |
| `--seed` | `42` | Random seed |
| `--out-dir` | `report/oopao_vs_numpy` | Output dir |
| `--aberrations` | `none,defocus,astig+coma` | Comma-separated subset (`spherical` available) |
| `--turbulence` | all 4 levels | Comma-separated subset |
| `--quick` | off | Smoke mode: first 2 aberrations × first 2 turbulence levels |

Requires OOPAO (editable install from the `libs/OOPAO` submodule:
`uv pip install --no-deps -e libs/OOPAO`). Backend contract, cache caveat and
routing scope are documented in
[`sim/AGENTS.md`](../src/ao_shaping/drivers/sim/AGENTS.md).

### generate_oopao_impact_report.py

The **end-to-end companion** to `generate_oopao_vs_numpy_report.py`. Where that
script compares the phase screen + focal plane *in isolation*, this one drives a
real AO environment (`SimTurbulenceAOEnv`) under both backends and measures the
downstream consequence on Strehl / PIB / RMS. **Fully offline** (simulated, no
hardware); runs both backends in-process.

**Usage:**
```bash
python scripts/generate_oopao_impact_report.py
python scripts/generate_oopao_impact_report.py --quick
python scripts/generate_oopao_impact_report.py --cn2 0,5e-15 --steps 100
```

**What it does** (writes `report.md` + `summary.csv` + `figures/`):
- **Matrix**: 4 Cn2 levels (0 / 1e-16 / 5e-15 / 5e-14) × 2 arms × 2 modes
  (`open` = sliding turbulence, zero action; `closed` = frozen turbulence,
  3-step greedy SPGD) = 16 rows.
- **Headline finding (§4.3)**: the `disturbance_rms` oopao/numpy ratio is
  **constant across every turbulence level** (open 5.428×, closed 13.354×;
  relative spread ≤1.2e-09 over two orders of magnitude of Cn2). Constancy
  implies a **multiplicative calibration offset** between the two phase-screen
  implementations, not statistical fluctuation. The verdict is *computed* from
  the data against a tolerance, not asserted.
- **⚠️ metric-identity warning**: `init_rms` (`compat.py::_phase_rms()` —
  *pupil-masked total* wavefront incl. aberration + DM) and `disturbance_rms`
  (`env._disturbance_rms` — *full-grid unmasked* raw screen) are **different
  quantities and must not be inferred from one another**. A stronger screen does
  not imply a larger `init_rms` (measured counterexample included in the report).
- **cn2=0 control**: arms must be bit-identical (`turbulence_phase` short-circuits
  to a zero screen) — reported as pass/fail.
- **Anti-vacuity guard**: asserts the two arms actually *differ* at cn2>0 and that
  `_oopao_enabled()` matched the intended arm on every row; refuses to write a
  silently-degenerate two-arm-numpy report.
- **Negative finding**: documents that `slm_shaping_bench`'s `cn2` is **dead
  config** (it imports `turbulence_phase` but only calls the never-routed
  `focal_plane`, so output is byte-identical at cn2=0 vs 5e-14) — i.e. that bench
  must **not** be used as a backend-impact vehicle.

**Non-comparability caveat (§9)**: absolute Strehl/PIB must not be compared across
arms. Rows that look like "OOPAO is better" (e.g. open/cn2=5e-14 init Strehl
0.4650 vs 0.3276) reflect each arm being subject to a differently-scaled phase
screen, **not** a better propagation kernel. Calibrating both screens to the same
r0 / phase_std is required before any absolute comparison.

| Option | Default | Description |
|--------|---------|-------------|
| `--n-grid` | `64` | Simulation grid side length |
| `--seed` | `42` | Random seed |
| `--out-dir` | `report/oopao_impact` | Output dir |
| `--cn2` | `0,1e-16,5e-15,5e-14` | Comma-separated Cn2 ladder |
| `--steps` | `60` | Steps per episode (closed-mode SPGD iters = `steps//3`) |
| `--quick` | off | Smoke mode: first 2 Cn2 levels, `steps=10` |

`summary.csv` and all figures are byte-reproducible across runs (verified by
md5). Requires OOPAO; see
[`sim/AGENTS.md`](../src/ao_shaping/drivers/sim/AGENTS.md) for the routing table
and the `slm_shaping_bench` dead-config trap.

### generate_slm_gsnet_sim_gif.py

Generates the slm-gsnet offline-sim verification GIFs (+ prints the markdown
snippet) from a `slm-gsnet spgd --cam_type sim --debug` artifact directory.
**Fully offline** — reads the saved PKL only, no hardware, no pipeline code.

**Usage:**
```bash
python scripts/generate_slm_gsnet_sim_gif.py
python scripts/generate_slm_gsnet_sim_gif.py --pkl data/debug/slm_gsnet_<ts>/<ts>/xxx.pkl
python scripts/generate_slm_gsnet_sim_gif.py -o report/fouriergsnet_sim
```

**What it does** (writes `report/fouriergsnet_sim/gifs/`):
- Loads the debug PKL (`{epoch: record}` with per-epoch `_img` CCD far-field
  frames + `_c` freeform phase vector, length `phase_grid²` = 576)
- `slm_gsnet_spgd_sim_far.gif` — 逐 epoch 远场 (CCD 帧, inferno)
- `slm_gsnet_spgd_sim_phase.gif` — 逐 epoch SLM freeform 相位 (24×24 网格,
  mod 2π, twilight)
- Both via the repo `_frames_to_gif` convention (reused from
  `generate_diff_shaping_report`, LANCZOS 128px + adaptive 256 palette, 15 fps)
- Prints the `![...](gifs/...)` markdown lines for embedding in
  `report/fouriergsnet_sim/report.md` §5.6

| Option | Default | Description |
|--------|---------|-------------|
| `--pkl` | latest `data/debug/slm_gsnet_*/*/*.pkl` | Debug artifact pkl path |
| `-o, --output` | `report/fouriergsnet_sim` | Output dir (GIFs → `<output>/gifs/`) |

### generate_pearson_pkl_gif.py

Generates **synchronized CCD-image + SLM-phase** animated GIFs from all
`data/debug/*/*.pkl` debug artifacts that carry a Pearson correlation metric.
Each GIF frame shows the CCD far-field (left, `inferno`) and the SLM phase pattern
(right, cyclic `twilight`) evolving in lockstep, with the epoch and Pearson value
annotated in the top bar.

Supports both phase representations: full-panel uint16 grayscale ``_phase``
(1200×1920) and freeform coefficient vectors ``_c`` (reshaped to a square grid,
as produced by `slm-gsnet`).

**Usage:**
```bash
python scripts/generate_pearson_pkl_gif.py
python scripts/generate_pearson_pkl_gif.py --pearson-only --max-frames 200 --max-dim 384
python scripts/generate_pearson_pkl_gif.py --pkl data/debug/<run>/<ts>/<name>.pkl
```

**What it does** (writes `report/pearson_gifs/<pkl_stem>.gif`):
- Collects every ``*.pkl`` under `data/debug/` (newest-first)
- Loads each record set; selects only epochs with both `_img` and a phase
  representation (`_phase` or `_c`)
- Renders each epoch as a 1×2 composite (CCD | phase), downscaled so each panel
  fits within `--max-dim` pixels, framed at `--fps` frames/second
- Annotates each frame with the epoch index and the `pearson` value when present
- `--pearson-only` skips pkls that lack a `pearson` field or "pearson" in their
  name; `--max-frames` evenly subsamples epochs (default 300)

### slm_pib_sim_run.py

Runs the **`slm-pib`** SPGD shaping pipeline **entirely in the simulation
environment** (no hardware). Wires the pure-numpy 2f-Fourier sim
(`src/ao_shaping/drivers/sim/slm_pib_sim.py`) into the **genuine**
`slm_shaping_runner` (the `slm-pib` half) CLI path so the standard debug artifacts
(PNG/PKL/JSON) are
produced exactly as a hardware run would write them — ready for report
generation.

It can optionally inject a **wavefront disturbance** — atmospheric turbulence
plus a **thermal halo (热晕)** — and record it beside the run so the offline
report can compare a **static** (frozen) against a **dynamic** (per-evaluation)
disturbance regime.

**Usage:**
```bash
python scripts/slm_pib_sim_run.py
python scripts/slm_pib_sim_run.py --epochs 300 --target-shape square
python scripts/slm_pib_sim_run.py --epochs 60 --objective pearson

# disturbance runs (the static/dynamic pair the report compares)
python scripts/slm_pib_sim_run.py --epochs 300 --disturbance static
python scripts/slm_pib_sim_run.py --epochs 300 --disturbance dynamic
```

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | `300` | SPGD epochs (the CPU sim runs ≈1.4 it/s) |
| `--objective` | `shape` | Shaping objective passed through to the runner |
| `--target-shape` | `square` | Target shape |
| `--algorithm` | (runner default = `spgd`) | Search driver override |
| `--data-root` | `data` | Root the runner writes `debug/slm_pib_*` under |
| `--disturbance` | `none` | `none` / `static` / `dynamic` (see below) |
| `--dist-tag` | (derived) | Report tag; defaults to the disturbance mode |
| `--cn2` | `2e-13` | Refractive-index structure constant |
| `--distance-m` | `500.0` | Generator path-length knob [m]. **Degenerate with `--cn2`** |
| `--l-max` / `--l-min` | `30.0` / `2e-3` | Outer / inner scale [m] |
| `--pixel-pitch-um` | `8.0` | SLM pixel pitch [µm]; sets the screen's physical extent |
| `--halo-pv-waves` | `0.30` | Thermal-halo peak-to-valley [waves]; `0` disables it |
| `--halo-radius-px` | `600.0` | Thermal-halo radius [SLM px] (`w0` = 400 px) |
| `--dist-seed` | `20261001` | Disturbance seed (both regimes are deterministic) |
| `--dist-archive-factor` | `8` | Spatial decimation for archived screen thumbnails |
| `--dist-archive-max` | `12` | Max distinct screens archived |

**Disturbance regimes.** `static` generates one frozen screen and reuses it for
the whole run — the `closed`/frozen-turbulence analogue. `dynamic` draws a
**fresh independent screen on every optical evaluation** — the `open`/sliding
analogue, i.e. the fully-decorrelated ("white in time") limit. That limit is the
correct asymptotic here because the real atmospheric decorrelation time
(~10–50 ms) is far shorter than this loop's ~0.375 s per evaluation; it is
**not** a wind/advection model.

The disturbance is applied in the pupil plane alongside the SLM command phase
(`SimPibSystem.far_field`, consumed once per *real* optical evaluation via its
cache). Turbulence reuses the canonical `beam_backend.turbulence_phase`
(von-Karman); the thermal halo is a negative thermal lens (Noll 4 defocus +
Noll 11 spherical via the canonical `zernike_utils.generate_zernike_phase`),
smoothly apodised out to `--halo-radius-px` and PV-normalised to
`--halo-pv-waves`.

> ⚠️ `cn2` and `distance_m` are **degenerate generator knobs**: the canonical
> generator's `r0 = (0.423·k²·Cn2·L)^(-3/5)` depends only on their product, and a
> single thin screen carries no propagation physics. This is a **parametric
> stress test** on a ~0.3 m laboratory bench, not an atmospheric-propagation
> simulation. The numpy screen generator also lacks subharmonic/low-frequency
> compensation (`drivers/sim/AGENTS.md` §4), so the measured σ is a **lower
> bound** — reports quote the *measured* value.

> ⚠️ `slm_pib_sim_run.py` 在调用 CLI 前自己 `reset_system(...)` 装好带干扰的系统, 并包装
> 该调用使每次都重新挂上干扰。`slm-pib` runner 侧的 sim 接线
> (`runner_common.patch_sim_pib_shaping`) **刻意不 reset** —— 否则会把 harness 装好的
> 带干扰系统换成无种子、无干扰的系统, 运行静默地与自己的 manifest 矛盾。方形家族那边
> (`patch_sim_square_shaping`) 则钉 `seed=42`, 因为它没有面向用户的 seed 能传到台架,
> 不钉住的话两次 `spgd-square --cam_type sim` 的干扰流不可比
> (locked by `tests/ao_shaping/scripts/test_slm_pib_sim_run_disturbance.py` +
> `tests/ao_shaping/runners/test_slm_shaping_runner_sim_backend.py`)。

**What it does:**
- registers the `"sim"` camera type so `create_camera("sim", ...)` returns a
  `SimPibCCD` reading the shared far-field state (the `slm-pib --cam_type`
  `click.Choice` was extended to include `"sim"`)
- monkeypatches `ao_shaping.optimizer.wfless.slm_zernike_pib.Santec` →
  `SimSLMPib` so the optimizer's SLM context manager instantiates the sim
  (no hardware, no DVI hang). `SimSLMPib` implements **both** construction
  entry points: the legacy `Santec(...)` and the dataclass-API
  `Santec.from_params(config.slm)` that `optimize_slm_zernike_pib` now uses —
  without the classmethod the sim path dies with
  `AttributeError: type object 'SimSLMPib' has no attribute 'from_params'`
  (locked by `tests/ao_shaping/drivers/sim/test_sim_slm_from_params.py`)
- invokes the genuine `slm_shaping_runner.run` Click entry with `--cam_type sim`
  `--debug` (square target, Zernike n≤4, SPGD + AdaMOD)
- optical model: SLM = 2f front focal plane, CCD = back focal plane, so the CCD
  image is the 2D FFT (Fraunhofer far field) of the SLM pupil field — the
  0-order spot lands at frame centre and Zernike phase measurably modulates it
- the run logs `ROI energy guard armed: reference energy …, max loss …%`; the
  `pearson` objective is energy-**blind** after mean-centring, so this guard is
  what stops the search pushing light out of the target box (the `slm-pib`
  family guards; the `slm-gsnet` square path does **not** — see
  `slm-gsnet --objective pearson` help)

**Outputs:** `data/debug/slm_pib_shape_<ts>/` (PNG/PKL/JSON), then
`report/slm_pib_sim/report.md` + `figures/` + `gifs/` via
`generate_slm_pib_sim_report.py`.

When `--disturbance` is not `none`, the harness also writes a **companion**
next to the run's own artifacts (the report is a pure offline reader, so this
is the only channel carrying the disturbance to it):

- `disturbance.json` — mode, full config, **measured** σ_turb / σ_halo /
  σ_total, screens used, evaluations, and the run tag
- `disturbance.npz` — `screens` (decimated float32 thumbnails of the distinct
  screens actually used, capped at `--dist-archive-max`), `call_rms` and
  `call_streak_index` (**every** evaluation, in full), `archive_factor`

Recording one scalar per evaluation — never the full-resolution screens (a
1200×1920 float64 array is ~18 MB, and a 300-epoch run performs ~604
evaluations) — is what keeps memory bounded.

> 🔬 **Measured on the real bench (2026-09-29, Daheng MER2-507-23GM NIR + Santec
> SLM-200, 2592×1944, `exposure_time_ms=1.2`, `zernike_radius=480`, `n_max=9` →
> 55 DOF).** The bench is healthy: the 0-order spot sits at `(x=673, y=1027)` —
> **not** the frame centre, so it must be located by `argmax`.
>
> **The objective's noise floor is `ΔJ_noise ≈ 1.7e-3`, and it is drift-dominated,
> not shot-noise.** Averaging 40 frames instead of 10 did *not* reduce it
> (2.8e-3 vs 1.7e-3) because the residual is slow intensity drift between
> frames, which averaging cannot cancel.
>
> **Consequence: `--delta` must not be too small.** With SPGD perturbing all 55
> DOF by a random ±pattern, the per-epoch `|ΔJ|` is far smaller than a
> single-mode probe suggests, so the per-epoch SNR collapses and the
> `noise_gate_k=3.0` gate rejects nearly every epoch:
>
> | `--delta` | updates applied | Pearson improvement (best / sustained) |
> |---|---|---|
> | `0.0005` (historical default) | 9/200 | noise only — **no usable gradient** |
> | `0.002` | 8/200 | +24.8% best / **+0.2% sustained** (drift) |
> | `0.01` | 10/200 | +28.9% best / +3.0% sustained |
> | **`0.1`** | **11–13/60** | **+29…37% best / +15…24% sustained** ✅ |
> | `0.2` | 4/60 | 45/60 epochs rejected by the **brightness-fold** guard |
>
> So `delta ≈ 0.1` is the functional optimum for `n_max=9`; `0.2` overshoots into
> the fold guard, and anything `≤0.01` is drift-dominated. A delta of `5e-4`
> (the value in the historical smoke runs) is ~100× below the floor and cannot
> optimise this bench — the "improvements" it appears to make are drift.
> `measure_shape_sensitivity.py` is the tool that establishes this floor
> (`--deltas 0.002,0.01,0.05,0.1,0.2`).

### generate_shape_objective_comparison.py

Scores every recorded `slm-pib` frame under **three** shape objectives and
reports whether they **agree** on the ranking — the evidence for whether the
FourierGSNet `1 - Pearson` loss may be promoted onto the hardware path.
**Fully offline** — reads saved recorder pickles only, never opens a device.

**Usage:**
```bash
python scripts/generate_shape_objective_comparison.py
```

**What it does** (writes `report/slm_pib_online/objective_comparison.md` +
`figures/`):
- loads each `data/debug/slm_pib_online/*/recorder_*.pkl`; the SNR sweeps
  (`history is None`) are skipped, as is any run with no scored frame
- **anchors the target ROI on recorder row 0** (the pre-optimization frame), so
  the box does not follow optimizer drift and every epoch of every run is scored
  against the same fixed target. Row 0 is deliberately excluded from the
  statistics
- scores each frame with `square_quality_score`, `compute_quality_score` and
  `1 - Pearson`, then reports per-run **Spearman** rank correlation. The
  Pearson loss is *lower-is-better*, so the expected sign of agreement is
  **negative** — the control pair (the two composites) is reported alongside to
  prove the frame set and anchor are sound
- **Verdict** distinguishes two failure modes rather than only counting signs:
  an inconsistent *sign* (ordering unstable) vs a correct sign with a *weak*
  effect (`mean |rho| < 0.5`, ordering noise-dominated). They call for different
  remedies, and conflating them overstates the instability
- per-run section shows the frames where the objectives disagree most, ordered
  by descending disagreement with **ties broken worst-Pearson-first** (the most
  alarming frames must not sink to the bottom of the table)
- `1 - Pearson` is reported as **bounded `[0, 2]`** for a valid frame (a
  constant frame gives exactly `1.0`); the `1e3` value is a discrete sentinel
  for a dark / NaN / non-normalisable frame, not a continuous tail

> ⚠️ **Provenance is inferred unless the sidecar records it.** `slm_shaping_runner`
> now writes `objective` / `cam_type` / `cam_id` / `exposure_time_ms` /
> `zernike_radius` / search knobs into the JSON sidecar, so new runs are
> self-attributing. Runs written before that still record only
> `delta`/`epochs`/`lr`; the report labels those `recorded run config: **absent**`
> and their backend attribution stays *provisional*.

> 🔑 **Two on-disk pickle shapes are accepted.** `save_recorder_debug_artifacts`
> writes a plain `{epoch: row}` dict, whereas the SNR-sweep dumps are pickled
> `Recorder` objects (or `None`). The loader handles both — requiring
> `.history` alone made it silently report "no recorded frames" for a correctly
> staged hardware run.

> 🔬 **Hardware verdict (2026-09-29).** Re-running the comparison on real Daheng
> frames from the `delta=0.1` runs reproduces the offline conclusion:
> `square_quality_score` vs `1 - Pearson` shows the expected sign in only **2/4**
> runs (mean |rho| = 0.442), with the control pair `square` vs `metrics` at
> +0.88…+0.96. `1 - Pearson` therefore remains **RISKY** as a drop-in
> replacement on real hardware: it optimises (the loss falls ~30%) but its
> *ranking* of candidate frames is too weak and too run-dependent to trust as a
> convergence signal. Use it as an explicitly selected additional objective, keep
> the ROI energy guard armed, and re-check the sign per run.

> 🔑 **No native SDK on import.** The report needs the pure function
> `square_quality_score`, which lives in the optimizer and therefore drags in the
> camera package. The MIICAM / Daheng backends are exposed through PEP 562
> module `__getattr__` and load only on first attribute access, so importing this
> script does **not** `ctypes.CDLL` the native SDK or import `gxipy`. Locked by
> `tests/ao_shaping/drivers/ccd/test_lazy_backend_imports.py`.

**Regression tests:** `tests/ao_shaping/runners/test_shape_objective_comparison_report.py`
(pins the tie-break direction and the `rows`/`epochs`/`gates` alignment).


### generate_iterative_zernike_shaping_report.py

Generates the illustrated **free-form refinement** report on a zero-padded
2f-Fourier sim bench, comparing the iterative refinement loop (GS warm start +
differentiable free-form shaping) against single-pass baselines on the same
64×64 pupil grid — the canonical sensorless SPGD, the Gerchberg-Saxton (GS)
single pass, and the unshaped initial state.
**Fully offline** — pure torch (FFT forward model), no hardware.

**Usage:**
```bash
.venv/bin/python scripts/generate_iterative_zernike_shaping_report.py
```

**What it does** (writes to `report/iterative_zernike_shaping/`):
- Runs the iterative refinement loop: Stage B optimizes the free-form SLM phase
  to the square target, warm-started from a Gerchberg-Saxton phase, iterating
  until early-stop convergence
- Baselines (all on the **identical 64×64 pupil grid**, padded 8× to a 512×512
  far field, identical bench metric
  `composite_score = 0.5·PIB + 0.5·(1/(1+CV))` evaluated in the grid-centred
  target support):
  - initial (unshaped golden) score
  - single-pass GS (`gs_shape`, 200 iters)
  - sensorless **SPGD** (`spgd_shape`, 600 iters, dim=8 freeform) — the
    canonical black-box reference. The `0.89` figure cited elsewhere is a
    1920×1200 **hardware-grid** result and is NOT comparable to this 64×64 sim.
- `initial_vs_final.png` — far-field before/after the loop (zoomed, log colour
  scale: shows the initial aberrated focus and the target box)
- `score_history.png` — composite score per outer iteration
- `zernike_coeffs.png` — per-mode Zernike coefficient traces (only when the
  optional Zernike calibration pass is enabled)
- `phase_evolution.png` — free-form SLM phase (mod 2π) start/mid/end montage
- `score_comparison.png` — bar chart: initial vs GS vs SPGD vs refinement
- `data.json` + `*.npy` — raw scores, the Zernike ablation, and phase arrays

> 📐 **Measured (2026-10-01, 64×64 pupil, 8× pad, `far_field_pixel_size` = 0.693 µm,
> target 43 px ≈ 30 µm ≈ 2.2 Airy diameters)**: initial **0.662**, GS **0.812**,
> SPGD **0.647**, refinement **0.849** → **+4.6% vs GS**, **+31.2% vs SPGD**.
> GS produces the most uniform single-pass flat-top (CV 0.41); the refinement
> reaches CV 0.12.
>
> 🔬 **The Zernike calibration pass is a documented NEGATIVE result.** Enabling it
> (`n_zernike=4`) drops the score to **0.810** (ties GS, −4.5% vs disabled): the
> calibration's low-order estimate is both leaky (free-form phase is absorbed into
> spurious high-order modes) and redundant with the free-form stage, and freezing
> it corrupts the warm start. It is therefore **off by default**; the report still
> computes the ablation and records it in `data.json`.
>
> ⚠️ **Corrected series.** Earlier numbers (initial 0.1590 / GS 0.1780 / SPGD
> 0.1906 / iterative 0.3851) and the first "corrected" numbers (initial 0.657 /
> GS 0.810 / SPGD 0.647 / iterative 0.642, "refinement loses") were produced with
> defects that are now fixed and must not be resurrected:
> 1. an **un-padded, same-size FFT** whose focal-plane sampling (~1.1 px per
>    waist radius, a model constant independent of `n_grid`) aliased the
>    lens-phase-modulated pupil into a lattice of hundreds of dots, so the
>    "initial spot" was not physical;
> 2. a **clipped uniformity term** `1 − min(CV/0.3, 1)` whose threshold sat below
>    every achievable flat-top CV, so the objective silently reduced to pure
>    bucket energy (`0.5·PIB`) and rewarded concentrating light over flattening
>    it;
> 3. an **argmax-rolled support** whose box follows the intensity peak: for
>    speckle-like fields the argmax hops between near-equal grains under a ~1e-3
>    model change, making PIB/CV discontinuous and letting the optimizer chase a
>    box that does not cover the beam. The box is now fixed to the grid-centred
>    target;
> 4. Stage B ran at a **flat `slm_lr`** and returned its last iterate, which
>    oscillated; it now uses a cosine LR decay and returns the best iterate;
> 5. the Zernike calibration ran at `zernike_lr=0.05`, which **diverges to
>    non-finite coefficients** on the rugged far-field MSE landscape. The default
>    is now `0.005` and the loop bails out on the first non-finite iterate
>    (`test_s10_calibration_is_finite_and_reduces_mismatch`).
>
> Locked by regression tests
> `tests/ao_shaping/algorithm/test_iterative_zernike_shaping.py`
> (`test_s7_objective_consistency_and_shaping_discrimination` — the algorithm
> `_score` must equal the bench `composite_score`, and GS must beat SPGD on
> uniformity; `test_s8_initial_spot_is_single_not_lattice` — the initial spot is
> a single focus, not an aliased dot lattice;
> `test_s9_warm_started_refinement_beats_gs` — the GS-warm-started refinement
> must beat plain GS; `test_s10_calibration_is_finite_and_reduces_mismatch` —
> Zernike calibration must stay finite and beat the zero-coefficient baseline).

### generate_slm_pib_sim_report.py

Generates the illustrated **slm-pib simulation** report from a
`slm-pib --debug` artifact directory. **Fully offline** — reads the saved
PKL/JSON only, no hardware, no pipeline code.

**Usage:**
```bash
python scripts/generate_slm_pib_sim_report.py
python scripts/generate_slm_pib_sim_report.py --debug-dir data/debug/slm_pib_shape_<ts>
python scripts/generate_slm_pib_sim_report.py --max-runs 3
```

**What it does** (writes `report/slm_pib_sim/report.md` + `figures/` + `gifs/`):
- **Header**: run description (sim camera / monkeypatched SLM / square target /
  SPGD Zernike n≤4), 2f-Fourier optical model, `**Fully offline**` marker
- **Per-run section**: metric table (epochs / initial+final J / in-target
  energy `_p%` / search config) + four figures — `*_objective.png` (J + `_p%`
  curves), `*_zernike.png` (per-mode coefficient trace), `*_phase_evolution.png`
  (sent Zernike phase, mod 2π, start/mid/end), `*_spot_evolution.png` (far-field
  CCD frame, start/mid/end) — plus two animated GIFs (`*_phase.gif` hsv,
  `*_spot.gif` inferno)
- **Interpretation + conclusion**: objective improvement, the AGENTS.md
  anti-pattern note that low-order Zernike (n≤4) cannot synthesize a true square
  (needs full-pixel / freeform phase), and the reusability claim (any `slm-pib
  --debug` run, sim or hardware, regenerates a report)
- Robust: scans `data/debug/slm_pib_*/*` newest-first; `--max-runs` selects how
  many runs to render

| Option | Default | Description |
|--------|---------|-------------|
| `--debug-root` | `data/debug` | Root dir containing `slm_pib_*` artifact dirs |
| `--debug-dir` | (None) | A single artifact dir (overrides the glob) |
| `--max-runs` | `1` | How many runs (newest first) to render |
| `--out` | `report/slm_pib_sim` | Output dir for figures/gifs/report.md |

### generate_slm_zernike_shaping_report.py

Generates the illustrated report for the **`slm_zernike_shaping`** optimizer
(the shaping module) from its `debug=True` artifact bundle. **Fully offline** —
reads the saved PKL/JSON only, no hardware, no pipeline code.

**Usage:**
```bash
python scripts/generate_slm_zernike_shaping_report.py
python scripts/generate_slm_zernike_shaping_report.py --debug-dir data/debug/slm_zernike_shaping_rmse_out_<ts>
python scripts/generate_slm_zernike_shaping_report.py --debug-root data/debug --max-runs 2
```

**Artifacts read** (`<debug_dir>/debug/slm_zernike_shaping_<objective>_<ts>/<ts>/`):
- `*.pkl` — `{epoch: record}`; rows carry `J / _p% / lr / delta / r / exp_t /
  max_brt / _img` (CCD far-field) / `_c` (Zernike coeffs) / `_grad`, the
  cross-objective `m_*` panel, and the objective's own column (e.g. `rmse_out`).
- `*.json` — run payload (`objective / target_shape / target_size / epochs /
> 🔬 **Pre-correcting the pupil by `-Z_est` before GS is a provable no-op.**
> `gs_shape(..., base_phase=...)` accepts a fixed pupil phase and applies it inside
> the pupil constraint, but that constraint re-imposes `amp·exp(i·angle(field))`
> every iteration, so any constant base is annihilated. Measured identical to six
> decimals with and without `-Z_est` (0.811805 vs 0.811806 at the GS stage, 0.8411
> after refinement) and *slightly worse* with the ideal `-Z_golden` (0.8034 /
> 0.8381). Recorded in `data.json` under `pre_correction_ablation`.
>
  algorithm / optimizer_type / delta / w_outside / r_bucket / cam_type / cam_size`).
- `*.png` — the run-time summary figure.

**What it does** (writes `report/slm_zernike_shaping/report.md` + `figures/`):
- **Header + run config** from the JSON sidecar; `**Fully offline**` marker
- **Per-run section**: objective-vs-epoch curve (min/max aware), best objective
  + epoch, Zernike-coefficient evolution, first-vs-last CCD frames
  (`run<N>_frames.png`), and an optional spot GIF (`run<N>_spot.gif`)
- Robust: scans `data/debug/slm_zernike_shaping_*/*` newest-first; missing
  keys/figures degrade to a warning, never a traceback

| Option | Default | Description |
|--------|---------|-------------|
| `--debug-root` | `data/debug` | Root dir containing `slm_zernike_shaping_*` artifact dirs |
| `--debug-dir` | (None) | A single artifact dir (overrides the glob) |
| `--max-runs` | `1` | How many runs (newest first) to render |
| `--out` | `report/slm_zernike_shaping` | Output dir for figures/report.md |

### generate_fouriergsnet_pipeline_report.py

Generates the FourierGSNet pipeline integration-test report. **Fully offline** —
pure markdown, no hardware, no figures.

**Usage:**
```bash
python scripts/generate_fouriergsnet_pipeline_report.py
python scripts/generate_fouriergsnet_pipeline_report.py -o report/fouriergsnet_pipeline
```

**What it does** (writes `report/fouriergsnet_pipeline/report.md`):
- Documents the 5 integration tests in
  `tests/ao_shaping/drivers/sim/test_sim_fouriergsnet_pipeline.py` that drive
  the REAL standalone `fouriergsnet_optimize.py` pipeline
  (ShapingSystem → adaptive_gs_init → closed_loop → _append_final_record)
  through `SimFourierGSNetEnv` with no hardware and no pipeline modification
- Harness decisions: narrow beam `BeamParams(region=256, w0=1.0)` (default
  region=512/w0=250 → sub-pixel spot → `RuntimeError("工作区无信号")` at
  fouriergsnet_optimize.py L814; wide beams → `uni=0.0000` always because
  `_metrics` `min(I·roi)/mean(I·roi)` over a 22×22 ROI the tight spot never
  fills); `torch.no_grad()` wrapper for `closed_loop` (L938
  `phi.cpu().numpy()` on a requires-grad tensor — genuine pipeline bug that
  also breaks the real CLI `run()`); `SETTLE_S=0.0`
- Findings: L938 bug; uni=0 root cause; `FourierGSNetLite(K, n_zern, ch,
  src_mask)` signature mismatch vs the task brief; panel-resolution transpose
  nuance (`PANEL_RES=(1920,1200)` is (W,H) in the real driver but the pipeline
  `place_on_panel` treats it as (h,w) — masked by `SimFourierGSNetEnv` which
  deliberately uses `PANEL_H, PANEL_W = 1920, 1200`)
- Results: 5 passed (≈29 s); regression 21 passed

### generate_gsnet_offline_report.py

Generates the illustrated **FourierGSNet offline training** report from a
`slm-gsnet train` run directory. **Fully offline** — reads the saved
`summary.json` / `comparison.png` / `train_history.png` only; no torch, no
hardware, no network.

**Usage:**
```bash
python scripts/generate_gsnet_offline_report.py
python scripts/generate_gsnet_offline_report.py --run-dir data/gsnet_train/run-20260929_225847
python scripts/generate_gsnet_offline_report.py -o report/fouriergsnet_pipeline/offline_training
```

**What it does** (writes `report/fouriergsnet_pipeline/offline_training/report.md` + `figures/`):
- **Header**: run dir, generation timestamp, `**Fully offline**` marker
- **训练配置**: table from `summary.json` `config` + `resolved` (epochs / lr /
  batch / grid / device / n_records / w_phase / w_shaping / layers / channels /
  parameter count)
- **收敛概览**: first vs last total loss, `best_epoch`, wall time rendered as
  hours + s/epoch
- **评估指标**: `evaluation.means` table (phase_mae / far_correlation / far_rmse /
  uniformity_cv / encircled_energy) with a value-derived interpretation — the
  prose branches on the actual numbers, so a high `far_correlation` is never
  allowed to mask a poor `uniformity_cv`
- **预测光斑 vs 真值对比** (核心交付): the run's own `comparison.png` is copied
  to `figures/comparison_pred_vs_gt.png` and embedded, giving the predicted spot
  next to the ground truth for the `n_compare` sampled records
- **损失曲线**: `figures/loss_curves.png` redrawn from `training.history` as two
  stacked panels sharing the x-axis (total+phase on top, shaping below) —
  necessary because phase (~2.6) and shaping (~0.017) losses differ by two orders
  of magnitude, so a single linear axis flattens the shaping curve; `best_epoch`
  is marked
- **产物清单** + **复现命令**: the reproduction block emits the real Click entry
  point (`python src/ao_shaping/main.py slm-gsnet train ...`); note that
  `ml/gsnet_debug/train.py` (moved there from `ao_shaping/runners/gsnet_train.py`)
  is a *library* module (no `__main__`, no Click command), so invoking it as a
  module path is **not** a valid way to retrain
- Writes into the `offline_training/` **subdirectory** so it never overwrites the
  sibling `report/fouriergsnet_pipeline/report.md` owned by
  `generate_fouriergsnet_pipeline_report.py`
- Robust: every `summary.json` key is read through a defensive `.get()` chain, so
  an older/partial summary still renders; a missing `comparison.png` degrades to
  a note instead of raising

| Option | Default | Description |
|--------|---------|-------------|
| `--run-dir` | newest `data/gsnet_train/run-*` (mtime) | Training run dir to render |
| `-o, --output` | `report/fouriergsnet_pipeline/offline_training` | Report output dir |

### generate_models_report.py

Generates the ML model training analysis report from TensorBoard event files.
**Fully offline** — reads saved training logs, no hardware.

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_models_report.py
```

**What it does** (writes to `report/models_analysis/`):
- Maps run names to experiment stages (stage1_easy -> 阶段1, stage2_medium ->
  阶段2, stage3_ -> 阶段3, static_long -> 静态湍流-长训练, static_focus ->
  静态聚焦, turbulence_long / turbulence_mamba_best / turbulence_long_retry ->
  湍流-长训练, turb_focus -> 湍流聚焦, sac_ -> 冒烟/架构对比实验,
  tmp_sac_run -> 临时调试)
- Reads TensorBoard tags `rollout/ep_rew_mean`, `ao/best_pib`,
  `ao/best_strehl`, `ao/pib`, `ao/rms`
- Renders training curves at DPI 130 and writes the analysis report

## Verification Scripts

### generate_bench_probe_report.py

Regenerates the **illustrated SLM bench-probe report** from artefacts a hardware
sweep already wrote. **Fully offline** — reads only `*.npz` / `*.json` / `*.pkl`
from disk, never opens a camera or SLM, so conclusions can be refreshed while the
instruments are powered down.

**Usage:**
```powershell
python scripts/generate_bench_probe_report.py            # default paths
python scripts/generate_bench_probe_report.py -o report/slm/bench_probe
python scripts/generate_bench_probe_report.py --no-figures
```

**What it does** (writes `report/slm/bench_probe/report.md` + `figures/`):
- **§1 provenance** — every input with a ✅/⚠️/❌ status, a per-mode point
  census, and two consistency audits (§1.2 sidecar vs npz, §1.3
  `bench_geometry.json` vs npz).
- **§3 ramp linearity** — the authoritative focal scale `S` from
  `displacement = S / period`, per axis, with the axes' disagreement.
- **§4 Zernike tilt linearity** — per-axis slope in cam px/rad, repeat spread,
  and which estimator was usable.
- **§5 width response** — `fwhm²` vs coefficient per mode with the fitted
  parabola, points coloured by hollowness, and the single-lobe gate drawn at
  0.60 with the points it discards marked.
- **§6 focal scale, three independent routes** — ramp / tilt / field-of-view.
- **§7 the four bench iron rules**, each with the numbers from *this* dataset
  that prove it still bites.
- **§8 conclusions + §8.1 actionable items for shaping experiments.**
- **Appendix A** — the full 55-row record table.

**Per-test-item analysis (§3.1 / §4.1 / §5.1 / §7.1)**: every test is reported
as **目的 / 结果分析 / 对整形的影响**, because a bench number is only actionable
once you know what it does to the shaping loop — e.g. a focal scale that is
*silently* wrong scales the target square's side by the same factor, and a width
feedback signal read outside the single-lobe region feeds the optimiser an
inverted gradient.

**Two guards that fail silently, and therefore have regression tests**
(`tests/ao_shaping/scripts/test_generate_bench_probe_report.py`):
- **Stale-geometry guard.** `bench_geometry.json` is **shared and overwritten** by
  both calibration routes (speckle and sweep). Reading it next to a fresh npz
  mixes two calibrations: every model-side number inherits the error while the
  derivation still looks self-consistent. §1.3 cross-checks `panel_disc_radius`
  against the npz's own `zernike_radius`/`collect_disc` and against `method`, and
  when they disagree §2 and §6 are **disabled rather than reported**, because a
  ~185 % "route disagreement" computed from two different calibrations is a
  pure artefact.
- **Constant-drift guard.** The bench constants are imported from the modules
  that define them (`slm_bench_probe.TILT_SHIFT_SCALE`,
  `model_in_loop_shaping._MAX_DEFOCUS_FIT_RMS`) rather than hardcoded. A literal
  can silently drift away from the code; any import failure is listed in
  `fallbacks` and surfaced in the captions.

The largest cross-check the report makes — `S = k_tilt·π·R` — deliberately needs
**no calibration file at all** (the illuminated radius ships inside the npz), so
it stays valid even while §2 and §6 are disabled.

### model_in_loop_hw_runbook.py

Hardware runbook for the **model-in-the-loop** square-shaping test: calibrate
the forward model's geometry on the real bench, then compute a square phase with
it, display it, and measure the result (needs hardware: Santec SLM-200 + camera).

> The sweep acquisition kernel (`zernike_panel`, `capture_settled`, the Recorder
> bookkeeping) now lives in the reusable tool
> `src/ao_shaping/tools/slm/slm_zernike_sweep_probe.py`, and this runbook imports
> it. `--stage sweep` keeps only the calibration-protocol specifics (ABBA tilt
> interleaving, phase-correlation shifts, per-point frame dumps).

**Usage:**
```bash
python scripts/model_in_loop_hw_runbook.py --stage all --target-cam-px 40
python scripts/model_in_loop_hw_runbook.py --stage collect --probes 12
python scripts/model_in_loop_hw_runbook.py --stage calibrate
python scripts/model_in_loop_hw_runbook.py --stage shape --dry-run
```

**What it does** — three stages, resumable, artefacts under
`data/model_in_loop_hw/`:
- `collect` — captures calibration records using **uniform random pupil phase**
  (not Zernike: those are built over a large panel radius and leave the
  illuminated core nearly flat, so there is no speckle for the geometry solve to
  correlate and it saturates at its search boundary). Locates the 0-order by
  `argmax`, records the beam offset, warns on clipping. Saves
  `calibration_records.npz` + `beam_offset.json`.
- `calibrate` — solves panel disc radius, beam waist and far-field scale via
  `calibrate_bench_geometry` (joint disc × waist search scored by speckle
  correlation), writes `bench_geometry.json`, and echoes the correlation so a
  non-converged solve is visible.
- `shape` — converts `--target-cam-px` into model pixels with the calibrated
  scale, runs Step B with the calibrated `region` / `far_field_size` / `w0`,
  places the phase on the panel at the measured beam offset, and reports
  before/after encircled energy, uniformity CV and flatness.

`--dry-run` exercises every non-device step (including a synthetic geometry
solve) and is how the logic is verified while the bench is offline.

| Option | Default | Description |
|---|---|---|
| `--stage` | `all` | `collect` / `calibrate` / `shape` / `all` |
| `--out` | `data/model_in_loop_hw` | artefact directory |
| `--probes` | `12` | calibration records to capture |
| `--region` / `--far-field-size` | `256` / `4096` | model grid and zero-padding |
| `--collect-disc` | `200` | panel radius used when probing, px |
| `--target-cam-px` | `40` | target square side in camera pixels |
| `--shape-iterations` | `600` | Step B budget |
| `--exposure-ms` | `3.0` | camera exposure (ms) |
| `--slm-number` / `--slm-wavelength` | `1` / `1064` | SLM device / wavelength |
| `--cam-type` / `--cam-id` | `daheng` / `0` | camera backend / id |
| `--dry-run` | off | run everything except device I/O |

> Bench constants, the measured 29.5× far-field sampling mismatch, and the
> failure modes this works around are documented in
> [`report/slm/model_in_loop_bench_calibration.md`](../report/slm/model_in_loop_bench_calibration.md).

### verify_correction_csv.py

Offline verification for the `--export-correction` gray-offset CSV
(2026-09-16 contract: **no baked shift** + full-scale 2π = `get_max_grayscale()`
= 1023, NOT a wavelength-dependent `two_pi_gray`). **Fully offline** — reads
saved artefacts, no hardware.

**Usage:**
```powershell
$env:PYTHONPATH = "src;libs"
python scripts/verify_correction_csv.py
python scripts/verify_correction_csv.py --h5 <matrix.h5> --w <w_before.json> --csv <corr.csv>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--h5` | `data/zernike_response_matrix/zm_recal_532_20260916.h5` | Zernike response matrix h5 |
| `--w` | `data/zernike_correction/_w_before_66.json` | WFS wavefront JSON (66-length, unit λ) |
| `--csv` | `<h5> 同目录 <stem>_correction_gray.csv` | Exported correction gray CSV to verify |

**Checks (all must pass):**
- [0] Sidecar JSON: `max_gray == 1023`, `shift_included == false`, consumption
  free of `two_pi_gray`
- [1] CSV shape `(1200, 1920)`, value range ⊂ `0..1023`
- [2] `Santec.load_gray_from_csv` / `WavefrontCorrection.load_gray_from_csv`
  byte-identical to the raw CSV
- [3] Recomputed production pipeline (`c = -pinv(M)@w` → `make_phase` →
  `correction_gray_offsets(1023)`) byte-identical to the export → proves **no
  baked shift**
- [4] `WavefrontCorrection.map_error` additive semantics
  (`displayed = mod(base + corr, 1024)`, incl. mod wrap at base=500)
- [5] Quantization error ≤ 0.5 gray levels (circular distance)

### validate_flat_phase_gray.py

Validates the SLM flat-phase gray level response (needs hardware: Santec SLM +
MiiCam). The SLM has amplitude coupling at 1064 nm: different flat-phase gray
levels produce different camera intensities, periodic with 2π ≈ 993 gray.

**Usage:**
```bash
python scripts/validate_flat_phase_gray.py --exposure-ms 0.8 --wait-time-s 0.3 --discard-count 3
python scripts/validate_flat_phase_gray.py --scan --gray-step 50
```

**What it does:**
- Opens the SLM in memory mode (`video_mode=0`) and the MiiCam via
  `MIICamera`
- Writes flat phases at the quick gray values `[0, g_pi2, g_pi, g_3pi2, g_2pi,
  1023]` (or a full `--scan` sweep in `--gray-step` increments) using raw
  uint16 grayscale (`np.full`, never through `create_phase_from_array()`)
- Captures frames and reports the 0-order bucket brightness per gray level
- Verdict: ≥ 2 distinct brightness levels pass, ≥ 5 is excellent

| Option | Default | Description |
|--------|---------|-------------|
| `--slm-number` | `1` | SLM device number (1-8) |
| `--miicam-id` | `0` | MiiCam camera ID |
| `--wavelength` | `1064` | SLM wavelength (nm, 450-1600) |
| `--exposure-ms` | `3.0` | Camera exposure (ms) |
| `--wait-time-s` | `0.3` | Settle time after SLM write (s) |
| `--discard-count` | `3` | Frames discarded before measurement |
| `--bit-depth` | `8` | Camera output bit depth (`8` or `16`) |
| `--scan` | off | Full gray sweep instead of the quick 6-point check |
| `--gray-step` | `50` | Gray step for `--scan` |

### wfs_compare_image_functions.py

Compares the old and new Thorlabs WFS spot-field image functions (needs
hardware: Thorlabs WFS).

**Usage:**
```bash
python scripts/wfs_compare_image_functions.py
```

**What it does:**
- Runs 6 tests comparing the NEW bound `WFS_GetSpotfieldImageCopy`
  (1024x1280) against the OLD unbound `WFS_GetSpotfieldImage`
- Verifies error handling: `handle_error(-120)` raises `WfsError`

### wfs_driver_hw_test.py

Hardware smoke test for the Thorlabs WFS driver (needs hardware: Thorlabs WFS).

**Usage:**
```powershell
.venv\Scripts\python.exe scripts\wfs_driver_hw_test.py
```

**What it does:**
- Opens `ThorlabWFS(mla_index="512", high_speed=False,
  stable_sample_enable=False)`
- Runs a PASS/FAIL checklist over the driver API (open, read, exposure,
  reference, close)

### wfs_probe.py

Probes the Thorlabs WFS driver for stability and crash reproduction (needs
hardware: Thorlabs WFS).

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python -X faulthandler scripts/wfs_probe.py --iters 80 --mla 512 --exp 4.0
```

**What it does:**
- Runs `--iters` WFS read cycles with garbage collection every 20 iterations
- Reproduces the 0xC0000374 heap-corruption crash for diagnosis
- Supports custom reference, tilt cancellation and Zernike readout options

| Option | Default | Description |
|--------|---------|-------------|
| `--iters` | `80` | Number of read cycles |
| `--mla` | `512` | MLA resolution |
| `--exp` | `4.0` | Exposure time (ms) |
| `--high-speed` | off | Enable high-speed mode |
| `--use-custom-ref` | off | Use a custom reference file |
| `--cancel-tile` | off | Cancel tip/tilt in measurements |
| `--zernike-order` | `10` | Zernike fit order |

## Subdirectories

### dm_sim/
Contains MATLAB scripts for deformable mirror simulation:
- `ComputeInfluenceMatrix.m` - Computes influence matrix for DM
- `CreateElectrodes.m` - Creates electrode configurations
- `CreateHDMMatrices.m` - Creates hysteresis matrices
- `CreatePreisachs.m` - Creates Preisach models
- `SimulateHDMControl.m` - Simulates HDM control
- `WavefrontReconstruction.m` - Wavefront reconstruction algorithms
- `zernike/` - Zernike polynomial implementations

### tuning_devices/
Contains scripts for device tuning and calibration:
- `dm_unit_compute.py` - Computes DM unit properties
- `calculateDerotation.m` - Calculates derotation
- `centroidcaculation.m` - Calculates centroids
- `stdWavefront/std_wavefront.npz` — Standard wavefront reference data: 66 measured
  Zernike base maps (modes 1..66, 360×360, float64) in one compressed archive,
  loaded by `ao_shaping.utils.wavefront.wavefront_calc.get_zernike_base_matrixs`.
  It replaced 66 ASCII `.txt` files (56.4 MB → 23.2 MB, bit-exact, ~5× faster to
  load) whose `Path.glob("*.txt")` loader ordered the modes **lexicographically**,
  so 65 of the 66 slots held the wrong map.
- Various utility scripts for device tuning

### _common/
Shared helpers for the `generate_*_report.py` family (extracted 2026-10-03,
TODO R-25). Every generator used to carry its own copy, and two of the `_fmt`
copies had already drifted apart — one rendered `1e-7` as `0.0000`, silently
flattening a real measurement to zero in a committed report.

| Helper | Replaces | Notes |
|---|---|---|
| `fmt_metric(v, nd=4)` | `_fmt` in `generate_fouriergsnet_sim_report.py`, `generate_gsnet_offline_report.py` | **The gsnet behaviour is the fix.** `-` for `None`/NaN/inf, integer compaction `>= 10`, scientific below `1e-4` and at/above `1e5`. |
| `fmt_ratio(value, digits=6)` | `_fmt` in `generate_oopao_{vs_numpy,impact}_report.py` | Scientific `< 1e-3` / `>= 1e5`, `g` formatting, `0` reads as `"0"`. |
| `fmt_general(v, spec=".4g")` | `_fmt` in `generate_slm_pib_online_report.py` | Passthrough to `format(v, spec)`; no `None`/NaN special case. |
| `fmt_fixed(v, nd=4)` | `_fmt` in `generate_slm_pib_rms_pib_report.py` | Em dash for NaN; **raises on `None`** (recorded limitation, kept). |
| `fmt_signed(v, spec="+.4f")` | `_fmt` in `generate_shape_objective_comparison.py` | `"n/a"` for non-finite. |
| `markdown_table(headers, rows)` | `_markdown_table` ×2 | GitHub-flavoured table. |
| `savefig(fig, path, dpi=150)` | `_savefig` ×3 | `bbox_inches="tight"` + closes the figure. |
| `iters_to_threshold(curve, threshold)` | ×2 | 1-based first crossing; `- 1e-12` absorbs float noise. |
| `format_iters(v)` | ×2 | Em dash when the threshold was never reached. |

**The five formatters are deliberately NOT merged.** A 170-probe before/after
comparison of the old inline copies against `_common` came out **168 identical**;
the only 2 differences are the sanctioned tiny-value fix above. Merging the other
three would have rewritten already-committed reports (`0.5` → `0.5000`,
`1e+05` → `100000`, `nan` → `-`, …), which is why each keeps its own name.

Using them from a generator requires the repo root on `sys.path` (a direct
`python scripts/<name>.py` does not add it; pytest does via
`pythonpath = ["src", ".", "scripts"]` in `pyproject.toml`):

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import fmt_metric, markdown_table
```

Behaviour is pinned by `tests/ao_shaping/scripts/test_common_helpers.py`, and
`test_common_helpers_not_reintroduced.py` fails if any generator grows a local copy
again.

### sync_report_provenance.py

Owns the provenance header in every report under `report/` plus the
`report/README.md` index. Both are **generated** — the same reason a report's
numbers are: hand-maintained provenance drifts the moment a generator is renamed,
which is exactly the failure the 2026-10-05 `docs/` → `report/` migration had to
clean up (reports pointed at scripts that no longer existed under those names).

```bash
python scripts/sync_report_provenance.py            # headers + index
python scripts/sync_report_provenance.py --check    # verify only, exit 1 on drift
python scripts/sync_report_provenance.py --verbose  # list every file touched
```

**Adding a report** means adding one entry to `REPORTS` keyed by repo-root-relative
path and re-running the sync. Nothing else needs to know the rule.

| API | Purpose |
|---|---|
| `provenance_block(key, depth)` | The `> 生成脚本` / `> 复现命令` / `> 运行环境` header text. `depth` is the report's directory depth below the repo root, so the relative script link is computed rather than hardcoded. |
| `provenance_block_for(key)` | Fenced block, or `""` when unregistered. For **writers**. |
| `insert_header(body, key)` | Places the block into a report body, idempotently. **The single placement implementation** — the sync pass and every writer call it, because two placement rules means `--check` reports drift forever on a file that is actually correct. |
| `render_index()` | The `report/README.md` body: report → script → 离线/硬件, with registered-but-unproduced reports listed separately so the index has no dead links. |

**Writers must stamp their own output.** A generator that overwrites its report
without calling `insert_header` deletes the header on every run — three did exactly
that, and the test suite regenerates two of them, so `pytest` alone was enough to
strip it. `render_index()` is filesystem-aware, so a registered report that has not
been produced yet is listed as pending instead of linked.

Pinned by `tests/ao_shaping/scripts/test_report_provenance.py` (registry ↔ disk,
header presence, no stacked headers, link resolution, index links).

## Common Patterns

Most Python scripts in this directory follow these patterns:
1. Set up paths to import from the `src` directory
2. Configure matplotlib to use 'Agg' backend for non-interactive plotting
3. Create timestamped output directories under `logs/`
4. Generate plots and save them as PNG files
5. Save data as CSV and JSON files for further analysis
6. Print the output directory path upon completion

## Dependencies

These scripts require:
- Python 3.12+
- Packages listed in `pyproject.toml` (numpy, pandas, matplotlib, seaborn, etc.)
- Stable Baselines3 for RL scripts
- TensorBoard for event processing
- MATLAB Runtime for dm_sim/ scripts (if applicable)

To install dependencies:
```bash
pip install -e .
```

## Notes

- Many scripts modify `sys.path` or set `PYTHONPATH` to import from the `src` directory
- Output directories are typically created under `logs/` with timestamps
- Plots are saved with high DPI (150-200) and tight bounding boxes
- Scripts often generate both visualizations (PNG) and data files (CSV, JSON)
- Some scripts have dependencies on specific hardware or MATLAB for full functionality

## Related: `src/ao_shaping/tools/slm/` (SLM bench probes)

The SLM bench probes are **not** in this directory — they are importable package
modules under `src/ao_shaping/tools/slm/`, so they can be unit-tested offline and
are documented separately:

- Directory index: [`src/ao_shaping/tools/slm/README.md`](../src/ao_shaping/tools/slm/README.md)
- **Pre-run guide (read this before any GS / GSNet / SPGD run)**:
  [`docs/slm/pre_run_characterization.md`](../docs/slm/pre_run_characterization.md)

Run them with `python -m ao_shaping.tools.slm.<name>` (they are not registered as
`main.py` Click commands). Unlike most entries in *this* file, the three
characterisation probes need no report generator — they persist recorder-style
pickles plus a `summary.npz` under `data/slm_*/`:

| Probe | Answers |
|---|---|
| `slm_drift_probe` | Is the flat field stable, and is exposure linear? |
| `slm_floor_probe` | What is the noise floor, how long does settling take, and is the noise read noise or drift? |
| `slm_abba_probe` | Is a dense random-phase perturbation resolvable above the drift floor at all? |

All three support `--no-hw` (print the acquisition plan, exit 0, touch no
hardware), which is also how their tests run in CI.