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
- Saves output to `docs/simulation/atmospheric_spot_phase_comparison.png`

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
  `docs/benchmarks/device_less_full/`

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
- Saves results to `docs/simulation/sim_spgd_zernike_tuning.json`
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

> **Shared analysis helpers**: `scripts/` report generators in the SLM/Zernike
> family delegate measurement/analysis logic to
> `src/ao_shaping/tools/slm/slm_scan_analysis.py` (`outlier_mask`, `clamp_shift`,
> `parabolic_min`, `latest_match`, `group_raw_scan`, `analyze_linearity`,
> `LINEARITY_AMPS`) — scripts keep only figure/markdown rendering.

### generate_zernike_wfs_report.py

Generates the illustrated **Zernike phase → WFS readout distribution** report
(needs hardware: Santec SLM-200 + Thorlabs WFS).

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_zernike_wfs_report.py
python scripts/generate_zernike_wfs_report.py -o docs/slm/zernike_wfs_report
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
| `-o, --output-dir` | `docs/slm/zernike_wfs_report` | Output directory |

### generate_zernike_response_matrix_report.py

Generates the illustrated **Zernike response matrix** report
(creation / analysis / detection). **Fully offline** — reads saved artefacts, no
hardware.

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_zernike_response_matrix_report.py
python scripts/generate_zernike_response_matrix_report.py --h5 <path> -o docs/slm/<dir>
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
python scripts/generate_zernike_linearity_report.py -o docs/slm/zernike_linearity
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

**What it does** (writes `report.md` + `figures/` + `metrics.csv` to `docs/zernike_farfield_sim/`):
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

**What it does** (writes to `docs/heuristic_pib/`):
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
`optimize_slm_zernike_pib`, writing `docs/slm_pib_heuristic_hw/` (`report.md`,
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

### measure_shape_sensitivity.py

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
into `-o/--output`, default `docs/slm_pib_heuristic_hw/`):
- **Noise floor**: evaluates the shaping score on `--n-frames` frames of the *same*
  fixed phase and reports its std (`ΔJ_noise`).
- **Signal**: for each `Δa` in `--deltas`, writes `c ± Δa` (first Zernike mode =
  Noll 4 defocus, no tilt), averages `--n-repeat` pairs, and reports
  `ΔJ_signal = |mean(J+) − mean(J−)|` — exactly what SPGD turns into a gradient.
- **Verdict**: `SNR = ΔJ_signal / ΔJ_noise` — `>= 3` strong, `>= 2` usable, `< 2`
  unusable (raise `Δa` toward 0.2 rad, or average more frames per perturbation).

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
| `-o, --output` | `docs/slm_pib_heuristic_hw` | output directory |

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

**Outputs** (into `-o/--output`, default `docs/slm_pib_heuristic_hw/`):
- `objectives/<n>_<slug>_spot.png` — best un-windowed frame per variant with the target box drawn and the yardstick annotated;
- `objectives_summary.png` — CV / energy / peak bars per variant;
- `objectives.csv` — the raw numbers;
- an `<!-- OBJECTIVES_START --> … <!-- OBJECTIVES_END -->` section **appended idempotently** to `--append-to` (default `docs/slm_pib_heuristic_hw/report.md`), so re-runs replace rather than duplicate it.

| Option | Default | Description |
|--------|---------|-------------|
| `--algorithm` / `--epochs` | `sa` / `80` | pinned algorithm / iterations |
| `--cam-type` / `--cam-id` / `--cam-size` | `daheng` / `0` / `320` | camera backend / id / window |
| `--exposure-ms` / `--target-brightness` | `0.0` / `180` | `0` = auto-expose |
| `--zoom` | `300` | display zoom box (px) around the spot |
| `--append-to` | `docs/slm_pib_heuristic_hw/report.md` | report to append the section to |
| `-o, --output` | `docs/slm_pib_heuristic_hw` | output directory |

> ⚠️ **Measured (2026-09-20, fixed SA, fixed ROI = 21 px = 2×waist)**: CV 0.259 (energy-only)
> / 0.263 (`roi_pib`) / 0.273 (`e-5u`) / 0.278 (`log-u`) / 0.302 (`e-2u-0.5pk`), while repeating
> the SAME variant gave CV 0.354 then 0.302 — the between-variant spread is **inside the
> single-run variance**, so no objective can be declared the most uniform from one run.

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

**What it does** (writes to `docs/strehl_benchmark/`):
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
  cross-benchmark comparison table against `docs/heuristic_pib/summary.csv`
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
python scripts/generate_dm_response_matrix_report.py --h5 <path> -o docs/dm_response_matrix_report
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
  `scripts/reports/centroid_test_visualization/`

### generate_diff_shaping_report.py

Generates the illustrated differential beam shaping report (GS vs
differentiable shaping). GPU recommended, CPU fallback available.

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_diff_shaping_report.py
```

**What it does** (writes to `docs/slm_differential_shaping/`):
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
python scripts/generate_fouriergsnet_sim_report.py --matrix-dir data/fouriergsnet_sim/<ts> -o docs/fouriergsnet_sim
```

**What it does** (writes `docs/fouriergsnet_sim/report.md` + `figures/` + `gifs/`):
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
- Robust: per-scenario try/except; missing frames degrade to static-only with a
  warning; scenario dirs are scanned directly so the report can be regenerated
  mid-run or after the matrix completes (summary.json optional)

| Option | Default | Description |
|--------|---------|-------------|
| `--matrix-dir` | latest `data/fouriergsnet_sim/<ts>` | Matrix output dir |
| `-o, --output` | `docs/fouriergsnet_sim` | Report output dir |

### generate_slm_gsnet_sim_gif.py

Generates the slm-gsnet offline-sim verification GIFs (+ prints the markdown
snippet) from a `slm-gsnet spgd --cam_type sim --debug` artifact directory.
**Fully offline** — reads the saved PKL only, no hardware, no pipeline code.

**Usage:**
```bash
python scripts/generate_slm_gsnet_sim_gif.py
python scripts/generate_slm_gsnet_sim_gif.py --pkl data/debug/slm_gsnet_<ts>/<ts>/xxx.pkl
python scripts/generate_slm_gsnet_sim_gif.py -o docs/fouriergsnet_sim
```

**What it does** (writes `docs/fouriergsnet_sim/gifs/`):
- Loads the debug PKL (`{epoch: record}` with per-epoch `_img` CCD far-field
  frames + `_c` freeform phase vector, length `phase_grid²` = 576)
- `slm_gsnet_spgd_sim_far.gif` — 逐 epoch 远场 (CCD 帧, inferno)
- `slm_gsnet_spgd_sim_phase.gif` — 逐 epoch SLM freeform 相位 (24×24 网格,
  mod 2π, twilight)
- Both via the repo `_frames_to_gif` convention (reused from
  `generate_diff_shaping_report`, LANCZOS 128px + adaptive 256 palette, 15 fps)
- Prints the `![...](gifs/...)` markdown lines for embedding in
  `docs/fouriergsnet_sim/report.md` §5.6

| Option | Default | Description |
|--------|---------|-------------|
| `--pkl` | latest `data/debug/slm_gsnet_*/*/*.pkl` | Debug artifact pkl path |
| `-o, --output` | `docs/fouriergsnet_sim` | Output dir (GIFs → `<output>/gifs/`) |

### slm_pib_sim_run.py

Runs the **`slm-pib`** SPGD shaping pipeline **entirely in the simulation
environment** (no hardware). Wires the pure-numpy 2f-Fourier sim
(`src/ao_shaping/drivers/sim/slm_pib_sim.py`) into the **genuine**
`slm_pib_runner` CLI path so the standard debug artifacts (PNG/PKL/JSON) are
produced exactly as a hardware run would write them — ready for report
generation.

**Usage:**
```bash
python scripts/slm_pib_sim_run.py
python scripts/slm_pib_sim_run.py --epochs 300 --target-shape square
```

**What it does:**
- registers the `"sim"` camera type so `create_camera("sim", ...)` returns a
  `SimPibCCD` reading the shared far-field state (the `slm_pib_runner --cam_type`
  `click.Choice` was extended to include `"sim"`)
- monkeypatches `ao_shaping.optimizer.wfless.slm_zernike_pib.Santec` →
  `SimSLMPib` so the optimizer's SLM context manager instantiates the sim
  (no hardware, no DVI hang)
- invokes the genuine `slm_pib_runner.run` Click entry with `--cam_type sim`
  `--debug` (square target, Zernike n≤4, SPGD + AdaMOD)
- optical model: SLM = 2f front focal plane, CCD = back focal plane, so the CCD
  image is the 2D FFT (Fraunhofer far field) of the SLM pupil field — the
  0-order spot lands at frame centre and Zernike phase measurably modulates it

**Outputs:** `data/debug/slm_pib_shape_<ts>/` (PNG/PKL/JSON), then
`docs/slm_pib_sim/report.md` + `figures/` + `gifs/` via
`generate_slm_pib_sim_report.py`.

### generate_slm_pib_sim_report.py

Generates the illustrated **slm-pib simulation** report from a
`slm_pib_runner --debug` artifact directory. **Fully offline** — reads the saved
PKL/JSON only, no hardware, no pipeline code.

**Usage:**
```bash
python scripts/generate_slm_pib_sim_report.py
python scripts/generate_slm_pib_sim_report.py --debug-dir data/debug/slm_pib_shape_<ts>
python scripts/generate_slm_pib_sim_report.py --max-runs 3
```

**What it does** (writes `docs/slm_pib_sim/report.md` + `figures/` + `gifs/`):
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
| `--out` | `docs/slm_pib_sim` | Output dir for figures/gifs/report.md |

### generate_fouriergsnet_pipeline_report.py

Generates the FourierGSNet pipeline integration-test report. **Fully offline** —
pure markdown, no hardware, no figures.

**Usage:**
```bash
python scripts/generate_fouriergsnet_pipeline_report.py
python scripts/generate_fouriergsnet_pipeline_report.py -o docs/fouriergsnet_pipeline
```

**What it does** (writes `docs/fouriergsnet_pipeline/report.md`):
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

### generate_models_report.py

Generates the ML model training analysis report from TensorBoard event files.
**Fully offline** — reads saved training logs, no hardware.

**Usage:**
```powershell
$env:PYTHONPATH = "src"
python scripts/generate_models_report.py
```

**What it does** (writes to `docs/models_analysis/`):
- Maps run names to experiment stages (stage1_easy -> 阶段1, stage2_medium ->
  阶段2, stage3_ -> 阶段3, static_long -> 静态湍流-长训练, static_focus ->
  静态聚焦, turbulence_long / turbulence_mamba_best / turbulence_long_retry ->
  湍流-长训练, turb_focus -> 湍流聚焦, sac_ -> 冒烟/架构对比实验,
  tmp_sac_run -> 临时调试)
- Reads TensorBoard tags `rollout/ep_rew_mean`, `ao/best_pib`,
  `ao/best_strehl`, `ao/pib`, `ao/rms`
- Renders training curves at DPI 130 and writes the analysis report

## Verification Scripts

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
- `stdWavefront/` - Standard wavefront reference data
- Various utility scripts for device tuning

### reports/
Contains generated report artefacts from `scripts/` report generators:
- `centroid_test_visualization/` - centroid algorithm test report and figures
  (from `generate_centroid_test_visualization.py`)

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