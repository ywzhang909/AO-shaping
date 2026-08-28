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
- **Per-IP references** — automatically uses `<IP>-000.png` as reference for each IP
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