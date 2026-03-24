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