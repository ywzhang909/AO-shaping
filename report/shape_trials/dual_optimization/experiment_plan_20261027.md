# Experiment Plan: Transfer Learning + Closed-Loop Dual Optimization on Physical Hardware

## 0. Document Info

- **Prepared**: 2026-10-27
- **Experiment type**: Hardware (SLM + CCD closed loop)
- **Entry points**: `slm-model-in-loop` (model-in-loop), `slm-transfer-shapes` (transfer + multi-shape)
- **Status**: Ready pending G0-2 optical drift resolution (see §8)

---

## 1. Objective

Run the transfer-learning + model-in-the-loop dual optimization flow on physical SLM+CCD hardware to shape the far field into target shapes. The dual optimization alternates:

1. **Step A (fit)**: Display strong, varied random pupil probes on the SLM; measure far-field frames on the CCD; fit a shared Zernike aberration vector by minimising loss(measured CCD, predicted far field).
2. **Step B (synthesis)**: Freeze the fitted aberration; optimise a full-pixel SLM phase against the target shape; display it; measure the result.
3. **Guards**: Trust region on coefficient change + round acceptance test (loss worsening / score degradation / CV regression) prevent the two estimators from chasing each other.
4. **Final bake-off**: Keep the shaped phase only if it beats the flat baseline.

Transfer learning (pre-capture) builds the initial forward model from phase-diverse probes before the iterative loop begins, providing a warm start for Step A.

---

## 2. Hardware Setup

| Component | Specification |
|-----------|--------------|
| SLM | Santec SLM#1, serial 22030108, @1064 nm, 2π = 993 gray levels |
| CCD | Daheng MER2-507-23GM, pupil center (960, 720) panel px |
| Lens | 2f Fourier bench, f = 125 mm |
| Exposure | 13.7846 ms (ceiling 18.8846 ms), auto-exposure at 0.02 ms initial |
| Optical parameters | panel_span_px = 900, pupil radius 450 px, wavelength 1064 nm |

### 2.1 Verified Gate-0 status (from `report/sim2f_bench/alignment_20261027/report.md`)

| Gate | Threshold | Status | Notes |
|------|-----------|--------|-------|
| G0-1 Exposure | flat peak ∈ [0.2, 0.8], saturated < 0.3% | **PASS** | flat peak 0.471–0.518, 0.000% saturated |
| G0-2 0-order stability | A1-vs-A2 drift < 2 px | **FAIL** | max drift 12.99 px (6.5× limit); real optical drift |
| G0-3 Phase reaches panel | min norm(B-A)/norm(A1-A2) > 3 | **PASS** | min ratio 5.21, 24/24 |
| G0-4 Geometry identifiable | min ncc(measured, twin) ≥ 0.5 | **FAIL** | min ncc 0.190; correlated with G0-2 drift |

**G0-2 must be resolved before the hardware experiment** (see §8).

---

## 3. Software Setup

| Item | Value |
|------|-------|
| Python | 3.12+ |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU (CUDA available, 1 device) |
| Device resolution | `None` / `"auto"` → CUDA if available, else CPU |
| Zernike convention | Noll 1976 (canonical: `zernike_utils.py` → `zernike_calc.py`) |
| Phase contract | Raw unwrapped radians; wrap only in `Santec.create_phase_from_array()` |

---

## 4. Experiment Flow

### 4.1 Phase 0: Transfer Learning (Forward Model Build)

Capture phase-diverse probes on the current optical path, fit the physical forward model, export at control resolution.

**Entry**: `python -m ao_shaping.tools.slm.transfer` (subcommands: plan/run/replay/audit/record)

**Stages**:
1. **Capture**: Display probe phases, record CCD frames + phase sidecars
2. **Nominal init**: Build initial checkpoint from capture alone (no historical model)
3. **Fit**: Fit physical forward model (Zernike aberration + optical parameters)
4. **CCD blocks**: Validate with CCD-anchored optimization blocks (non-GS seed → Mono12 → continuation → dense)
5. **Replay**: Re-display archived phases at common bit depth/exposure
6. **Record**: Save Recorder data (CSV + HDF5 + GIF)

**Key parameters**:
- `--device auto` (CUDA auto-detect)
- `--shapes square,rectangle,triangle`
- `--count 24` (probes per shape)
- `--blocks 4`, `--updates 10`, `--frames 3`
- `--side 80`, `--period 64`

### 4.2 Phase 1: Model-in-the-Loop Dual Optimization

Iteratively refine aberration estimate and shape targets using the forward model.

**Entry**: `python src/ao_shaping/main.py slm-model-in-loop [OPTIONS]`

**Per round**:
1. **Step A**: Display `probe_count` (default 8) strong random probes; fit shared Zernike aberration via Adam; hold out `holdout_probe_count` (default 2) independent probes for correlation check.
2. **Step B**: Freeze aberration; optimize full-pixel SLM phase for target shape (SPGD/differentiable); display; measure composite score (EE + CV).
3. **Guards**: Trust region (L2 cap on coefficient step) + acceptance test (loss worsening / score degradation / CV regression) → accept or rollback.
4. **Escalation**: On rejection, damp LR, add probes, optionally switch to uncertainty sampling.

**Termination**: `n_rounds` (default 6) completed, or early stop on score threshold, or rejection streak exceeded (`max_rejection_streak=3`).

### 4.3 Phase 2: Replay & Analysis

Re-display archived seed/final phases at common conditions for comparison. Generate report from Recorder data.

---

## 5. Metrics to Collect

### 5.1 Recorder keys (canonical, from `recorder_keys.py`)

| Key | Type | Description |
|-----|------|-------------|
| `_img` | ndarray (H, W) | CCD frame (raw or prepared) |
| `_phase` | ndarray (H, W) | SLM phase (raw radians) |
| `_c` | ndarray (n_coeffs,) | Zernike coefficients |
| `_grad` | ndarray | Gradient |
| `_epoch` | int | Epoch/round number |

### 5.2 Per-round metrics (from model-in-loop recorder history)

| Metric | Description |
|--------|-------------|
| `round` | Round index (1-based) |
| `accepted` | Whether round was accepted (bool) |
| `reason` | Acceptance verdict reason tag |
| `score_before` | Best committed composite before this round |
| `score_after` | Composite score of measured shaped phase |
| `uniformity_cv` | Coefficient of variation in target box |
| `ee_after` | Encircled energy after shaping |
| `loss_after` | Step A fit loss after this round |
| `coeff_step_l2` | Unclamped L2 norm of coefficient change |
| `probe_count` | Number of probes used this round |
| `step_a_lr` | Step A learning rate (may be damped) |
| `loss_before` | Step A loss at start of fit |

### 5.3 Geometry metrics

| Metric | Description |
|--------|-------------|
| `method` | Geometry calibration method |
| `correlation` | Geometry bake-off correlation |
| `camera_px_per_model_px` | Calibrated scale factor |
| `target_side_camera` | Target width in camera pixels |
| `target_side_model` | Target width in model pixels |

### 5.4 Summary metrics

| Metric | Description |
|--------|-------------|
| `status` | Terminal status (completed/aborted_unidentifiable/aborted_rejection_streak/bake_off_rejected) |
| `flat_score` | Flat baseline composite score |
| `best_score` | Best shaped composite score |
| `rounds accepted` | Number of accepted rounds / total rounds |

### 5.5 Transfer-specific metrics (from transfer system)

| Metric | Description |
|--------|-------------|
| `core_cv` | Core coefficient of variation (target uniformity) |
| `encircled_energy` | Energy in target box / total energy |
| `phase_rms` | RMS of SLM phase |
| `holdout_correlation` | Pearson correlation on held-out probes |
| `ncc_net` | Normalized cross-correlation (net vs flat) |

---

## 6. Model Parameters to Freeze

### 6.1 Frozen Zernike modes (unidentifiable from intensity)

```python
frozen_modes = (1, 2, 3)  # Noll: piston, tip, tilt
```

**Rationale**: Piston and tilt are unidentifiable from far-field *intensity* (translation/phase offset). Leaving them free puts 56% of fitted coefficient norm into degenerate directions without improving fit.

### 6.2 Fixed optical parameters (do not change during optimization)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_orders` | 10 | Zernike orders to fit |
| `region` | 64 | Forward-model pupil grid edge (model px) |
| `wavelength_nm` | 1064 | Operating wavelength |
| `slm_pixel_um` | 8.0 | SLM pixel pitch |
| `camera_pixel_um` | 2.2 | CCD pixel pitch (measurement anchor) |
| `panel_span_px` | 900.0 | Model grid coverage in panel px |
| `far_field_padding` | 8 | FFT zero-padding factor |
| `pupil_oversampling` | 1 | Sub-pixel pupil interpolation |
| `focal_length_m` | 0.0 | Derived from measured scale (0 = auto) |

### 6.3 Working precision

```python
dtype = "float32"  # GPU-optimized; quantiser discards float64 mantissa anyway
```

**Rationale**: SLM panel quantises to 10-bit gray (2π = 993 levels ≈ 6.33e-3 rad/level). Float32 and float64 produce byte-identical gray maps. Float32 is 1.46x–1.69x faster on GPU.

### 6.4 Self-calibration (off by default)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selfcal_affine` | False | Pupil-plane affine τ_θ (6 scalars) |
| `selfcal_travel` | False | Per-mode-family travel scale γ (7 scalars) |
| `fourier_weight` | 0.0 | Fourier-domain L1 term weight |
| `loss_kind` | "mse" | Image-domain loss (mse or l1) |

**Rationale**: Off by default to keep forward model output bit-for-bit identical to historical runs. Enable only when bench aberration warrants identification.

### 6.5 Trust region & guards (fixed thresholds)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trust_region_c_l2` | 0.5 | Fixed per-round L2 cap on coefficient step (rad) |
| `acceptance_loss_delta` | 1e-4 | Tolerance on loss worsening |
| `acceptance_score_eps` | 1e-3 | Tolerance on score degradation |
| `max_cv_regression` | 0.02 | Allowed CV increase between shaped rounds |
| `max_rejection_streak` | 3 | Consecutive rejections before abort |
| `min_geometry_correlation` | 0.5 | Abort if geometry correlation below this |

---

## 7. GPU Configuration

### 7.1 Verified GPU availability

```
torch version: 2.11.0+cu128
CUDA available: True
CUDA devices: 1
  Device 0: NVIDIA GeForce RTX 5080 Laptop GPU
```

### 7.2 Device resolution chain

1. **CLI**: `--device auto` (or `None` default) → passed to `SlmModelInLoopConfig.device`
2. **Config**: `device: str | None = None` → `ZernikeCoefficientOptimizer.__init__`
3. **Optimizer**: `None` → `torch.device("cuda")` if `torch.cuda.is_available()` else `torch.device("cpu")`
4. **Transfer system**: `resolve_device("auto")` → same logic, records resolved device in `session.json`

### 7.3 Performance (measured on RTX 5080 Laptop, 2026-10-19)

| Region | float64/CUDA | float32/CUDA | float64/CPU | float32/CPU |
|--------|-------------|-------------|-------------|-------------|
| 128 | 1.71 s | 1.71 s | 3.16 s | 2.36 s |
| 256 | 3.40 s | 3.05 s | 10.64 s | 5.96 s |

CUDA is 1.85x–3.13x faster than CPU at equal precision.

### 7.4 Recommended invocation

```bash
# Model-in-loop with GPU (auto-detect)
python src/ao_shaping/main.py slm-model-in-loop \
  --cam-type daheng --cam-id 0 --exposure-time-ms 13.7846 \
  --slm-number 1 --slm-wavelength 1064 \
  --target-shape square --target-side 60 \
  --n-rounds 6 --device auto \
  --debug

# Transfer shapes with GPU (auto-detect)
python -m ao_shaping.tools.slm.transfer run \
  --shapes square,rectangle,triangle \
  --device auto \
  --count 24 --blocks 4 --updates 10
```

---

## 8. Prerequisites (BLOCKING)

### 8.1 G0-2 Optical Drift Resolution

**Status**: BLOCKING — must be resolved before hardware experiment.

**Problem**: 0-order drift 12.99 px max (limit 2 px), 10/24 samples exceed threshold. Real optical drift (thermal/vibration/SLM settle).

**Required actions**:
1. Increase SLM settle time (`settle_wait_s`, `settle_max_wait_s`)
2. Verify SLM memory slot rotation (consecutive writes must target different slots)
3. Physical stabilization (thermal management, vibration isolation)
4. Re-run hardware bench to verify reproducibility

**Reference**: `report/sim2f_bench/alignment_20261027/report.md` §2, §5

### 8.2 Other prerequisites

- [ ] SLM phase table verified (2π = 993 gray at 1064 nm)
- [ ] CCD exposure in linear range (G0-1 PASS: 13.7846 ms)
- [ ] Geometry calibration file available (or auto-session enabled)
- [ ] Forward checkpoint available (optional, for warm-start)
- [ ] Debug directory writable for artifact saving

---

## 9. Execution Commands

### 9.1 Pre-experiment (simulation dry run)

```bash
# Verify model-in-loop on sim backend
python src/ao_shaping/main.py slm-model-in-loop \
  --cam-type sim --target-shape square --target-side 60 \
  --n-rounds 3 --device auto

# Verify transfer system simulation
python -m ao_shaping.tools.slm.transfer simulate
```

### 9.2 Hardware experiment (after G0-2 resolution)

```bash
# Single-shape model-in-loop (hardware)
python src/ao_shaping/main.py slm-model-in-loop \
  --cam-type daheng --cam-id 0 --exposure-time-ms 13.7846 \
  --slm-number 1 --slm-wavelength 1064 \
  --target-shape square --target-side 60 \
  --n-rounds 6 --device auto \
  --debug

# Multi-shape transfer (hardware, from scratch)
python -m ao_shaping.tools.slm.transfer run \
  --shapes square,rectangle,triangle \
  --device auto --count 24 --blocks 4 --updates 10
```

### 9.3 Post-experiment analysis

```bash
# Replay archived phases
python -m ao_shaping.tools.slm.transfer replay --run-dir <run_dir>

# Generate report
python scripts/slm/generate_model_in_loop_cv_report.py  # if applicable
```

---

## 10. Expected Outcomes

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| G0-2 0-order drift | < 2 px | Hardware bench re-run |
| G0-4 Geometry correlation | ≥ 0.5 | Model-in-loop geometry bake-off |
| Rounds accepted | ≥ 1 of 6 | Per-round summary |
| Best score > flat score | Yes | Final bake-off |
| Recorder data saved | Yes | CSV + HDF5 + GIF in run dir |
| GPU utilization | CUDA active | Device resolution in session.json |

---

## 11. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Optical drift (G0-2) | Resolve before experiment; increase settle time |
| Model mismatch (G0-4) | Abort if correlation < 0.5; use slm-gs-refine fallback |
| Rejection streak | Abort after 3 consecutive rejections; bench outside model |
| SLM settle | Memory slot rotation; power cycle if hung |
| GPU OOM | float32 default; region=64 keeps FFT small |
| Data loss | Recorder saves incrementally; debug_dir for artifacts |

---

## 12. References

- `report/sim2f_bench/alignment_20261027/report.md` — Gate-0 alignment conclusion
- `src/ao_shaping/optimizer/wfless/slm_model_in_loop.py` — Model-in-loop optimizer
- `src/ao_shaping/tools/slm/transfer/workflow.py` — Transfer learning session
- `src/ao_shaping/runners/slm/model_in_loop_runner.py` — CLI runner
- `src/ao_shaping/runners/slm/transfer_shapes_runner.py` — Multi-shape transfer runner
- `src/ao_shaping/utils/io/recorder_keys.py` — Recorder key constants
- `src/ao_shaping/tools/slm/transfer/device.py` — Device resolution
