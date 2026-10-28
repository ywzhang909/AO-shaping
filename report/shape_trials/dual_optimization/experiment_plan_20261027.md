# Experiment Plan: Transfer Learning + Closed-Loop Dual Optimization on Physical Hardware

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`src/ao_shaping/runners/slm/model_in_loop_runner.py`](../../../src/ao_shaping/runners/slm/model_in_loop_runner.py)
> **数据/关联脚本**: [`src/ao_shaping/runners/slm/transfer_shapes_runner.py`](../../../src/ao_shaping/runners/slm/transfer_shapes_runner.py)
> **运行环境**: 离线（方案文档；设备已预检，实验尚待 G0-2 光路漂移处置）
> **说明**: 迁移学习 + model-in-loop 双优化流程的实验方案：G0 前置门禁、逐轮交替的判决口径与记录要求；不含任何新测量。
<!-- provenance:end -->

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

---

已扫描 `report/shape_trials/multishape_scale/` 的规划、实验、分析与耗时报告，并核对了 `s1200`、`s600_fixed` 的原始 `config.json`、`summary.json` 和 `manifest.json`。

目前的判断是：**多形状优化流程已经具备执行能力，部分条件下有明确的均匀度改善；但原定的“无 GS、同模型、三尺度对照”尚未完成，尚不能判断尺度是否是主要限制，也不能认定已经验证了替代 GS 的效果。** 下一轮应优先完成测量口径与更新方向验证，再补齐对照矩阵。

本次仅进行了离线核对，没有连接设备或下发相位。分析采用了仓库 [MIL/transfer skill](../../../libs/skills/slm-mil-transfer/SKILL.md) 中区分“流程完成、模型有效、整形成功”的方法。

### 1. 原定目标完成情况

原定目标来自[规划报告](../multishape_scale/multishape_scale_planning_20261028.md:22)：正方形、三角形、五角星、圆形，分别测试 **200 / 600 / 1200 原生 CCD px**，排除 GS 初相位，在一致条件下比较尺度与形状的影响。

| 原定目标 | 当前证据 | 评估 |
|---|---|---|
| 四形状 × 三尺度，共 12 个实验组合 | 1200、600 各完成四形状的四段优化；200 未执行 | 执行覆盖 **8/12** |
| 不使用 GS 起点 | 1200 四形状实际采用反投影初相位；600 四形状实际采用 GS | 符合无 GS 条件的覆盖 **4/12** |
| 同一模型、同光路条件下比较尺度 | 600 前光强发生变化，并重新采集、拟合模型 | **未形成受控跨尺度对照** |
| 优化后同曝光独立重放验收 | 1200 有多批重放，但存在跨时段、失效帧等问题；600 最终重放因网格检查失败 | 部分完成 |
| 证明均匀度可以改善 | 1200 部分阶段改善；600 稠密阶段四形状均改善 | 有阶段性证据 |
| 证明生成了质量达标的目标形状 | 主要报告核心 CV，缺少统一的形状覆盖、边界和相对能量验收 | 尚不能判定 |
| 用响应记录判断模型适用区间 | 有过程记录，但尚未完成系数、响应、方向性与性能的联合分析 | 部分完成 |

需要更新一个时间顺序上的认识：[分析报告](../multishape_scale/multishape_scale_analysis_20261028.md:28) 中“600 尚未开始”的结论已过时；[后续耗时报告](../multishape_scale/multishape_scale_wallclock_20261028.md:316) 已记录 `s600_fixed` 完成全部 16 个优化阶段。不过，该轮使用 GS 起点，不能补齐原定的无 GS 对照。

此外，**1200 的“无 GS”不等于平面或随机相位冷启动**：实际是形状相关的反投影初相位。两种能力应分别评价。

### 2. 哪些优化确实带来了改善

1200 px 的三段 Mono12 阶段末 CV 如下。CV 越小，均匀度越好；这些是阶段末读数，不能直接当作同曝光重复验收的最终排名。

| 形状 | `mono12_cv` | `mono12_continue` | `mono12_dense` | 当前判断 |
|---|---:|---:|---:|---|
| 正方形 | 0.6574 | 0.6442 | 0.6443 | 后续改善有限，趋于平台 |
| 三角形 | 0.6640 | 0.6582 | 0.6647 | 后续收益有限，需核对饱和与重复性 |
| 五角星 | 0.7069 | 0.7144 | 0.7316 | 继续优化未显示稳定收益 |
| 圆形 | 0.6774 | 0.6925 | 0.6958 | 继续优化未显示稳定收益 |

不能由此说五角星从未改善：其 `mono12_cv` 段首到段末为 **0.7838 → 0.7069**，有改善；问题是后续阶段没有维持持续收益。

600 px 则提供了更有区分力的现象：

- `mono12_cv` 四种形状 **32 个候选块全部被拒绝**。
- `mono12_continue` 四种形状共 48 块，仅接受 2 块。
- `mono12_dense` 四种形状共 16 块，全部接受。

对最后一个阶段，原始记录显示同段曝光一致、记录中饱和像素数为 0：

| 形状 | 稠密阶段段首 CV | 段末 CV | 相对下降 |
|---|---:|---:|---:|
| 正方形 | 1.0264 | 0.6873 | 约 33.0% |
| 三角形 | 1.0879 | 0.8358 | 约 23.2% |
| 五角星 | 1.0811 | 0.7584 | 约 29.8% |
| 圆形 | 0.9083 | 0.7540 | 约 17.0% |

**这是本轮最值得追踪的正面结果。** 它说明改变后的模型与采样配置下，可微更新能够产生 CCD 可见的均匀度改善。

但还不能把收益归因于“采样更密”：该阶段同时切换了 CCD 采样步长及所用模型/几何配置。它也没有完成独立重放验收。尤其正方形的目标峰值下降明显，应同时检查目标平均强度和相对能量，排除通过降低有效信号获得较低 CV 的情况。

### 3. 猜想验证情况

| 猜想 | 当前结论 | 原因 |
|---|---|---|
| H1：目标尺度主导 CV，形状影响较小 | **未验证** | 200 缺失；600 与 1200 的初相位、模型、光路条件不同 |
| H2：1200 较差主要来自控制频带，而非模型误差 | **未验证** | 没有频带单变量对照；600 常规阶段也表现出更新无效，而模型/采样切换后改善 |
| H3：200 与零级完全分离，可作干净参照 | **原先的几何前提已被推翻** | 实测载波位移约为 `(14.76, −129.70)` px，半径 60 px 的排除区与 200 目标框存在重叠 |
| H4：约 300 ms 曝光可以避免核心饱和 | **不能作为通用结论** | 是否饱和取决于相位、形状、光强和响应；已有较短曝光出现全帧饱和的记录，需要逐核心核查 |
| 更长优化、更多阶段会继续改善 | **不支持作为默认策略** | 1200 五角星、圆形后续阶段未持续改善；600 常规阶段大量拒绝 |
| 物理预测模型方向不准是重要限制 | **有较强迹象，尚未完成直接验证** | 600 常规阶段候选几乎全部被拒绝，但需要正负方向差分才能区分模型方向错误、步长问题及目标函数失配 |
| 记录 I/O 是重要耗时来源 | **已定位并获得修复收益证据** | 整体重写历史记录造成写放大；报告给出正方形总耗时 780.6 → 658.4 s，但不能将全部差额视为严格隔离的 I/O 收益 |

目前更合理的工作假设是：**限制不只是相位梯度或控制频带；模型局部方向性、几何配置和测量口径也在影响优化。**

### 4. 报告中应纠正的几项解释

这些问题会直接影响下一轮计划：

1. **“探索导致阶段末变差”的解释不成立。**  
   本轮配置 `exploration_temperature=0`，没有启用探索。仅看到 `exploration_budget=0.25`，不能认定允许了非单调探索，也不能据此解释 CV 恶化。

2. **112 个过程块不等于 112 次 CCD 候选验收。**  
   `transport_cv` 的配置为 `ccd_check=False`、`accept_metric=loss`，它有 CCD 锚定采集，但候选接受依据是模型损失。

3. **曝光影响已经大到足以改变结论。**  
   例如同一正方形相位在重放中，CV 随 20 / 50 / 90 ms 从约 0.4190 变为 0.6445 / 0.7588。理想线性、无偏置条件下，整体强度缩放不应改变 CV；需要检查偏置、非线性、实际曝光、帧对应和光路漂移，不能直接把最低 CV 当成最佳光场。

4. **无需绝对辐照度标定，也可以报告相对 EE。**  
   原始 `manifest.json` 已有 `ee_roi`。应固定分母窗口和有效像素，报告相对能量比例；绝对衍射效率则另行标定。

5. **已有下一轮脚本与无 GS 目标冲突。**  
   耗时报告生成的 `r2_next` 计划使用 `--initial-phase gs --no-seed-bakeoff`。它适合 GS 起点的效率测试，不适合直接作为下一轮替代 GS 的主实验。

### 5. 建议的下一轮实验计划

建议按以下顺序执行，前一阶段决定后一阶段的配置，而不是直接重复完整四段流程。

| 阶段 | 实验内容 | 要回答的问题与产出 |
|---|---|---|
| ① 测量一致性 | 固定参考相位，曝光升降阶梯重复采集；记录请求/读回曝光、增益、帧时间、位深、核心饱和、均值、CV、相对 EE | 找到有效比较区间，判断曝光相关 CV 偏差来源 |
| ② 补齐现有验收 | 重放 `s600_fixed` 的段首、历史最佳、段末相位，按实际网格分组；相同相位、曝光重复测量 | 验证稠密阶段的改善能否复现，不必重做全部优化 |
| ③ 检验模型方向 | 优先选 600 正方形和圆形，测多个方向、多个幅度的 `φ±εd`；包含模型梯度、低阶模式和带限随机方向 | 比较预测与 CCD 的强度差分、CV 变化符号和幅值，定位方向错误或步长问题 |
| ④ 拆分稠密阶段收益 | 分别固定模型改变 CCD 采样、固定采样改变模型/几何；控制网格和频带再单独比较 | 确定收益来自采样、模型、几何还是频带 |
| ⑤ 补齐无 GS 矩阵 | 同一稳定光路、共享模型下补 200/600/1200；先圆形和正方形，再三角形、五角星；交错执行尺寸 | 回答尺度与形状的主效应 |
| ⑥ 验证改进模型 | 根据方向检验结果增加最少必要参数，再用未参与拟合的相位与形状验证 | 防止拟合相关性提高、CV 预测或优化方向反而恶化 |

无 GS 实验建议分成两条明确标记的路线：

- **严格冷启动：**平面或随机相位，不继承历史整形相位。
- **解析初始化：**反投影或能量映射初相位，不调用 GS。

两者可以共享物理预测模型，但不能合并声称“平面冷启动成功”。

下一轮每个实验组合至少输出三类结果：**固定核心 CV、形状覆盖/边界误差、相对 EE**；同时保存零级剔除比例、饱和比例、相位来源、模型检查点、Zernike 系数及参数变化、物理中间响应、预测与实测差分和精确时间戳。独立验收使用重复测量估计不确定性，不能沿用一个固定的“2 个百分点”阈值。A–B–B–A 可以用于漂移明显或改善接近测量波动的候选，保持为可选项。

在模型改进优先级上，建议先做**局部方向验证与信任域校正**，随后再评估带限/MTF 正则、入射振幅残差和几何余项。单个 A→B 差分可以监控实际更新是否兑现预测，但不足以完整验证 Jacobian；若要证明方向可靠，仍需要多个正负扰动。

本轮最有价值的发现，是 **600 px 从“常规阶段几乎全部拒绝”转为“稠密阶段全部接受并明显改善”**。下一轮优先拆解这项差异，比直接增加迭代次数或同时增加大量模型参数，更有机会找到可复用的改进。
