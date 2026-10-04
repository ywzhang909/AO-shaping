# FourierGSNet 管线集成测试报告

<!-- provenance:start -->
> **生成脚本**: [`scripts/generate_fouriergsnet_pipeline_report.py`](../../scripts/generate_fouriergsnet_pipeline_report.py)
> **复现命令**: `python scripts/generate_fouriergsnet_pipeline_report.py`
> **运行环境**: 离线
> **说明**: 5 项集成测试的结果记录
<!-- provenance:end -->

**生成时间**: 2026-09-22 12:13:32

## 1. 概述

`tests/ao_shaping/drivers/sim/test_sim_fouriergsnet_pipeline.py` 提供 5 个
**端到端集成测试**, 驱动仓库根目录的独立脚本 `fouriergsnet_optimize.py` 全流程
(ShapingSystem → adaptive_gs_init → closed_loop → _append_final_record),
通过 `SimFourierGSNetEnv` (K_px=256, 窄束 w0=1.0, 无噪声) 提供离线 CCD/SLM 替身。
**不打开任何硬件, 不改动管线与仿真环境任何内部逻辑。**

| # | 测试 | 断言要点 |
|---|------|----------|
| 1 | `test_pipeline_adaptive_gs_init_and_measure` | 平场测量能量守恒 (sum≈1), GS 初值形状/设备, 均匀度非退化 (uni>0.02) |
| 2 | `test_pipeline_closed_loop_recorder_keys_and_improvement` | 闭环记录键结构, 末步均匀度>首步, `_append_final_record` 最终记录键 |
| 3 | `test_pipeline_closed_loop_recovers_from_static_aberration` | 静态像差下闭环精修 GS 初值 (uni_final > uni_init) |
| 4 | `test_pipeline_closed_loop_tracks_ou_turbulence` | 配置前 advance_time 空操作, 配置后逐帧漂移, 均匀度有界 (>0.02) |
| 5 | `test_pipeline_target_fn_shapes_flow_through` | circle/gaussian 目标形状贯穿 ShapingSystem 与闭环记录 |

**结果**: 5 passed (≈29 s); 回归集 21 passed (test_sim_fouriergsnet.py +
test_sim_fouriergsnet_turbulence.py + test_fouriergsnet_optimize.py)。

## 2. Harness 决策 (不改动管线的前提)

### 2.1 窄束 w0=1.0 — 解决 "工作区无信号"

默认 `BeamParams(region=512, w0=250)` 在 K_px=256 下, 远场光斑经 workzone
双线性重采样 (256→64) 后变成**亚像素点**, `measure()` 求和 ≤1e-12 触发
`RuntimeError("工作区无信号: 检查曝光/光路/标定文件")` (fouriergsnet_optimize.py L814)。

改用 `BeamParams(region=256, w0=1.0)` (窄源束 → 宽远场) 后:
- 平场覆盖 3522/4096 px, 均匀度非退化 (uni≈0.19);
- 管线自洽: `gauss_amp_from_farfield` 从宽光斑估计出窄 A_src (w0 裁剪到 0.2),
  与宽远场一致, GS 可正常整形。

### 2.2 torch.no_grad() 包裹 closed_loop — 规避真实 bug ✅ 已修复

> ⚠️ **2026-10-01 复核：此 bug 已在源头修复。** `fouriergsnet_optimize.py:943`
> 现在是 `phi.detach().cpu().numpy()`。下文 §3.1「修复方向」已完成，
> **不必再靠 `torch.no_grad()` 包裹绕过**（脚本里的 workaround 仍无害，但已非必需）。

`closed_loop` 曾对 `requires_grad` 的 `phi` 调用 `phi.cpu().numpy()`，会抛
`RuntimeError: Can't call numpy() on Tensor that requires grad` (网络前向输出
`phi` 经可训练卷积层, 带梯度)。当时**真实 CLI `run()` 同样受影响** ——
这是管线自身的 bug, 非测试引入。

`scripts/fouriergsnet_sim_train.py` 曾用 `replay=False` + `torch.no_grad()` 绕过。

### 2.3 SETTLE_S=0.0

> ⚠️ 2026-10-01：主脚本现为 `fouriergsnet_optimize.py:88 SETTLE_S = 0.2`，
> 0.0 只是测试里 pin 的值。

`_display_grayscale` 的 `time.sleep(settle_s)` 在 SETTLE_S=0.0 下为 no-op;
SimSLM 内部无 sleep, 无需全局 monkeypatch `time.sleep`。

## 3. 发现 (供后续修复参考)

> ⚠️ **2026-10-01：本节全部行号已漂移**（原文均基于旧版本 `fouriergsnet_optimize.py`）。
> 已核对并更新：`closed_loop` L936-938 → **L941-943**；`run()` L1135 → **L1105**；
> `_metrics` L905-910 → **L910-913**；`place_on_panel` L450 → **L445**；
> `fouriergsnet_env.py` `configure_turbulence` L282 → **L309**、`advance_time` L320 → **L347**；
> `utils/io/file.py` `Recorder` L221 → **L362**。
> §3.3 的签名不匹配与 §3.4 的转置风险**仍然成立**（已复核 `slm200_constants.py:13`
> 与 `fouriergsnet_env.py:67-68` 仍为 `(1920,1200)`，两个文件混用 `(宽,高)` 与
> `(PANEL_H, PANEL_W)`）。

### 3.1 `phi.detach()` bug — 影响真实硬件 CLI ✅ 已修复

```python
recorder.append({"step": step, "uniformity": uni, "encircled": ee,
                 "inference_ms": dt, "ccd": I.cpu().numpy(),
                 "phase": phi.detach().cpu().numpy()})   # ← 2026-10-01 已修复
```
`replay=True` (默认, 真实 CLI) 下曾同样崩溃。**现已修复，勿按旧描述再修一遍。**

### 3.2 均匀度恒为 0 的根因 (窄束下)

`_metrics` (L905-910): `uni = min(I·roi) / mean(I·roi)`。窄束远场光斑 (~8 px)
远小于 22×22 目标 ROI (484 px) 时, ROI 内最小值为 0 → **uni=0.0000 恒成立**
(基线/初值/末态皆然)。这是指标定义对"未填满 ROI"的固有行为, 不是管线崩溃。
测试 3/4 因此改用确定性断言: `uni_final > uni_init` (闭环精修 GS 初值,
实测 +0.03) 与 `ee > 0.5` (环围能量)。

### 3.3 FourierGSNetLite 签名与任务书不符

任务书描述 `FourierGSNetLite(n, dim=256, depth=6)`; 真实签名为
`FourierGSNetLite(K, n_zern, ch, src_mask)`。测试用
`fg.FourierGSNetLite(fg.K_UNROLL, fg.N_ZERN, 8, sys_.src_mask)`。

### 3.4 面板分辨率转置隐患 (仿真掩盖)

- 真实驱动: `PANEL_RES = (1920, 1200)  # (宽, 高)` (slm200_constants.py L13),
  期望灰度数组 shape=(1200, 1920) (driver.py L1295, L1458 `target_h=Panel_Res[1]`)。
- 管线: `place_on_panel` L450 `h, w = self.panel_res` → 把 (1920,1200) 当 (h,w),
  产出 (1920,1200) 面板。
- 仿真: `SimFourierGSNetEnv` 特意用 `PANEL_H, PANEL_W = 1920, 1200` (校准器
  h,w 约定) 匹配管线 → **仿真掩盖了真实硬件上的转置不匹配**。

## 4. 运行方式

```bash
# 集成测试
.venv/bin/python -m pytest tests/ao_shaping/drivers/sim/test_sim_fouriergsnet_pipeline.py -q

# 回归集
.venv/bin/python -m pytest tests/ao_shaping/drivers/sim/test_sim_fouriergsnet.py \
    tests/ao_shaping/drivers/sim/test_sim_fouriergsnet_turbulence.py \
    tests/ao_shaping/runners/test_fouriergsnet_optimize.py -q
```

## 5. 相关文件

| 文件 | 作用 |
|------|------|
| `tests/ao_shaping/drivers/sim/test_sim_fouriergsnet_pipeline.py` | 5 个集成测试 |
| `fouriergsnet_optimize.py` | 被测独立脚本 (L938 bug, L814 无信号, L905 _metrics) |
| `src/ao_shaping/drivers/sim/fouriergsnet_env.py` | 数字孪生环境 (configure_turbulence L282, advance_time L320) |
| `src/ao_shaping/utils/io/file.py` | Recorder (L221) |
| `scripts/fouriergsnet_sim_train.py` | 场景矩阵离线训练 (同一 no_grad workaround) |
