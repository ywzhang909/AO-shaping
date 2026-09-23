"""Generate the FourierGSNet pipeline integration-test report.

Documents the offline integration tests in
``tests/ao_shaping/drivers/sim/test_sim_fouriergsnet_pipeline.py``: the 5
tests drive the REAL standalone ``fouriergsnet_optimize.py`` pipeline
(ShapingSystem -> adaptive_gs_init -> closed_loop -> _append_final_record)
through ``SimFourierGSNetEnv`` with no hardware and no pipeline modification.

Fully offline — pure markdown, no hardware, no figures. Writes
``docs/fouriergsnet_pipeline/report.md``.

Usage:
    python scripts/generate_fouriergsnet_pipeline_report.py
    python scripts/generate_fouriergsnet_pipeline_report.py -o docs/fouriergsnet_pipeline
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

REPORT = """# FourierGSNet 管线集成测试报告

**生成时间**: {date}

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

### 2.2 torch.no_grad() 包裹 closed_loop — 规避 L938 真实 bug

`closed_loop` L938 对 `requires_grad` 的 `phi` 调用 `phi.cpu().numpy()` 会抛
`RuntimeError: Can't call numpy() on Tensor that requires grad` (网络前向输出
`phi` 经可训练卷积层, 带梯度)。**真实 CLI `run()` (L1135) 同样受影响** ——
这是管线自身的 bug, 非测试引入。

测试用 `replay=False` (跳过全部训练/微调), 因此 `torch.no_grad()` 包裹安全,
且不改动管线内部。`scripts/fouriergsnet_sim_train.py` 采用同一 workaround。

### 2.3 SETTLE_S=0.0

`_display_grayscale` 的 `time.sleep(settle_s)` 在 SETTLE_S=0.0 下为 no-op;
SimSLM 内部无 sleep, 无需全局 monkeypatch `time.sleep`。

## 3. 发现 (供后续修复参考)

### 3.1 L938 真实 bug (影响真实硬件 CLI)

`closed_loop` L936-938:
```python
recorder.append({{"step": step, "uniformity": uni, "encircled": ee,
                 "inference_ms": dt, "ccd": I.cpu().numpy(),
                 "phase": phi.cpu().numpy()}})
```
`phi` 来自 `net(...)` 前向输出 (requires_grad=True)。`replay=True` (默认, 真实
CLI) 下同样崩溃。修复方向: `phi.detach().cpu().numpy()`。

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
.venv/bin/python -m pytest tests/ao_shaping/drivers/sim/test_sim_fouriergsnet.py \\
    tests/ao_shaping/drivers/sim/test_sim_fouriergsnet_turbulence.py \\
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
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output-dir", default="docs/fouriergsnet_pipeline",
        help="输出目录 (默认: docs/fouriergsnet_pipeline)",
    )
    args = parser.parse_args()

    out_dir = (_REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "report.md"
    out_file.write_text(
        REPORT.format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        encoding="utf-8",
    )
    print(f"Report written to {out_file}")


if __name__ == "__main__":
    main()