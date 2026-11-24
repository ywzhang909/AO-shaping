# 真机光路复用门试跑计划（2026-11-24）

**状态：离线已验证并提交（2c873fe）；等待设备窗口（SLM #1 + CCD 空闲）。**

## 1. 目的与边界

- 目的：在真实硬件上端到端验证 `scripts/slm/optical_path_reuse_gate.py`，确认它能在一次打开周期内给出可信的 `optical_path_reuse_gate.json` 与退出码，为后续实验「跳过完整 Step 3 阶梯」提供依据。
- 边界：这是**工具验证**，不是实验采集，不产生用于归因/冻结的图样数据；不写映射/标定/模型产物；不改 issue #247 冻结归因状态；试跑结果只用于判断复用门本身可用性，不代表任何物理结论。

## 2. 前提（全部满足才可开设备）

1. 复用门已提交（`2c873fe`），离线 pytest 112 passed、ruff 通过。
2. 存在**当天**（2026-11-24）`verdict=ok` 探针参照：`data/debug/optical_path_probe/issue247_reproducibility_20261124/20261124_045821Z/optical_path_probe.json`（配套标准记录与平场帧在同一目录）。
3. SLM #1 / 1064 nm / cam 0（daheng）空闲；执行前报告设备占用，并 `--wait-until-idle`。
4. 遵守 `docs/dev/hardware_discipline.md` 与六步光路预检纪律；一次 SLM open + close，不自动重试。

## 3. 执行

- 命令（`--run-dir` 为全新唯一目录；`--cam-type` 以参照为准，参照用 `daheng`）：
  ```
  python scripts/slm/optical_path_reuse_gate.py \
    --out-root data/debug/optical_path_probe \
    --run-dir data/debug/optical_path_reuse_gate_trial_20261124 \
    --slm-number 1 --wavelength 1064 --cam-type daheng --cam-id 0 \
    --wait-until-idle 600
  ```
- 只在 SLM 关闭、锁释放后检查结果；不自动重试。

## 4. 通过判据（试跑本身）

- 退出码 ∈ {0, 3}，且与 `reuse_gate`（passed/failed）一致。
- `<run-dir>/optical_path_reuse_gate.json` 存在，schema `optical_path_reuse_gate/1`；`closed_normally=true`。
- `passed`：`exposure_check.passed=true`（峰值 ≤ 0.7×满量程、0 饱和像素），`devices.drive_board_id` 与参照一致，criteria 值齐全。
- `failed`：`reasons` 非空且可行动（缺记录 / 参照非当天 / 驱动板 ID 不符 / 饱和校验失败 / 任一异常）。
- **两种结果都可接受**——`failed` 同样证明门能 fail-closed；但须记录原因并在报告中说明。
- 不得把「试跑通过」当作「本次光路有效」的证据；光路有效性仍由完整 Step 3–4 或复用门当次结果决定。

## 5. 后续动作

- 试跑合法（JSON 合法、退出码一致、设备正常释放）→ 报告结论；后续实验可正常启用复用门跳过完整阶梯。
- 试跑异常（崩溃 / 挂起 / 未释放 / JSON 非法）→ 先修复 helper 再试；未验证的复用门不得用于正式实验。
- 当天 Fresnel D / 2π 灰度仍按 skill 手工补录；`fresnel_manual_required=true` 时当天 D 必须补录。
