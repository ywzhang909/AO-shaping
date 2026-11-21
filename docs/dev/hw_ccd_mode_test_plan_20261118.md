# ccd_mode（light/none）实验测试计划 — 2026-11-18

> **📌 状态以合并版为准**：本文是 ccd_mode light/none 的原子计划，已被
> [`hw_plan_consolidated_20261118.md`](hw_plan_consolidated_20261118.md) 汇编为 2f 整形唯一入口。
> 完成标记、设备段取舍与执行顺序以合并版 §1–§4 为准；本文保留作判据细节依据。
> 关键变化：实现 + 13 项离线测试已就绪，light/none 实验段已并入合并版 S2/S3/S4。

配套实现：`TransferSession.ccd_mode`（full/light/none），CLI `--ccd-mode`。
本计划仅在下一次设备窗口执行；执行前必读 [hardware_discipline.md](hardware_discipline.md) 与光路标定 SKILL。

## 目标

1. **light（少CCD）**：验证复用探针 + 每块一次实测锚点 + 终检 A–B–B–A 的质量损失是否在可接受范围（对照 full 的历史基线）。
2. **none（无CCD开环）**：验证纯模型相位上屏的远场形态，对照 anchored_m3_20261116 的 model_cv 0.66 vs 实测 CV 5.01 假收敛量级，确认开环实测 CV 落在已知区间。

## 前置（一次）

- 光路探测 + 曝光标定（六步探针，写 `optical_path_record.json`）。
- 确认存在可复用的 dataset + checkpoint（最近一次 full run，光路未动）。**light/none 必须显式传 `--dataset` 与 `--checkpoint`**：`--reuse-latest` 不会为这两个模式解析路径（run_session 先 fail-fast，避免静默退化为采集+拟合）。若没有可用模型，先跑一次短 full，再进入下表。

## 采集序（一次实验、一次 SLM/CCD open/close 串行，段间冷却 ≥200 s）

| 段 | 命令要点 | 时长估计 | 记录 |
|----|---------|---------|------|
| L1 light | `--ccd-mode light --dataset <latest>/capture --checkpoint <latest>/fit/frozen_parameters.json --shapes square --side 1200`（或当日目标） | ~3 min | session 目录（resolved.acceptance=`paired_abba`） |
| 冷却 | SLM 关闭 ≥200 s，确认 `DeviceLock.default().holder()` 为 None | 200 s | 时间戳 |
| N1 none | `--ccd-mode none --dataset <同上> --checkpoint <同上> --shapes square --side 1200`（同目标同种子） | ~2 min（无 CCD 段） | session 目录 + open_loop_display.json（acceptance=`open_loop`） |

> **执行入口（2026-11-19 起）**：L1/N1 不再手敲两段，统一交给窗口脚本
> `scripts/experiments/run_transfer_window_20261119.py --segments k1 k4 light none`
> （它等锁、持锁贯穿全窗口、段间冷却 200 s、每段独立 `--out`、结束时写 `released.json`/`window.json`）。
> 数据落 `data/experiments/transfer_window_<date>/{k1,k4,light,none}/`；无 `--execute` 只打印计划。
> 状态：**脚本就绪，未采集**（见合并版 §7.3）。

验收记录（每段）：SLM/CCD 打开时间、关闭时间、`slm_closed/camera_closed/lock_released`、exit code。

## 离线判读（设备下线后）

1. light vs full 基线：终检 A–B–B–A 的实测 CV / IoU 与同形状同尺寸历史 full run 对比；可接受判据 = 实测 CV 退化 < 20% 相对。
2. none：无实测帧（不开相机），判读只能依赖随后一次 full/light 运行以 none 相位为种子？——否。none 段本身只记录上屏时间戳；none 的实测质量需另设「none 相位 + 单次 CCD 拍帧」对照段（N2，可选）：用 `transfer validate` 手动上屏 none 合成相位拍 3 帧。是否执行 N2 在窗口内视时间决定。
3. 耗时拆解：light 相对 full 应省 CCD_model_blocks（~80 s）与 anchored 多余 updates（每块省 4 次采帧）。
4. **模型科学强度门（独立于本页工程模式验收）**：冻结模型的零步绝对强度图在固定 ROI/mask 上，以未缩放 `relL2` 评估；所有有效独立 pass 均须 `relL2 ≤ 0.20`，并与既有方向门 10/14 联合通过。预注册口径、无效条件及 `δ=0.20` 的适用限制见 [`transfer_strength_gate_preregistration_20261119.md`](transfer_strength_gate_preregistration_20261119.md)。本页 `light` CV 退化 `<20% relative` 是另一项应用质量判据，不等价于 `relL2`，不得互相替代。

## 红线

- 任何饱和 → 当段 abort，不重试。
- 背对背 SLM open/close 禁止；冷却不满不进下一段。
- 驱动板假死（ReadSU=-1000）→ 停止实验，手动重启控制器，当日不再续。

## 不做

- 不改 MIL 相关文件；不并入口径到 GUI（runner 暂不暴露 ccd_mode，CLI 先行）。
