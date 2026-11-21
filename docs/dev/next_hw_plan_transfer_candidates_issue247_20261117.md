# 下次在线窗口上机计划：Transfer K 候选验证 + #247 零级校正（编排于 2026-11-17）

> **📌 状态以合并版为准**：本文是 K 候选 + #247 的原子计划，已被
> [`hw_plan_consolidated_20261118.md`](hw_plan_consolidated_20261118.md) 汇编为 2f 整形唯一入口。
> 完成标记、设备段取舍与执行顺序以合并版 §1–§4 为准；本文保留作任务细节依据。
> 关键变化：K 代码/K=1 逐位一致/GPU 微基准已完成，K=4 真闭环待上机；#247 闭环受决策门 D2（deswivel 离线回验）门控。

> 本轮（2026-11-17）设备采集已全部完成并释放，设备即将下线。本文把剩余上机工作编排成
> 下次在线窗口可直接执行的清单。所有离线部分已完成；本文只列**需要设备**的动作与前置。

## 已完成（本轮，离线可查）

- 光路探针 ok（曝光 0.05 ms 未饱和）：`data/debug/optical_path_probe/transfer_parallel_timing_20261117/20261117_131421Z/optical_path_record.json`
- P1 定心 + P2 双幅度诊断 + 真线性 ±tilt 补采，全部正常释放（详见报告）
- #247 判读：零级稳定、目标级偏转、**+88.54° 旋转配准**（见 `report/shape_trials/issue247_zero_order_20261117/`）
- Transfer `--candidates K` 批前向落地 + GPU 微基准（见 `report/shape_trials/transfer_candidates_timing_20261117/`）

## 前置检查（每次上机必做，任一失败即停）

1. `DeviceLock.default().holder()` == None；无竞争硬件进程（**不要**在采集期跑含
   `Santec SLM` / `tilt` / `defocus` 等关键词的长查询——曾误判为硬件进程致 census 延迟 84 s）。
2. 六步光路探测重跑一遍（曝光不得超量程、SLM 相位自检、A=0 焦点确认），见
   `libs/skills/ao-optical-path-calibration/SKILL.md`。
3. `require_drive_board()` + `serial_source()=="driveboard"`；`set_bit_depth(12)` 与
   `effective_full_scale` 双重确认 Mono12。
4. 三段之间冷却 ≥3 min（背对背 open/close 会触发驱动板假死 `ReadSU=-1000`）。

## 任务 A：Transfer `--candidates K>1` 真实闭环验证（优先，短）

目的：只验证 K>1 在真实 SLM/CCD 上跑通（**不是**测加速——优化器本体占端到端 <0.5%）。

```powershell
# 一次 open/close，K=1 与 K=4 各一次（段间冷却 200 s，由窗口脚本负责）
.venv\Scripts\python.exe scripts/experiments/run_transfer_window_20261119.py `
  --window data/experiments/transfer_window_20261119 `
  --segments k1 k4 --shapes square --side 1200
# 加 --execute 才占用设备；段表/冷却/锁/证据见 scripts/experiments/_window.py

# 数据落 data/experiments/transfer_window_20261119/{k1,k4}/；
# 2026-11-19 状态：脚本就绪，未采集（见合并版 §7.3）。
```

记录：墙钟、每步 loss、`current_phase` 2D 形状、是否 0 饱和、released.json 全 true。
判据：K=4 全程无异常、结果与 K=1 同量级；端到端耗时仅作量级记录，不作加速结论。

## 任务 B：#247 零级校正闭环（需先补一个标定）

**关键前置**：本轮已确认面板轴↔CCD 轴是 **+88.54° 旋转**（非写死 90°，含 1.086 轴间尺度比），
但**数值比例因子尚缺**——tilt 探针 period=128 panel px，要把目标位移换算成 panel→CCD 的
比例，需要 panel pitch / CCD pixel pitch / 焦距等标定常数，不能凭单次 tilt 数据猜测。

下一步离线待办（不占设备）：

1. 从现有 ramp_matrix 拟合 JSON / 光路标定记录中取出 panel→CCD 的几何与放大率；
2. 用本轮 tilt 数据（目标半位移：旋转对齐帧 tilt_x 54.54 px / tilt_y 59.28 px，跨轴 ≤0.12 px）
   校验 +88.54° 旋转映射的方向与量级；
3. 在 `FrozenDefocusForward` 的 ramp_matrix 里落实旋转映射（含轴间尺度比），用 13 帧离线回验
   预测位移方向。

完成上述**离线**校验后，再上机做零级校正闭环：

- 施加 panel-x / panel-y 斜坡，闭环目标是"零级中心保持 (870,1255)（对齐帧）不动、目标级被引导到指定位置"；
- 判据：零级中心漂移 ≤ 2 px（本轮 flat 本底）且目标级按预测方向移动；
- 本轮**不启用**硬件零级校正——模型映射未对齐前贸然校正会把 panel-x 校正加到 CCD-y 上。

### 任务 B 状态补充（2026-11-20，纯离线复核）

用当前跟踪的 `scripts/slm/issue247_zero_order_stability_20261117.py` 对同一归档 session 重放：

```powershell
.venv\Scripts\python.exe scripts/slm/issue247_zero_order_stability_20261117.py --session data/debug/issue247_tilt_20261117 --out C:/Users/zhangh/AppData/Local/Temp/opencode/issue247_20261120_replay.json
```

命令输出 13 captures、0 saturated；旋转角 **+88.543997°**，panel-x/y 目标半位移分别约 **54.54 / 59.28 px**，跨轴分量均约 **0.12 px**，orthogonality error **0.2435°**、scale ratio **1.085895**；aligned shape 为 `2641×2010`。该次复现与 2026-11-17 的 `issue247_analysis.json` 一致。实际 JSON 在 `C:/Users/zhangh/AppData/Local/Temp/opencode/issue247_20261120_replay.json/issue247_analysis.json`（脚本把 `--out` 视为目录）。未覆盖归档输入。

但 `data/debug/issue247_analysis_rot_20261118/issue247_analysis.json` 对同一 session 记录 **86.938333° / 1.170798 / 7.0970°**，panel 位移向量与 aligned shape 也不同（`2693×2080`）。在仓库内未找到生成该结果的脚本、精确命令或输入变换记录；两份 JSON 的峰值计数一致，但其间处理口径不可追溯。因此：**88.544° 是当前跟踪脚本可复现的结果，不等于两份历史分析的分歧已解释，也不得据此冻结物理映射或写入 `ramp_matrix`**。

新增纯离线待办：① 找回/登记 11-18 rot 结果的生成代码、命令和输入哈希；若不可恢复，将其标为不可复现探索结果，不与规范结果混用；② 对同一原始输入用唯一受控脚本逐步对照两种处理，说明角度、尺度及画布差异；③ 再核对 panel pitch、CCD pitch、焦距/放大率和 ramp 符号约定。缺任一依据时，保留“不应用于 `FrozenDefocusForward.ramp_matrix`”的停止线。完成前不开展依赖该几何常数的硬件零级校正；任何新增设备采集仍须单独明确批准。

#### 任务 B 离线溯源复核（2026-11-20 补充）

- 11-17 与 11-18 rot 两份分析 JSON 的 13 组峰值/饱和计数相同；归档 13 个 `.npz` 与 `.slim/worktrees/sim2f-model-correction` 副本逐个 SHA-256 一致，后者不是独立重复。11-18 结果于 05:24 写入，Temp 中 `issue247-bgfix-run` 于 05:30 只剩与 11-17 口径相同的输出，仓库分析脚本/JSON 于 05:32 更新；因此时间顺序支持同一数据的中间/修正分析，但“bgfix”只是目录名，不能据此断言具体修复内容。
- 当前跟踪脚本已在本轮再次只读重放至 `C:/Users/zhangh/AppData/Local/Temp/opencode/issue247_20261120_replay_run/issue247_analysis.json`，返回 13 captures、0 saturated、`correction_deg=88.5439972581`、`scale_ratio=1.0858949312`、`orthogonality_error_deg=0.2434973656`，与 11-17 结果一致。11-18 rot 的 `86.938333° / 1.170798 / 7.0970°` 仍无可追溯生成代码、命令及输入变换记录；当前归档最多支持在受控脚本口径下的经验几何描述，不足以冻结物理映射或关闭 #247。

#### 任务 B 对称性与可辨性复核（2026-11-20 追加，纯离线）

用归档 13 帧、`_pair_displacement` + `estimate_axis_rotation` 直接重算（未改任何仓库文件）：

- **重复性**：5 次重复的逐帧角度 `[88.628, 88.620, 88.548, 88.504, 88.415]°`，SD **0.088°**。88.544° 是统计上稳定的测量值，不是噪声。
- **标签交换被拒**：交换两轴标签时抛 `AxisRotationError: mirror axis pair`，不会静默给出互补角（设计保护有效）。
- **180° 不可分辨（关键）**：把 (vx, vy) 同时翻号后同一估计量给出 **−91.457°**。数据本身无法区分 88.543° 与 −91.457°，符号分支只由 `image_alignment.convention` 字符串钉住，没有任何独立观测（旋转标定板 / 正交光栅双向扫描）验证该分支。
- **几何常数仍缺**：`data/slm_geometry_calib.npz` 只含波前拟合的 center `[1104.84, 1224.42]`、`Kx/Ky`、`rotation 0.1675°`、`fwhm0 16.0`（来源 `persist_live_geometry.py`），**不含** panel pitch、CCD pitch、焦距/放大率，也**不含** ramp 符号约定。§1 的前置条件因此仍未满足。
- **可微零级校正确实未实现（对物理预测成立）**：`src/ao_shaping/utils/wavefront/zero_order_overlay.py` 为死代码（src/scripts/tests/report 全库零调用、零测试）；`FrozenDefocusForward` 全文无零级项；唯一可微零级模型在 sim2f 代理（`src/ml/sim2f/optics.py` 的 `OpticsErrors.zero_order_eta`），从未接入物理 transfer 前向，也未对 #247 数据做过数据驱动 eta 估计。
- **测试覆盖现状**：合成模板恢复（`test_issue247_zero_order_rotation.py` + `utils/image/test_rotation.py`）与无校准逐位兼容实测 **18 passed**，判定力高；但梯度一致性单测因缺 `data/debug/physical_forward_defocus/uniformity_refit_20261022_r1` 被 **skip（13 skipped）**，batch/vmap 一致性**完全没有测试**。可用 `data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/checkpoints/M3_illumination/frozen_parameters.json`（grid=128/fft=512）补一个可跑的梯度单测——当前未被任何测试引用。

结论：**88.544° 作为受控脚本口径下的可复现测量值成立，但不是已锚定的物理映射**；不冻结、不写 `ramp_matrix`、不据此开展硬件零级校正。

#### 几何采集脚本（2026-11-20，离线新增，未上机）

新增 `scripts/slm/issue247_geometry_capture_20261120.py`（plan-only 默认；`--self-test` 无设备自检；`--execute` 才开设备）+ `tests/ao_shaping/scripts/test_issue247_geometry_capture_20261120.py`。目的：一次采集同时给出（a）已知面板周期 → panel↔CCD 每轴比例，（b）非中心对称靶打破 180° 分支，（c）同场复核 tilt 符号/手性，（d）datasheet 常数与实测值分离。

- **图样表（11 图样 / 44 帧，schema `issue247-geometry-capture/v2`）**：`flat`(4)；`grating_{x,y}_p128`、`grating_{x,y}_p256` 各 4（0/π 二值、50% 占空比、灰阶 `{0,512}`，双周期互检斜率/一致性）；`ladder_x`/`ladder_y` 各 4（两个 600×600 blaze patch，`2π/96` → +1 级、`2π/32` → +3 级，远场 spot 集 `[0,+1,+3]` 单侧、非镜像/非 180° 不变）；`tilt_{x,y}_{plus,minus}` 各 4（真实棱镜斜坡 `2π/period_px`、居中零活塞、逐点反对称）。
- **tilt 幅度**：旧版 `TILT_AMPLITUDE_RAD=0.5` 全跨仅 1.0 rad（≈ period 12063.7 px），远场位移比可测小约 190 倍，已在 `raw_tilt_feasible` 中记为 `rejected_weak_design`。新版默认 `--tilt-period-px 128.0`（与 11-17 脚本一致）；短轴（1200 行）为保 ≥10 条纹钳到 120 px，plan 同时记 `requested_period_px` 与 `period_px`。
- **可行性算术（悲观三元组 pitch 20 µm / focal 100 mm / CCD 3.45 µm / λ=1064 nm）**：x 轴 period 128 → 跨度 94.199 rad、15 条纹、12.05 CCD px；y 轴 period 120 → 跨度 62.779 rad、10 条纹、12.85 CCD px；`--self-test` 要求悲观位移 ≥5 px 且条纹 ≥10。
- **仍缺物理常数**（plan `missing` 明列，未猜测）：`declared.camera_pixel_size_um`、`declared.slm_pixel_pitch_um`、`declared.focal_magnification`；`declared.tilt_expected_ccd_px` / `declared.ladder_expected_ccd_px` 仅以符号公式记录。`--execute` 时从驱动读取 pitch/pixel size，读不到即写 `null` + `missing`，不改 `data/slm_geometry_calib.npz`。
- **验证（离线）**：`test_issue247_geometry_capture_20261120.py` 16 passed；`test_script_imports_resolve` + `test_script_file_form_imports` 915 passed；`py_compile`/`ruff`/`--self-test`/`--help` 全 0；plan-only 不建目录；无 BOM。
- **硬件前置**：需单独明确授权；唯一新 `--out`、`mkdir(exist_ok=False)`、`DeviceLock` + `--wait-until-idle`、固定曝光 + 饱和保护（`SaturationAbort` 记录原因后正常释放）、写 `optical_path_record.json`(`ao-optical-path-record/v1`)/`released.json`、`finally` 中 `set_grayscale(0)→cam.close→slm.close→lock.release`。summary 仅描述性统计，不做旋转拟合/阈值判定。

## 代码准备状态（均已就绪，勿重复造）

| 用途 | 文件 | 状态 |
|---|---|---|
| K 候选批前向 mixin | `src/ao_shaping/algorithm/signal_processing/differentiable_candidates.py` | 新增，ruff/golden 通过 |
| n_candidates 主战场 | `differentiable_defocus_uniformity.py` | 已改，K=1 逐位一致 |
| CLI --candidates | `tools/slm/transfer/workflow.py` / `validate.py` | 已改 |
| K 候选测试 | `tests/ao_shaping/algorithm/test_differentiable_candidates.py` | 新增 7 测试通过 |
| tilt 补采 | `scripts/slm/issue247_capture_tilt_20261117.py` | 就绪，本轮已用 |
| #247 离线分析 | `scripts/slm/issue247_zero_order_stability_20261117.py` | 就绪，物理口径已修正 |
| GPU 微基准 | `scripts/slm/transfer_candidates_gpu_bench_20261117.py` | 就绪，本轮已用 |

**红线**：不改任何 MIL 文件（`model_in_loop_shaping.py` / `slm_model_in_loop.py` /
`shaping_controls.py` / `test_model_in_loop_shaping.py`）——本轮已确认全部未动。

## 设备纪律

任何上机动作前后在回复中明确报告：占用设备（SLM#/CCD/激光）、做了什么、何时释放、是否正常关闭。
