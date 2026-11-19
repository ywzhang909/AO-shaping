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
