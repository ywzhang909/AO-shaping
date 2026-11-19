# Model-in-loop 跨光路交接手册

本手册用于 Santec SLM 与项目支持的远场 CCD。每次换光路、功率、衰减片、相机或入射位置，都运行一次新的会话。`model_in_loop_auto.py` 调用现有 `slm-model-in-loop`，由它执行全板入射定位、当前画面曝光探测、sweep 几何标定、逐轮相差重拟合和候选验收。旧的 `bench_geometry.json` 只用于复现，不是新光路的默认输入。自动闭环和固定曝光重放目前在 Daheng 上有实验记录；换其他相机时须先验证该驱动的曝光回读、原始帧量程和固定曝光接口。

## 1. 准备光路配置

复制 `scripts/slm/model_in_loop_profile.example.json` 到本地配置文件，逐项核对：

| 字段 | 如何填写 |
| --- | --- |
| `cam_type`, `cam_id` | 当前可被 `create_camera` 打开的相机类型及编号；本示例是 Daheng #0 |
| `slm_number`, `slm_wavelength` | 现场设备编号与实际激光波长；先确认 SLM LUT 与波长匹配 |
| `target_shape`, `target_side` | 所需形状和相机像素尺度；支持方形、矩形、三角、星形、圆形 |
| `region`, `far_field_padding` | 模型瞳面网格及远场补零倍数；改变 `region` 后必须重新标定 |
| `panel_span_px` | 本次探针覆盖的 SLM 光斑宽度估计，单位面板像素 |
| `camera_pixel_um`, `slm_pixel_um` | 两个设备当前的物理像元间距；不要沿用另一台相机的数据 |
| `exposure_time_ms`, `max_exposure_ms` | `exposure_time_ms` 取三态之一：`auto` = 由本次画面探测，`default` = 保持设备当前曝光，正数 = 当前光路已实测的曝光（ms）。**裸 `0` 会被拒绝**（`--exposure-time-ms` 报 `0 is not a synonym for 'auto'`）。`max_exposure_ms` 是本次会话的曝光上限（ms），`0` = 自动探测 |
| `max_cv`, `min_ee`, `min_iou` | 由应用确定的三项验收边界；示例值不是通用性能保证 |

`w_efficiency` 必须保持正值，单纯压 CV 可能把目标区域变暗。自动脚本不接受 `geometry_file`，避免无意复用旧会话几何。先确认 SLM 和相机未被其他进程占用，激光与衰减片在一次扫描及闭环期间保持稳定；若功率变化，结束本轮后重新运行。这个约束是测量可比性的条件，不是把某个历史曝光档设成永久设备上限。

## 2. 离线预览

在仓库根目录执行：

```powershell
python scripts/slm/model_in_loop_auto.py path/to/bench_profile.json --dry-run
```

检查输出命令中的设备编号、波长、目标像素尺寸、曝光策略与网格。`--dry-run` 只校验配置与打印命令，不打开设备。尚未针对当前相机证明短曝光线性时，不要把微小的 CV 差异当成算法增益。

## 3. 实验运行

```powershell
python scripts/slm/model_in_loop_auto.py path/to/bench_profile.json
```

脚本串行调用主项目 CLI。它会先写 `data/debug/slm_model_in_loop/<时间>/run_config.json`，再保存 `panel_scan/`、`session_geometry/`、探针、逐轮 CSV、最优相位和 CCD 证据。扫描响应不足、平场漂移、几何相关性或验证集相位相关性不达门槛时，立即查看该目录的拒绝原因；不要把失败会话的相位用作新光路起点。运行结束还会生成 `handoff_summary.json`。

`handoff_summary.json` 分别检查相位产物、已验收轮次、相对平场的分数、CV、EE、IoU、几何和验证集相关性。`passes_recorded_gates` 只表示保存的逐轮记录过了配置门槛，**不是最终光学验收**；该 CSV 不保证其中最优分数行就是最后保存相位的固定曝光指标。

### 3.1 一次跑完 A/B 对照与大面积目标

需要「验证集门槛不动」与「强制放开门槛」两个变体对照、以及 1000–1600 相机像素的大面积正方形目标时，用编排脚本（它复用上面的 `model_in_loop_auto.py` 入口，并在每个硬件阶段前加一道测光门禁）：

```powershell
python scripts/slm/model_in_loop_hw_ab.py --plan      # 只打印计划，不开设备
python scripts/slm/model_in_loop_hw_ab.py             # light → baseline_a → baseline_b → large_1000 / large_1600
```

各阶段的期望先写进 manifest 再执行，产物在 `data/debug/model_in_loop_hw_ab/<时间戳>/`（`manifest.json` + `summary.md` + 各阶段 `stage.log`）。测光判定 `no_light` 直接停，`weak` 默认停（`--allow-weak` 放行）。大面积变体会把相机窗口放大到目标边长 +100 px，目标框必须落在读出窗口内。

## 4. 固定曝光复核与交接

要比较两次会话或一个候选与历史相位，先确认两者 `run_config.json` 中相机、SLM、目标、相位扰动与入射位置一致，然后执行现有交错重放：

```powershell
python -m ao_shaping.tools.slm.model_in_loop_replay data/debug/slm_model_in_loop/<A> data/debug/slm_model_in_loop/<B> data/debug/slm_model_in_loop_replay/<本次对照> --fixed-shaped-exposure-ms <当前成形曝光> --fixed-flat-exposure-ms <当前平场曝光>
```

目前固定曝光重放仅实现 Daheng，相机不同则使用各自的原始帧与曝光回读做同曝光对照，不能直接跨曝光比较 CV/EE。重放会拒绝配置不匹配或固定曝光超过当前会话上限；若两帧太暗、削顶、平场阶跃或时间漂移明显，调整当前光路后重测。最终验收同时看 CV、EE、IoU、原始峰值、曝光回读、零级附近掩码及平场漂移；剔除零级后的 CV 属补充指标，不替代完整目标区域指标。

失败恢复顺序：确认设备占用和光路稳定性；检查 `panel_scan/` 的响应与漂移；检查 `session_geometry/bench_geometry.json` 的相关度、焦面尺度一致性和束腰备注；检查探针验证集相关性；最后再调目标尺寸、网格或 Step B 参数。几何相关性合格却有明显尺度冲突时，先独立验证几何，不应继续解释整形增益。没有证据表明当前物理预测模型能描述新光路时，可运行 `slm-gs-refine` 作为基线。

## 5. 本次实验的边界

2026-10-16 当前光路下，两轮热启动在固定 0.200 ms 重放中使目标框 CV `0.5627 -> 0.4454`，EE `0.4251 -> 0.4438`。软边缘 CV 改动在固定 0.18 ms 重放中使 CV `0.6775 -> 0.5937`，EE 基本持平，但 IoU `0.1933 -> 0.1508`。两组都没有证明形成均匀实心方斑，后一组还伴随较低原始计数和几何尺度冲突。详见 [实验报告](../../report/shape_trials/model_in_loop/model_in_loop_uniformity_live_20261016.md)。
